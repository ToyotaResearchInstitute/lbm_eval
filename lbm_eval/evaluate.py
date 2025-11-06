"""API and simple main program to evaluate a policy.

See the README.md section "Evaluating using your own policy wrapper" for how to
write a wrapper around your policy that's compatible with this benchmark.
"""

import argparse
from collections import deque
import dataclasses
from datetime import datetime
import functools
import gc
import io
import json
import logging
from multiprocessing import Lock, Pool, set_start_method
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable

from grpc_workspace.lbm_policy_client import LbmPolicyClientConfig
from robot_gym.policy import Policy
import tqdm

from pydrake.common import configure_logging as configure_pydrake_logging
from pydrake.common.yaml import yaml_load

# When running in open-source mode, we need to set a path var.
try:
    import python.runfiles as _ignored  # noqa: F401
except ImportError:
    # N.B. This branch is only tested in the nightly build, not pre-merge.
    lbm_eval_dir = Path(__file__).parent
    anzu_dir = lbm_eval_dir.parent / "anzu"
    os.environ["ANZU_NUC_RUNFILES"] = str(anzu_dir)

from anzu.common.anzu_model_directives import MakeDefaultAnzuPackageMap
from anzu.intuitive.typing_ import from_dict
from anzu.intuitive.visuomotor.bases import (
    GymEnvWrappingAnzuEnv,
    NoopRecorder,
    closing_multiple,
    collect_episode_gym,
)
from anzu.intuitive.visuomotor.demonstration_seed import get_demonstration_seed
from anzu.intuitive.visuomotor.multiarm_simulations import (
    HardwareStationScenarioSimulationEnvConfig,
)


@dataclasses.dataclass
class SingleEvaluationResult:
    """Information about a single simulation."""

    skill_type: str
    """The task being performed."""

    scenario_index: int
    """An index (starting from zero) that influences the random_seed.
    Also known as 'demonstration index'."""

    is_pending: bool = True
    """Whether the evaluation is waiting for execution (True) or has completed
    (False)."""

    total_time: float | None = None
    """Total elapsed simulation time at some (unspecified) time during the
    step in which the simulation succeeded, crashed, or timed out. None if
    is_ending is True."""

    is_success: bool | None = None
    """Whether the evaluation successfully completed the skill. None if
    is_pending is True."""

    failure_message: str | None = None
    """If the evaluation was not successful, this field might contain an
    unstructured explanation of the failure."""


@dataclasses.dataclass
class EvaluationResults:
    """Information about all evaluations."""

    evaluations: list[SingleEvaluationResult] = dataclasses.field(
        default_factory=list,
    )
    """The result details for all evaluations."""

    num_processes: int = 0
    """The number of processes being used for evaluation."""

    elapsed_time: float = 0.0
    """The (wall clock) time spent so far for evaluation."""

    @property
    def num_requested_evaluations(self) -> int:
        return len(self.evaluations)

    @property
    def num_evaluated(self) -> int:
        return sum([not x.is_pending for x in self.evaluations])

    @property
    def num_success(self) -> int:
        return sum([bool(x.is_success) for x in self.evaluations])

    @property
    def success_rate(self) -> float:
        numerator = self.num_success
        denominator = self.num_evaluated
        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator


def _configure_logging() -> None:
    """Configures program-wide logging settings."""
    # Add some tasteful defaults (formatting, etc).
    # Beware that this has different effects in Anzu vs an lbm_eval wheel,
    # because anzu/__init__.py nerfs pydrake's logging redirection.
    configure_pydrake_logging()
    # Don't show INFO (there's too much of it).
    # Don't show WARNING (the messages there aren't actionable).
    logging.getLogger().setLevel(logging.ERROR)


def _get_json_path(output_directory: Path) -> Path:
    """Returns a path to a JSON file for storing evaluation results."""
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    json_file = f"results-{timestamp}.json"
    json_path = output_directory / json_file
    return json_path


def _save_json(
    *,
    results: EvaluationResults,
    path: Path,
) -> None:
    """Saves the evaluation results to a JSON file.
    All records (pending or not) are written to the named file.
    """
    # Convert all dataclass fields to plain dicts (recursively).
    json_data = dataclasses.asdict(results)
    # Add the computed @property values.
    for name, value in EvaluationResults.__dict__.items():
        if isinstance(value, property):
            json_data[name] = getattr(results, name)
    # Move the raw evaluation details to the end of the json.
    evaluations = json_data.pop("evaluations")
    json_data["evaluations"] = evaluations
    # Write the json output transactionally.
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    temp_path.rename(path)


class _LastStepRecorder(NoopRecorder):
    def __init__(self):
        self.last_time_step = None

    def record_step(self, time_step_prev, act, time_step):
        self.last_time_step = time_step


def _get_known_skill_paths() -> dict[str, Path]:
    """Returns a mapping from all skill_types within our known packages to the
    Path for the skill yaml file. Packages that contain skills must have a file
    named "skill_filenames.txt" at their root, containing a newline-separated
    list of skill filename relative paths.
    """
    result = dict()
    package_map = MakeDefaultAnzuPackageMap()
    for package_name in package_map.GetPackageNames():
        base = Path(package_map.GetPath(package_name))
        inventory = base / "skill_filenames.txt"
        if not inventory.exists():
            continue
        for line in inventory.read_text(encoding="utf-8").split():
            config_file = base / line
            skill_type = config_file.stem
            result[skill_type] = config_file
    return result


def get_known_skill_types():
    """Returns a list of all skill_types within our known packages."""
    return list(_get_known_skill_paths().keys())


def evaluate_one(
    *,
    skill_type: str,
    scenario_index: int,
    output_directory: Path,
    use_eval_seed: bool = True,
    t_max: float | None = None,
    server_uri: str = LbmPolicyClientConfig().server_uri,
    use_rpc: bool = True,
    policy: Policy = None,
) -> SingleEvaluationResult:
    """Perform a single evaluation of a policy against a scenario.

    The `skill_type` specifies which skill (aka "task") to evaluate.
    Refer to `get_known_skill_types()` to learn the valid choices.

    The `scenario_index` (typically starting from zero and counting up)
    influences the random_seed used to choose the initial conditions.
    Also known as 'demonstration index'.

    The `output_directory` must be set to an existing directory path. Log files
    and temporary files will be written into a subdirectory of this path during
    evaluation, based on the `skill_type` and `scenario_index`.

    The `t_max` specifies how much simulation time is allowed for the skill.
    If an evaluation does complete within this time, it will be treated as a
    failure. When not provided, a skill-specific default value will be used.

    The `server_uri` may be used to change the default gRPC host and/or port
    where the policy RPC server will be found.

    The `use_rpc` should nearly always be set to True. Some unit tests will set
    it to False for certain kinds of experiments, in which case the `policy`
    argument must be used to provide the policy serving API.
    """
    # Wrap the evaluation logic in a try-except so that we can reify exceptions
    # into a SingleEvaluationResult with the details.
    try:
        return _evaluate_one_impl(
            skill_type=skill_type,
            scenario_index=scenario_index,
            output_directory=output_directory,
            use_eval_seed=use_eval_seed,
            t_max=t_max,
            server_uri=server_uri,
            use_rpc=use_rpc,
            policy=policy,
        )
    except Exception:
        string_file = io.StringIO()
        traceback.print_exc(file=string_file)
        return SingleEvaluationResult(
            skill_type=skill_type,
            scenario_index=scenario_index,
            is_pending=False,
            total_time=None,
            is_success=False,
            failure_message=string_file.getvalue(),
        )


def _evaluate_one_impl(
    *,
    skill_type: str,
    scenario_index: int,
    output_directory: Path,
    use_eval_seed: bool,
    t_max: float | None,
    server_uri: str,
    use_rpc: bool,
    policy: Policy,
) -> SingleEvaluationResult:
    """The implementation of evaluate_one."""
    # Find the skill_type within our known packages. Packages that contain
    # skills must have a file named "skill_filenames.txt" at their root,
    # containing a newline-separated list of skill filename relative paths.
    package_map = MakeDefaultAnzuPackageMap()
    config_file = None
    for package_name in package_map.GetPackageNames():
        base = Path(package_map.GetPath(package_name))
        inventory = base / "skill_filenames.txt"
        if not inventory.exists():
            continue
        for line in inventory.read_text(encoding="utf-8").split():
            if line.endswith(f"/{skill_type}.yaml"):
                config_file = base / line
                break
    if not config_file:
        raise RuntimeError(f"Unknown {skill_type=}")

    # Materialize the visuomotor scenario environment.
    raw_config_all = yaml_load(filename=config_file, private=True)
    if use_rpc:
        scenario_config_raw = raw_config_all["GrpcServerToSim"]
        if policy is not None:
            raise RuntimeError("Must not supply a policy when use_rpc=True")
        policy = LbmPolicyClientConfig(server_uri=server_uri).create()
    else:
        scenario_config_raw = raw_config_all["DiffusionInProcessSim"]
        assert policy is not None
    env_config_raw = scenario_config_raw["env"]
    # TODO(#13072) This is added to avoid CI issues and should be
    # removed/refactored into something the test runner manages.
    simulation_config_raw = env_config_raw["simulation_scenario_config"]
    simulation_config_raw["num_sample_processes"] = 1
    random_seed = get_demonstration_seed(scenario_index, use_eval_seed)
    simulation_config_raw["random_seed"] = random_seed
    env_config = from_dict(
        HardwareStationScenarioSimulationEnvConfig, env_config_raw
    )
    env_config.simulation_scenario_package = "anzu.sim.station.open_source"

    anzu_env = env_config.create()
    recorder = _LastStepRecorder()
    with closing_multiple(anzu_env, policy):
        env = GymEnvWrappingAnzuEnv(anzu_env, recorder)
        with closing_multiple(env):
            options = {
                "demonstration_index": scenario_index,
                "save_dir": str((output_directory / skill_type).absolute()),
            }
            if t_max is not None:
                options["t_max"] = t_max
            collect_episode_gym(env, policy, seed=random_seed, options=options)
        total_time = recorder.last_time_step.info["time"]
        is_success = recorder.last_time_step.info["is_success"]
        return SingleEvaluationResult(
            skill_type=skill_type,
            scenario_index=scenario_index,
            total_time=total_time,
            is_success=is_success,
            is_pending=False,
        )


def _evaluate_one_wrapper(
    index: int, kwargs: dict[str, Any]
) -> tuple[int, SingleEvaluationResult]:
    """Wrapper for `evaluate_one` to be used with multiprocessing. This
    function is only ever called inside a multiprocessing pool process,
    not the main evaluation process.
    """
    # When using `multiprocessing`, the global logging configuration of the
    # main process is *not* inherited by the spawned processes. We need to
    # configure it again each time.
    _configure_logging()
    single_result = evaluate_one(**kwargs)
    return index, single_result


def evaluate_many(
    evaluations: list[dict[str, Any]],
    output_directory: Path | None = None,
    num_processes: int = 1,
    progress_callback: Callable[[EvaluationResults], None] | None = None,
) -> EvaluationResults:
    """Given a series of kwargs (i.e., `[kwargs_1, kwargs_2, ...]`), calls
    `evaluate_one(**kwargs_i)` on each one and returns a list of the evaluation
    results. The order of results in the returned list will match the order of
    `evaluations`.

    This function uses multiprocessing to run the evaluations in parallel.

    During evaluation, the `progress_callback` (if given) is called regularly
    with the latest EvaluationResults, and a `results-{...}.json` file is
    saved to the `output_directory`. If the `output_directory` is not given,
    the `evaluations[0]['output_directory']` will be used as a default.
    """
    if output_directory is None:
        output_directory = evaluations[0]["output_directory"]
    json_path = _get_json_path(output_directory)
    start_time = time.time()

    # Populate all pending evaluations.
    results = EvaluationResults(
        num_processes=num_processes,
        evaluations=[
            SingleEvaluationResult(
                skill_type=kwargs["skill_type"],
                scenario_index=kwargs["scenario_index"],
            )
            for kwargs in evaluations
        ],
    )

    # Prime the output file with all pending results.
    save_json = functools.partial(
        _save_json,
        results=results,
        path=json_path,
    )
    save_json()
    print(f"Evaluation results will be saved to {json_path}")

    if progress_callback is not None:
        progress_callback(results)

    lock = Lock()

    def child_process_callback(index_and_result):
        """After each task completes, it sends the index and result back here
        to update the json file."""
        index, result = index_and_result
        with lock:
            results.evaluations[index] = result
            results.elapsed_time = time.time() - start_time
            save_json()
        gc.collect()

    with Pool(processes=num_processes, maxtasksperchild=1) as pool:
        # We'll upper-bound the number of evaluations pending in the queue to
        # avoid flooding the main process's memory with an unbounded amount of
        # task overhead all at once. To get per-task file updates, we need to
        # do apply_async and not any of the map-like functions (which only
        # evaluate the callback once at the end).
        worklist = deque(enumerate(evaluations))
        pending_results = deque()
        while worklist or pending_results:
            # Grow the pending_results queue to its maximum size.
            while worklist and len(pending_results) < 100:
                index, kwargs = worklist.popleft()
                async_result = pool.apply_async(
                    func=_evaluate_one_wrapper,
                    kwds={"index": index, "kwargs": kwargs},
                    callback=child_process_callback,
                    error_callback=logging.critical,
                )
                pending_results.append(async_result)
            # Wait for the oldest pending result to finish.
            pending_results[0].wait()
            # Clear out the oldest result(s) that have completed.
            while pending_results and pending_results[0].ready():
                pending_results.popleft()
            # Report progress to the user.
            if progress_callback is not None:
                progress_callback(results)
        # The context manager only calls terminate(); we'd like a more graceful
        # shutdown, so we'll explicitly close and join, first.
        pool.close()
        pool.join()

    print(f"Evaluation results saved to {json_path}")
    return results


def _run(
    *,
    output_directory: Path,
    skill_types: list[str],
    num_evaluations: int,
    num_processes: int,
    server_uri: str,
    # For unit testing only.
    t_max: float = None,
) -> EvaluationResults:
    """The implementation of main() after argparse finishes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluations = [
        dict(
            output_directory=output_directory,
            skill_type=skill_type,
            scenario_index=scenario_index,
            server_uri=server_uri,
            t_max=t_max,
        )
        for skill_type in skill_types
        for scenario_index in range(num_evaluations)
    ]

    progress_bar = None
    num_evaluated = 0

    def progress_callback(results: EvaluationResults):
        nonlocal num_evaluated, progress_bar
        if progress_bar is None:
            progress_bar = tqdm.tqdm(
                desc="Evaluations",
                unit="eval",
                total=len(evaluations),
                smoothing=0.03,
            )
        new_num_evaluated = results.num_evaluated
        delta = new_num_evaluated - num_evaluated
        if delta > 0:
            progress_bar.update(delta)
        num_evaluated = new_num_evaluated
        if num_evaluated == len(evaluations):
            progress_bar.close()

    return evaluate_many(
        evaluations=evaluations,
        output_directory=output_directory,
        num_processes=num_processes,
        progress_callback=progress_callback,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=Path,
        required=True,
        metavar="PATH",
        help="Directory for temporary files and logged output.",
    )
    parser.add_argument(
        "--skill_type",
        dest="skill_types",
        action="append",
        help="The skill(s) to evaluate. "
        "Multiple skills may be specified by repeating this option.",
        choices=get_known_skill_types(),
        metavar="...",
    )
    parser.add_argument(
        "--num_evaluations",
        type=int,
        metavar="N",
        default=10,
        help="The number of evaluations to run per skill. "
        "(Defaults to %(default)s.)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        metavar="N",
        help="The number of processes to use for parallel evaluation. "
        "Defaults to 1, which means serial evaluation.",
    )
    parser.add_argument(
        "--server_uri",
        type=str,
        default=LbmPolicyClientConfig().server_uri,
        metavar="ADDRESS:PORT",
        help="A URI of the format 'address:port' to indicate the address "
        "of the gRPC server (default: '%(default)s').",
    )
    args = parser.parse_args()
    _run(**vars(args))
    return 0


if __name__ == "__main__":
    _configure_logging()
    set_start_method("forkserver")
    main()
