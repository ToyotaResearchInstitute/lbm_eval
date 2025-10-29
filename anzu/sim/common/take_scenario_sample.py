"""A tool to perform flattening and random sampling on a scenario.

Given an input scenario, this tool will:
 * Fully flatten all model directives, and
 * Replace every Drake stochastic variable with a concrete value

By sampling in pure python, prior to typed parsing, this allows the use
of stochastic variables anywhere that the corresponding sampled type
would appear.

The sampling will use the scenario's random seed, or a random seed of
the caller's choosing.  The scenario's random seed will be replaced with
a random value after all other processing, to ensure that no rng output
is reused.

There are two exceptions:
 * At least for now, stochastic variables within external model directives
   files (included via `add_directives`) are not handled.
 * The random seed itself cannot be a stochastic variable.
 * Stochastic variables cannot be nested; that is, a parameter of a
   distribution must be an ordinary numeric value.
"""

import argparse
from copy import deepcopy
import dataclasses
import logging
from multiprocessing import Pool
from numbers import Number
from typing import Callable
import zlib

import numpy as np

from pydrake.common import Parallelism, RandomGenerator
import pydrake.common.schema
from pydrake.common.yaml import (
    yaml_dump,
    yaml_dump_typed,
    yaml_load,
    yaml_load_typed,
)
from pydrake.multibody.parsing import (
    FlattenModelDirectives,
    LoadModelDirectivesFromString,
    PackageMap,
)

from anzu.common.anzu_model_directives import MakeDefaultAnzuPackageMap
from anzu.sim.common.rejection_sampling import satisfies_constraints
from anzu.sim.common.scenario_constraints_config import ScenarioConstraints
from anzu.sim.common.scenario_directives_resolution import (
    is_directives_field,
    resolve_directives_num_copies,
)
from anzu.sim.common.scenario_list_resolution import (
    is_list_node,
    resolve_list_num_copies,
)
from anzu.sim.common.xacro_model_functions import resolve_xacro_model

_ARBITRARY_VALUE = 5489  # The default C++ hash seed.


# Map of tag -> dataclass, for every supported Drake stochastic type.
_DRAKE_STOCHASTIC_CLASSES = {
    "!UniformVector": pydrake.common.schema.UniformVector[None],
    "!UniformDiscrete": pydrake.common.schema.UniformDiscrete,
    "!Uniform": pydrake.common.schema.Uniform,
    "!GaussianVector": pydrake.common.schema.GaussianVector[None],
    "!Gaussian": pydrake.common.schema.Gaussian,

    # Transform implements the stochastic interface for itself and all
    # of its interior factory methods.
    "!Transform": pydrake.common.schema.Transform,
}


# Map of tag -> dataclass, for classes that should be loaded using typed
# parsing (i.e., via the visitor idiom) after their contents are sampled.
_STRICT_CLASSES = {
    "!Rotation": pydrake.common.schema.Rotation,
}


# Special-purpose factory tags; that is, tags whose dict members are kwargs
# to a factory function.  This is intended as a transitional state for classes
# whose semantics will be similar to Transform but have not yet implemented
# the variant-ctor mechanism.
_FACTORY_TAGS = {
    "!Hsv": lambda node: {"rgba": _hsv_to_rgba(np.array(node["hsv"]))},
    "!Include": lambda node: _include_file(**node),
}


# Special cases for pre-sampling behaviour.  Like _FACTORY_TAGS in that
# matching nodes will be passed to the given function; unlike _FACTORY_TAGS
# the node contents will not be sampled before the function is called.
#
# The resolution function will be passed _resolve_randomness as an argument
# so that it can sample recursively if necessary.
_PRE_SAMPLE_SPECIAL_CASES = {
    # NOTE: This is a broadened version of the num_copies sampling mechanisms
    # below, but it does not do any uniquification.
    "sample_list_num_copies": (
        is_list_node,
        resolve_list_num_copies),
    # TODO(dale.mcconachie) This only operates on `add_model` entries in a
    # directives list. In theory we could broaden it to other options as well
    # if the need arises. Note that this cannot be naively combined with Items
    # num_copies management because of the need to go inside the each item
    # in the list to find the name element and make it unique.
    "sample_directives_num_copies": (
        is_directives_field,
        resolve_directives_num_copies),
}


def _check_keys_match(expected: set[str], actual: set[str]):
    """
    Checks if expected == actual, raising an error if they do not.
    """
    if actual != expected:
        raise ValueError(
            "Mismatch between expected keys and actual keys.\n"
            f"  Expected but not present: {expected - actual}\n"
            f"  Present but not expected: {actual - expected}"
        )


def _random_choice(options: list, rng: RandomGenerator):
    """
    Chooses one value from options at random, using Drake's RandomGenerator
    instead of python's Random.choice mechanism.
    """
    # Note that Drake's UniformDiscrete only operates on floats, so we launder
    # to float and back as part of the sampling process.
    values = [float(i) for i in range(len(options))]
    uniform = pydrake.common.schema.UniformDiscrete(values)
    choice = int(uniform.Sample(rng))
    return options[choice]


# TODO(dale.mcconachie) This function should work on any element type inside
# the `values` field, but is only tested on strings so we keep it named that
# way for now.
def _sample_uniform_discrete_string(node: dict, rng: RandomGenerator):
    """
    Sample a single string from a discrete set of strings.

    Args:
        node: Dict containing a `values` and a `_tag` field. `values` must be
            a list of strings.
        rng: A Random generator to be used for any randomness.
    Returns:
        str: String containing the selected value.
    """
    expected_keys = {"values", "_tag"}
    _check_keys_match(expected_keys, set(node.keys()))
    return _random_choice(node["values"], rng)


def _sample_uniform_manipuland(node: dict, rng: RandomGenerator):
    """
    Sample a single manipuland from a list of possibilities.

    Args:
        node: Dict containing a `name`, `values`, and `_tag` fields. `values`
            is expected to be a list of dicts of strings and transforms in the
            following format:

                values:
                - file: package://anzu/models/example_file_0.sdf
                  default_free_body_pose:
                    body_name_in_example_file_0:
                      translation: [x0, y0, z0]
                      rotation: !Rpy
                        deg: [r0, p0, y0]
                - file: package://anzu/models/example_file_1.sdf
                  default_free_body_pose:
                    body_name_in_example_file_1:
                      translation: [x1, y1, z1]
                      rotation: !Rpy
                        deg: [r1, p1, y1]
                ...

        rng: A RandomGenerator to be used for any randomness.
    Returns:
        dict: Dict compatible with the AddModel yaml schema.
    """
    expected_keys = {"values", "name", "_tag"}
    _check_keys_match(expected_keys, set(node.keys()))
    value = _random_choice(node["values"], rng)
    default_free_body_pose = _resolve_randomness(
        value["default_free_body_pose"],
        _get_seed_for_child(rng(), "default_free_body_pose"))

    file = _resolve_randomness(
        value["file"], _get_seed_for_child(rng(), "file"))

    return {
        "file": file,
        "name": node["name"],
        "default_free_body_pose": default_free_body_pose
    }


# Map of tag -> sampling function which must take 2 arguments; the node that
# is being resolved and a RandomGenerator to be used for resolving the node.
# Note: Like _DRAKE_STOCHASTIC_CLASSES types, each sampling method is required
# to fully resolve any randomness within its own sample method.
_INTERNAL_STOCHASTIC_CLASSES = {
    "!UniformDiscreteString": _sample_uniform_discrete_string,
    "!UniformManipuland": _sample_uniform_manipuland,
}


# TODO(imcmahon): move these "FACTORY_TAG" methods into their own file(s)
def _hsv_to_rgba(hsv: np.ndarray) -> list:
    """
    From intuitive/visuomotor/lighting_schemas.py
    Convert HSV to RGBA color space, as explained in
    https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
    Args:
      hsv: (h, s, v). The hue is in the range [0, 1), saturation in the range
      [0, 1] and value v in the range [0, 1]
    Returns:
      rgba: The RGB values are in the range [0, 1]. Alpha is always set to 1.0
      to conform with the default alpha value in the drake::geometry::Rgba
      class. See the following Drake doxygen entry for reference:
      https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_rgba.html
    """
    assert np.all(hsv >= 0) and np.all(hsv <= 1)
    hue = hsv[0]
    sat = hsv[1]
    value = hsv[2]
    C = value * sat
    h_prime = hue * 6
    X = C * (1 - np.abs(h_prime % 2 - 1))
    if h_prime >= 0 and h_prime < 1:
        r1, g1, b1 = C, X, 0.0
    elif h_prime >= 1 and h_prime < 2:
        r1, g1, b1 = X, C, 0.0
    elif h_prime >= 2 and h_prime < 3:
        r1, g1, b1 = 0.0, C, X
    elif h_prime >= 3 and h_prime < 4:
        r1, g1, b1 = 0.0, X, C
    elif h_prime >= 4 and h_prime < 5:
        r1, g1, b1 = X, 0.0, C
    elif h_prime >= 5 and h_prime < 6:
        r1, g1, b1 = C, 0.0, X
    m = value - C
    return np.array([r1 + m, g1 + m, b1 + m, 1.0]).tolist()


def _include_file(filename: str,
                  _tag: str,
                  schema: str = None):
    """Loads @p filename as a yaml file (applying a schema named by @p schema
    if present), and returns it; as used by _FACTORY_TAGS this will replace a
    yaml node with the parsed file contents.

    If specified, `schema` must be the name of a member of _STRICT_CLASSES.
    """
    assert _tag == "!Include"
    result = yaml_load(filename=filename)
    if schema is not None:
        result["_tag"] = schema
    return result


@dataclasses.dataclass(kw_only=True)
class CommonScenarioConfig:
    """The common configuration items used by most/all programs that ingest
    scenarios."""
    scenario_file: (str | None) = None
    scenario_name: (str | None) = None
    scenario_text: (str | None) = None
    random_seed: (int | None) = None
    num_sample_processes: (int | None) = 1

    @classmethod
    def from_args(cls, args):
        return cls(scenario_file=args.scenario_file,
                   scenario_name=args.scenario_name,
                   scenario_text=args.scenario_text,
                   random_seed=args.random_seed,
                   num_sample_processes=args.num_sample_processes)


def _hash(data):
    """Deterministic hash function."""
    return zlib.adler32(bytes(str(data), encoding="utf-8"))


def _parse_num_processes(x):
    if x == "None":
        return None
    return int(x)


def add_cmdline_args(parser):
    """Add the command line arguments for scenario handling to @p parser."""
    parser.add_argument(
        "--scenario_file",
        type=str,
        default=None,
        help="Scenario filename.",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default=None,
        help="Scenario name within the scenario file",
    )
    parser.add_argument(
        "--scenario_text",
        type=str,
        default=None,
        help="Additional YAML scenario text to load",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="If set, equivalent to '--scenario_text {random_seed: VALUE}'"
    )
    parser.add_argument(
        "--num_sample_processes",
        type=_parse_num_processes,
        default=1,
        help="If set, run this number of process to sample the scenario."
        + "If None, then use #CPU_count processes to sample the scenarios")


def postprocess_cmdline_args(parser, args):
    """Perform any post-`parse_args()` adjustments to @p args (e.g. applying
    dependent default values) and log the errors through @p parser."""
    if args.scenario_file is None and args.scenario_text is None:
        parser.error("Either --scenario_file or --scenario_text is required.")
    if args.scenario_text is None:
        args.scenario_text = "{}"


def _parse_cmdline():
    """Parse command line arguments; these should all be the same in general
    semantics as the corresponding command line arguments in other scenario
    based binaries."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    add_cmdline_args(parser)
    args = parser.parse_args()
    postprocess_cmdline_args(parser, args)
    return CommonScenarioConfig.from_args(args)


def ingest_yaml(*,
                scenario_file=None,
                scenario_name=None,
                scenario_text=None,
                random_seed=None,
                # The following arguments are accepted but ignored, for
                # convenience.
                num_sample_processes=None):
    """Given command line arguments, load and parse the effective input yaml
    (i.e. loaded from the file, with or without a scenario name, and/or with a
    scenario text overlay)."""
    if scenario_name is not None:
        assert scenario_file is not None, \
               "`scenario_file` is required if `scenario_name` is specified."
        scenarios = yaml_load(filename=scenario_file, private=True)
        scenario = scenarios[scenario_name]
    else:
        if scenario_file is not None:
            scenario = yaml_load(filename=scenario_file, private=True)
        else:
            scenario = {}
    diffs = {}
    if scenario_text not in (None, ''):
        diffs = yaml_load(data=scenario_text)
    if random_seed:
        diffs["random_seed"] = random_seed
    merged = _merge_nodes(scenario, diffs)
    if "random_seed" not in merged:
        merged["random_seed"] = _ARBITRARY_VALUE
    return merged


def _merge_nodes(original, diff):
    """Merge two yaml-based nodes with recursive dict overlay
    semantics.

    NOTE:  The arguments are copied, not altered in place."""
    result = deepcopy(original)
    # `dict`s have merge semantics.
    if isinstance(result, dict) and isinstance(diff, dict):
        for key in diff:
            if key in result:
                result[key] = _merge_nodes(original[key], diff[key])
            if key not in result:
                result[key] = diff[key]
    # All other types, if the types match, have replacement semantics.
    elif ((type(original) is type(diff))
          or (isinstance(original, Number) and isinstance(diff, Number))):
        result = diff
    else:
        raise RuntimeError(
            "Nodes to merge have different types: "
            + f"{type(result)} in original and {type(diff)} in overlay")
    return result


def _get_seed_for_child(seed, child_name):
    """Deterministically produce a new random seed for processing a child
    node; this prevents seed reuse while preserving seed values across
    small changes to the yaml to reduce user surprise."""
    return (RandomGenerator(seed)() + _hash(child_name)) % (1 << 63)


def _fix_sampled_rpy(_original, sample):
    """`schema.Rotation.Rpy` does not trigger yaml serialization correctly
    when it appears within a `schema.Rotation`."""
    return {'_tag': '!Rpy', 'deg': sample.deg.tolist()}


def _fix_sampled_rotation(_original, sample):
    """`schema.Rotation` is sampled to `RotationMatrix` which lacks
    a serialization mechanism.

    This function special-cases the raising of the sampled `RotationMatrix`
    back to the more-expressive `Rotation` class.
    """
    rpy = pydrake.common.schema.Rotation(sample).value
    return _fix_sampled_rpy(None, rpy)


def _fix_sampled_transform(transform, sample):
    """`schema.Transform` is sampled to `RigidTransform` which both lacks
    a serialization mechanism and suffers from drake#20754 (which was
    WONTFIX in drake itself).

    This function special-cases the raising of the sampled `RigidTransform`
    back to the more-expressive `Transform` class.
    """
    return {
        'base_frame': transform.base_frame,
        'translation': sample.translation().tolist(),
        'rotation': _fix_sampled_rotation(None, sample.rotation()),
    }


_INCORRECT_SERIALIZATION_CLASSES = {
    # Classes that don't roundtrip through load->sample->dump.
    # TODO(ggould) These "special cases" are really drake bugs and should
    # be filed as such.
    pydrake.math.RigidTransform: _fix_sampled_transform,
    pydrake.math.RotationMatrix: _fix_sampled_rotation,
    pydrake.common.schema.Rotation.Rpy: _fix_sampled_rpy,
}


def _convert_all_xacro_files_to_model_files(
        config,
        *,
        package_map: PackageMap):
    """The directive add_model can define the file as a !XacroModel. This
    specification applies xacro to the named file with the given properties,
    converting it to a model file and then substitutes the !XacroModel value in
    `file` to the path for the generated model file.

    The !XacroModel tag has two properties:
      - xacro_filename: a string providing the URL to the .xacro file.
      - arg_mappings: a dictionary containing (key, value) pairs compatible
        with the named xacro file.

    Using it would look like this:

        - add_model:
          file: !XacroModel
            xacro_filename: package://anzu/models/foo.xacro
            arg_mappings:
              size: 17
              offset: 0.2
          name: foo_0
          default_free_body_pose:
            foo_base:
              parent: some_other_frame
    """
    expected_keys = {"xacro_filename", "arg_mappings", "_tag"}

    target = deepcopy(config)
    directives = target.get("directives", [])
    for directive in directives:
        assert len(directive) == 1, directive
        add_model = directive.get("add_model", None)
        if add_model is not None:
            file = add_model.get("file", None)
            if (not isinstance(file, dict) or
                    file.get("_tag", "") != "!XacroModel"):
                # To be xacro, we need a dictionary with the right tag.
                continue
            given_keys = set(file.keys())
            if given_keys != expected_keys:
                missing_keys = expected_keys.difference(given_keys)
                extra_keys = given_keys.difference(expected_keys)
                error_descrip = ""
                if extra_keys:
                    error_descrip = f"has extra keys {extra_keys}"
                    if missing_keys:
                        error_descrip += " and "
                if missing_keys:
                    error_descrip += f"is missing keys {missing_keys}"
                raise RuntimeError(
                    "The add_model directive has the !XacroModel tag. It "
                    "takes two keys: `xacro_filename` and `arg_mappings`. "
                    f"This specification {error_descrip}.")
            xacro_filename = file["xacro_filename"]
            arg_mappings = file["arg_mappings"]
            file_uri = resolve_xacro_model(
                xacro_filename=xacro_filename,
                arg_mappings=arg_mappings,
                package_map=package_map,
            )
            add_model["file"] = file_uri
    return target


def _resolve_tagged_dict(node, seed):
    """Turn a dict representing a tagged yaml node into an object.  The
    semantics of tagging are described at https://yaml.org/spec/1.2.2/#tags,
    but by this point in parsing only Drake custom tags should remain."""
    tag = node['_tag']
    # Most tag cases should have been resolved before we get here.
    assert not tag.startswith("!!")
    assert not tag.startswith("?")
    assert not tag.startswith("tag:yaml.org")

    if tag in _STRICT_CLASSES:
        dataclass = _STRICT_CLASSES[tag]
        resolved_node = _resolve_untagged_dict(node, seed)
        typed_value = yaml_load_typed(schema=dataclass,
                                      data=yaml_dump(resolved_node))
        typed_value_as_dict = yaml_load(
            data=yaml_dump_typed(schema=dataclass, data=typed_value))
        return typed_value_as_dict

    if tag in _DRAKE_STOCHASTIC_CLASSES:
        dataclass = _DRAKE_STOCHASTIC_CLASSES[tag]
        typed_value = yaml_load_typed(schema=dataclass,
                                      data=yaml_dump(node))
        rng = RandomGenerator(seed)
        sampled_value = typed_value.Sample(rng)
        if type(sampled_value) in {int, float, str, list, dict}:
            # If the sampled value has a simple json type, return that.
            return sampled_value
        elif hasattr(sampled_value, "GetDeterministicValue"):
            # If the sampled value's type uses the Sample/IsDeterministic
            # idiom, use that.
            return sampled_value.GetDeterministicValue()
        elif hasattr(sampled_value, "tolist"):
            # If the sampled value is an array, treat it as a list.
            return sampled_value.tolist()
        else:
            # Sometimes sampling type T returns type U, where U is not
            # itself an instance of T (e.g. Transform samples to
            # RigidTransform).  We have a couple of ways to try to handle this:
            if type(sampled_value) in _INCORRECT_SERIALIZATION_CLASSES:
                # If the sampled type is not serializable, see if it's a known
                # special case.
                fixup = _INCORRECT_SERIALIZATION_CLASSES[type(sampled_value)]
                return fixup(typed_value, sampled_value)
            elif hasattr(sampled_value, '__fields__'):
                # If U is serializable, we can hail-mary by dumping with U's
                # serializer and loading the result as a dict.
                # TODO(ggould) This case may no longer exist in practice, but
                #              we retain it as likely to make future drake
                #              stochastics work out-of-the-box.
                return yaml_load(yaml_dump_typed(sampled_value))
            # Otherwise fall through and hope something else matches.

    if tag in _INTERNAL_STOCHASTIC_CLASSES:
        sampler = _INTERNAL_STOCHASTIC_CLASSES[tag]
        rng = RandomGenerator(seed)
        sampled_value = sampler(node, rng)
        return sampled_value

    if tag in _FACTORY_TAGS:
        # One subtlety of factory calls is that the arguments to the factory
        # could have randomness or other resolvable elements, but so could
        # the output, so we have to resolve recursively.
        factory = _FACTORY_TAGS[tag]
        rng = RandomGenerator(seed)
        resolved_node = _resolve_untagged_dict(node, rng())
        generated_node = factory(resolved_node)
        recursively_resolved = _resolve_randomness(generated_node, rng())
        return recursively_resolved

    # None of the above matched.
    return _resolve_untagged_dict(node, seed)


def _resolve_untagged_dict(node, seed):
    result = {}
    for key, value in node.items():
        result[key] = _resolve_randomness(
            value, _get_seed_for_child(seed, key))
    return result


def _resolve_randomness(node, seed):
    """Given a node from parsed yaml, return an identical node with
    Drake stochastics replaced with sampled values from those stochastics."""
    working_node = node
    for case_name, (predicate, action) in _PRE_SAMPLE_SPECIAL_CASES.items():
        if predicate(working_node):
            child_seed = _get_seed_for_child(seed, case_name)
            working_node = action(
                working_node, child_seed, _resolve_randomness)
    if isinstance(working_node, dict):
        if '_tag' in working_node:  # YAML "JSON schema" type
            working_node = _resolve_tagged_dict(working_node, seed)
        else:  # YAML "mapping" type
            working_node = _resolve_untagged_dict(working_node, seed)
    elif isinstance(working_node, list):  # YAML "sequence" type
        result = []
        for index, value in enumerate(working_node):
            result.append(_resolve_randomness(
                value, _get_seed_for_child(seed, index)))
        working_node = result
    # If `working_node` is a YAML "scalar" type (not the same as Drake
    # scalar!) then it can pass through unaltered.
    return working_node


def _flatten_scenario_directives(node, *, package_map: PackageMap):
    """Return a copy of the scenario with any model directives replaced with
    their flattened versions."""
    # Without a scenario class in hand, we don't actually know where if
    # anywhere there are model directives, which are not a tagged type.
    # So we search the tree for anything that quacks like a model directive.
    if isinstance(node, list):
        try:
            directives = LoadModelDirectivesFromString(
                yaml_dump({"directives": node}))
            flattened = FlattenModelDirectives(directives, package_map)
            flattened_as_yaml_data = [
                yaml_load(data=yaml_dump_typed(d))
                for d in flattened.directives]
            return flattened_as_yaml_data
        except Exception:
            return [
                _flatten_scenario_directives(s, package_map=package_map)
                for s in node
            ]
    elif isinstance(node, dict):
        return {
            k: _flatten_scenario_directives(v, package_map=package_map)
            for k, v in node.items()
        }
    else:
        return node


def _sample_with_seed(*, schema, make_hardware_station,
                      initial_scenario, seed):
    package_map = MakeDefaultAnzuPackageMap()
    sample = deepcopy(initial_scenario)
    sample["random_seed"] = seed
    sample = _resolve_randomness(sample, seed)
    no_xacro_sample = _convert_all_xacro_files_to_model_files(
        sample,
        package_map=package_map,
    )
    flat_scenario = _flatten_scenario_directives(
        no_xacro_sample,
        package_map=package_map,
    )
    if ("preconditions" not in flat_scenario or
            satisfies_constraints(
                schema=schema,
                make_hardware_station=make_hardware_station,
                scenario_dict=flat_scenario,
            )):
        return flat_scenario
    return None


class _SampleWorker:
    def __init__(self, *, schema, make_hardware_station,
                 initial_scenario: dict):
        self.schema = schema
        self.make_hardware_station = make_hardware_station
        self.initial_scenario = initial_scenario

    def __call__(self, seed):
        flat_scenario = _sample_with_seed(
            schema=self.schema,
            make_hardware_station=self.make_hardware_station,
            initial_scenario=self.initial_scenario,
            seed=seed,
        )
        return flat_scenario


def do_rejection_sampling(*, initial_scenario: dict,
                          schema=None,
                          make_hardware_station=None,
                          num_processes=None):
    """Perform rejection sampling according to the scenario's configured
    constraints and sample count.

    Args:
      initial_scenario: untyped raw scenario data.
      schema: The dataclass schema associated with the initial_scenario.
       Required (and used) iff the initial_scenario has `preconditions`.
      make_hardware_station: The factory function for the given `schema`.
       Required (and used) iff the initial_scenario has `preconditions`.
      num_processes: The number of processes to sample the scenario.
       1 for single process, None for maximum concurrency as constrained by
       DRAKE_NUM_THREADS, OMP_NUM_THREADS, or the platform's maximum
       concurrency.
       Used iff the initial_scenario has `preconditions`.
       During `bazel test` an upper bound is enforced per the BUILD file.
    """
    num_samples = 1
    seeds = [initial_scenario["random_seed"]]

    if "preconditions" in initial_scenario:
        constraints = yaml_load_typed(
            data=yaml_dump(initial_scenario["preconditions"]),
            schema=ScenarioConstraints)
        seed_generator = RandomGenerator(seeds[0])
        num_samples = constraints.num_sample_attempts
        for _ in range(constraints.num_sample_attempts - 1):
            seeds.append(seed_generator())

    # When run under `bazel test` or `ray.remote` or other such processes that
    # set the DRAKE_NUM_THREADS or OMP_NUM_THREADS environment variables, obey
    # that limit. `Parallelism.Max().num_threads()` looks up those values as
    # well as the platform's maximum concurrency and returns the maximum value
    # consistent with those limits.
    orig_num_processes = num_processes
    max_num_threads = Parallelism.Max().num_threads()
    if num_processes is None:
        num_processes = max_num_threads
    else:
        num_processes = min(num_processes, max_num_threads)
    if num_processes != orig_num_processes:
        logging.info(
            f"do_rejection_sampling will use {num_processes} process(es) "
            f"instead of the {orig_num_processes or 'max'} process(es) "
            "requested because the execution environment has set a maximum "
            "amount of parallelism for this executable. If you are running "
            "under 'bazel test' and need more processes, set "
            "'num_threads = ...' in the BUILD file for your test. If you are "
            "using ray and need more processes, increase the 'num_cpus=...` "
            "value being passed through the ray stack. I.e.; the call to "
            "'ray.remote().options()' or 'RemoteEvaluationCluster.evaluate()'."
        )

    if num_processes == 1 or num_samples == 1:
        for seed in seeds:
            flat_scenario = _sample_with_seed(
                schema=schema,
                make_hardware_station=make_hardware_station,
                initial_scenario=initial_scenario,
                seed=seed,
            )
            if flat_scenario:
                return flat_scenario
    else:
        worker = _SampleWorker(
            schema=schema,
            make_hardware_station=make_hardware_station,
            initial_scenario=initial_scenario,
        )

        with Pool(processes=num_processes) as pool:
            for flat_scenario in pool.imap(worker, seeds):
                if flat_scenario:
                    return flat_scenario

    raise RuntimeError("Rejection sampling found no valid scenario "
                       f"after trying {num_samples} seeds.")


def do_sample(config: CommonScenarioConfig,
              *,
              schema: type,
              make_hardware_station: Callable) -> dict:
    """A function-like entry point for testing or wrapping; takes the command
    line arguments as a CommonScenarioConfig structure; returns a sampled,
    flattened node suitable for dumping as output.

    The returned node will pass the provided constraints in the scenario; the
    random seed will be updated (via rejection sampling) if necessary to make
    this true."""
    base_scenario = ingest_yaml(**config.__dict__)
    return do_rejection_sampling(
        schema=schema,
        make_hardware_station=make_hardware_station,
        initial_scenario=base_scenario,
        num_processes=config.num_sample_processes,
    )
