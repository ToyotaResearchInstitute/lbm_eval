# LBM Eval

LBM Eval is a simulation benchmark from Toyota Research Institute containing 49
tasks that measure the performance of Large Behavior Model policies. A prior
revision of this benchmark was used for the simulation-based evaluation in the
2025 paper
[A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://toyotaresearchinstitute.github.io/lbm1/),
and is built on the [Drake](https://drake.mit.edu/) toolbox.

https://github.com/user-attachments/assets/4cbe099b-3653-4030-acf2-f52f20d05889

Note that this software is newer than the version used in the published paper.
(The paper used an earlier snapshot of this software from Spring 2025; this
version is a snapshot from Fall 2025.) This newer version will not necessarily
reproduce the results in the paper.

## Getting Started with `lbm_eval`

### Prerequisites

* `lbm_eval` is tested on Ubuntu 24.04 using Python 3.12, but different
   platforms (e.g., Linux or macOS) or versions (e.g., Python 3.13) are likely
   to work. The primary limitation is probably the set of platforms supported
   by https://pypi.org/project/drake/ wheels. Refer to Drake's
   [pip documentation](https://drake.mit.edu/pip.html) for details.

* `lbm_eval` works best when using GPU with modern OpenGL rendering for Drake's
  camera simulation. A CPU-based GL stack (e.g., MESA) will result in very slow
  evaluation throughput.

* To view evaluations interactively, the supported web browser is Chrome (or
  anything Chromium-based); Firefox may not show 3D visualizations correctly.
  You might also need to disable any heavy browser extensions that interfere
  with `three.js`.

* To use LBM Eval with your own policy, you'll also need to write your own
  ~200-line Python program that adapts your policy inference code to our
  `robot_gym.policy.Policy` abstract class. This guide provides a demo using a
  built-in sample policy, but to benchmark your own policy you'll need to write
  your own policy wrapper.  See below for instructions.

### Installing `lbm_eval`

You can use any virtual environment manager you prefer (e.g., Poetry or UV).
For simplicity, this guide will use Python's built-in tools `venv` and `pip`.

Note that `lbm_eval` is composed of four wheels. To run the `evaluate` binary,
you must install all four wheels. If you prefer to make a separate virtual
environment for your policy wrapper, the policy's virtual environment only
requires the `robot_gym` wheel by itself.

The wheels are provided as attachments to the GitHub releases page.

```
cd ~/tmp
python3 -m venv venv
venv/bin/pip install \
  robot_gym-1.1.0-py3-none-any.whl \
  lbm_eval-1.1.0-py3-none-any.whl \
  lbm_eval_models-1.1.0-py3-none-any.whl \
  lbm_eval_scenarios-1.1.0-py3-none-any.whl
venv/bin/evaluate --help
```

(If you prefer to `venv/bin/activate` instead of typing out the `venv/bin/` path
prefix every time, that's fine.)

### Evaluating using our provided sample policy wrapper

To perform an evaluation, you must run two processes:
- the policy wrapper program that serves inference at a gRPC endpoint;
- the `evaluate` program that runs the benchmark using a gRPC client.

(1) In a new terminal, run our sample policy wrapper:

```sh
venv/bin/wave_around_policy_server
```

The terminal output will look something like this:
```console
Started Server loop on localhost:50051...
```

The server will run forever, until you cancel it with Ctrl-C or by closing the
window.

If you need to change the listen port number of the server, use the
`--server-uri` argument.

(2) In a new terminal, run the `evaluate` program with your choice of benchmark
tasks and details:

```sh
venv/bin/evaluate --skill_type=pick_and_place_box --num_evaluations=1 --num_processes=1 --output_dir=output
```

The terminal output will look something like this:
```console
Evaluation results will be saved to output/results-YYYY-MM-DDTHH:mm:SS.sss-TZ:00.json
Evaluations: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:42<00:00, 42.61s/eval]
Evaluation results saved to output/results-YYYY-MM-DDTHH:mm:SS.sss-TZ:00.json
```

Note the choices of arguments:

* The `--skill_type` option chooses which skill(s) to benchmark. The option can
  be repeated multiple times to benchmark multiple skills as a group.  In the
  example above, we used the `pick_and_place_box` option, which is intended to
  be the simplest possible skill, useful when bootstrapping a new policy; run
  with `--skill_type=help` to see the complete list of available skill names.

* The `--num_evaluations` option chooses the number of evaluations to run per
  skill, i.e., the number of different stochastic initial conditions to roll
  out. For good statistics, we recommend using 200 or more, but sometimes it's
  convenient to use smaller numbers in early experiments, for quicker results.

* The `num_processes` option chooses the number of simultaneously processes used
  to evaluate the skills. See below for discussion on common pitfalls.

* The `--output_dir` chooses a directory name that will be populated with a
  benchmark summary file named `results-{timestamp}.json` as well as
  per-roll-out details in subdirectories.

The `results-{timestamp}.json` file will be incrementally updated as evaluations
finish; you can monitor it for progress. If you Ctrl-C the `evaluate` program,
the partial results will remain intact.

If you need to change the server port number to connect to, use the
`--server_uri` argument.

Note that `evaluate` will pause until the server is ready, so if you forget to
run a server it will pause forever. Always start the server prior to starting
`evaluate`.

### Evaluating using your own policy wrapper

To benchmark your own policy, you'll need to replace the "wave around" sample
with your own policy wrapper that performs inference using your policy. Your
wrapper will listen on the same gRPC server port as the "wave around" policy,
so that `lbm_eval` will use your policy wrapper instead.

For your policy, you can either use the same venv as `evaluate` if it happens to
be compatible; or if you prefer, you may create a separate virtual environment.
In the policy's virtual environment, the only `lbm_eval`-related wheel you need
to install is `robot_gym-1.1.0-py3-none-any.whl`.

The easiest way to write your wrapper is probably to copy and adapt the
`robot_gym/wave_around_policy_server.py` sample into a new file.
For clarity, we recommend renaming the `WaveAround` and `WaveAroundBatch`
classes to something different.

The key methods to edit are `WaveAround.reset` and `WaveAround.step`.
The `step` function receives observations and should return actions; replace
its code with your policy inference. The `reset` function should clear any
state from a prior evaluation.

We recommend testing your policy wrapper using the trivial `pick_and_place_box`
skill, before attempting more complicated skills.

Note that if your policy server crashes, you can restart to continue the
evaluation (the `evaluate` program will pause, not crash). However, the
episodes(s) in progress when the server crashed will show up as failure;
they will not be re-attempted.

### Evaluating using the API (advanced)

The `venv/bin/evaluate` program provides a convenient starting point for running
benchmarks. However, if you would prefer to write your own program to customize
the evaluation, you have that option by importing `evaluate` as a module.

To do that, run your custom main program using the venv's python interpreter,
and import one of the public functions from the `lbm_eval.evaluate` module:

```py
from lbm_eval import evaluate_many

def main():
   evaluate_many(...)

if __name_ == '__main__':
   main()
```

### Multiprocessing

When the `--num_processes` option is omitted, `evaluate` will default to a
single process. However, the skill gets evaluated in a _separate_, forked
process. This guarantees that each skill evaluation occurs in its own
hermetically-sealed environment and all memory is guaranteed to be freed
up for the next evaluation.

The goal of multiprocessing is to increase the *throughput* on evaluations.
It might be tempting to set the number of processors to be equal to the number
of physical processes available on your platform. However, `evaluate` is also
bound on your GPU memory. Each skill evaluation may take more GPU memory than
you might expect. If the total amount of GPU requirements (per-evaluation GPU
footprint × number of processes) exceeds the available memory on your GPU the
per process performance will become significantly degraded. The GPU will have
to continually swap rendering contexts between the processes, slowing _all_
processes down. The rule of thumb is to pick the largest number of processors
which avoids this memory thrashing.

One possible sign that you have too many processes and have overtaxed your GPU
memory is the appearance of this error message:

    Unable to eglMakeCurrent: 12291

If this is printed to the console, you are *definitely* thrashing GPU memory.
However, you might still be thrashing memory even if you don't see this message.

# Policy training data

A selection of simulated training data suitable for training policies is
available. For each skill type in LBM Eval, there are hundreds of episode logs
of human operators performing the skills in teleoperated simulation. For
information about downloading and a detailed description of the data provided,
see the [training data documentation](TRAINING_DATA_FORMAT.md).

# Support

Feel free to file Issues or open new Discussions with questions.
We are not currently accepting pull requests.

# Credits

This benchmark was created by the authors of the 2025 paper
[A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://toyotaresearchinstitute.github.io/lbm1/),
and is built on the [Drake](https://drake.mit.edu/) toolbox.

# License

The majority of LBM Eval is primarily distributed under the terms of both the
MIT license and the Apache License (Version 2.0); see the files LICENSE-MIT and
LICENSE-APACHE in this directory.

Small portions of LBM Eval are covered by other compatible open-source licenses
as indicated by the presence of a different LICENSE.TXT file in a subdirectory.

Any references or links to third-party products (including, but not limited to,
product pages) are provided solely for informational and convenience purposes.
We do not endorse, sponsor, or have any official affiliation with the
manufacturers, sellers, or platforms associated with these products. All product
names, logos, and brands are the property of their respective owners.
