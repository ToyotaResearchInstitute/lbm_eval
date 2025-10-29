"""Helper functions for dealing with Bazel runfiles, i.e., declared data
dependencies of libraries and programs.
"""

import functools
import os
import os.path
from pathlib import Path

try:
    from python.runfiles import Runfiles as _Runfiles
    _runfiles_import_error = None
except ImportError as e:
    # In ANZU_NUC_RUNFILES mode, we don't have bazel runfiles infrastructure,
    # so we'll defer the exception until the first time anytime tries to use
    # runfiles other than `anzu/...`.
    _Runfiles = None
    _runfiles_import_error = e


# This is the name of an environment variable that can be set to a /dir/to/anzu
# to override the default behavior of the Rlocation() function, below.
ANZU_NUC_RUNFILES_ENV_VAR = "ANZU_NUC_RUNFILES"


@functools.cache
def get_runfiles_singleton() -> _Runfiles:
    """Initializes and returns the default Runfiles object."""
    if _Runfiles is None:
        raise _runfiles_import_error
    return _Runfiles.Create()


@functools.cache
def get_directory_based_runfiles_singleton() -> _Runfiles:
    """Initializes and returns a Runfiles object using the 'directory based'
    method instead of the default 'manifest based' method. The upshot is that
    all runfiles in the same package will be co-located, no matter if they are
    source files or generated files.
    """
    runfiles_dir = get_runfiles_singleton().EnvVars()["RUNFILES_DIR"]
    return _Runfiles.CreateDirectoryBased(runfiles_dir)


def label_to_respath(label):
    """Given a label path, returns the matching respath.

    For example, //foo/bar:baz returns anzu/foo/bar/baz.

    Note: This only allows for Anzu labels (//foo) for now, not @drake//foo.
    """
    assert label.startswith("//"), label
    assert ":" in label, label
    return "anzu/" + label[2:].replace(":", "/")


def Rlocation(respath, executable=False):
    """Returns the pathname of a declared data dependency of a py_binary (or
    py_library, etc.).  The result is guaranteed to exist and be readable, and
    be under the `.runfiles` directory for use with relative paths.

    The `respath` looks like "anzu/pkg/file.ext" or "drake/pkg/file.ext"
    or etc.

    If `executable` is set to True, then the result is also guaranteed to be
    excutable.

    The path we return is always somewhere inside the runfiles directory, not
    the source tree. (This is important in case resource files cross-reference
    each other; we need them to all be in the same place, not some in the
    source tree and others in genfiles.)
    """
    # Check for the hard-coded override used by NUC-deployed software.
    override_value = os.environ.get(ANZU_NUC_RUNFILES_ENV_VAR, None)
    if override_value is not None and respath.startswith("anzu/"):
        result = os.path.join(override_value, respath[5:])
        cross_check_realpath = False
    else:
        # First, we'll look for the file using the runfiles manifest. If the
        # file was not declared as input to this program, this will fail-fast.
        runfiles_by_manifest = get_runfiles_singleton()
        result_by_manifest = runfiles_by_manifest.Rlocation(respath)
        if result_by_manifest is None:
            raise RuntimeError(
                f"Resource path {respath} could not be resolved to a "
                "filesystem path; maybe there is a typo in the path, or a "
                "missing data = [] attribute in the BUILD.bazel file.")
        # Second, we'll look for the file again using the RUNFILES_DIR only,
        # without any manifest. This will not detect failures, but does meet
        # the goal of returning a consistent path structure across different
        # resources.
        runfiles_by_dir = get_directory_based_runfiles_singleton()
        result = runfiles_by_dir.Rlocation(respath)
        cross_check_realpath = True

    if not os.path.exists(result):
        raise RuntimeError(
            ("Resource path {} resolved to filesystem path {} but that "
             "filesystem path does not exist.").format(
                 respath, result))
    if not os.access(result, os.R_OK):
        raise RuntimeError(
            ("Resource path {} resolved to filesystem path {} but that "
             "filesystem path does not allow read access.").format(
                 respath, result))
    if executable:
        if not os.access(result, os.X_OK):
            raise RuntimeError(
                ("Resource path {} resolved to filesystem path {} but that "
                 "filesystem path does not allow execute access.").format(
                     respath, result))
    if cross_check_realpath:
        # As a sanity check, ensure that realpaths match.
        realpath_by_dir = os.path.realpath(result)
        realpath_by_manifest = os.path.realpath(result_by_manifest)
        if realpath_by_dir != realpath_by_manifest:
            raise RuntimeError(
                f"Resource path {respath} has a mismatch between directory "
                f"and manifest realpaths:\n"
                f"  directory: {result}\n"
                f"    realpath: {realpath_by_dir}\n"
                f"  manifest: {result_by_manifest}\n"
                f"    realpath: {realpath_by_manifest}\n")

    return result


def find_anzu_resource_or_throw(anzu_resource_path: str) -> str:
    """Compatibility shim for legacy C++ FindAnzuResourceOrThrow.

    Attempts to locate an Anzu resource named by the given
    `anzu_resource_path`.  The path refers to the relative path within the
    source repository, e.g., `apps/home/foo.yaml`. If there is an Rlocation
    error it is packaged into an exception and raised.

    Newer code should use Rlocation directly.
    """
    this_workspace = "anzu"  # TODO(ggould) Get this programmatically somehow.
    return Rlocation(os.path.join(this_workspace, anzu_resource_path))


@functools.cache
def anzu_package_xml_path() -> Path:
    """Returns the path to anzu's top-level package.xml file."""
    return Path(Rlocation("anzu/package.xml"))


def SubstituteMakeVariableLocation(arg):
    """Given a string argument that might be a $(location //foo) substitution,
    looks up ands return the specified runfile location for $(location //foo)
    if the argument is in such a form, or if not just returns the argument
    unchanged.  Only absolute labels ("//foo" or "@drake//bar") are supported.
    It is an error if the argument looks any other $(...).  For details see
    https://docs.bazel.build/versions/master/be/make-variables.html.
    """
    if arg.startswith("$(location "):
        label = arg[11:-1]
        assert label.startswith("@") or label.startswith("//"), label
        if not label.startswith("@"):
            label = "@anzu" + label
        elif label.startswith("@//"):
            label = label.replace("@//", "@anzu//")
        normalized = label[1:]  # Strip the leading @.
        normalized = normalized.replace("//:", "/")
        normalized = normalized.replace("//", "/")
        normalized = normalized.replace(":", "/")
        arg = Rlocation(normalized)
    assert not arg.startswith("$("), arg
    return arg


def UpdateRunfilesEnviron(env):
    """Returns a copy of env with the Runfiles environment variables for child
    processes set.

    Do not abuse this to purely access RUNFILES_DIR or other variables to then
    access specific files.
    """
    result = dict(env)
    result.update(get_runfiles_singleton().EnvVars())
    return result
