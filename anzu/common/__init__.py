import os
import pathlib


def get_test_tmpdir(default=None):
    """When running under `bazel test`, returns the absolute path to a private
    writable directory that is the preferred location for scratch files or
    scratch subdirectories.  Otherwise, returns `default`.  The typical use of
    this function is to fill the `dir=` argument in a `tempfile` call, e.g.,

    with TemporaryDirectory(prefix="foo", dir=get_test_tmpdir()) as dir:
        print(d.name)
    """
    result = os.getenv("TEST_TMPDIR", default)
    if result is not None:
        assert os.path.isabs(result)
        assert os.path.isdir(result)
    return result


def get_test_undeclared_outputs_dir(
    default: str | pathlib.Path | None = None
) -> pathlib.Path | None:
    """Returns the absolute path to a writable directory for storing test
    outputs when running under `bazel test`. It is typically used to store
    files that help diagnose test failures.
    """
    result = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", default)
    if result is not None:
        result = pathlib.Path(result)
        assert result.is_absolute()
        assert result.is_dir()
    return result
