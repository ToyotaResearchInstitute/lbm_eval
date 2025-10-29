# N.B. We can't name this module "tempfile" because that conflicts with the
# built-in tempfile module.

import gc
import re
import tempfile
import threading

from anzu.common import get_test_tmpdir

# User code is allowed change this in to keep all tempdirs intact upon program
# termination.
keep_tempdir = False


class AutoTemporaryDirectory:
    """Like tempfile.TemporaryDirectory but formulated as a base class that
    cleans up upon destruction (without spamming warning about it like the
    tempfile module does).

    This class is desiged to be safe for use with the `threading` module
    (directory creation is guarded by a lock), but not `multiprocessing`.
    """

    def __init__(self, *, dir=None, suffix=None, prefix=None, cls=None):
        """Captures the arguments but does not create a directory until
        tempdir() is called. The parameters have the same semantics as
        tempfile.TemporaryDirectory, but with different defaults.

        If @p prefix is not provided, the default value will be used based on
        @p cls or else the type of `self`.

        If @p dir is not provided, the default value will be $TEST_TMPDIR when
        set, otherwise the same defaults as tempfile.TemporaryDirectory.
        """
        self.__dir = dir
        self.__prefix = prefix
        self.__suffix = suffix
        self.__cls = cls

        # This lock guards the member fields that follow it.
        self.__lock = threading.Lock()
        self.__tempdir_holder = None
        self.__tempdir = None

    def __del__(self):
        if self.__tempdir_holder is not None:
            self.__tempdir_holder.cleanup()

    def __dir_or_default(self):
        if self.__dir:
            return self.__dir
        return get_test_tmpdir()

    def __prefix_or_default(self):
        if self.__prefix:
            return self.__prefix
        cls = self.__cls or type(self)
        if cls == AutoTemporaryDirectory:
            raise RuntimeError(
                "When AutoTemporaryDirectory is not used as a base class, "
                "either cls= or prefix= is required.")
        # Turn the classname into a basename prefix for our tempdir.
        # The result looks like "anzu_sim_batch_tempfile_Example_".
        prefix = str(cls)
        prefix = prefix.replace("class", "")
        prefix = re.sub("[^A-Za-z0-9]+", "_", prefix)
        prefix = re.sub("^_", "", prefix)
        return prefix

    def tempdir(self):
        """Returns the pathname to a per-self temporary directory, creating it
        on demand the first time this is called.
        """
        with self.__lock:
            if self.__tempdir is None:
                if keep_tempdir:
                    self.__tempdir = tempfile.mkdtemp(
                        dir=self.__dir_or_default(),
                        prefix=self.__prefix_or_default(),
                        suffix=self.__suffix)
                else:
                    self.__tempdir_holder = tempfile.TemporaryDirectory(
                        dir=self.__dir_or_default(),
                        prefix=self.__prefix_or_default(),
                        suffix=self.__suffix)
                    self.__tempdir = self.__tempdir_holder.name
            return self.__tempdir


class _Example(AutoTemporaryDirectory):
    """A private testing fixture to make use of AutoTemporaryDirectory.  We
    need to define this in a module (not main) to cover the use cases we care
    about, and putting it into this module is easier than making a whole new
    test_utilities module for these few lines of code.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
