import functools
import glob
import os
from pathlib import Path

from pydrake.multibody.parsing import PackageMap

from anzu.common.runfiles import (
    ANZU_NUC_RUNFILES_ENV_VAR,
    anzu_package_xml_path,
)


@functools.cache
def _lbm_eval_models_package_xml_path() -> Path:
    """Returns the path to the lbm_eval_models's package.xml file."""
    return anzu_package_xml_path().parent / "lbm_eval/models/package.xml"


@functools.cache
def _lbm_eval_scenarios_package_xml_path() -> Path:
    """Returns the path to the lbm_eval_scenarios's package.xml file."""
    return anzu_package_xml_path().parent / "lbm_eval/scenarios/package.xml"


def _add_venv_packages(package_map: PackageMap) -> None:
    """Adds to the given map any packages found in a glob of the same venv
    site-packages directory as anzu is installed into.
    """
    site_packages = anzu_package_xml_path().parent.parent
    if site_packages.name != "site-packages":
        return
    for filename in glob.glob(f"{site_packages}/*/package.xml"):
        try:
            package_map.AddPackageXml(filename)
        except Exception:
            pass


def add_default_anzu_packages(package_map: PackageMap) -> None:
    """Adds anzu-relevant package paths to the given map."""
    # Add ability to read `package://anzu` URIs.
    package_map.AddPackageXml(anzu_package_xml_path())
    # How we find the packages related to lbm_eval depends on whether we are
    # running an Anzu Bazel binary or a packaged release. When using normal
    # runfiles, we can find lbm_eval inside of Anzu. Otherwise (i.e., without
    # runfiles), we'll fall back to scraping a venv for packages (if any).
    use_bazel_runfiles = os.environ.get("ANZU_NUC_RUNFILES", "") == ""
    if use_bazel_runfiles:
        # Add ability to read `package://lbm_eval_models` URIs, using
        # relative paths based on the `package://anzu` directory.
        package_map.AddPackageXml(_lbm_eval_models_package_xml_path())
        # Add ability to read `package://lbm_eval_scenarios` URIs, using
        # relative paths based on the `package://anzu` directory.
        package_map.AddPackageXml(_lbm_eval_scenarios_package_xml_path())
    else:
        # Add all `package.xml` contents we can find in the current venv. In
        # the case of lbm_eval wheel packages, this is where we'd expect to
        # find the lbm_eval_models and lbm_eval_scenarios packages. If there
        # isn't a venv, this is a no-op (not an error).
        _add_venv_packages(package_map)


def MakeDefaultAnzuPackageMap() -> PackageMap:
    """Creates a PackageMap with add_default_anzu_packages() already added.
    This is a pure-python implementation of the identically named C++ function.
    """
    # N.B. Keep this function in sync with anzu_model_directives.cc.
    result = PackageMap()
    add_default_anzu_packages(package_map=result)
    return result
