"""
Contains functions for file path processing, such as trailing slash removal,
file permission modification and URI parsing etc.
"""

import copy
import datetime
import os
from os.path import (
    abspath,
    basename,
    dirname,
    exists,
    expanduser,
    expandvars,
    isdir,
    join,
    relpath,
)
import re
from shutil import rmtree
from subprocess import run
from typing import Optional

from dateutil.relativedelta import relativedelta

from anzu.common.runfiles import Rlocation


def get_parent_dir(*, path, up_level=1):
    """
    Find the parent of a @p path at @p up_level.
    """
    assert up_level > 0
    for _ in range(up_level):
        path = dirname(path)
    return path


def create_path_remap_func(input_dir, output_dir):
    """Returns a function that remaps file paths from an input directory to a
    the equivalent path and name in output directory."""
    input_dir = strip_trailing_slash(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = strip_trailing_slash(output_dir)

    def save_file_remap(file, makedirs=True):
        assert file.startswith(f"{input_dir}/"), f"\n  {input_dir}\n  {file}"
        rel = relpath(file, input_dir)
        out = join(output_dir, rel)
        if makedirs:
            os.makedirs(dirname(out), exist_ok=True)
        return out

    return save_file_remap


def strip_trailing_slash(path):
    if path != "/":
        path = path.rstrip("/")
    return path


def make_readonly(path):
    assert exists(path), f"{path} doesn't exist"
    run(["chmod", "-R", "a-w", path], check=True)


def make_writeable(path):
    assert exists(path), f"{path} doesn't exist"
    run(["chmod", "-R", "u+w", path], check=True)


def delete_readonly(path):
    """"Deletes a file or directory from a read-only parent directory."""
    # NOTE:  File "deletion" (unlinking) does not write to the file itself,
    # but writes to the directory to delete the hardlink to the file.  So
    # read-only files are deletable; if deleting a file gives a permissions
    # error it is because the _parent_ was readonly.
    parent = get_parent_dir(path=path)
    saved_permissions = os.stat(parent).st_mode
    make_writeable(parent)
    if isdir(path):
        rmtree(path)
    else:
        os.unlink(path)
    os.chmod(parent, saved_permissions)


def get_repeated_dir_nest(directory):
    """Checks if a directory only contains a directory of the same name.

    For instance, if `my_data/` has only one direct child, `my_data/my_data/`.
    This happens generally with tar archive creation and extraction.

    @returns The inner directory if it's nested, or None otherwise."""
    items = os.listdir(directory)
    if len(items) != 1:
        return
    (item,) = items
    nest_dir = join(directory, item)
    if isdir(nest_dir) and item == basename(directory):
        return nest_dir
    return None


def parse_uri(uri, relpath=None):
    """Parse and separate the URI into protocol and the full path"""
    uri = uri.lstrip("/")
    sep = "://"
    if sep not in uri:
        uri = expanduser(uri)
        if relpath is not None:
            uri = join(relpath, uri)
        assert uri.startswith("/"), uri
        uri = f"file://{uri}"
    protocol, path = uri.split(sep)
    return protocol, path


def canonical_uri(uri, relpath=None):
    """Remove trailing slash and return the path in {protocol}://{path}
    format"""
    protocol, path = parse_uri(strip_trailing_slash(uri))
    return f"{protocol}://{path}"


def datetime_as_filename(date=None):
    """A representation of `date` suitable for dir names

    Returns the ISO 8601 representation of `date`, but replacing : with -.
    If `date` is not specified, uses the current time."""
    if date is None:
        date = datetime.datetime.now().astimezone()
    is_naive = (date.tzinfo is None) and (date.tzinfo.utcoffset() is None)
    assert not is_naive
    return date.isoformat(timespec="seconds").replace(":", "-")


def filename_to_datetime(filename: str) -> Optional[datetime.datetime]:
    """
    This is the inverse operation of datetime_as_filename. Note that we ignore
    the microseconds since datetime_as_filename() function uses
    timespec=seconds, hence ignoring the microseconds info.
    If filename is not the output of datetime_as_filename(), then return None.
    """
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}[+-]\d{2}-\d{2}$"
    base, filename = os.path.split(filename)
    if re.match(pattern, filename):
        ret = datetime.datetime(
            year=int(filename[:4]),
            month=int(filename[5:7]),
            day=int(filename[8:10]),
            hour=int(filename[11:13]),
            minute=int(filename[14:16]),
            second=int(filename[17:19]))
        return ret
    else:
        return None


def reserve_datetime_folder(root_folder: str, date: datetime.datetime) -> str:
    """Finds the first available folder name starting from the given timestamp
    as formatted by `filename_to_datetime`. If the given timestamp already has
    a folder, then we increment the timestamp by 1 second as that is the
    resolution used by `filename_to_datetime`.

    Once an available folder is found, it is reserved by making the directory
    on the target filesystem, thus marking it as unavailable for future calls
    to this function.
    """
    date_copy = copy.deepcopy(date)
    os.makedirs(root_folder, exist_ok=True)
    while True:
        folder_name = datetime_as_filename(date_copy)
        full_path = join(root_folder, folder_name)
        try:
            os.mkdir(full_path)
            break
        except FileExistsError:
            date_copy += relativedelta(seconds=1)
    return full_path


def resolve_path(path):
    """Expands a path, expanding variables, ~, and package URLs.
    Package URLs are (wrongly!!) assumed to be congruent with Bazel Rlocation
    module names, so generally the URL logic is wrong for nearly all use cases.
    If you want to resolve URLs correctly, use a Drake PackageMap.
    """
    pkg_prefix = "package://"
    if path.startswith(pkg_prefix):
        # TODO(#16200) This should not be supported.
        return Rlocation(path[len(pkg_prefix):])
    else:
        path = expandvars(expanduser(path))
        return abspath(path)


def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def human_sorting_key(text):
    return tuple(_tryint(s) for s in re.split(r"(\d+)", text))


def human_sorted_strings(items):
    """
    Applies human sorting to a list of strings.

    See also: https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    assert isinstance(items, list)
    return list(sorted(items, key=human_sorting_key))
