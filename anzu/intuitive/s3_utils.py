import errno
import os
from os.path import expanduser, expandvars
import re
import shutil
import subprocess
import sys
from typing import List, Optional

import boto3

from anzu.common.runfiles import Rlocation

DEFAULT_AWS_PROFILE = "manip-cluster"
S3_LBM_BUCKET_NAME = "robotics-manip-lbm"
S3_LBM_BUCKET_NAME_US_WEST = "robotics-manip-lbm-us-west-2"
S3_LBM_BUCKET = f"s3://{S3_LBM_BUCKET_NAME}"
S3_LBM_EFS_PATH = f"{S3_LBM_BUCKET}/efs"


def assert_s3_access(
    path: str,
    extras: List[str] = [],
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
):
    proc = exec_s3_ls(path, extras, aws_profile)
    if "InvalidAccessKeyId" in proc.stderr:
        error = f"""
Unable to access s3 path {path}; this program cannot proceed.

If on a local machine, make sure you have set AWS_PROFILE to the
correct value or have otherwise configured your AWS credentials.

`aws s3 ls` output:
stdout:

{proc.stdout}

stderr:

{proc.stderr}
"""
        raise AssertionError(error)


def exec_s3_ls(
    path: str,
    extras: List[str] = [],
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
):
    aws_bin = Rlocation("anzu/tools/aws")
    cmd = [aws_bin, "s3", "ls", path] + extras
    my_env = os.environ.copy()
    if aws_profile:
        my_env["AWS_PROFILE"] = aws_profile
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env,
        text=True,
    )


def exec_s3_cp(
    src: str,
    dest: str,
    extras: List[str] = [],
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
    print_to_console: bool = True,
):
    # TODO(sfeng): move all the s3 stuff together.
    aws_bin = Rlocation("anzu/tools/aws")
    cmd = [aws_bin, "s3", "cp", src, dest] + extras
    my_env = os.environ.copy()
    if aws_profile:
        my_env["AWS_PROFILE"] = aws_profile

    if print_to_console:
        print("====================================")
        print(f"exec:  {' '.join(cmd)}")
        print("====================================")

    os.makedirs(expanduser("~/tmp"), exist_ok=True)
    tmp_log = expanduser("~/tmp/s3_cp.log")
    with open(tmp_log, "wb") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=my_env)
        for c in iter(lambda: process.stdout.read(1), b""):
            if print_to_console:
                sys.stdout.buffer.write(c)
            f.write(c)

        # check aws s3 return code.
        process.communicate()
        assert process.returncode == 0, process.returncode

    with open(tmp_log, "rb") as f:
        outputs = f.readlines()
    # bytes to string, and string new line
    outputs = [line.decode().rstrip() for line in outputs]
    return cmd, outputs


def maybe_exec_s3_cp(src, dest, *args, **kwargs):
    """
    Same as exec_s3_cp(), but will return True if success, or will catch
    errors and return False.
    """
    try:
        exec_s3_cp(src, dest, *args, **kwargs)
        return True
    except Exception as e:
        print(f"Error copying from {repr(src)} to {repr(dest)}:\n{repr(e)}")
        return False


def exec_s3_sync(
    *,
    src,
    dest,
    extras=[],
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
    print_to_console=True,
):
    # boto3 doesnt seem to have the equivalent of aws s3 sync, so exec
    # through subprocess.
    aws_bin = Rlocation("anzu/tools/aws")
    cmd = [aws_bin, "s3", "sync", src, dest] + extras
    my_env = os.environ.copy()
    if aws_profile:
        my_env["AWS_PROFILE"] = aws_profile
    print("====================================")
    print(f"exec: {' '.join(cmd)}")
    print("====================================")
    # this logs to both the terminal and a tmp file, incase we need to
    # parse the output from aws s3 sync later.
    os.makedirs(expanduser("~/tmp"), exist_ok=True)
    tmp_log = expanduser("~/tmp/s3_sync.log")
    with open(tmp_log, "wb") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=my_env)
        for c in iter(lambda: process.stdout.read(1), b""):
            if print_to_console:
                sys.stdout.buffer.write(c)
            f.write(c)

        # check aws s3 sync return code.
        process.communicate()
        assert process.returncode == 0, process.returncode

    with open(tmp_log, "rb") as f:
        outputs = f.readlines()
    # bytes to string, and string new line
    outputs = [line.decode().rstrip() for line in outputs]
    return cmd, outputs


def exec_s3_rm(
    *,
    path_to_delete: str,
    extras: List[str] = [],
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
):
    aws_bin = Rlocation("anzu/tools/aws")
    cmd = [aws_bin, "s3", "rm", path_to_delete] + extras
    my_env = os.environ.copy()
    if aws_profile:
        my_env["AWS_PROFILE"] = aws_profile
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env,
        text=True,
    )


def efs_path_to_s3_path(
    efs_path,
    s3_bucket=S3_LBM_BUCKET,
):
    efs_path = expandvars(expanduser(efs_path))
    home_path = expandvars(expanduser("~"))
    assert efs_path.startswith(
        os.path.join(home_path, "efs")
    ), f"  {efs_path} not a valid efs path"

    s3_bucket.rstrip(os.sep)
    parts = efs_path.split(os.sep)
    idx = parts.index("efs")
    path = "/".join(parts[idx:])

    s3_path = s3_bucket + "/" + path
    return s3_path


def convert_local_data_path_to_s3_path(
    local_path: str,
    local_root_prefix: str,
) -> str:
    """
    Example:
        Input:
            local_path: "~/tmp/data/tasks/BimanualVacuum/maverick/real/bc/teleop/2023-09-26T17-48-14-04-00"  # noqa
            local_root_prefix: "~/tmp"

        Output:
            "s3://robotics-manip-lbm/efs/data/tasks/BimanualVacuum/maverick/real/bc/teleop/2023-09-26T17-48-14-04-00"  # noqa
    """
    assert local_path.startswith(
        local_root_prefix
    ), f"{local_path} doesn't match {local_root_prefix}"
    remain_path = local_path[len(local_root_prefix) :]
    assert remain_path.startswith("/data/tasks")
    return f"{S3_LBM_EFS_PATH}{remain_path}"


def s3_path_to_bucket_and_key(s3_path: str) -> tuple[str, str]:
    """
    Example:
        Input: s3://bucket_name/path/to/data

        Output:
            ("bucket_name", "path/to/data")
    """
    if not is_s3_path(s3_path):
        raise ValueError(
            f"Invalid s3 path, must start with s3://: [{s3_path}]"
        )
    bucket_and_key = s3_path[len("s3://") :]
    index = bucket_and_key.find("/")
    bucket = bucket_and_key[:index]
    key = bucket_and_key[index + 1 :]
    return bucket, key


def _abs_efs_path_to_tilde_path(path: str):
    """Lexically manipulates `path` to try to guess what the user intended.
    As with all heuristics, this is extrordinarily brittle and should not be
    used in production.
    """
    parts = path.split(os.sep)
    assert "efs" in parts, f"{path} not valid efs path"

    index = parts.index("efs")
    parts = ["~"] + parts[index:]

    return os.sep.join(parts)


def download_from_s3(
    path: str = None,
    save_dir: str = "~/tmp",
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
) -> str:
    """
    Downloads a file from either an EFS or S3 path to a local save_dir.

    The input path may be specified either as a mounted file-path
    (e.g., '~/efs/...'), or as an S3 path (e.g., 's3://...').
    In both cases, the specified save_dir is used to download the file
    (if not already present). If an file-system type path is provided,
    a corresponding path in the S3_LBM_BUCKET is used to locate the
    referenced file and download it from S3 to the local save_dir.

    Args:
        path: The S3 path to the file to be downloaded
            (e.g., 's3://bucket_name/path/to/file').
        save_dir (str): The local directory where the file should be saved.
            Defaults to '~/tmp'.
        aws_profile (Optional[str]): The AWS CLI profile to use for
            authentication. Defaults to DEFAULT_AWS_PROFILE.

    Returns:
        str: The path to the downloaded file in the cache (save_dir).
    """
    if not save_dir.endswith("/"):
        save_dir = save_dir + "/"
    home_path = expandvars(expanduser("~"))
    save_path = ""
    s3_path = ""
    efs_path = ""

    if path.startswith("s3://"):
        # Native s3 path. Proceed directly cache / return.
        # sagemaker training checkpoint: s3://robotics-manip-lbm/sagemaker_outputs/main-single-task/main-bimanual-place-tape-int-training-2024-08-06-08-35-04/epoch_400-step_22456.ckpt # noqa
        s3_path = path
        save_path = path.replace("s3://", save_dir)
    else:
        # file-system path. Remap to equivalent path on S3_LBM_BUCKET
        # and proceed to cache / return.
        # eg. non-sagemaker training checkpoint: "~/efs/results/diffusion_policy/outputs/2024.02.10/03.12.09_unet_e2e_bimanual_transfer_banana_from_purple_bowl_to_ziploc/checkpoints/epoch=999-val_action_mse_ema=0.003215.ckpt" # noqa
        if _is_efs_path(path):
            efs_path = path
            # If it is an efs, try to map to s3
            # Generate a tilde path
            path = _abs_efs_path_to_tilde_path(path)
            s3_path = efs_path_to_s3_path(
                expandvars(expanduser(path)), S3_LBM_BUCKET
            )
        expanded_path = expandvars(expanduser(path))
        save_path = expanded_path.replace(
            os.path.join(home_path, "efs") + "/", save_dir
        )

    efs_local_path = expandvars(expanduser(efs_path))
    save_local_path = expandvars(expanduser(save_path))

    if os.path.exists(save_local_path):
        # Already present in cache
        return save_local_path
    elif maybe_exec_s3_cp(
        s3_path,
        save_local_path,
        aws_profile=aws_profile,
        print_to_console=True,
    ):
        # downloaded from s3 successfully, so return
        return save_local_path
    elif os.path.exists(efs_local_path):
        print(f"copy from {efs_path} to {save_path}")
        os.makedirs(os.path.dirname(save_local_path), exist_ok=True)
        shutil.copyfile(
            efs_local_path,
            save_local_path,
        )
        return save_local_path
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            f"{path}",
        )


def upload_efs_to_s3(
    efs_path: str,
    s3_bucket: str = S3_LBM_BUCKET,
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
):
    efs_path = expandvars(expanduser(efs_path))
    s3_path = efs_path_to_s3_path(efs_path, s3_bucket)

    if maybe_exec_s3_cp(
        efs_path,
        s3_path,
        aws_profile=aws_profile,
        print_to_console=True,
    ):
        return s3_path
    else:
        return None


def _is_efs_path(path: str | None) -> bool:
    if path is None:
        return False
    parts = path.split(os.sep)
    return "efs" in parts


def is_s3_path(path: str | None) -> bool:
    if path is None:
        return False
    return path.startswith("s3://")


def _is_efs_or_s3_path(path: str | None) -> bool:
    return _is_efs_path(path) or is_s3_path(path)


def maybe_download_from_s3(
    path: str = None,
    save_dir: str = "~/tmp",
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
) -> str:
    """
    Calls `download_from_s3` if the input path is an s3 or efs path. Otherwise
    returns the unmodified path doing no work.

    See `download_from_s3` for more details.
    """
    if _is_efs_or_s3_path(path):
        return download_from_s3(path, save_dir, aws_profile)
    else:
        return path


def s3_put_object(s3_path: str, data: bytes | str, **kwargs):
    bucket, key = s3_path_to_bucket_and_key(s3_path)
    s3_client = boto3.client("s3")
    return s3_client.put_object(Body=data, Bucket=bucket, Key=key, **kwargs)


def copy_checkpoint_cross_region(
    s3_uri: str,
    aws_profile: Optional[str] = DEFAULT_AWS_PROFILE,
) -> str | None:
    """Copy checkpoint folder from the us-west-2 to the us-east-1 LBM
    checkpoint bucket or vice versa. Will infer region from the s3_uri.

    This function is specific to the LBM aws sagemaker output buckets
    (i.e., s3://robotics-manip-lbm-us-west-2 and s3://robotics-manip-lbm).

    Args:
        s3_uri: Full S3 URI (a directory, not a checkpoint file - e.g.,
            s3://robotics-manip-lbm-us-west-2/path/to/checkpoint/). Must be in
            the S3_LBM_BUCKET_NAME_US_WEST or S3_LBM_BUCKET_NAME bucket.
        aws_profile: Name of the profile to use when performing aws operations.

    Returns:
        The s3 uri of the copied checkpoint or `None` if the copy failed.
    """
    if not s3_uri.endswith("/"):
        s3_uri += "/"
    src_bucket, key = s3_path_to_bucket_and_key(s3_uri)

    if src_bucket == S3_LBM_BUCKET:
        dst_bucket = S3_LBM_BUCKET_NAME_US_WEST
    elif src_bucket == S3_LBM_BUCKET_NAME_US_WEST:
        dst_bucket = S3_LBM_BUCKET_NAME
    else:
        raise ValueError(f"Unknown bucket [{src_bucket}]")
    dst_uri = f"s3://{dst_bucket}/{key}"

    # First verify that there is a checkpoint file at the requested location.
    proc = exec_s3_ls(s3_uri, aws_profile=aws_profile)
    if proc.returncode != 0:
        print(
            f"""[aws s3 ls {s3_uri}] failed with {proc.returncode}

            stdout:

            {proc.stdout}

            stderr:

            {proc.stderr}

            """
        )
        return None
    # TODO(lbm#779) Update this list when our requirements for what it means to
    # be a valid checkpoint are updated.
    expected_suffixes = ["metadata.yaml"]
    missing_suffixes = []
    for suffix in expected_suffixes:
        if re.search(f"{suffix}\n", proc.stdout) is None:
            missing_suffixes.append(suffix)
    if len(missing_suffixes) > 0:
        print(
            f"Expected to find {expected_suffixes}, but could not find "
            f"{missing_suffixes} in stdout:\n{proc.stdout}"
        )
        return None

    try:
        print(f"Syncing checkpoint {s3_uri} to {dst_uri}")
        # Use aws cli sync (as opposed to boto3) since it appears to be
        # faster and handles only copying when necessary.

        exec_s3_sync(
            src=s3_uri,
            dest=dst_uri,
            aws_profile=aws_profile,
            # Avoid printing to the console because
            # this causes an error when run from a jupyter notebook.
            print_to_console=False,
        )
    except Exception as e:
        print(f"Sync failed with exception: {e}")
        return None
    print("Sync succeeded")
    return dst_uri
