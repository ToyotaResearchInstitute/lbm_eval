import os
import uuid

from pydrake.multibody.parsing import PackageMap


def _xacro_process(*, input_file_name: str, mappings: dict):
    """A wrapper around xacro.process that defers importing xacro until we
    actually need it at runtime, which reduces our dependency footprint for
    open-source LBM Eval.
    """
    import xacro
    return xacro.process(input_file_name=input_file_name, mappings=mappings)


def resolve_xacro_model(xacro_filename: str,
                        arg_mappings: dict[str, str | float],
                        package_map: PackageMap) -> str:
    """
    Using the ROS library Xacro (http://wiki.ros.org/xacro), this resolves
    xacro_filename using arg_mappings (argument mappings) which are
    specified by the given Xacro filename. The resolved Xacro contents are
    written to a file that can be resolved in the current package map.

    Args:
        xacro_filename: Package URI prefixed-path to the Xacro XML file to be
                        processed. Raises if this argument is empty. This can
                        only resolve URIs in the runfiles package.
        arg_mappings: Dictionary of argument mappings which are specified in
                      the xacro_filename file.
    Returns:
        str: String containing package URI to the written file containing the
             resolved Xacro XML output.
    """
    if not xacro_filename:
        raise FileNotFoundError("xacro_filename argument must not be empty.")
    pkg_prefix = "package://"
    if not xacro_filename.startswith(pkg_prefix):
        raise FileNotFoundError(
            "xacro_filename must include the Package URI prefix "
            f"'{pkg_prefix}'."
        )
    model_absolute_path = package_map.ResolveUrl(xacro_filename)

    # Wrap the arg mappings as strings for Xacro processing.
    arg_string_mappings = dict()
    for key in arg_mappings:
        arg_string_mappings[key] = str(arg_mappings[key])
    # Call out to Xacro with args mappings to resolve the model.
    resolved_model = _xacro_process(input_file_name=model_absolute_path,
                                    mappings=arg_string_mappings)

    # Write the resolved model string to a file, and then
    # return the string of the resolved model file location.
    #
    # To construct the resolved model filename,
    # take the xacro_filename base name, remove any additional extensions,
    # append an uuid, and then add an appropriate urdf or sdf extension.
    (model_package_common_path, base_name_xacro) = os.path.split(
        xacro_filename)
    (model_absolute_common_path, base_name) = os.path.split(
        model_absolute_path)
    assert base_name_xacro == base_name
    extension = None
    extension_list = list()
    # This loop should yield a base_name free of extensions, and a list
    # of all extensions beyond the base_name.
    while extension not in ('', '.'):
        base_name_with_extensions = base_name
        (base_name, extension) = os.path.splitext(base_name_with_extensions)
        extension_list.append(extension)
    extension = '.urdf'
    if extension not in extension_list:
        extension = '.sdf'
    # Use the first 8 characters of a UUID to avoid naming collisions
    unique_id = str(uuid.uuid4())[:8]
    model_name_with_uuid = ''.join((base_name, '_', unique_id, extension))

    # Use the absolute path to write the resolved model, but return the
    # resolved model's package URI prefixed filepath for Scenario 'item'
    # processing.
    # TODO(imcmahon): Anzu Issue #11823.
    model_name_with_absolute_path = os.path.join(
        model_absolute_common_path,
        model_name_with_uuid)
    model_name_with_package_path = os.path.join(
        model_package_common_path,
        model_name_with_uuid)
    with open(model_name_with_absolute_path, 'w') as file:
        file.write(resolved_model)
    return model_name_with_package_path
