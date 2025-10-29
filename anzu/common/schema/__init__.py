import copy


# TODO(jeremy.nimmer) This function should move into Drake eventually.
def merge_with_defaults(*, data, defaults):
    """Python dict version of the merge semantics in YamlLoadWithDefaults.
    Returns a copy of defaults with the new data applied atop it.
    Both data and defaults should be dicts.
    """
    merged = copy.deepcopy(defaults)
    for key, value in data.items():
        if isinstance(value, dict):
            if "_tag" in value:
                merged[key] = value
            else:
                merged[key] = merge_with_defaults(
                    data=value, defaults=defaults.get(key, dict()))
        else:
            merged[key] = value
    return merged
