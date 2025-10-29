"""
Operations in Python for handling configuration overrides via dict
(un)flattening.

This intended for command-line usage and for ease of integration with things
like WandB.
"""

# TODO(eric.cousineau): Synchronize this with `merge_with_defaults`.

from pydrake.common.yaml import yaml_load


def flatten_dict(config, prefix="", delim="."):
    """
    Transforms a dict of the form:

        {"top": {"mid": {"bottom": 0.25}}}

    to the following form:

        {"top.mid.bottom": 0.25}
    """
    assert isinstance(config, dict)
    out = dict()
    for k, v in config.items():
        assert isinstance(k, str), repr(k)
        assert "." not in k, repr(k)
        if isinstance(v, dict):
            v = flatten_dict(v, f"{prefix}{k}{delim}", delim=delim)
            for ki, vi in v.items():
                assert ki not in config
            out.update(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def is_flat_dict(config):
    assert isinstance(config, dict), type(config)
    for v in config.values():
        if isinstance(v, dict):
            return False
    return True


def unflatten_dict(config, delim="."):
    """
    Transforms a dict from:

        {
            "top.mid.bottom": 0.25,  # From wandb sweep controller.
            # "top.mid": 0.35,  # Error: conflict with above.
            # "top.mid": {"bottom": 0.45}  # Error: conflict with above.
        }

    to:

        {"top": {"mid": {"bottom": 0.25}}}

    See:
    - https://github.com/wandb/client/issues/982
    - https://drakedevelopers.slack.com/archives/CUJKN7Q14/p1593617201105000?thread_ts=1593616323.103800&cid=CUJKN7Q14  # noqa
    """
    # We will only process one class type.
    assert type(config) is dict
    assert is_flat_dict(config), "Cannot have nested dicts"
    nested = dict()
    for k, v in config.items():
        assert isinstance(k, str), k
        assert not isinstance(v, dict), (k, v)
        k_list = k.split(delim)
        if len(k_list) > 1:
            assert not isinstance(
                v, dict
            ), f"Cannot have nested field be a dict: {(k, v)}"
            nested_n = nested
            for k_i in k_list[:-1]:
                if k_i not in nested_n:
                    nested_n[k_i] = dict()
                nested_n = nested_n[k_i]
                assert isinstance(
                    nested_n, dict
                ), "Intermediate field must be dict"
            k_n = k_list[-1]
            assert k_n not in nested_n, "Cannot mix middle level nesting"
            nested_n[k_n] = v
        else:
            nested[k] = v
    return nested


def recursive_dict_update(a, b, strict=False):
    """
    Rescursively merges entries from b into a, letting b override any values in
    a. `a` will be updated.

    Will raise an error if field structure does not match for shared keys (e.g.
    a[k] is a dict, but b[k] is not).

    If strict, will not permit new entries.
    """
    assert isinstance(a, dict) and isinstance(b, dict), (a, b)
    out = a
    for k, b_v in b.items():
        if k in a:
            a_v = a[k]
            # Do not allow structure mismatch.
            if isinstance(b_v, dict):
                if not strict and a_v is None:
                    a_v = dict()
                assert isinstance(a_v, dict), (repr(a_v), repr(b_v))
                out[k] = recursive_dict_update(a_v, b_v, strict=strict)
            else:
                assert not isinstance(a_v, dict), repr((a_v, b_v))
                out[k] = b_v
        else:
            assert not strict, f"Cannot add new key: {k}, {b_v}"
            out[k] = b_v
    return out


def apply_flat_dict_override(raw_config, config_override, *, strict=True):
    """
    In-place override with flat values.

    Mostly used for command-line argument parsing and Jupyter notebooks, but
    can be useful in tests as well for a cohesive config override setup.

    Arguments:
        raw_config: Dictionary; can be flat or not.
        config_override: Configuration override:
            - if it's a dict, it's used directly.
            - otherwise, it's passed to `yaml_load` as data
            This must satisfy the `is_flat_dict` invariant.
    Returns:
        raw_config (mutated in place) with overridden values.
    """
    if len(config_override) == 0:
        return raw_config

    assert isinstance(raw_config, dict), type(raw_config)
    if isinstance(config_override, dict):
        config_override_flat = config_override
    else:
        config_override_flat = yaml_load(data=config_override, private=True)
        if config_override_flat is None:
            return raw_config
    assert is_flat_dict(config_override_flat), config_override_flat
    config_override = unflatten_dict(config_override_flat)
    # Not really recursive in this case, but we still want strictness.
    recursive_dict_update(raw_config, config_override, strict=strict)
    return raw_config


def apply_dict_override(raw_config, config_override):
    """
    In-place override with direct replacement; no recursive merging is done.

    Top-level keys are used to resolve nested fields. For example,
    `a.b: <new_value>` corresponds to `raw_config["a"]["b"] = <new_value>`.
    """
    if len(config_override) == 0:
        return raw_config

    assert isinstance(raw_config, dict), type(raw_config)
    if isinstance(config_override, dict):
        config_override = config_override
    else:
        config_override = yaml_load(data=config_override, private=True)
        if config_override is None:
            return raw_config

    for key, val in config_override.items():
        # split key into a path of keywords.
        keywords = key.split(".")
        to_be_overriden = raw_config
        for keyword in keywords[:-1]:
            to_be_overriden = to_be_overriden[keyword]
        to_be_overriden[keywords[-1]] = val
    return raw_config
