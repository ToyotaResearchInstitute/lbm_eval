import copy
import dataclasses as dc
from datetime import datetime
from enum import Enum
import json
import sys
import types
import typing

import numpy as np

from pydrake.common import pretty_class_name
from pydrake.common.yaml import (
    yaml_dump,
    yaml_dump_typed,
    yaml_load,
    yaml_load_typed,
)

from anzu.common.debug import iex
from anzu.common.path_util import resolve_path
from anzu.common.schema.flat_dict import (
    apply_dict_override,
    apply_flat_dict_override,
    flatten_dict,
    unflatten_dict,
)
from anzu.intuitive.time_util import datetime_to_isoformat
from anzu.intuitive.typing_.generic import (
    ContentTracedException,
    is_cc_schema_cls,
    is_optional,
    pformat,
)

# Set this to false if you are using `debug.iex` and want to jump to actual
# point of error.
USE_CONTENT_TRACE = True

# TODO(eric.cousineau): This incantantion of general `typing`-esque behavior
# should be hollowed out and instead use `pydrake`s similar affordances where
# possible.


def _try_convert(value, T):
    try:
        return T(value)
    except ValueError:
        return value


def _reconcile(value, T):
    assert isinstance(T, type), (T, value)
    if T == float:
        if isinstance(value, str):
            # Workaround:
            # https://github.com/wandb/client/issues/1129
            # https://github.com/yaml/pyyaml/issues/173
            value = _try_convert(value, float)
        assert isinstance(value, (float, int)), (T, type(value), value)
    else:
        assert isinstance(value, T), (T, value)
    return value


def _is_maybe_generic_array(cls):
    # TODO(eric): Figure out another orgnaization to avoid dep loop?
    return cls == np.ndarray


def _is_enum_cls(cls):
    return isinstance(cls, type) and issubclass(cls, Enum)


@iex
def _union_value_to_type_and_name(T, value, name=None):
    assert typing.get_origin(T) in [typing.Union, types.UnionType]
    cls_list = typing.get_args(T)
    none_type = type(None)
    can_be_none = none_type in cls_list
    if value is None:
        if can_be_none:
            return none_type, None
        else:
            raise RuntimeError(f"Specified type is not optional: {T}")

    T_actual = type(value)

    # Handle basic types which won't have a tag.
    if T_actual in (bool, int, float, str) and T_actual in cls_list:
        return T_actual, None

    if T_actual not in cls_list:
        if len(cls_list) == 1:
            # Should only be one option.
            (T_actual,) = cls_list
        elif name is not None:
            # Name given, use that.
            cls_map = {pretty_class_name(x): x for x in cls_list}
            T_actual = cls_map[name]
        else:
            # Try to infer :(
            is_dict = isinstance(value, dict)
            is_list = isinstance(value, list)
            is_array = isinstance(value, np.ndarray)
            is_str = isinstance(value, str)
            assert is_list or is_array or is_dict or is_str, repr((value, T))
            # Try to resolve against possibilities.
            matches = []
            for cls in cls_list:
                cls_origin = typing.get_origin(cls)
                if is_dict and cls_origin == dict:
                    # TODO(eric): Check inner type.
                    matches.append(cls)
                elif is_list and cls_origin == list:
                    # TODO(eric): Check inner type.
                    matches.append(cls)
                elif (is_list or is_array) and _is_maybe_generic_array(cls):
                    matches.append(cls)
                elif is_str and _is_enum_cls(cls_origin):
                    matches.append(cls)
            if len(matches) == 0:
                # Assume first.
                assert len(cls_list) > 0
                T_actual = cls_list[0]
            elif len(matches) != 1:
                raise RuntimeError(f"Ambiguous: {value}, Matches: {matches}")
            else:
                (T_actual,) = matches
    # Only provide name iff it's a dataclass, C++ schema, or enum.
    if (
        dc.is_dataclass(T_actual)
        or is_cc_schema_cls(T_actual)
        or _is_enum_cls(T_actual)
    ):
        name = pretty_class_name(T_actual)
    else:
        name = None
    return T_actual, name


def to_dict(value, *, T=None, enum_value_cls=str, path=""):
    """
    Converts a value to a dict (for YAML).

    If T is supplied, it defines the intended schema for the type.
    T must be supplied for any type without an explicit schema.
    """
    assert enum_value_cls in [str, int]
    if T is None:
        assert dc.is_dataclass(value)
        T = type(value)

    def recurse(value, *, T, subpath):
        nested_path = f"{path}{subpath}"
        try:
            return to_dict(
                value, T=T, enum_value_cls=enum_value_cls, path=nested_path
            )
        except ContentTracedException:
            if not USE_CONTENT_TRACE:
                raise
            # Meh.
            print(pformat(value), file=sys.stderr)
            raise
        except Exception as e:
            if not USE_CONTENT_TRACE:
                raise
            # Meh.
            print(pformat(value), file=sys.stderr)
            raise ContentTracedException(nested_path, e) from e

    if T == type(None):  # noqa
        assert value is None
        return None
    elif typing.get_origin(T) in [typing.Union, types.UnionType]:
        T_actual, tag = _union_value_to_type_and_name(T, value)
        if T_actual is None:
            # Optional value.
            return None
        out = recurse(value, T=T_actual, subpath=f"/@tag={tag}")
        if isinstance(out, dict) and tag is not None:
            # Ensure tag is present.
            out["_tag"] = f"!{tag}"
        return out
    elif isinstance(value, np.ndarray):
        value = _reconcile(value, T)
        return value.tolist()
    elif isinstance(value, (bool, int, float, str)):
        value = _reconcile(value, T)
        return value
    elif dc.is_dataclass(T):
        value = _reconcile(value, T)
        out = dict()
        for field in dc.fields(value):
            value_i = getattr(value, field.name)
            out[field.name] = recurse(
                value_i, T=field.type, subpath=f"/{field.name}"
            )
        return out
    elif is_cc_schema_cls(T):
        data = yaml_dump_typed(value)
        out = yaml_load(data=data)
        return out
    elif isinstance(value, dict):
        if T == dict:
            return copy.deepcopy(value)
        else:
            assert typing.get_origin(T) == dict
            K, V = typing.get_args(T)
            out = dict()
            for k, v in value.items():
                k = recurse(k, T=K, subpath=f"/key={k}")
                out[k] = recurse(v, T=V, subpath=f"[{k}]")
            return out
    elif isinstance(value, list):
        if T == list:
            return copy.deepcopy(value)
        else:
            assert typing.get_origin(T) == list
            (U,) = typing.get_args(T)
            out = [
                recurse(value_i, T=U, subpath=f"[{i}]")
                for i, value_i in enumerate(value)
            ]
            return out
    elif isinstance(value, tuple):
        if T == tuple:
            return copy.deepcopy(value)
        else:
            assert typing.get_origin(T) == tuple
            (U,) = typing.get_args(T)
            out = tuple(
                [
                    recurse(value_i, T=U, subpath=f"[{i}]")
                    for i, value_i in enumerate(value)
                ]
            )
            return out
    elif _is_enum_cls(T):
        # Use string name.
        return value.name
    elif isinstance(value, datetime):
        # Converts to iso format using system timezone.
        return datetime_to_isoformat(value)
    else:
        assert False, (T, value)


def _is_field_optional(field):
    return is_optional(field.type) or (
        field.default is not dc.MISSING
        or field.default_factory is not dc.MISSING
    )


def from_dict(
    T,
    value,
    *,
    strict=True,
    allow_heuristic=True,
    path="",
    infer_type_from_value=False,
):
    """
    Converts a dict (or POD type) to a value.

    Arugments:
        T: Class/schema to guide parsing.
        value: The dict (or POD type) to be parsed.
        strict: Fail on missing or unused keys.
        allow_heuristic: Allow schema transition/transform hueristics to be
            done (e.g. for culling / migrating old fields from old checkpoints,
            etc.). As an example:

            @dataclass
            class MyClass:
                new_name: str

                @staticmethod
                def from_dict_heuristic(value):
                    value, _ = optional_exclusive_key_rename(
                        value, old="old_name", new="new_name"
                    )
                    return value
        infer_type_from_value: Set this to True to infer the types of the
            attributes from given `value` for dataclass objects. The default is
            False, in which case types will be extracted from default values.
    """

    def recurse(T, value, subpath):
        nested_path = f"{path}{subpath}"
        try:
            return from_dict(
                T,
                value,
                strict=strict,
                allow_heuristic=allow_heuristic,
                path=nested_path,
                infer_type_from_value=infer_type_from_value,
            )
        except ContentTracedException:
            if not USE_CONTENT_TRACE:
                raise
            # Meh.
            print(pformat(value), file=sys.stderr)
            raise
        except Exception as e:
            if not USE_CONTENT_TRACE:
                raise
            # Meh.
            print(pformat(value), file=sys.stderr)
            raise ContentTracedException(nested_path, e) from e

    if T is np.ndarray:
        out = np.asarray(value)
        return out
    elif T in (bool, int, float, str):
        value = _reconcile(value, T)
        return value
    elif is_cc_schema_cls(T):
        assert isinstance(value, dict), (type(value), T)
        data = yaml_dump(value)
        return yaml_load_typed(schema=T, data=data)
    elif dc.is_dataclass(T):
        assert isinstance(value, dict), (type(value), T)
        kwargs = dict()
        fields = dc.fields(T)
        if allow_heuristic:
            heuristic = getattr(T, "from_dict_heuristic", None)  # Meh.
            if heuristic is not None:
                # Transform.
                value = heuristic(value)
        if strict:
            config_names = set(value.keys())
            schema_all_names = set(field.name for field in fields)
            schema_required_names = set(
                field.name for field in fields if not _is_field_optional(field)
            )
            if "_tag" in config_names:
                config_names.remove("_tag")
            assert config_names <= schema_all_names, (
                f"Mismatch between config and schema for {repr(T)}:\n"
                f"  In config, not in schema: "
                f"{config_names - schema_all_names}\n"
                f"  Required in schema, not in config: "
                f"{schema_required_names - config_names}"
            )
        for field in fields:
            if _is_field_optional(field) and field.name not in value:
                continue
            # TOOD(eric.cousineau): Find out why this assert was not
            # effectively handled by above checks.
            assert field.name in value, (field.name, T)
            value_i = value[field.name]
            if infer_type_from_value:
                value_i = recurse(type(value_i), value_i, f"/{field.name}")
            else:
                value_i = recurse(field.type, value_i, f"/{field.name}")

            kwargs[field.name] = value_i
        return T(**kwargs)
    elif typing.get_origin(T) == dict:
        # I.e. a native python GenericAlias.
        K, V = typing.get_args(T)
        out = dict()
        for k, v in value.items():
            if k == "_tag":
                continue
            k = recurse(K, k, subpath=f"/key={k}")
            out[k] = recurse(V, v, subpath=f"[{k}]")
        return out
    elif typing.get_origin(T) == list:
        # I.e. a native python GenericAlias.
        (U,) = typing.get_args(T)
        out = [
            recurse(U, value_i, subpath=f"[{i}]")
            for i, value_i in enumerate(value)
        ]
        return out
    elif typing.get_origin(T) == tuple:
        (U,) = typing.get_args(T)
        out = tuple(
            [
                recurse(U, value_i, subpath=f"[{i}]")
                for i, value_i in enumerate(value)
            ]
        )
        return out
    elif _is_enum_cls(T):
        T_map = {value.name: value for value in T}
        return T_map[value]
    elif T == type(None):  # noqa
        assert value is None
        return value
    elif typing.get_origin(T) in [typing.Union, types.UnionType]:
        tag = None
        hint = None

        # Handle cases where there is a _tag
        if isinstance(value, dict):
            tag = value.get("_tag")
        if tag is not None:
            assert tag.startswith("!"), tag
            hint = tag[1:]

        T_actual, tag = _union_value_to_type_and_name(T, value, hint)
        if T_actual is None:
            # Optional, not present.
            return None

        return recurse(T_actual, value, subpath=f"/@tag={tag}")
    elif T == datetime:
        return datetime.fromisoformat(value)
    elif T in (dict, list, tuple):
        return T(copy.deepcopy(value))
    else:
        assert False, T


def yaml_load_schema(T, *, data=None, filename=None):
    raw = yaml_load(data=data, filename=filename, private=True)
    return from_dict(T, raw)


def yaml_dump_schema(value, filename=None, *, T=None):
    raw = to_dict(value, T=T)
    return yaml_dump(raw, filename=filename)


def json_load_schema(T, *, data, is_flat=False):
    raw = json.loads(data)
    if is_flat:
        raw = unflatten_dict(raw)
    param = from_dict(T=T, value=raw)
    return param


def optional_exclusive_key_rename(
    d, *, old, new, inplace=False, transform=lambda x: x
):
    """
    If `old` is a key present in the dictionary, then `old` is replaced with
    `new`.

    For use with `from_dict(*, allow_heuristic=True)`.

    Arugments:
        d: Dictionary.
        old: Old key.
        new: New key.
        inplace:
            If True, `d` is mutated directly (useful for nesting).
            If False, a shallow copy of `d` is always returned (with or without
                replacement).
        transform: Transform the value (e.g. negate).

    Returns:
        (d, used), where `used` indicates whether a repalcement happened.
    """
    used = False
    if not inplace:
        d = copy.copy(d)
    if old in d:
        assert new not in d, (old, new)
        d[new] = transform(d[old])
        del d[old]
        used = True
    return d, used


def maybe_delete_key(d, k):
    used = False
    if k in d:
        used = True
        del d[k]
    return d, used


def maybe_delete_path(value, p):
    if isinstance(p, str):
        p = p.split(".")
    assert isinstance(p, list)
    sub = value
    for piece in p[:-1]:
        sub = sub.get(piece)
        if sub is None:
            break
    else:
        sub.pop(p[-1], None)
    return value


def maybe_delete_paths(value, ps):
    for p in ps:
        maybe_delete_path(value, p)
    return value


def maybe_subtract_subset(full, required_keys):
    """
    If all of ``required_keys`` are present in ``full``, then these keys and
    values are subtracted (in-place) from ``full`` and returned.
    Otherwise, None is returned.
    """
    assert isinstance(full, dict)
    assert isinstance(required_keys, set)
    sub = dict()
    for key in required_keys:
        if key in full:
            sub[key] = full[key]
    if set(sub.keys()) == required_keys:
        for key in required_keys:
            del full[key]
        return sub
    else:
        return None


def apply_flat_schema_override(config, config_override, **kwargs):
    """
    Wraps `apply_flat_dict_override` with schema parsing.

    However, this does *not* apply in-place overrides.
    """
    T = type(config)
    assert dc.is_dataclass(T), T
    # N.B. It's important to use this for schema overrides so we populate all
    # default entries.
    raw_config = to_dict(config, T=T, **kwargs)
    apply_flat_dict_override(raw_config, config_override)
    config = from_dict(T, raw_config, **kwargs)
    return config


def apply_direct_schema_override(config, config_override, **kwargs):
    """
    Wraps around apply_dict_override() with schema parsing.
    Does not apply in-place, returns a new config.

    """
    T = type(config)
    assert dc.is_dataclass(T), T
    raw_config = to_dict(config, T=T, **kwargs)
    apply_dict_override(raw_config, config_override)
    config = from_dict(T, raw_config, **kwargs)
    return config


def load_schema_with_overrides(
    T,
    *,
    config_file,
    scenario_name,
    config_override=None,
    config_override_unflat=None,
):
    raw_config_all = yaml_load(
        filename=resolve_path(config_file), private=True
    )
    raw_config = raw_config_all[scenario_name]
    config = from_dict(T, raw_config)
    if config_override is not None:
        config = apply_flat_schema_override(config, config_override)
    if config_override_unflat is not None:
        config = apply_direct_schema_override(config, config_override_unflat)
    return config


def flatten_schema_with_prefix(obj, prefix):
    flat = flatten_dict(to_dict(obj))
    new_flat = dict()
    for key, value in flat.items():
        new_flat[f"{prefix}{key}"] = value
    return new_flat


def asdict_shallow(obj, *, dict_factory=dict):
    """Same as `asdict()`, but shallow copies."""
    # Adapted from
    # https://github.com/python/cpython/blob/v3.7.0/Lib/dataclasses.py#L1028
    if not dc.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    if dc.is_dataclass(obj):
        result = []
        for f in dc.fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
            for k, v in obj.items()
        )
    else:
        return obj
