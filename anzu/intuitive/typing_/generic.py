from contextlib import contextmanager
import copy
import dataclasses as dc
import difflib
from io import BytesIO
import pickle
import pprint as pp
from textwrap import dedent, indent
import types
import typing
from typing import Any
import unittest

import dill
import numpy as np

from pydrake.common import pretty_class_name
from pydrake.common.yaml import yaml_dump_typed
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialMomentum,
    SpatialVelocity,
)


def _get_typing_name(x):
    """
    Gets the name for a class or class-like object.

    For brevity, this will strip off module names for known class-like objects
    (e.g. typing objects).
    """
    if isinstance(x, type):
        if typing.get_origin(x) in (list, dict):
            return repr(x)
        else:
            return pretty_class_name(x)
    elif x is Ellipsis:
        return "..."
    else:
        s = repr(x)
        typing_prefix = "typing."
        if s.startswith(typing_prefix):
            return s[len(typing_prefix) :]
        else:
            return s


def is_cc_schema_cls(T):
    fields = getattr(T, "__fields__", None)
    return fields is not None


def isinstance_namedtuple(obj) -> bool:
    # https://stackoverflow.com/a/62692640
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )


def pformat_dataclass(cls):
    assert dc.is_dataclass(cls)
    fields = dc.fields(cls)
    assert len(fields) > 0
    out = f"@dataclass\n"
    out += f"class {_get_typing_name(cls)}:\n"
    for field in fields:
        out += f"    {field.name}: {_get_typing_name(field.type)}"
        if field.default is None:
            out += " = None"
        elif (
            field.default is not dc.MISSING
            or field.default_factory is not dc.MISSING
        ):
            out += " = <default>"
        out += "\n"
    return out


def _indent_readjust(text, prefix):
    # Readjust the inner indented portion of text; only indent the inner lines
    # (neither first nor last). Useful for items whose repr() is already
    # "well-formatted".
    lines = text.strip().splitlines()
    new_text = lines[0]
    if len(lines) > 1:
        new_text += "\n" + indent(dedent("\n".join(lines[1:-1])), prefix)
        new_text += "\n" + lines[-1].strip()
    return new_text


def pformat(obj, incr="  ", max_array_size=30):
    """
    Pretty formatting for values with more vertical whitespace, less hanging
    indents.
    """

    def sub_pformat(obj):
        txt = pformat(obj, incr=incr, max_array_size=max_array_size)
        return indent(txt, incr)

    # Try short version.
    short_len = 60
    try:
        maybe_short = pp.pformat(obj)
        if "\n" not in maybe_short and len(maybe_short) <= short_len:
            return maybe_short
    except TypeError as e:
        # TODO(eric.cousineau): Remove this hack once our pybind11 fork is
        # updated to fix the following:
        # https://bugs.python.org/issue33395
        # Seems like our (old) fork may have a bad hashable __repr__ method.
        # Via `scenario_pformat_test`, this fails first on pybind11-bound
        # enum classes.
        if "unhashable type: 'instancemethod'" in str(e):
            pass
        else:
            raise

    if isinstance(obj, list):
        out = f"[\n"
        for obj_i in obj:
            out += sub_pformat(obj_i) + ",\n"
        out += f"]"
        return out
    elif isinstance(obj, types.SimpleNamespace):
        out = f"SimpleNamespace(\n"
        for k_i, obj_i in obj.__dict__.items():
            txt = sub_pformat(obj_i)
            out += f"{incr}{k_i}={txt.strip()},\n"
        out += f")"
        return out
    elif isinstance(obj, dict):
        out = f"{{\n"
        for k_i, obj_i in obj.items():
            txt = sub_pformat(obj_i)
            out += f"{incr}{repr(k_i)}: {txt.strip()},\n"
        out += f"}}"
        return out
    elif isinstance(obj, np.ndarray) and obj.size > max_array_size:
        return f"array(<data>, shape={obj.shape}, dtype={obj.dtype})"
    elif dc.is_dataclass(obj):
        if isinstance(obj, type):
            return _get_typing_name(obj)
        else:
            cls_name = _get_typing_name(type(obj))
            out = f"{cls_name}(\n"
            for field in dc.fields(obj):
                obj_i = getattr(obj, field.name)
                txt = sub_pformat(obj_i)
                out += f"{incr}{field.name}={txt.strip()},\n"
            out += f")"
            return out
    elif is_cc_schema_cls(obj):
        # TODO(eric.cousineau): Should have better method.
        return _indent_readjust(repr(obj), incr)
    elif isinstance_namedtuple(obj):
        cls = type(obj)
        cls_name = _get_typing_name(cls)
        out = f"{cls_name}(\n"
        for field in cls._fields:
            obj_i = getattr(obj, field)
            txt = sub_pformat(obj_i)
            out += f"{incr}{field}={txt.strip()},\n"
        out += f")"
        return out
    else:
        return _indent_readjust(pp.pformat(obj), incr)


def pprint(obj):
    print(pformat(obj))


def copy_field(value):
    """
    Provides a field type for dataclass s.t. we can define a value to be cloned
    as a default.

    Example:

        @dc.dataclass
        class MyStruct:
            my_empty_list: list[int] = dc.field(default_factory=list)
            my_default_list: list[int] = copy_field([1, 2, 3])

    """
    # Make sure that our model value is unaffected.
    value = copy.deepcopy(value)
    return dc.field(default_factory=lambda: copy.deepcopy(value))


# The following methods are generally for testing.


class ContentTracedException(Exception):
    """Helps trace where exception was raised in the content."""

    def __init__(self, path, e):
        self.path = path
        self.e = e

    def __str__(self):
        message = indent(str(self.e), "  ")
        return f"{self.path}:\n{message}"


def is_pickleable(obj):
    buffer = BytesIO()
    pickle.dump(obj, buffer)
    # Simply load, but don't check value.
    buffer.seek(0)
    pickle.load(buffer)


@contextmanager
def dill_trace():
    # TODO(eric.couisneau): Remove when we have newer version of dill.
    dill.detect.trace(True)
    yield
    dill.detect.trace(False)


def is_pickleable_with_dill(obj):
    buffer = BytesIO()
    dill.dump(obj, buffer)
    # Simply load, but don't check value.
    buffer.seek(0)
    dill.load(buffer)


def is_optional(T):
    if typing.get_origin(T) in [typing.Union, types.UnionType]:
        if type(None) in typing.get_args(T):
            return True
    return False


def depth_first_on_error(value, *, check):
    """
    Applies `check` directly to `value`.

    If an error occurs, will recurse through entries in `value` and apply
    `check()` at each level in a depth-first fashion to identify the more acute
    error. This is useful to debug issues such as pickling.
    """

    def run(value, *, path):
        def recurse(value, *, subpath):
            nested_path = f"{path}{subpath}"
            try:
                run(value, path=nested_path)
            except ContentTracedException:
                raise
            except Exception as e:
                raise ContentTracedException(nested_path, e) from e

        if value is None:
            pass
        elif isinstance(value, np.ndarray):
            pass
        elif isinstance(value, (bool, int, float, str)):
            pass
        elif dc.is_dataclass(value):
            for field in dc.fields(value):
                value_i = getattr(value, field.name)
                recurse(value_i, subpath=f"/{field.name}")
        elif isinstance(value, dict):
            for k, v in value.items():
                recurse(k, subpath=f"/key={k}")
                recurse(v, subpath=f"[{k}]")
        elif isinstance(value, list):
            for i, value_i in enumerate(value):
                recurse(value_i, subpath=f"[{i}]")
        elif isinstance(value, tuple):
            for i, value_i in enumerate(value):
                recurse(value_i, subpath=f"[{i}]")
        else:
            assert False, repr(value)
        check(value)

    try:
        # First try overall.
        check(value)
    except Exception:
        # Error occurred. Dig in.
        run(value, path="")


def adjust_field_type(cls, name, new_type):
    fields = dc.fields(cls)
    field_map = {field.name: field for field in fields}
    field = field_map[name]
    field.type = new_type


def assert_equal_recursive(
    test: unittest.TestCase, a: Any, b: Any, base_path="", np_tol=0.0, msg=""
):
    """Recursively asserts equality between two objects, a and b."""

    def recurse(a, b, path):
        T = type(a)
        test.assertEqual(type(b), T, path + msg)
        if T == np.ndarray:
            test.assertEqual(a.shape, b.shape, path + msg)
            test.assertEqual(a.dtype, b.dtype, path + msg)
            np.testing.assert_allclose(
                a, b, err_msg=path + msg, atol=np_tol, rtol=0.0
            )
        elif isinstance(a, (list, tuple)):
            test.assertEqual(len(a), len(b), path + msg)
            pair_iter = zip(a, b, strict=True)
            for i, (ai, bi) in enumerate(pair_iter):
                recurse(ai, bi, f"{path}[{i}]")
        elif T == dict:
            diff = difflib.context_diff(
                list(a.keys()), list(b.keys()), fromfile="a", tofile="b"
            )
            error_msg = f"{path}\n" + "\n".join(diff)
            test.assertEqual(len(a), len(b), error_msg + msg)
            test.assertEqual(a.keys(), b.keys(), error_msg + msg)
            for k, ai in a.items():
                bi = b[k]
                recurse(ai, bi, f"{path}[{repr(k)}]")
        elif dc.is_dataclass(T):
            fields = dc.fields(a)
            test.assertEqual(dc.fields(b), fields, path + msg)
            for field in fields:
                ai = getattr(a, field.name)
                bi = getattr(b, field.name)
                recurse(ai, bi, f"{path}.{field.name}")
        elif T == RigidTransform:
            recurse(a.GetAsMatrix4(), b.GetAsMatrix4(), path + msg)
        elif T == RotationMatrix:
            recurse(a.matrix(), b.matrix(), path + msg)
        elif T == RollPitchYaw:
            recurse(a.vector(), b.vector(), path + msg)
        # TODO(dale.mcconachie) Address quaternions.
        elif T == SpatialAcceleration:
            recurse(a.get_coeffs(), b.get_coeffs(), path + msg)
        elif T == SpatialForce:
            recurse(a.get_coeffs(), b.get_coeffs(), path + msg)
        elif T == SpatialMomentum:
            recurse(a.get_coeffs(), b.get_coeffs(), path + msg)
        elif T == SpatialVelocity:
            recurse(a.get_coeffs(), b.get_coeffs(), path + msg)
        elif is_cc_schema_cls(T):
            # We dump to yaml string to take advantage of the representation
            # of C++ schemas defined using `DefSchemaUsingSerialize` which
            # don't have a equality operator for the underlying C++ type. We
            # don't cast to string directly as not all schemas have that
            # enabled.
            # TODO(dale.mcconachie, eric.cousineau): Make this more structural,
            # perhaps by converting to dict form and then recursing.
            test.assertEqual(
                yaml_dump_typed(a), yaml_dump_typed(b), path + msg
            )
        elif T == float:
            np.testing.assert_allclose(
                a, b, err_msg=path + msg, atol=np_tol, rtol=0.0
            )
        elif hasattr(a, "__eq__"):
            test.assertEqual(a, b, path + msg)
        else:
            # Check if the type as a __repr__ that doesn't have any < or > in
            # it. If so, then __repr__ is a sufficient way to evaluate equality
            # due to the round-trip requirement on __repr__ if it does not
            # contain angle brackets.
            repr_a = repr(a)
            repr_b = repr(b)
            valid = not any(char in repr_a for char in "<>")
            valid &= not any(char in repr_b for char in "<>")
            if valid:
                test.assertEqual(repr_a, repr_b, path + msg)
            # Otherwise fall back to whatever a == b would result in.
            else:
                test.assertEqual(a, b, path + msg)

    recurse(a, b, base_path)


def assert_not_equal_recursive(test, a, b):
    with test.assertRaises(AssertionError):
        assert_equal_recursive(test, a, b)
