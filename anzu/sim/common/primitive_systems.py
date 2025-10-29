"""
Additional primitives for the Drake systems framework.
"""

import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.primitives import (
    ConstantValueSource,
    ConstantVectorSource_,
)


def is_ndarray_like(x):
    # Determines if x could be ndarray-compatible.
    if isinstance(x, np.ndarray):
        return True
    elif isinstance(x, list):
        if len(x) == 0:
            raise RuntimeError("Unable to infer vector-or-value semantics")
        scalar_types = (int, float)
        if isinstance(x[0], scalar_types):
            return True
        else:
            return False
    else:
        return False


def ConstantSource(value):
    """Sugar function to create either a ConstantVectorSource or
    ConstantValueSource based on an input value."""
    if is_ndarray_like(value):
        # Use template to permit different dtypes.
        return ConstantVectorSource_(value)
    else:
        return ConstantValueSource(AbstractValue.Make(value))


def fix_port(port, parent_context, value, *, resolve=True):
    if resolve:
        context = port.get_system().GetMyContextFromRoot(parent_context)
    else:
        context = parent_context
    return port.FixValue(context, value)


def eval_port(port, parent_context, *, resolve=True):
    if resolve:
        context = port.get_system().GetMyContextFromRoot(parent_context)
    else:
        context = parent_context
    return port.Eval(context)


def fix_port_via_allocate(context, port):
    return port.FixValue(context, port.Allocate())
