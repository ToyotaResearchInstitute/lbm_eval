import math

import numpy as np


def acos_safe(x: float, tol: float = 1e-12) -> float:
    """
    Return acos(x) if x is approximately within [-1, 1], throw an exception
    otherwise.
    """
    if x > 1 + tol or x < -1 - tol:
        raise Exception(f"acos_safe: {x=}, is out-of-range.")
    x = np.clip(x, -1, 1)
    return math.acos(x)
