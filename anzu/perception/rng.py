"""
Random number generation utilities using common libraries:

- `random` from Python standard library
- `np.random` from NumPy
- (optional) `torch` from PyTorch
"""

import random

import numpy as np

_has_torch = False
try:
    import torch

    _has_torch = True
except ImportError:
    pass


def seed(value, seed_torch_cuda=True):
    """Set seed for deterministic behavior with RNG."""
    random.seed(value)
    np.random.seed(value)
    if _has_torch:
        torch.manual_seed(value)
        if torch.cuda.is_available() and seed_torch_cuda:
            torch.cuda.manual_seed(value)
