"""
CVXPY Utilities
===============
"""

from __future__ import annotations

import numpy as np
from numpy.exceptions import AxisError

_PATCH_ATTR = "_quantdata_cvxpy_sum_shape_patch"


def patch_cvxpy_sum_shape_from_args() -> bool:
    """
    Patch CVXPY's Sum.shape_from_args() to avoid reducing uninitialized memory.

    CVXPY 1.8.2 computes the output shape of cp.sum() by calling
    ``np.sum(np.empty(shape))``. With NumPy 2.4 this can raise a spurious
    ``RuntimeWarning: overflow encountered in reduce`` while the expression is
    being validated. Replacing ``np.empty`` with ``np.zeros`` preserves the
    shape inference while avoiding the warning.

    Returns ``True`` when the patch is applied and ``False`` when it is already
    active.
    """
    import cvxpy.atoms.affine.sum as cvxpy_sum

    current = cvxpy_sum.Sum.shape_from_args
    if getattr(current, _PATCH_ATTR, False):
        return False

    def safe_shape_from_args(self):
        try:
            return np.sum(
                np.zeros(self.args[0].shape, dtype=np.float64),
                axis=self.axis,
                keepdims=self.keepdims,
            ).shape
        except (ValueError, AxisError, TypeError) as exc:
            raise ValueError(f"Invalid arguments for cp.sum: {exc}") from exc

    setattr(safe_shape_from_args, _PATCH_ATTR, True)
    setattr(safe_shape_from_args, "_quantdata_original", current)
    cvxpy_sum.Sum.shape_from_args = safe_shape_from_args
    return True

