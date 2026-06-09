"""
Jax Base
^^^^^^^^

Base import for jax based computations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pyexpat import model

import numpy as nnp
import jax as jax
import jax.numpy as jnp
import flax as flax
import optax as optax
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from cdxcore.version import version

@version("1.0")
class BaseEnvironment(object):
    """
    Base environment class to control overall jax model behaviour.

    Parameters
    ----------
    precision : int, default ``32``
        Default floating point and integer precision for the model.
    device : ``jax.Device`` | str | None, default ``None``
        Jax device to use for the model. If ``None``, it will use "cuda" if available, otherwise "cpu".
    debug : bool, default ``False``
        Whether to enable debug mode. If True, the model will perform additional checks and print debug information.
        This is heavy; avoid in a production setting.
    """

    Tensor = jax.numpy.ndarray

    def __init__(
        self, *, precision: int = 32, device: jax.Device | str | None = None, debug: bool = False
    ):
        """Initialize environment"""
        self._precision = precision
        self.device = (
            (jax.devices("cuda")[0] if jax.devices("cuda") else jax.devices("cpu")[0])
            if device is None
            else device
        )
        self.debug = debug

    @property
    def fdtype(self) -> jax.numpy.dtype:
        """Get default floating point dtype"""
        if self._precision == 16:
            return jax.numpy.float16
        elif self._precision == 32:
            return jax.numpy.float32
        elif self._precision == 64:
            return jax.numpy.float64
        else:
            raise ValueError(
                f"Unsupported precision {self._precision}; supported values are 16, 32, and 64."
            )

    @property
    def idtype(self) -> jax.numpy.dtype:
        """Get default integer dtype"""
        if self._precision == 16:
            return jax.numpy.int16
        elif self._precision == 32:
            return jax.numpy.int32
        elif self._precision == 64:
            return jax.numpy.int64
        else:
            raise ValueError(
                f"Unsupported precision {self._precision}; supported values are 16, 32, and 64."
            )
