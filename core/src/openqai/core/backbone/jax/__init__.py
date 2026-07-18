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

class BaseModel(flax.linen.Module):
    """Base class for jax models"""

    Tensor = jnp.ndarray

    def __init__(self, env: BaseEnvironment):
        """Initialize model"""
        super().__init__()
        self._env = env

    @property
    def env(self) -> BaseEnvironment:
        """Get environment"""
        return self._env

    @property
    def debug(self) -> bool:
        """Whether debug mode is enabled"""
        return self._env.debug

    @property
    def fdtype(self) -> jnp.dtype:
        """Get default floating point dtype from environment"""
        return self._env.fdtype

    @property
    def device(self) -> jax.Device:
        """Get device from environment"""
        return self._env.device

    @property
    def idtype(self) -> jnp.dtype:
        """Get default integer dtype from environment"""
        return self._env.idtype

    def verify( self, condition: callable, message: str | callable, exception: Exception = RuntimeError ) -> None:
        """
        Verify condition and print message if ``debug`` is enabled
        via :class:`BaseEnvironment`. This is a helper function to perform checks during model execution.
        If ``debug`` is not enabled, this function is a NOOP.

        Parameters
        ----------
        condition : callable
            A function that returns a boolean tensor. This function will be called only if ``debug`` is True.
        message : str or callable
            A string or a function that returns a string. This will be called only if ``debug`` is True and the condition is not satisfied.
        exception : Exception, default :class:`RuntimeError`
            The type of exception to raise if the condition is not satisfied.
        """
        if not self.debug:
            return

            
        condition = jnp.all(condition() if callable(condition) else condition)
        if not condition.item():
            raise exception(message() if callable(message) else message)

    def debug_detect_grad_issues(self):
        """
        Detect common gradient issues such as missing gradients or non-finite values. This is a helper function to perform checks during model execution.
        If any issues are detected, an AssertionError will be raised with a message describing the problems.

        If ``debug`` is not enabled, this function is a NOOP.
        """
        if not self.debug:
            return
        grad_problems = []
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                grad_problems.append(f"{name}: missing grad")
            elif not torch.isfinite(parameter.grad).all():
                grad_problems.append(f"{name}: non-finite grad")
        if len(grad_problems) > 0:
            raise RuntimeError(
                f"{self.__class__.__name__}: gradient diagnostics failed: "
                + "; ".join(grad_problems[:5])
            )

    def debug_finite_parameters(self):
        """
        Verify that all parameters in the model are finite. This is a helper function to perform checks during model execution.
        If any non-finite parameters are detected, an AssertionError will be raised with a message describing the problems

        If ``debug`` is not enabled, this function is a NOOP.
        """
        if not self.debug:
            return

        for p in self.parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"{self.__class__.__name__}: parameter contains non-finite values, which can lead to training instability. Check the model and the data."
                )

    def compute_grad_norm(self):
        """
        Compute the average L2 norm of the gradients across all parameters in the model. This is a helper function to perform checks during model execution.
        If there are no gradients, this function will return None.
        """
        with torch.no_grad():
            total_grad_sq = None
            num_grad_tensors = 0
            for parameter in self.parameters():
                if parameter.grad is not None:
                    grad = parameter.grad.detach()
                    total_grad_sq = (
                        total_grad_sq + torch.sum(grad * grad)
                        if total_grad_sq is not None
                        else torch.sum(grad * grad)
                    )
                    num_grad_tensors += 1
            if num_grad_tensors == 0:
                return None
            grad_norm = torch.sqrt(total_grad_sq / num_grad_tensors)
            return grad_norm


class BaseVisualizer(object):
    """Base visualizer class to visualize model performance during training and validation."""

    def update(self, model: BaseModel, epoch: int):
        """Update visualizer with new model and epoch number"""
        raise NotImplementedError("Visualizer must implement update method.")


class BaseSampler(object):
    """Base sampler class to sample data for training and validation."""

    def sample(self, epoch: int) -> dict:
        """Sample data for a given epoch number. The expected output is a dictionary containing tensors for the model. The specific keys and tensor shapes depend on the model implementation."""
        raise NotImplementedError("Sampler must implement sample method.")


@dataclass(frozen=True)
class BaseSamplerConfig:
    """Base class for a sampler configuration"""

    def instantiate_from_cache(self, *, verbose: Context = Context.quiet) -> BaseSampler:
        """
        Creates the Sampler corresponding to this config from cache only.
        This function will fail if not all data are availanle.
        """
        raise NotImplementedError("SamplerConfig must implement instantiate_from_cache method.")


# General torch utilities
# -----------------------


def check(
    tensor: torch.Tensor,
    item: str,
    shape: tuple | int | None = None,
    dtype: type | None = None,
    optional: bool = False,
) -> torch.Tensor:
    """
    Check that a tensor satisfies the specified shape and dtype requirements. This is a helper function to perform checks during model execution.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to check.
    item : str
        The name of the item being checked, used for error messages.
    shape : tuple | int | None, default ``None``
        The expected shape of the tensor. If int, it will be converted to a tuple with
        one element. If None, no shape check will be performed.
    dtype : type | None, default ``None``
        The expected dtype of the tensor. If None, no dtype check will be performed.
    optional : bool, default ``False``
        Whether the tensor is optional. If True, the function will return None if the tensor is
        None. If False, the function will raise an error if the tensor is None.
    """
    if tensor is None:
        if optional:
            return None
        raise KeyError(f"Expected item '{item}' not found in dictionary.")

    if shape is not None:
        if isinstance(shape, int):
            if len(tensor.shape) != shape:
                raise ValueError(
                    f"Expected item '{item}' to have {shape} dimensions, but got {len(tensor.shape)} for shape {tensor.shape}."
                )
        else:
            if tensor.shape != tuple(shape):
                raise ValueError(
                    f"Expected item '{item}' to have shape {shape}, but got {tensor.shape}."
                )

    if dtype is not None:
        if dtype is float:
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"Expected item '{item}' to have a floating point dtype, but got {tensor.dtype}."
                )
        elif dtype is int:
            if tensor.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError(
                    f"Expected item '{item}' to have an integer dtype, but got {tensor.dtype}."
                )
        elif dtype is bool:
            if tensor.dtype != torch.bool:
                raise TypeError(
                    f"Expected item '{item}' to have a boolean dtype, but got {tensor.dtype}."
                )
        else:
            if tensor.dtype != dtype:
                raise TypeError(
                    f"Expected item '{item}' to have dtype {dtype}, but got {tensor.dtype}."
                )

    return tensor


def get(
    d: dict,
    item: str,
    shape: tuple | int | None = None,
    dtype: type | None = None,
    optional: bool = False,
) -> torch.Tensor:
    """
    Helper function to get an item from a dictionary and perform checks on its shape and dtype.
    This is useful for verifying the structure of the input data during model execution.

    Parameters
    ----------
    d : dict
        The dictionary to get the item from.
    item : str
        The key of the item to get from the dictionary.
    shape : tuple | int | None, default ``None``
        The expected shape of the item. If int, it will be converted to a tuple with
        one element. If None, no shape check will be performed.
    dtype : type | None, default ``None``
        The expected dtype of the item. If None, no dtype check will be performed.
    optional : bool, default ``False``
        Whether the item is optional. If True, the function will return None if the item is not found in the dictionary.
    """

    return check(d.get(item, None), item, shape=shape, dtype=dtype, optional=optional)


def pop(
    d: dict,
    item: str,
    shape: tuple | int | None = None,
    dtype: type | None = None,
    optional: bool = False,
) -> torch.Tensor:
    """
    Helper function to pop an item from a dictionary and perform checks on its shape and dtype.
    This is useful for verifying the structure of the input data during model execution.

    Parameters
    ----------
    d : dict
        The dictionary to pop the item from.
    item : str
        The key of the item to get from the dictionary.
    shape : tuple | int | None, default ``None``
        The expected shape of the item. If int, it will be converted to a tuple with
        one element. If None, no shape check will be performed.
    dtype : type | None, default ``None``
        The expected dtype of the item. If None, no dtype check will be performed.
    optional : bool, default ``False``
        Whether the item is optional. If True, the function will return None if the item is not found in the dictionary.
    """

    return check(d.pop(item, None), item, shape=shape, dtype=dtype, optional=optional)


def are_broadcastable(shape1, shape2) -> bool:
    """Check if two shapes are broadcastable according to PyTorch's broadcasting rules."""
    try:
        torch.broadcast_shapes(shape1, shape2)
        return True
    except RuntimeError:
        return False


# ---------------------------------------
# Debugging
# ---------------------------------------


def lazy_verify(
    debug: bool, condition: callable, message: str | callable, exception: Exception = RuntimeError
) -> None:
    """
    Verify condition and print message if ``debug`` is enabled
    via :class:`BaseEnvironment`. This is a helper function to perform checks during model execution.
    If ``debug`` is not enabled, this function is a NOOP.

    Parameters
    ----------
    condition : callable
        A function that returns a boolean tensor. This function will be called only if ``debug`` is True.
    message : str or callable
        A string or a function that returns a string. This will be called only if ``debug`` is True and the condition is not satisfied.
    exception : Exception, default :class:`RuntimeError`
        The type of exception to raise if the condition is not satisfied.
    """
    if not debug:
        return

    with torch.no_grad():
        condition = torch.all(condition() if callable(condition) else condition)
        if not condition.item():
            raise exception(message() if callable(message) else message)


class MetricTacker(object):
    """
    Simple utility class to track metrics during training
    """

    def __init__(self):
        self.metrics = {}

    def record(self, epoch: int, metrics: Mapping[str, tuple[float, float | None]]):
        """Update the metric tracker with new metrics and their corresponding weights"""
        assert "loss" in metrics, f"'loss' should be included in the metrics directory"
        for k, v in metrics.items():
            if isinstance(v, tuple):
                assert len(v) == 2, (
                    f"Expected metric '{k}' to be a tuple of (mean, std) or (value, None), but got {v} with length {len(v)}."
                )
                self.record_metric(epoch, k, mean=v[0], std=v[1])
            else:
                self.record_metric(epoch, k, mean=v, std=None)

    def record_metric(self, epoch: int, name: str, mean: float, std: float | None = None):
        """Add a single metric"""
        tm = self.metrics.get(name)
        if tm is None:
            tm = dict(epochs=[epoch], values=[mean], stds=[std] if not std is None else None)
            self.metrics[name] = tm
        else:
            assert tm["epochs"][-1] < epoch, (
                f"Epoch {epoch} is not greater than last epoch {tm['epochs'][-1]} for metric {name}"
            )
            tm["epochs"].append(epoch)
            tm["values"].append(mean)
            if std is None:
                assert tm["stds"] is None, (
                    f"Metric {name} was previously recorded with stds, but now std is None"
                )
            else:
                assert tm["stds"] is not None, (
                    f"Metric {name} was previously recorded without stds, but now std is provided"
                )
                tm["stds"].append(std)

    def items(self):
        """Get all metrics as an iterable of (name, epochs, values) tuples"""
        return self.metrics.items()

    def __getitem__(self, name):
        """Get metric by name"""
        return self.metrics[name]

    def get(self, name, default=None):
        """Get metric by name with default"""
        return self.metrics.get(name, default)

    def state_dict(self):
        """Get a serializable copy of the tracked metrics as numpy arrays"""
        return {
            k: {
                "epochs": nnp.asarray(v["epochs"], dtype=nnp.int32),
                "values": nnp.asarray(v["values"], dtype=nnp.float32),
                "stds": nnp.asarray(v["stds"], dtype=nnp.float32)
                if v["stds"] is not None
                else None,
            }
            for k, v in self.metrics.items()
        }

    @staticmethod
    def load_state_dict(state):
        """Load tracked metrics from a serializable state"""
        assert isinstance(state, dict), (
            f"Expected tracker state to be a dict, but got {type(state)}"
        )
        tracker = MetricTacker()
        tracker.metrics = {
            k: {
                "epochs": x["epochs"].tolist(),
                "values": x["values"].tolist(),
                "stds": x["stds"].tolist() if not x["stds"] is None else None,
            }
            for k, x in state.items()
        }
        return tracker
