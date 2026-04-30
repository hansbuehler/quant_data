"""
Torch base
"""

from __future__ import annotations

import torch as torch
import torch.nn as tnn
from cdxcore.version import version

@version("1.0")
class BaseEnvironment(object):
    """
    Base environment class to control overall torch model behaviour.

    Parameters
    ----------
    precision : int, default ``32``
        Default floating point and integer precision for the model.
    device : ``torch.device`` | str | None, default ``None``
        Torch device to use for the model. If ``None``, it will use "cuda" if available, otherwise "cpu".
    debug : bool, default ``False``
        Whether to enable debug mode. If True, the model will perform additional checks and print debug information.
        This is heavy; avoid in a production setting.
    """

    Tensor = torch.Tensor

    def __init__(self, *, precision : int = 32,
                          device    : torch.device | str |None = None,
                          debug     : bool = False):
        """ Initialize environment """    
        self._precision = precision
        self.device     = ( torch.device("cuda" if torch.cuda.is_available() else "cpu") ) if device is None else device
        self.debug      = debug

    @property
    def fdtype(self) -> torch.dtype:
        """ Get default floating point dtype """
        if self._precision == 16:
            return torch.float16
        elif self._precision == 32:
            return torch.float32
        elif self._precision == 64:
            return torch.float64
        else:
            raise ValueError(f"Unsupported precision {self._precision}; supported values are 16, 32, and 64.")

    @property
    def idtype(self) -> torch.dtype:
        """ Get default integer dtype """
        if self._precision == 16:
            return torch.int16
        elif self._precision == 32:
            return torch.int32
        elif self._precision == 64:
            return torch.int64
        else:
            raise ValueError(f"Unsupported precision {self._precision}; supported values are 16, 32, and 64.")   

class BaseModel(tnn.Module):
    """ Base class for torch models """

    Tensor = torch.Tensor

    def __init__(self, env : BaseEnvironment):
        """ Initialize model """
        super().__init__()
        self._env = env

    @property
    def debug(self) -> bool:
        """ Whether debug mode is enabled """
        return self._env.debug

    @property
    def fdtype(self) -> torch.dtype:
        """ Get default floating point dtype from environment """
        return self._env.fdtype

    @property
    def device(self) -> torch.device:
        """ Get device from environment """
        return self._env.device

    @property
    def idtype(self) -> torch.dtype:
        """ Get default integer dtype from environment """
        return self._env.idtype

    def verify( self, condition: callable, message: str | callable, exception : Exception = RuntimeError )-> None:    
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
        
        with torch.no_grad():
            condition = torch.all( condition() if callable(condition) else condition )
            if not condition.item():
                raise exception( message() if callable(message) else message )

# General torch utilities
# -----------------------

def check( tensor : torch.Tensor, item : str, shape : tuple|int|None = None, dtype : type|None = None, optional : bool = False ) -> torch.Tensor:
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
                raise ValueError(f"Expected item '{item}' to have {shape} dimensions, but got {len(tensor.shape)} for shape {tensor.shape}.")
        else:
            if tensor.shape != tuple(shape):
                raise ValueError(f"Expected item '{item}' to have shape {shape}, but got {tensor.shape}.")

    if dtype is not None:
        if dtype == float:
            if not torch.is_floating_point(tensor):
                raise TypeError(f"Expected item '{item}' to have a floating point dtype, but got {tensor.dtype}.")
        elif dtype == int:
            if not torch.is_integral(tensor):
                raise TypeError(f"Expected item '{item}' to have an integer dtype, but got {tensor.dtype}.")
        elif dtype == bool:
            if tensor.dtype != torch.bool:
                raise TypeError(f"Expected item '{item}' to have a boolean dtype, but got {tensor.dtype}.")
        else:
            if tensor.dtype != dtype:
                raise TypeError(f"Expected item '{item}' to have dtype {dtype}, but got {tensor.dtype}.")

    return tensor

def get( d : dict, item : str, shape : tuple|int|None = None, dtype : type|None = None, optional : bool = False ) -> torch.Tensor:
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

    return check( d.get(item, None), item, shape=shape, dtype=dtype, optional=optional )

def pop( d : dict, item : str, shape : tuple|int|None = None, dtype : type|None = None, optional : bool = False ) -> torch.Tensor:
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

    return check( d.pop(item, None), item, shape=shape, dtype=dtype, optional=optional )

def are_broadcastable(shape1, shape2) -> bool:
    """ Check if two shapes are broadcastable according to PyTorch's broadcasting rules. """
    try:
        torch.broadcast_shapes(shape1, shape2)
        return True
    except RuntimeError:
        return False
    