"""
Grid Sanos torch backbone
"""

from __future__ import annotations
from quantdata.torch import BaseEnvironment
from quantdata.utils.backbone import Backbone, BackboneNumpy #NOQA
import torch as torch
import numpy as nnp

class BackboneTorch(Backbone):
    """
    Torch compute backbone
    """

    def __init__(self, env : BaseEnvironment|str = "cpu" ):
        if isinstance(env, str):
            env = BaseEnvironment(device=env)
        
        self.float32 : type = torch.float32
        self.int32 : type = torch.int32
        self.bool_ : type = torch.bool
        self.Array : type = torch.Tensor
        self.device = env.device

    def norm_cdf( self, arr ):
        return torch.special.ndtr( arr )
    
    def to_atomic( self, data, dtype : type | None = None ) -> Backbone.Array:
        """ Convert data to a scalar tensor. Stops gradient calculation if data is a tensor. """
        if isinstance( data, nnp.ndarray ):
            return torch.from_numpy( data ).to( dtype=dtype, device=self.device ).item()
        return torch.tensor( data, dtype=dtype, device=self.device ).item()

    def to_array( self, data, dtype : type | None = None ) -> Backbone.Array:
        """
        Create an array; do not copy if data is already an array of the correct dtype and on the correct device.
        Stops gradient calculation if data is a tensor
        """
        if not hasattr( data, "shape" ) :
            data = nnp.asarray( data )
        if isinstance( data, nnp.ndarray):
            return torch.from_numpy( data ).to( dtype=dtype, device=self.device )
        assert isinstance( data, torch.Tensor ), f"Expected data to be a torch tensor or numpy array, found type {type(data)}"
        if data.device == self.device and data.dtype == dtype:
            return data.detach()
        return torch.tensor( data, dtype=dtype, device=self.device )
    
    def to_numpy( self, arr : Backbone.Array ) -> nnp.ndarray:
        assert isinstance( arr, torch.Tensor ), f"Expected a torch tensor, got type {type(arr)}"
        return arr.detach().cpu().numpy()
    
    def stop_gradient( self , arr : Backbone.Array ) -> Backbone.Array:
        """ Stop gradient calculation. Does NOT make a copy """
        return arr.detach()

    def zeros( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return torch.zeros( shape, dtype=dtype, device=self.device )

    def empty( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return torch.empty( shape, dtype=dtype, device=self.device )

    def zeros_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return torch.zeros_like( arr, dtype=dtype, device=self.device )

    def empty_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return torch.empty_like( arr, dtype=dtype, device=self.device )
    
    def unique( self, arr : Backbone.Array, axis : int = -1 ) -> Backbone.Array:
        return torch.unique( arr, dim=axis )
    
    def take_along_axis( self, arr : Backbone.Array, indices : Backbone.Array, axis : int ) -> Backbone.Array:
        axis = axis if axis >= 0 else arr.ndim + axis
        assert arr.ndim == indices.ndim, f"Expected arr and indices to have the same rank, found {arr.ndim} and {indices.ndim}"

        arr_shape = list(arr.shape)
        index_shape = list(indices.shape)
        arr_expand_shape = arr_shape.copy()
        index_expand_shape = index_shape.copy()

        for dim, (arr_dim, index_dim) in enumerate(zip(arr_shape, index_shape)):
            if dim == axis:
                continue
            if arr_dim == index_dim:
                continue
            if arr_dim == 1:
                arr_expand_shape[dim] = index_dim
                continue
            if index_dim == 1:
                index_expand_shape[dim] = arr_dim
                continue
            raise AssertionError(f"Cannot broadcast shapes {tuple(arr_shape)} and {tuple(index_shape)} for axis {axis}")

        arr = arr.expand(*arr_expand_shape)
        indices = indices.to(dtype=torch.int64, device=arr.device).expand(*index_expand_shape)
        return torch.gather( arr, dim=axis, index=indices )

    def __getattr__(self, name):
        if hasattr(torch, name):
            return getattr(torch, name)
        else:
            raise AttributeError(f"BackboneTorch has no attribute {name}")

    def verify( self, condition: bool|Backbone.Array, message : str|callable, exception : Exception = RuntimeError  ):
        """ Verify a condition, raise an error with message if not satisfied. """
        with torch.no_grad():
            condition = torch.all(condition)
            if not condition.item():
                raise exception( message() if callable(message) else message )

    def are_broadcastable(self, *shapes) -> bool:
        """ Check if ``shapes`` are broadcastable according to torch/numpy broadcasting rules. """
        shapes = [ x.shape if isinstance(x, self.Array) else x for x in shapes ]
        return Backbone.are_broadcastable(*shapes)
    
    def stack( self, arrays : list[Backbone.Array], axis : int = 0, dtype : type | None = None ) -> Backbone.Array:
        """ Stack arrays along a new axis. If dtype is provided, cast to dtype after stacking. """
        r = torch.stack( arrays, dim=axis )
        return r if dtype is None else r.to( dtype=dtype )