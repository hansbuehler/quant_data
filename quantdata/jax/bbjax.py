"""
Jax Backbone
^^^^^^^^^^^^

Numerical backbone for jax operations.
"""

from __future__ import annotations
from quantdata.torch import BaseEnvironment
from quantdata.utils.backbone import Backbone, BackboneNumpy #NOQA
import torch as torch
import numpy as nnp
from quantdata.jax import jax, jnp

class BackboneJax(Backbone):
    """
    Jax compute backbone
    """

    def __init__(self, env : BaseEnvironment|str = "cpu" ):
        if isinstance(env, str):
            env = BaseEnvironment(device=env)
        
        self.float32 : type = jax.numpy.float32
        self.int32 : type = jax.numpy.int32
        self.bool_ : type = jax.numpy.bool_
        self.Array : type = jax.numpy.ndarray
        self.device = env.device

    def norm_cdf( self, arr ):
        return jax.scipy.special.ndtr( arr )
    
    def to_atomic( self, data, dtype : type | None = None ) -> Backbone.Array:
        """ Convert data to a scalar tensor. Stops gradient calculation if data is a tensor. """
        if isinstance( data, nnp.ndarray ):
            return jax.device_put( data, self.device ).item()
        return jax.device_put( jnp.array( data, dtype=dtype ), self.device ).item()

    def to_array( self, data, dtype : type | None = None ) -> Backbone.Array:
        """
        Create an array; do not copy if data is already an array of the correct dtype and on the correct device.
        Stops gradient calculation if data is a tensor
        """
        if not hasattr( data, "shape" ) :
            data = nnp.asarray( data )
        if isinstance( data, nnp.ndarray):
            return jax.device_put( data, self.device )
        assert isinstance( data, torch.Tensor ), f"Expected data to be a torch tensor or numpy array, found type {type(data)}"
        if data.device == self.device and data.dtype == dtype:
            return data.detach()
        return torch.tensor( data, dtype=dtype, device=self.device )
    
    def to_numpy( self, arr : Backbone.Array ) -> nnp.ndarray:
        assert isinstance( arr, jnp.ndarray ), f"Expected a jax numpy array, got type {type(arr)}"
        return jax.device_get( arr )
    
    def stop_gradient( self , arr : Backbone.Array ) -> Backbone.Array:
        """ Stop gradient calculation. Does NOT make a copy """
        return jax.stop_gradient( arr )

    def zeros( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return jnp.zeros( shape, dtype=dtype )

    def empty( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return jnp.empty( shape, dtype=dtype )

    def zeros_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return jnp.zeros_like( arr, dtype=dtype )

    def empty_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return jnp.empty_like( arr, dtype=dtype )
    
    def unique( self, arr : Backbone.Array, axis : int = -1 ) -> Backbone.Array:
        return jnp.unique( arr, axis=axis )
    
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

        arr = jnp.broadcast_to(arr, arr_expand_shape)
        indices = jnp.broadcast_to(indices, index_expand_shape)
        return jnp.take_along_axis(arr, indices, axis=axis)

    def __getattr__(self, name):
        if hasattr(jnp, name):
            return getattr(jnp, name)
        else:
            raise AttributeError(f"BackboneJax has no attribute {name}")

    def verify( self, condition: bool|Backbone.Array, message : str|callable, exception : Exception = RuntimeError  ):
        """ Verify a condition, raise an error with message if not satisfied. """
        if not self.debug:
            return
        def callback( condition ):
            if jnp.all( condition ):
                return
            raise exception( message() if callable(message) else message )
        jax.debug.callback( callback, condition )

    def are_broadcastable(self, *shapes) -> bool:
        """ Check if ``shapes`` are broadcastable according to torch/numpy broadcasting rules. """
        shapes = [ x.shape if isinstance(x, self.Array) else x for x in shapes ]
        return Backbone.are_broadcastable(*shapes)
    
    def stack( self, arrays : list[Backbone.Array], axis : int = 0, dtype : type | None = None ) -> Backbone.Array:
        """ Stack arrays along a new axis. If dtype is provided, cast to dtype after stacking. """
        r = jnp.stack( arrays, axis=axis )
        return r if dtype is None else r.astype( dtype )