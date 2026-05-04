"""
Grid Sanos torch backbone
"""

from __future__ import annotations
from dys.grid_sanos.backbone import Backbone, BackboneNumpy #NOQA
import torch as torch
import numpy as nnp

class BackboneTorch(Backbone):
    """
    Torch compute backbone
    """

    def __init__(self, device : torch.device | str | None ):
        self.float32 : type = torch.float32
        self.int32 : type = torch.int32
        self.bool_ : type = torch.bool
        self.Array : type = torch.Tensor

        if isinstance( device, str ):
            self.device = torch.device(device)
        elif isinstance( device, torch.device ):
            self.device = device
        else:
            self.device = None

    def norm_cdf( self, arr ):
        return torch.special.ndtr( arr )

    def to_array( self, data, dtype : type | None = None ) -> Backbone.Array:
        """ Create an array; do not copy if data is already an array of the correct dtype. Stops gradient calculation if data is a tensor. """
        if isinstance( data, nnp.ndarray ):
            return torch.from_numpy( data ).to( dtype=dtype, device=self.device )
        return torch.tensor( data, dtype=dtype, device=self.device )
    
    def to_numpy( self, arr : Backbone.Array ) -> nnp.ndarray:
        assert isinstance( arr, torch.Tensor ), f"Expected a torch tensor, got type {type(arr)}"
        return arr.detach().cpu().numpy()

    def astype( self, data, dtype : type | None = None ) -> Backbone.Array:
        """ Type conversion. This function also places the tensor on the correct device. Keeps the gradient. """
        return data.to( dtype=dtype, device=self.device )

    def zeros( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return torch.zeros( shape, dtype=dtype, device=self.device )

    def empty( self, shape : tuple, dtype : type ) -> Backbone.Array:
        return torch.empty( shape, dtype=dtype, device=self.device )

    def zeros_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return torch.zeros_like( arr, dtype=dtype, device=self.device )

    def empty_like( self, arr, dtype : type | None = None ) -> Backbone.Array:
        return torch.empty_like( arr, dtype=dtype, device=self.device )

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

    def stop_gradient(self, x:Backbone.Array ) -> Backbone.Array:
        """ Stop gradient calculation. Does NOT make a copy """
        return x.detach()

    def are_broadcastable(self, *shapes) -> bool:
        """ Check if ``shapes`` are broadcastable according to torch/numpy broadcasting rules. """
        shapes = [ x.shape if isinstance(x, self.Array) else x for x in shapes ]
        return Backbone.are_broadcastable(*shapes)
    
    def stack( self, arrays : list[Backbone.Array], axis : int = 0, dtype : type | None = None ) -> Backbone.Array:
        """ Stack arrays along a new axis. If dtype is provided, cast to dtype after stacking. """
        r = torch.stack( arrays, dim=axis )
        return r if dtype is None else r.to( dtype=dtype )