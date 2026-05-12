"""
Numerical backbone

Consistent wrapper around ``numpy`` and ``torch``, and in the future ``jax`` for being able to write unified model code where
sensible (e.g. outside ``jax.scan`` loops).

The backbone also has a black scholes function and a ``verify`` function.
"""

from __future__ import annotations
import numpy as nnp

class Backbone(object):
    """
    Generalized compute backbone: ``numpy``, ``torch``, (soon) ``jax``.
    """

    class Array(object):
        """ The array type """
        pass


    @staticmethod
    def are_broadcastable(*shapes) -> bool:
        """ Check if ``shapes`` are broadcastable according to numpy broadcasting rules. """

        if len(shapes) <= 1:
            return True
        shape1 = shapes[0]
        for shape2 in shapes[1:]:
            if len(shape1) != len(shape2):
                return False   
            for x1, x2 in zip(shape1, shape2):
                if x1 != x2 and x1 != 1 and x2 != 1:
                    return False
        return True

    BS_EPS = 1E-10
    EPS_VOL = 1E-10

    def bs_price( self, strikes : Array, 
                        sqrtVars : Array, *, 
                        mask : Array | None = None, 
                        spot : Array | None = None,
                        logK : Array | None = None,
                        high_precision : bool = False, 
                        debug_verify_bounds : bool = False,
                        eps : float = 1E-6 ) -> Array:
        """
        Black-Scholes price of call options with given spot, strikes and sqrtVars.

        Parameters
        ----------
        strikes : Array
            Strike prices for each option.
        sqrtVars : Array
            Square root of variance for each option.
        mask : Array | None, default ``None``
            Optional boolean mask indicating which options are valid. If provided, the function will return 0 for invalid options and compute prices only for valid options. This can be used to handle cases where some options are not traded or not relevant without affecting the computed prices of valid options.            
        spot : Array | None, default ``None``
            Spot prices, or ``None`` for ``1``.
        high_precision : bool, default ``False``
            If ``True``, perform calculations in higher precision (float64) to improve accuracy for deep OTM options. The output will be cast back to the input dtype.
        debug_verify_bounds : bool, default ``False``
            If ``True``, perform additional checks to verify that the computed option prices are within theoretical bounds (intrinsic value and spot price).
        eps : float, default ``1E-6``
            Tolerance for numerical checks when ``debug_verify_bounds`` is ``True``.
        Returns
        -------
        prices : Array
            The Black-Scholes price of the call options.            
        """
        out_dtype = strikes.dtype
        if high_precision:
            strikes = self.to_array(strikes, dtype=self.float64)
            spot    = self.to_array(spot, dtype=self.float64) if not spot is None else None
            sqrtVars= self.to_array(sqrtVars, dtype=self.float64)
        spot = spot if spot is not None else (strikes*0.+1.) # TODO: provide better "1"
        k    = strikes/spot if spot is not None else strikes
        intr = (k<self.BS_EPS) | (sqrtVars<self.EPS_VOL)
        intr = intr if mask is None else (intr|~mask)
        logK = self.log( self.where( intr, 1., k ) ) if logK is None else logK
        vf_  = self.where( intr, 1., sqrtVars )
        d1    = (-logK / vf_) + 0.5 * vf_
        d2    = d1 - vf_
        nd1   = self.norm_cdf(d1)
        nd2   = self.norm_cdf(d2)
        price = spot * ( nd1 - k * nd2 ) 
        inc   = self.relu( spot - strikes ) 
        price = self.where( intr, inc, price )
        if mask is not None:
            price = self.where(mask, price, 0.)

        if debug_verify_bounds:
            if mask is None:
                self.verify( price >= inc - eps, lambda : f"Option price must exceed intrinsic, found min {self.min(price-inc):.4g}; eps {eps}" )
                self.verify( price <= spot + eps, lambda : f"Option price must be less than or equal to spot, found max {self.max(price):.4g}; eps {eps}" )       
            else:
                self.verify( self.where( mask, price >= inc - eps, True ), lambda : f"Option price must exceed intrinsic, found min {self.min((price-inc)[mask]):.4g}; eps {eps}" )
                self.verify( self.where( mask, price <= spot + eps, True ), lambda : f"Option price must be less than or equal to spot, found max {self.max((price-spot)[mask]):.4g}; eps {eps}" )       
                 
        price = self.clip( price, min=inc, max=spot )  # enforce price >= intrinsic to avoid numerical issues in sanos. In ``debug`` these get caught above.
        price = self.minimum( price, spot ) # enforce price <= spot to avoid numerical issues in sanos. In ``debug`` these get caught above.
        return self.to_array( price, dtype=out_dtype ) if high_precision else price

# Numpy
# -----

def sigmoid(x: nnp.ndarray) -> nnp.ndarray:
    """ Numpy Sigmoid function 1/(1+exp(-x)) """
    _1 = x.dtype.type(1.)
    return _1 / (_1 + nnp.exp(-x))

def relu(x: nnp.ndarray) -> nnp.ndarray:
    """ Numpy ReLU function max(0,x) """
    _0 = x.dtype.type(0.)
    return nnp.maximum(_0, x)

def are_broadcastable(*shapes) -> bool:
    """ Check if ``shapes`` or arrays are broadcastable according to torch/numpy broadcasting rules. """    
    shapes = [ x.shape if isinstance(x, nnp.ndarray) else x for x in shapes ]
    return Backbone.are_broadcastable(*shapes)

class BackboneNumpy(Backbone):
    """
    Numpy compute backbone
    """

    def __init__(self):
        self.float32 : type = nnp.float32
        self.float64 : type = nnp.float64
        self.int32 : type = nnp.int32
        self.bool_ : type = nnp.bool_
        self.Array : type = nnp.ndarray
        
    def to_array( self, data, dtype : type | None = None ) -> Backbone.Array:
        """ Create an array; do not copy if data is already an array of the correct dtype """
        return nnp.asarray(data, dtype=dtype) if dtype is not None else nnp.asarray(data)
    
    def to_numpy( self, arr : Backbone.Array ) -> nnp.ndarray:
        assert isinstance( arr, nnp.ndarray ), f"Expected a numpy array, got type {type(arr)}"
        return arr

    def stop_gradient( self , arr : Backbone.Array ) -> Backbone.Array:
        """ Stop gradient calculation. Does NOT make a copy (NOOP for numpy)"""
        return arr

    def __getattr__(self, name):
        if hasattr(nnp, name):
            return getattr(nnp, name)
        else:
            raise AttributeError(f"BackboneNumpy has no attribute {name}")

    def norm_cdf( self, arr : Backbone.Array ) -> Backbone.Array:
        from scipy.special import ndtr  # fast N(0,1) CDF
        return ndtr( arr )

    def verify( self, condition: bool|Backbone.Array,  message : str|callable, exception : Exception = RuntimeError  ):
        """ Verify a condition, raise an error with message if not satisfied. """
        condition = nnp.all(condition)
        if not condition.item():
            raise exception( message() if callable(message) else message )

    def stop_gradient(self, x:Backbone.Array ) -> Backbone.Array:
        """ Stop gradient calculation. Does NOT make a copy """
        return x #NOOP

    def sigmoid(self, x:Backbone.Array) -> Backbone.Array   :
        """ Sigmoid function 1/(1+exp(-x)) """
        return sigmoid(x)
    
    def relu(self, x:Backbone.Array) -> Backbone.Array:
        """ ReLU function max(0,x) """
        return relu(x)
    
    def are_broadcastable(self, *shapes) -> bool:
        """ Check if ``shapes`` or arrays are broadcastable according to torch/numpy broadcasting rules. """    
        shapes = [ x.shape if isinstance(x, nnp.ndarray) else x for x in shapes ]
        return Backbone.are_broadcastable(*shapes)
    
    def stack( self, arrays : list[Backbone.Array], axis : int = 0, dtype : type | None = None ) -> Backbone.Array:
        """ Stack arrays along a new axis. If dtype is provided, cast to dtype after stacking. """
        r = nnp.stack( arrays, axis=axis )
        return r if dtype is None else r.astype( dtype=dtype )
    
