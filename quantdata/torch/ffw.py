"""
Fast Forward Network
"""

from __future__ import annotations
from quantdata.torch import BaseModel, BaseEnvironment, torch, tnn
from collections.abc import Callable
from dataclasses import dataclass
from cdxcore.err import verify_inp
import math as math

@dataclass(frozen=True)
class FeedForwardConfig:
    """
    Configuration for FeedForward.
    This is a separate class so it can be imported without importing torch.

    The default is a one-layer linear network with a width of 16.
    """
    activation : str = 'Softplus'   # Activation module; see https://docs.pytorch.org/docs/stable/nn.html
    num_depth : int = 0             # Depth; if zero then the network becomes a single linear layer.
    num_width : int = 16            # Inner width; use a negative number as multiplier of ``num_output`` (output dimension). If ``num_depth`` is zero, then this is ignored.
    use_bias : bool = True          # Whether to use bias in the inner layers. If ``num_depth`` is zero, then this is ignored.

    def __post_init__(self):
        verify_inp( not isinstance(self.activation,str) or hasattr(torch.nn, self.activation), f"Unknown torchh activation function 'torch.nn.{self.activation}' see https://docs.pytorch.org/docs/stable/nn.html")
        verify_inp( self.num_depth >= 0, "'num_depth' must be non-negative" )
        verify_inp( self.num_width != 0, "'num_width' cannot be zero")

    def get_activation(self) -> Callable:
        """ Retrieves the requested activation function """
        if not isinstance(self.activation, str):
            verify_inp( callable(self.activation), "'activation' must be a string" )
            return self.activation
        f = getattr(torch.nn, self.activation, None)
        verify_inp( not f is None, f"Cannot find activation function 'torch.nn.{self.activation}'" )
        return f()
    
    @staticmethod
    def linear_config( activation : str = 'Identity' ) -> FeedForwardConfig:
        """ Returns a config for a simple one-layer linear network with the specified activation. """
        return FeedForwardConfig( activation=activation, num_depth=0, num_width=16, use_bias=True )

class FeedForward(BaseModel):
    """
    Dense network applies to list of inputs (instead of a stack).

    Essentially this function does::

        y: input as list, implicitly y=stack(y,axis=-1)

        x = L0( y )
        for d in num_depth:
            x = activation( x )
            x = Ld( x )
        if use_final_activation:
            x = activation( x )
        x = mask & x

    where ``Li`` is a set of linear layers applied to each element in the list.

    (c)copyright 2019, Hans Buehler

    Parameters
    ----------
    env : :class:`quantdata.torch.BaseEnvironment`
        Environment of the model which includes the random generator, default floating point dtype, and
        whether to run invasive debugging.

    num_input : int
        Number of inputs, which will be expected as list of the form ``[ (A,...,B) ]``.
        Numerically equivalent to a stack of dimension ``(A,....B,num_input)``.

    num_output : int | tuple[int]
        Last dimension of output ie output is ``(A,...,B,num_output)``.
        Set to zero to return ``(A,...,B)``.
        Accepts a tuple for multi-dimensional output, see :class:`torch.nn.Linear`.

    config : :class:`quantdata.torch.ffw.FeedForwardConfig`, default is a one-layer linear network.
        User config of the network.

    use_final_bias : bool, default ``True``
        whether the final linear layer adds a bias.

    use_final_activation : bool, default ``False``
        Whether to apply activation after the final linear layer.

    """

    Config = FeedForwardConfig

    def __init__(self, env            : BaseEnvironment, 
                       num_input      : int,                                  # number of inputs [ (A,...,B) ] ~ (A,....B,num_input)
                       num_output     : int|tuple[int],                       # last dimension of output ie output is (A,...,B,num_output). Set to zero to return (A,...,B). Can handle tuple for multi-dimensional output.
                       config         : FeedForwardConfig = FeedForwardConfig(), *,   # config, see above 
                       use_final_bias : bool = True,                          # whether the final linear layer adds a bias
                       use_final_activation : bool = False                    # whether to apply activation after the final linear layer  
            ):
        super().__init__( env=env )
        verify_inp( num_input > 0, "num_input must be positive" )
        num_output                = num_output if isinstance(num_output, int) else tuple(num_output)
        if isinstance(num_output,int):
            verify_inp( num_output >= 0, "num_output must be non-negative" )
        else:
            verify_inp( all( n > 0 for n in num_output ), lambda : f"'num_output' must be a non-negative integer or a tuple with all dimensions positive. Found {num_output}" )

        self.num_output           = num_output
        self.num_input            = num_input
        self.activation           = config.get_activation()
        self.use_final_activation = use_final_activation

        output_dim = num_output if isinstance(num_output, int) else math.prod(num_output)

        if config.num_depth > 0:
            num_width = config.num_width if config.num_width > 0 else ( abs(config.num_width) * max(1, output_dim) )
        else:
            # for a zero depth network, width is just the output dimension
            # note that num_output==0 means that the output dimension is 1
            # but it will be reduced to zero at the end of the forward pass
            num_width = output_dim if output_dim > 0 else 1

        use_bias = config.use_bias if config.num_depth > 0 else use_final_bias

        self.I0 = tnn.Linear( in_features=num_input, out_features=num_width, bias=use_bias, device=env.device, dtype=env.fdtype) 
        ##OLD        self.I0 = tnn.ModuleList( [ tnn.Linear( in_features=1, out_features=num_width, bias=use_bias, device=env.device, dtype=env.fdtype) for i in range(num_input) ] )
        if config.num_depth > 0:
            In = [ tnn.Linear( in_features=num_width, out_features=num_width, bias=use_bias, device=env.device, dtype=env.fdtype ) for d in range(config.num_depth-1) ]
            In.append( tnn.Linear( in_features=num_width, out_features=output_dim if output_dim > 0 else 1, bias=use_final_bias, device=env.device, dtype=env.fdtype ) )
            self.In = tnn.ModuleList( In )
            #del In
        else:
            self.In = []

    def forward( self,
                 input : list[torch.Tensor]|torch.Tensor,
                 mask  : torch.Tensor|None = None ) -> torch.Tensor:
        """
        Compute masked fast forward network.

        Parameters
        ----------
        input : list[torch.Tensor] | torch.Tensor
            Either a list with ``num_input`` elements ``(A,...,B)``, or a ``torch.Tensor`` of dimension ``(A,...,B,num_input)``.

        mask : torch.Tensor | None:
            A mask of shape ``(A,...,B)`` to apply to the input, or ``None``.
            If provided, the mask is applied to each input element as ``input = torch.where(mask[...,None], input, 0)``.
            This can be used to zero out certain inputs without changing the shape of the output. 
        """
        def lin_has_finite_weights( lin : tnn.Linear ) -> bool:
            if lin.bias is None:
                return torch.isfinite(lin.weight).all()
            return torch.isfinite(lin.weight).all() & torch.isfinite(lin.bias).all()


        if isinstance(input, torch.Tensor ):
            assert input.shape[-1] == self.num_input, f"Input shape mismatch: expected last dimension to be {self.num_input}, but found {input.shape}"
            x    = torch.where( mask[...,None], input, 0.) if not mask is None else input
        else:
            assert len(input) == self.num_input, f"Input length mismatch: expected {self.num_input} elements in the input list, but found {len(input)}"
            shape = torch.broadcast_shapes( *[x.shape for x in input] )
            x     = torch.stack([ torch.where(mask, x, 0.).expand(shape) for x in input], dim=-1) if not mask is None else\
                    torch.stack([ x.expand(shape) for x in input], dim=-1)

        self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: not finite input 'x' after masking and stacking" )
        x = self.I0(x)

        """
        OLD
        input = [ input[...,i] for i in range(input.shape[-1]) ] if isinstance(input, torch.Tensor ) else input
        verify_inp( len(input) == self.num_input, lambda : f"Input length error: expected {self.num_input}, got {len(input)}" )
        self.verify( lambda : torch.any(mask), lambda : f"{FeedForward}: Mask is all False, no valid input to process" ) if not mask is None else None

        x = None
        for iinp, (inp, I0) in enumerate(zip(input, self.I0)):
            inp  = inp if mask is None else torch.where( mask, inp, 0. )
            self.verify( lambda : torch.isfinite(inp), lambda :  f"{FeedForward}: Input #{iinp} is not finite" )
            self.verify( lambda : lin_has_finite_weights(I0), lambda : f"{FeedForward}: Input #{iinp} weights or bias are not finite"    )
            inp = I0( inp[...,None] )
            self.verify( lambda :torch.isfinite(inp).all(), lambda : f"{FeedForward}: Iteration #{iinp} activation 'inp' is not finite" )
            x = inp if x is None else (x+inp)
            #del inp
        #del input
        self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Composed input 'x' is not finite" )  
        """
        for iI, I in enumerate(self.In):
            x = self.activation(x)
            self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Iteration #{iI} activation 'x' is not finite" )
            self.verify( lambda : lin_has_finite_weights(I), lambda : f"{FeedForward}: Input #{iI} weights or bias are not finite"    )
            x = I(x)
            self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Iteration #{iI} 'x' is not finite" )
        if self.use_final_activation:
            x = self.activation(x)
            self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Final activation 'x' is not finite" )

        if self.num_output == 0:
            assert x.shape[-1] == 1, ("Internal error: output shape mismatch:", x.shape)
            x = x[...,0]
            x = torch.where(mask, x, 0.)     if not mask is None else x
            self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Output 'x' is not finite (num_output=0)" )
        elif isinstance(self.num_output, int):
            assert x.shape[-1] == self.num_output, ("Internal error: output shape mismatch:", x.shape, "vs", self.num_output)    
            x = torch.where(mask[...,None], x, 0.) if not mask is None else x
            self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Output 'x' is not finite (num_output={self.num_output})" )
        else:
            x = x.reshape( x.shape[:-1] + self.num_output )
            if not mask is None:
                m = mask.reshape( mask.shape + (1,)*(len(x.shape)-len(mask.shape)) )
                x = torch.where(m, x, 0.) if not mask is None else x
                self.verify( lambda : torch.isfinite(x), lambda : f"{FeedForward}: Output 'x' is not finite (num_output={self.num_output})" )
        return x