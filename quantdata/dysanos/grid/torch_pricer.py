"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

from collections.abc import Callable
import warnings as warnings
import numpy as np
import math
from dataclasses import dataclass
from cdxcore.err import verify, verify_inp
from cdxcore.version import version
from cdxcore.pretty import PrettyObject

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, lazy_verify
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.biz_calendar import NMS
import gc

_bbtorch = BackboneTorch()

@version("0.1")
@dataclass(frozen=True)
class SanosDecoderConfig:
    smoothness: float|None = None    # Model smoothness
    num_state : int = 20

    num_nms   : int = 20
    learn_nms : bool = False
    nms_range : tuple[float,float] = (-2, 1)
    nms_eps   : float|None = None      # eps if used; use NMS.EPS otherwise.

    min_dlv   : float = 0.009          # Min DLV
    max_dlv   : float = 3.             # Max DLV
    min_fvol  : float = 0.01           # minimum fwd vol
    max_fvol  : float = 2.             # maximum fwd vol
    
    nn_config : FeedForward.Config = FeedForward.Config()  # decoder

    _xnms_scale : float = 2.

    def __post_init__(self):
        verify_inp( self.num_nms >= 4, f"Expected num_nms >= 4, but found {self.num_nms}" )
        verify_inp( self.num_state > 0, f"Expected num_state > 0, but found {self.num_state}" )
        verify_inp( self.nms_range[0] < 0. and self.nms_range[1] > 0., f"Expected nms_range to contain 0, but found {self.nms_range}." )
        verify_inp( self.min_dlv > 0. and self.max_dlv > 0., f"Expected min_dlv and max_dlv to be positive, but found {self.min_dlv} and {self.max_dlv}." )
        verify_inp( self.min_fvol > 0. and self.max_fvol > 0., f"Expected min_fvol and max_fvol to be positive, but found {self.min_fvol} and {self.max_fvol}." )
        verify_inp( self.smoothness is None or self.smoothness >= 0., f"Expected smoothness to be positive, but found {self.smoothness}." )
        verify_inp( self.nms_eps is None or self.nms_eps >= 0., f"Expected nms_eps to be positive, but found {self.nms_eps}." )

    def __str__(self) -> str:
        return f"{self.str_smooth} {self.nms_str} {self.state_str} {self.minmax_str}"
               
    @property
    def str_smooth(self) -> str:
        return 'linear' if self.smoothness is None else f"{self.smoothness:.2f}-smooth"

    @property
    def nms_str(self) -> str:
        return f"nms={self.num_nms},({self.nms_range[0]:.2f},{self.nms_range[1]:.2f})" +\
               ('*' if self.learn_nms else '')+\
               ('' if self.nms_eps is None else f',e={self.nms_eps:.2f}')

    @property
    def minmax_str(self):
        return f"dlv=({self.min_dlv:.2f},{self.max_dlv:.2f}),fvol=({self.min_fvol:.2f},{self.max_fvol:.2f})"

    @property
    def state_str(self) -> str:
        return f"state={self.num_state}@{self.nn_config}"

    @property
    def xnms(self) -> np.ndarray:
        """ Generate NMS strikes which contain '1' """
        assert self.nms_range[0] < 0. and self.nms_range[1] > 0., f"Internal error: expected grid_range to contain 0, but found {self.nms_range}."
        n    = self.num_nms
        dist = float(self.nms_range[1] - self.nms_range[0])
        Nlo  = min(max(2, float(n*abs(self.nms_range[0]))//dist),n-3)
        Nhi = n - Nlo - 1
        assert Nlo>=2 and Nhi>=2, f"Internal error: error {Nlo} {Nhi} for grid_range {self.nms_range} and grid_num_strikes {n}."
        xnms = np.concatenate( ( [self.nms_range[0]*self._xnms_scale], np.linspace(self.nms_range[0], 0., int(Nlo), endpoint=False), np.linspace(0., self.nms_range[1], int(Nhi)+1), [self.nms_range[1]*self._xnms_scale] ) )
        assert len(xnms)==n+2, f"Internal error: generated number of moneyness levels {len(xnms)} does not match expected {n+2}."
        return xnms

@version("0.1")
class SanosPricer:

    def __init__(self, *,
        smoothness: float | None,
        expiries  : torch.Tensor,   # (1|S,T)
        xstrikes  : torch.Tensor,   # (S, T, N+2)
        xdensities: torch.Tensor,   # (S, T, N+2)
        xdensity0 : torch.Tensor,   # (S, N+2)
        sqrtVars  : torch.Tensor,   # (S, T)
        # information only - no gradient
        dlvsqs    : torch.Tensor|None = None, # (S, T, N)
        xnms      : torch.Tensor|None = None, # (S|1, T|1, N+2)
        # debugging
        debug     : bool = False
    ):
        self.smoothness  = smoothness
        self.expiries    = expiries
        self.xstrikes    = xstrikes
        self.xdensities  = xdensities
        self.xdensity0   = xdensity0
        self.sqrtVars    = sqrtVars
        self.dlvsqs      = dlvsqs
        self.xnms        = xnms
        self.debug       = debug
        
    def verify( self, condition : Callable, message : Callable|str ) -> None:
        """ verify ``condition`` and raise an error with ``message`` if not satisfied; if ``self.debug`` is False, this function does nothing. """
        lazy_verify( self.debug, condition, message )

    @property
    def strikes(self) -> torch.Tensor:
        """ Return the strikes (S, T, N) """
        return self.xstrikes[:,:,1:-1]

    @staticmethod
    def create( smoothness : float, model_output : dict, *, debug : bool = False ) -> SanosPricer:
        """ Create a :class:`SanosPricer` from the output of a :class:`SanosDecoder` """
        return SanosPricer( smoothness=smoothness,
                            expiries=model_output['expiries'],
                            xstrikes=model_output['xstrikes'],
                            xdensities=model_output['xdensities'],
                            xdensity0=model_output['xdensity0'],
                            sqrtVars=model_output['sqrtVars'],
                            dlvsqs=model_output.get('dlvsqs', None),
                            xnms=model_output.get('xnms', None),
                            debug=debug )

    #@torch.compile(fullgraph=True)
    def __call__(self, market_strikes : torch.Tensor, *,
                       dt             : float|None = None,
                       training       : bool = False ) -> tuple[torch.Tensor, torch.Tensor|None]:
        """ Price call prices for the provided market strikes """
        S, T, n2 = self.xstrikes.shape
        assert market_strikes.shape[:2] == (S, T), f"Expected market_strikes to have shape {(S, T, ...)}, but found {market_strikes.shape}"
        N        = market_strikes.shape[2]
        xS       = self.expiries.shape[0]
    
        if dt is None:
            # model to market
            if self.smoothness is None or self.smoothness == 0.:
                C = torch.sum( torch.maximum( self.xstrikes[0,0,0]*0., self.xstrikes[:,:,None,:] - market_strikes[:,:,:,None] ) * self.xdensities[:,:,None,:], dim=-1 )
            else:
                C = torch.sum( _bbtorch.bs_price(  
                                        spot=self.xstrikes[:,:,None,:],
                                        strikes=market_strikes[:,:,:,None],
                                        sqrtVars=math.sqrt(self.smoothness) * self.sqrtVars[:,:,None,None]) * self.xdensities[:,:,None,:], dim=-1 )
            assert C.shape == (S, T, N), f"Expected C to have shape {(S, T, N)}, but found {C.shape}"  
            return C

        # 'dt' pricing: assume we do not cross the expiry boundary
        # The pricing rule is is follows:
        #
        # SANOS pricing at a fixed expory j is
        #  C = BSCall( Spot=Strike_j, Strikes, sqrtVar ) * density_j
        #
        # If we price at an intermediate time t with
        #  alpha_j = (t - t_{j-1}) / (t_j - t_{j-1})
        #
        # we first interpolate variance linearly:
        #  sqrtVar_t = sqrt( var_j * alpha_j + var_{j-1} * (1-alpha_j) ) 
        #
        # The we interpolate call prices to t>t_1 as
        # 
        #  C =     alpha_j * BSCall( Spot=Strike_j, Strikes, sqrtVar_t ) * density_j +
        #      (1-alpha_j) * BSCall( Spot=Strike_{j-1}, Strikes, sqrtVar_t ) * density_{j-1}
        # 
        # for t<t_1 we use
        #  C =     BSCall( Spot=Strike_1, Strikes, sqrtVar_1 ) * ( alpha_1 density_1 + (1-alpha_1) density_0 )
        
        assert dt > 0., f"Expected dt to be positive, but found {dt}"

        expiries_tp1 = torch.clip( self.expiries - dt, min=0. )
        assert expiries_tp1.shape == (S,T) or expiries_tp1.shape == (1,T), f"Expected expiries_tp1 to have shape ({S}|1,{T}), but found {expiries_tp1.shape}"
        self.verify( lambda : torch.all( dt >= 0. ), "Negative 'dt' values found" )
        self.verify( lambda : torch.all( expiries_tp1[:,1:] >= self.expiries[:, :-1] ), "'dt' transcends expiry boundaries" )
        self.verify( lambda : torch.all( expiries_tp1[:,0] >= 0. ), "'expiries_tp1[0]'<0" )

        alpha_j_tp1  = ( expiries_tp1[:,1:] - self.expiries[:, :-1] ) / ( self.expiries[:,1:] - self.expiries[:,:-1] )
        alpha_0_tp1  = expiries_tp1[:,0] / self.expiries[:,0] # (S,)
        assert alpha_j_tp1.shape == (xS, T-1), f"Expected alpha_j_tp1 to have shape {(xS, T-1)}, but found {alpha_j_tp1.shape}"
        assert alpha_0_tp1.shape == (xS,), f"Expected alpha_0_tp1 to have shape {(xS,)}, but found {alpha_0_tp1.shape}"
        self.verify( lambda : torch.all( torch.isfinite(alpha_j_tp1) & ( alpha_j_tp1 >= 0. ) & ( alpha_j_tp1 <= 1. ) ), "alpha_j_tp1 not within [0,1]" )
        good =torch.isfinite(alpha_0_tp1) & ( alpha_0_tp1 >= 0. ) & ( alpha_0_tp1 <= 1. )
        self.verify( lambda : torch.all( torch.isfinite(alpha_0_tp1) & ( alpha_0_tp1 >= 0. ) & ( alpha_0_tp1 <= 1.+1E-8 ) ), lambda : f"alpha_0_tp1 not within [0,1] {alpha_0_tp1[~good]}" )
        variances    = None

        variances    = self.sqrtVars**2
        int_var_j    = variances[:,1:] * alpha_j_tp1 + variances[:,:-1] * (1. - alpha_j_tp1)
        int_var_0    = variances[:,:1] * alpha_0_tp1[:,None]
        int_sqrV     = torch.cat( (int_var_0, int_var_j), dim=1 )
        assert int_sqrV.shape == (S, T), f"Expected int_sqrV to have shape {(S, T)}, but found {int_sqrV.shape}"
        int_sqrtV    = torch.sqrt( int_sqrV )
        blend_q0     = self.xdensity0[:,None,:] * (1.- alpha_0_tp1)[:,None,None] + self.xdensities[:,:1,:] * alpha_0_tp1[:,None,None]
        int_var_j    = None
        int_var_0    = None
        int_sqrV     = None
        variances    = None
        self.verify( lambda : torch.all( torch.isfinite(int_sqrtV) & (int_sqrtV >= 0.) ), "Infinite or negative 'int_sqrtV' values found" )

        # model to market
        # ---------------

        # vector version uses too much memory
        if self.smoothness is None or self.smoothness == 0.:
            C_hi    = torch.sum( torch.maximum(self.xstrikes[0,0,0]*0., self.xstrikes[:,1:,None,:] - market_strikes[:,1:,:,None]) * self.xdensities[:,1:,None,:], dim=-1, )
            C_lo    = torch.sum( torch.maximum(self.xstrikes[0,0,0]*0., self.xstrikes[:,:-1,None,:] - market_strikes[:,1:,:,None]) * self.xdensities[:,:-1,None,:], dim=-1, )
            C_j     = C_hi * alpha_j_tp1[:,:,None] + C_lo * (1. - alpha_j_tp1[:,:,None])
            C_hi    = None
            C_lo    = None
        else:
            C_hi    = torch.sum( _bbtorch.bs_price(  
                                    spot=self.xstrikes[:,1:,None,:],
                                    strikes=market_strikes[:,1:,:,None],
                                    sqrtVars=math.sqrt(self.smoothness) * int_sqrtV[:,1:,None,None]) * self.xdensities[:,1:,None,:], dim=-1 )
            C_lo    = torch.sum( _bbtorch.bs_price(  
                                    spot=self.xstrikes[:,:-1,None,:],
                                    strikes=market_strikes[:,1:,:,None],
                                    sqrtVars=math.sqrt(self.smoothness) * int_sqrtV[:,1:,None,None]) * self.xdensities[:,:-1,None,:], dim=-1 )
            C_j     =  C_hi * alpha_j_tp1[:,:,None] + C_lo * (1.- alpha_j_tp1[:,:,None])
            assert C_hi.shape == (S, T-1, N), f"Expected C_hi to have shape {(S, T-1, N)}, but found {C_hi.shape}"  
            assert C_lo.shape == (S, T-1, N), f"Expected C_lo to have shape {(S, T-1, N)}, but found {C_lo.shape}"  
            assert C_j.shape == (S, T-1, N), f"Expected C_j to have shape {(S, T-1, N)}, but found {C_j.shape}"  
            C_hi    = None
            C_lo    = None

            X       = _bbtorch.bs_price(  
                                    spot=self.xstrikes[:,0,None,:],
                                    strikes=market_strikes[:,0,:,None],
                                    sqrtVars=math.sqrt(self.smoothness)* int_sqrtV[:,0,None,None])
            C_0     = torch.sum( X  * blend_q0, dim=-1 )
            assert C_0.shape == (S, N), f"Expected C_0 to have shape {(S, N)}, but found {C_0.shape}"  
            C       = torch.cat( (C_0[:,None,:], C_j), dim=1 )
        self.verify( lambda : torch.all( torch.isfinite(C) & (C >= 0.) ), "Infinite or negative 'C' values found" )
        assert C.shape == (S, T, N), f"Expected C to have shape {(S, T, N)}, but found {C.shape}"  
        return C

        # loop in T
        if self.smoothness is None or self.smoothness == 0.:
            C_0     = torch.sum( torch.relu(self.xstrikes[:,0,None,:] - market_strikes[:,0,:,None]) * blend_q0, dim=-1 )
        else:
            C_0     = torch.sum( _bbtorch.bs_price(  
                                    spot=self.xstrikes[:,0,None,:],
                                    strikes=market_strikes[:,0,:,None],
                                    sqrtVars=math.sqrt(self.smoothness)* int_sqrtV[:,0,None,None]) * blend_q0, dim=-1 )
            assert C_0.shape == (S, N), f"Expected C_0 to have shape {(S, N)}, but found {C_0.shape}"  

        C       = [ C_0 ]
        C_0     = None

        for t in range(1,T):
            if self.smoothness is None or self.smoothness == 0.:
                C_hi    = torch.sum( torch.relu(self.xstrikes[:,t,None,:] - market_strikes[:,t,:,None]) * self.xdensities[:,t,None,:], dim=-1, )
                C_lo    = torch.sum( torch.relu(self.xstrikes[:,t-1,None,:] - market_strikes[:,t,:,None]) * self.xdensities[:,t-1,None,:], dim=-1, )
                C_j     = C_hi * alpha_j_tp1[:,t-1,None] + C_lo * (1. - alpha_j_tp1[:,t-1,None])
                C_hi    = None
                C_lo    = None
            else:
                C_hi    = torch.sum( _bbtorch.bs_price(  
                                        spot=self.xstrikes[:,t,None,:],
                                        strikes=market_strikes[:,t,:,None],
                                        sqrtVars=math.sqrt(self.smoothness) * int_sqrtV[:,t,None,None]) * self.xdensities[:,t,None,:], dim=-1 )
                C_lo    = torch.sum( _bbtorch.bs_price(  
                                        spot=self.xstrikes[:,t-1,None,:],
                                        strikes=market_strikes[:,t,:,None],
                                        sqrtVars=math.sqrt(self.smoothness) * int_sqrtV[:,t-1,None,None]) * self.xdensities[:,t-1,None,:], dim=-1 )
                C_j     =  C_hi * alpha_j_tp1[:,t-1,None] + C_lo * (1.- alpha_j_tp1[:,t-1,None])
                assert C_hi.shape == (S, N), f"Expected C_hi to have shape {(S, N)}, but found {C_hi.shape}"  
                assert C_lo.shape == (S, N), f"Expected C_lo to have shape {(S, N)}, but found {C_lo.shape}"  
                assert C_j.shape == (S, N), f"Expected C_j to have shape {(S, N)}, but found {C_j.shape}"  
                C_hi    = None
                C_lo    = None
            C.append( C_j )
            C_j = None
            gc.collect()

        C = torch.stack( C, dim=1 )
        assert C.shape == (S, T, N), f"Expected C to have shape {(S, T, N)}, but found {C.shape}"  
        return C

    def asdict(self) -> dict:
        """ Return a dictionary representation of the :class:`SanosPricer` """
        return dict( smoothness=self.smoothness,
                     expiries=self.expiries,
                     xstrikes=self.xstrikes,
                     xdensities=self.xdensities,
                     xdensity0=self.xdensity0,
                     sqrtVars=self.sqrtVars,
                     dlvsqs=self.dlvsqs,
                     xnms=self.xnms )

@version("0.1", dependencies=[SanosPricer, SanosDecoderConfig])
class SanosDecoder(BaseModel):
    """
    SANOS functional which decodes a ``state`` of ``num_state`` elements into SANOS parameters, returned as a :class:`SanosPricer` object.
    
    The decoder is a feedforward neural network with ``num_state`` inputs and ``num_output`` outputs, where ``num_output`` is determined by the number
    of expiries and the number of moneyness levels.  
    """

    def __init__(self, env          : BaseEnvironment,
                       num_expiries : int, *, 
                       config       : SanosDecoderConfig):
        super().__init__(env)
        self.config       = config
        self.num_expiries = num_expiries
        self.num_output   = num_expiries+num_expiries*self.config.num_nms if not self.config.learn_nms else\
                            num_expiries+num_expiries*self.config.num_nms+2
        self.decoder      = FeedForward( env, num_input=config.num_state, num_output=self.num_output, config=config.nn_config )  
        self.xnms         = torch.tensor( config.xnms, device=env.device, dtype=env.fdtype )

    def __call__(self, state    : torch.Tensor,
                       expiries : torch.Tensor, *,
                       training : bool = False ) -> dict[str, torch.Tensor]:
        """
        Decode a ``state`` into dlvsqs and foward variances.
        """
        S             = state.shape[0]
        T             = expiries.shape[1]
        n             = self.config.num_nms
        assert len(expiries.shape) == 2 and expiries.shape[0] in [1, S] and expiries.shape[1] == self.num_expiries, f"Expected 'expiries' to have shape ({S}|1, {self.num_expiries}), but found {expiries.shape}"
        assert state.shape==(S,self.config.num_state), f"Expected 'state' to have shape {(S,self.config.num_state)}, but found {state.shape}"

        # extract DLVs, dsqrtVars
        # -----------------------

        r = PrettyObject()
        r.smoothness = self.config.smoothness
        r.expiries   = expiries

        max_dlvsq    = self.config.max_dlv**2
        min_dlvsq    = self.config.min_dlv**2
        max_dsqrtVar = torch.diff( self.config.max_fvol * torch.sqrt(expiries), dim=1, prepend=torch.zeros_like(expiries[:,:1]) ).detach()
        min_dsqrtVar = torch.diff( self.config.min_fvol * torch.sqrt(expiries), dim=1, prepend=torch.zeros_like(expiries[:,:1]) ).detach()

        raw          = self.decoder( state )
        assert raw.shape == (S, self.num_output), f"Expected raw output to have shape {(S, self.num_output)}, but found {raw.shape}"
        raw          = torch.sigmoid( raw )
        r.dlvsqs     = raw[:, :T*n].reshape( S, T, n ) * (max_dlvsq - min_dlvsq) + min_dlvsq
        dsqrtVars    = raw[:, T*n:T*n+T].reshape( S, T ) * (max_dsqrtVar - min_dsqrtVar) + min_dsqrtVar
        r.xnms       = self.xnms[None,None,:]

        if self.config.learn_nms:
            fnms   = raw[:, T*n+T:].reshape( S, T, 1 ) * 1.9 + 0.1  # learn to scale NMS between 10% and 200%
            r.xnms = self.xnms * fnms * 2. # learnable moneyness levels  

        self.verify( lambda : torch.all( torch.isfinite(r.dlvsqs) & (r.dlvsqs > 0.) ), "Infinite or non-positive 'dlvsqs'")
        self.verify( lambda : torch.all( torch.isfinite(dsqrtVars) & (dsqrtVars > 0.) ), "Infinite or non-positive 'dsqrtVars'")
        r.sqrtVars  = torch.cumsum(dsqrtVars, dim=1 )
        raw = None

        # xstrikes
        # --------

        if self.config.nms_eps is None:
            r.xstrikes = torch.exp(r.xnms * r.sqrtVars[:,:,None] )
        else:
            """
            NMS(z,t) = exp( nms * sqrt{ var_e + var_t }/{ e + 1 } )
                    ~ exp( nms * sqrtVar_t * sqrt{ var_e/var_t + 1 }/{ e + 1 } )
                    = exp( nms * sqrtVar_t * sqrt{ var_e/var_t + 1 }/{ e + 1 } )
            """ 
            var_e      = ( r.sqrtVars[:,0]**2 / expiries[:,0] )[:,None] * self.config.nms_eps
            var_t      = r.sqrtVars**2
            nms_adj    = torch.sqrt( ( var_e / var_t + 1 ) / ( self.config.nms_eps + 1. ) ) if not self.config.nms_eps is None else 1.
            r.xstrikes = torch.exp(r.xnms * r.sqrtVars[:,:,None] * nms_adj[:,:,None] )

        # Pricing
        # -------

        # initial density
        _0_S1           = torch.zeros( (S,1), dtype=r.xstrikes.dtype, device=r.xstrikes.device )
        _1_S1           = torch.ones( (S,1), dtype=r.xstrikes.dtype, device=r.xstrikes.device )
        r.xdensity0     = torch.diff( torch.diff( torch.maximum( _0_S1[0    ,0], _1_S1[0,0] - r.xstrikes[:,0,:] ), prepend=_1_S1, dim=1 )
                                      / torch.diff(r.xstrikes[:,0,:], prepend=_0_S1, dim=1),
                                      append=_0_S1, dim=1 )

        # prep tridiagonal solver
        dt         = torch.diff( expiries, prepend=torch.zeros_like(expiries[:,:1]), dim=1 )
        box        = r.xstrikes[:,:,1:-1]**2 * dt[:,:,None] / ( r.xstrikes[:,:,2:] - r.xstrikes[:,:,:-2])
        iw_plus    = box / ( r.xstrikes[:,:,2:] - r.xstrikes[:,:,1:-1] )
        iw_minus   = box / ( r.xstrikes[:,:,1:-1] - r.xstrikes[:,:,:-2] )
        self.verify( lambda : torch.min( iw_plus ) >= 0., lambda : f"'iw_plus' must be non-negative, but found min {torch.min(iw_plus)}" )
        self.verify( lambda : torch.min( iw_minus ) >= 0., lambda : f"'iw_minus' must be non-negative, but found min {torch.min(iw_minus)}" )
        box = None

        # linear interpolation
        L          = torch.maximum( r.xstrikes[0,0,0]*0., r.xstrikes[:,:-1,None,:] - r.xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
        L1         = torch.diff( L, dim=2, prepend=torch.ones_like( L[:,:,:1,:] ) )
        L2         = torch.diff( r.xstrikes[:,1:,:,None], dim=2, prepend=torch.zeros_like( r.xstrikes[:,1:,:1,None] ) )
        L          = torch.diff( L1/L2, dim=2, append=torch.zeros_like( L[:,:,:1,:] ) )
        assert L.shape == (S, T-1, n+2, n+2), f"Expected L to have shape {(S, T-1, n+2, n+2)}, but found {L.shape}"
        L1 = None
        L2 = None

        # Compute densities
        # -----------------

        q_tm1        = None
        r.xdensities = []
        for t in range(T): # loop by expiry
            iw_minus_t  = iw_minus[:,t,:]
            iw_plus_t   = iw_plus[:,t,:]

            # move density forward
            if t>0:
                L_t     = L[:,t-1,:,:] 
                q_tm1   = torch.sum( L_t * q_tm1[:,None,:], dim=-1 ) # TODO: make faster
            else:
                q_tm1   = r.xdensity0

            self.verify( lambda : torch.min( q_tm1 ) >= -1E-6, lambda : f"Negative density values found at expiry #{t}, before solving DLV equation: min density value is {q_tm1.detach().cpu().numpy().min()}.")

            # solve DLV equation
            dlvsqs_t    = r.dlvsqs[:,t,:]
            w_minus_t   = dlvsqs_t * iw_minus_t
            w_plus_t    = dlvsqs_t * iw_plus_t
            q_t         = solve_sanos_tridiagonal( w_minus=w_minus_t, w_plus=w_plus_t, rhs=q_tm1, backbone=_bbtorch )
            r.xdensities.append( q_t )
            q_tm1 = q_t 

        r.xdensities = torch.stack( r.xdensities, axis=1 ) # (S, T, n+2)
        assert r.xdensities.shape == (S, T, n+2), f"Expected r.xdensities to have shape {(S, T, n+2)}, but found {r.xdensities.shape}"

        r.xnms   = r.xnms.detach()
        r.dlvsqs = r.dlvsqs.detach()
        return SanosPricer.create( smoothness=self.config.smoothness, model_output=r, debug=self.debug )
