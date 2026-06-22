"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

import warnings as warnings

import numpy as np
from dataclasses import dataclass, field
from cdxcore.err import error, verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.util import is_filename
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import Backbone, BackboneNumpy
import logging
import math

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, get, pop
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.dysanos.grid.torch_market import GridMarketDefinition, PoolConfig
from dataclasses import asdict, dataclass, field
from quantdata.biz_calendar import NMS

_bbtorch = BackboneTorch()

def _notebook_name() -> str:
    """ Return a name for the notebook based on the current notebook name and the training config. """
    try:
        import ipynbname
        nb_name = ipynbname.name()
    except Exception as e:
        warnings.warn(f"Failed to get notebook name: {e}. Using 'unknown' instead.")
        return None
    i = nb_name.rfind("viz_")
    if i >= 0:
        nb_name = nb_name[i+len("viz_"):]
    return nb_name

@version("0.3", dependencies=[GridMarketDefinition])
@dataclass(frozen=True)
class SanosModelConfig:
    label            : str = ""                  # Unique label for this experiment. If left to "" will be set to the current notebook value.
    group            : str = ""                  # Group for this experiment, if any.
    user_version     : str = "0.1"        # Version to trigger updates

    smoothness       : float|None = None    # Model smoothness
    min_vega_sqrt    : float = 1E-4      # minimum vega/sqrtT; used to compute weights
    scale_by_weights : str = "sample" # Whether to scale the loss by weights

    num_nms   : int = 20               # number of moneyness levels
    nms_range : tuple[float,float] = (-2, 1) # range of moneyness levels
    nms       : np.ndarray = field(init=False, repr=False) # moneyness levels, computed from num_nms and nms_range
    xnms      : np.ndarray = field(init=False, repr=False) # extended moneyness levels, computed from nms
    nms_eps   : float = NMS.EPS        # shift for NMS
    learn_nms : bool = False
    
    min_dlv   : float = 0.009          # Min DLV
    max_dlv   : float = 3.             # Max DLV
    min_fvol  : float = 0.01           # minimum fwd vol
    max_fvol  : float = 2.             # maximum fwd vol

    num_state  : int = 20
    state_config : FeedForward.Config = FeedForward.Config()  # decoder
    learn_nnms  : bool = False

    seed       : int = 213123123

    _debug_dont_learn_sqrtVars : bool = False # debug flag to not learn sqrtVars

    def __post_init__(self):
        verify_inp( self.smoothness is None or self.smoothness >= 0., f"Expected 'smoothness' to be None or non-negative, but found {self.smoothness}" )
        verify_inp( self.min_vega_sqrt > 0., f"Expected 'min_vega_sqrt' to be positive, but found {self.min_vega_sqrt}" )
        verify_inp( self.scale_by_weights in ["none", "sample", "total"], f"Expected 'scale_by_weights' to be either 'none', 'sample' or 'total', but found {self.scale_by_weights}" )

        verify_inp( self.group == "" or is_filename(self.group), f"Expected 'group' to be a valid file name, but found {self.group}" )

        if self.label == "":
            label = _notebook_name()
            verify_inp( label is not None, f"Expected to be able to determine notebook name for 'label', but failed. Please set 'label' explicitly in the config." )
            self.__dict__['label'] = label

        def _nms():
            """ Generate NMS strikes which contains '1' """
            # generate strikes which contain '1'
            N    = self.num_nms
            dist = float(self.nms_range[1] - self.nms_range[0])
            assert dist > 0., f"Internal error: expected grid_range to have positive distance but found {self.nms_range}."
            Nlo  = min(max(2, float(N*abs(self.nms_range[0]))//dist),N-3)
            Nhi = N - Nlo - 1
            assert Nlo>=2 and Nhi>=2, f"{message()}Internal error: error {Nlo} {Nhi} for grid_range {self.nms_range} and grid_num_strikes {N}."
            nms = np.concatenate( ( np.linspace(self.nms_range[0], 0., int(Nlo), endpoint=False), np.linspace(0., self.nms_range[1], int(Nhi)+1) ) )
            assert len(nms)==N, f"{message()}Internal error: generated number of moneyness levels {len(nms)} does not match expected {N}."
            return nms

        nms = _nms()
        xnms = np.concatenate( [[self.nms_range[0]*2], nms, [self.nms_range[1]*2]] )
        self.__dict__['nms'] = nms
        self.__dict__['xnms'] = xnms

    @property
    def label_str(self) -> str:
        return f"{self.label}{f'.{self.group}' if self.group else ''} {self.user_version} " + ( f"{self.smoothness:.2f}-smooth" if not self.smoothness is None else "linear" )

    @property
    def obj_str(self) -> str:
        return f"{self.scale_by_weights},mvsq={self.min_vega_sqrt:.3g}"

    @property
    def nms_str(self) -> str:
        return f"nms={self.num_nms},({self.nms_range[0]:.2f},{self.nms_range[1]:.2f})"

    @property
    def minmax_str(self):
        return f"dlv=({self.min_dlv:.2f},{self.max_dlv:.2f}),fvol=({self.min_fvol:.2f},{self.max_fvol:.2f})"

    @property
    def state_str(self) -> str:
        return f"state={self.num_state}@{self.state_config}"

    def create(self, env : BaseEnvironment, *, S, T, N) -> type:
        """ Create Model """
        raise NotImplementedError("This method should be implemented in subclasses to create a specific model instance.")

    def torch_data(self, 
                   env : BaseEnvironment,
                   market : PrettyObject, *,
                   pool_config : PoolConfig = PoolConfig.off(),
                   message_context: str|callable[[], str] | None = None,
                   verbose : Context = Context.all ) -> dict:
        """
        Preprocessing training data set.

        This function returns a full formed training data set to be passed to the model, all on torch.
        The ``env`` environent defines the device, debugging, and the floating point precision.

        Parameters
        ----------
        env : :class:`BaseEnvironment`  
            The torch environment to use for the model.
        market : :class:`PrettyObject`
            The market definition or market data to use for the model. Contains with ``grid_market_def`` an attribute of type :class:`GridMarketDefinition`.
        pool_config : :class:`PoolConfig`, default :meth:`PoolConfig.off`.
            The pool configuration to use for loading the market data. Default is :meth:`PoolConfig.off`.
        message_context : str|callable[[], str]|None, default ``None``
            The message context to use for error messages. If None, no additional message context will be used. 
        verbose : :class:`Context`, default :attr:`Context.all`
            The verbosity context to use for logging.
        """
        def message(with_cols : bool = True) -> str:
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg == "":
                return ""
            msg = msg[0].upper() + msg[1:]
            if not with_cols:
                return msg
            return msg + ": "

        assert self.smoothness is None or self.smoothness >= 0., f"{message()}Expected 'smoothness' to be None or non-negative, but found {self.smoothness}"
        assert self.min_vega_sqrt > 0., f"{message()}Expected 'min_vega_sqrt' to be positive, but found {self.min_vega_sqrt}"

        market = market.load( pool_config=pool_config, verbose=verbose ) if isinstance(market, GridMarketDefinition) else market 
        assert hasattr(market, 'grid_market_def'), f"{message()}Expected market to have attribute 'grid_market_def', but found {fmt_list(sorted(market))}"

        expiries: np.ndarray     = market['expiries']
        mids:np.ndarray          = market['mids']
        mid_vols: np.ndarray     = market['mid_vols']
        mid_vega: np.ndarray     = market['mid_vega']
        mid_delta: np.ndarray    = market['mid_delta']
        xstrikes: np.ndarray     = market['xstrikes']
        nms : np.ndarray         = market['nms']
        variances: np.ndarray    = market['variances'] # (S, T)
        weights: np.ndarray|None = market.get('weights', None)

        S, T, N  = mids.shape        
        assert xstrikes.shape == (S, T, N+2), f"Expected xstrikes to have shape {(S, T, N+2)}, but found {xstrikes.shape}"
        assert expiries.shape == (S, T,), f"Expected expiries to have shape {(S, T)}, but found {expiries.shape}"
        assert variances.shape == (S, T), f"Expected variances to have shape {(S, T)}, but found {variances.shape}"
        assert nms.shape == (S, T, N), f"Expected nms to have shape {(S, T, N)}, but found {nms.shape}"
        assert mid_vols.shape == (S, T, N), f"Expected mid_vols to have shape {(S, T, N)}, but found {mid_vols.shape}"
        assert mid_vega.shape == (S, T, N), f"Expected mid_vega to have shape {(S, T, N)}, but found {mid_vega.shape}"
        assert mid_delta.shape == (S, T, N), f"Expected mid_delta to have shape {(S, T, N)}, but found {mid_delta.shape}"
        assert weights is None or weights.shape == mids.shape, lambda : f"{message()}Expected weights to hshape {(S, T, N)}, but found {weights.shape}"

        assert np.min(xstrikes) > 0., f"{message()}All 'xstrikes' must be positive, but found min {np.min(xstrikes)}"
        assert np.min( np.diff(xstrikes, axis=-1) ) > 0., f"{message()}All 'xstrikes' must be increasing, but found min diff {np.min( np.diff(xstrikes, axis=-1) )}"
        assert np.min(expiries) > 0., f"{message()}All 'expiries' must be positive, but found min {np.min(expiries)}"
        assert np.min( np.diff(expiries, axis=-1) ) > 0, f"{message()}All 'expiries' must be increasing, but found min diff {np.min( np.diff(expiries, axis=-1) )}"
        assert np.min(variances) >= 0., f"{message()}All 'variances' must be non-negative, but found min {np.min(variances)}"
        assert np.min( np.diff(variances, axis=-1) ) > 0, f"{message()}All 'variances' must be increasing, but found min diff {np.min( np.diff(variances, axis=-1) )}"
        assert weights is None or np.min(np.sum(weights, axis=-1)) > 0., f"{message()}At least one weight must be positive for each expiry, but found min sum {np.min(np.sum(weights, axis=-1))}"  

        r = PrettyObject()

        # fit
        # compute transition 1
        # --------------------
        dt          = np.diff( expiries, prepend=0. )[:,:,None] 
        box         = xstrikes[:,:,1:-1]**2 * dt / ( xstrikes[:,:,2:] - xstrikes[:,:,:-2])
        w_plus      = box / ( xstrikes[:,:,2:] - xstrikes[:,:,1:-1] )
        w_minus     = box / ( xstrikes[:,:,1:-1] - xstrikes[:,:,:-2] )
        assert np.min( w_plus ) >= 0., f"{message()}computed 'w_plus' must be non-negative, but found min {np.min(w_plus)}"
        assert np.min( w_minus ) >= 0., f"{message()}computed 'w_minus' must be non-negative, but found min {np.min(w_minus)}"

        r.iw_minus = torch.tensor( w_minus, device=env.device, dtype=env.fdtype )
        r.iw_plus = torch.tensor( w_plus, device=env.device, dtype=env.fdtype )
        del w_minus, w_plus, box, dt

        # L transition
        # ------------
        L = np.maximum( 0., xstrikes[:,:-1,None,:] - xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
        L = np.diff( np.diff( L, axis=2, prepend=1. ) / np.diff( xstrikes[:,1:,:,None], axis=2, prepend=0. ), axis=2, append=0. )
        r.L = torch.tensor( L, device=env.device, dtype=env.fdtype )
        assert r.L.shape == (S, T-1, N+2, N+2), f"Expected L to have shape {(S, T-1, N+2, N+2)}, but found {r.L.shape}"
        del L

        # weights
        # -------

        if weights is None:
            vega_sqrt = mid_vega / np.sqrt(expiries[:,:,None])
            has_vega  = vega_sqrt >= self.min_vega_sqrt
            weights   = np.where( has_vega, 1./np.where( has_vega, vega_sqrt, 1.), 0. )

        assert np.all( np.any(weights > 0., axis=(1,2))), f"{message()}At least one weight must be positive for each sample, but found {np.sum(~np.any(weights > 0., axis=(1,2)))} samples with no positive weights"

        # objective
        # ---------
        def np_call( spots, strikes, variances, expiries ):                
            intr = np.maximum( 0., spots - strikes )
            if self.smoothness is None or self.smoothness == 0.:
                return intr
            sqrtVar  = np.sqrt( self.smoothness * variances )
            assert np.all( np.isfinite(sqrtVar) ), f"Internal error: {message()}infinite vol"
            return bs.price( k=strikes/spots, vol=sqrtVar, sqrtT=1., eps=1E-6 )*spots

        r.CX       = torch.tensor( np_call( spots=xstrikes[:,:,None,:], strikes=xstrikes[:,:,1:-1,None], variances=variances[:,:,None,None], expiries=expiries[:,:,None,None] ), device=env.device, dtype=env.fdtype )
        assert r.CX.shape == (S, T, N, N+2), f"Expected CX to have shape {(S, T, N, N+2)}, but found {r.CX.shape}"

        # rest
        r.xdensity0 = torch.tensor( np.diff( np.diff( np.maximum( 0., 1. - xstrikes[:,0,:] ), prepend=1. ) / np.diff(xstrikes[:,0,:], prepend=0.), append=0. ), device=env.device, dtype=env.fdtype )
        r.mids      = torch.tensor( mids, device=env.device, dtype=env.fdtype ) 
        r.mid_vols  = torch.tensor( mid_vols, device=env.device, dtype=env.fdtype )
        r.mid_vega  = torch.tensor( mid_vega, device=env.device, dtype=env.fdtype )
        r.mid_delta = torch.tensor( mid_delta, device=env.device, dtype=env.fdtype )
        r.expiries  = torch.tensor( expiries, device=env.device, dtype=env.fdtype )
        r.xstrikes  = torch.tensor( xstrikes, device=env.device, dtype=env.fdtype )
        r.nms       = torch.tensor( nms, device=env.device, dtype=env.fdtype )
        r.weights   = torch.tensor( weights, device=env.device, dtype=env.fdtype ) 
        r.variances = torch.tensor( variances, device=env.device, dtype=env.fdtype )

        for k, v in r.items():
            if not v is None:
                assert torch.all( torch.isfinite(v) ), f"{message()}Found non-finite value in {k}"
                assert v.device == r.nms.device, f"{message()}Expected {k} to be on device {r.nms.device}, but found {v.device}"

        return dict(r)

@version("0.2", dependencies=[BaseEnvironment, SanosModelConfig])
class SanosModel(BaseModel):

    NUM_FEATURES = 7

    def __init__( self, env : BaseEnvironment, T, config : SanosModelConfig ):
        super().__init__(env)
        self.config = config
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(
            recompiles=True,
            graph_breaks=True,
            dynamo=logging.WARNING,
        )

        n                   = config.num_nms
        self.num_output     = ( T * n + T ) if not self.config.learn_nnms else ( T * n + T + T * n)
        self.raw_from_state = FeedForward( env, num_input=config.num_state, num_output=self.num_output, config=config.state_config )  
        self.xnms           = torch.tensor( config.xnms, device=env.device, dtype=env.fdtype ) 

    def build_default_features( self, data : Pytree, *, training : bool = False ) -> torch.Tensor:
        """ Compute feature block """
        data             = dict(data)
        mid_vols         = data.get("mid_vols")
        S, T, N          = mid_vols.shape
        market_xstrikes  = get(data, "xstrikes", (S, T, N+2) )
        expiries         = get(data, "expiries", (S, T) )
        variances        = get(data, "variances", (S, T) )
        market_nms       = get(data, "nms", (S, T, N) )
        mid_vega         = get(data, "mid_vega", (S, T, N) )
        mid_delta        = get(data, "mid_delta", (S, T, N) )

        logK      = torch.log( market_xstrikes[:,:,1:-1] )
        dsqrtVars = torch.diff( torch.sqrt(variances), dim=1, prepend=torch.zeros_like(expiries[:,:1]) )
        features  = torch.stack( [ logK, logK*0.+expiries[:,:,None], logK*0.+dsqrtVars[:,:,None], mid_vols, market_nms, mid_vega, mid_delta ], axis=-1 )
        assert features.shape == (S, T, N, self.NUM_FEATURES), f"Expected features to have shape {(S, T, N, self.NUM_FEATURES)}, but found {features.shape}"
        features  = features.reshape( S, -1 )   
        assert features.shape == (S, T*N*self.NUM_FEATURES), f"Expected features to have shape {(S, T*N*self.NUM_FEATURES)}, but found {features.shape}"
        return features

    def compute_state( self, data : Pytree, *, sample_indices : np.array|None = None, training : bool = False ) -> tuple[torch.Tensor, dict|None]:
        raise NotImplementedError("This method should be implemented in subclasses to compute the raw DLVs from the (abstract) model state.")

    def translate_state( self, data : Pytree, *, sample_indices : np.array|None = None, training : bool = False ) -> tuple[torch.Tensor, dict|None]:
        """
        Decode DLVsq grid and forward variances from the (abstract) model state.
        """
        expiries      = get(data, "expiries")
        S, T          = expiries.shape
        n             = self.config.num_nms
        state, other  = self.compute_state( data=data, sample_indices=sample_indices, training=training )
        assert state.shape==(S,self.config.num_state), f"Expected raw output to have shape {(S,self.config.num_state)}, but found {state.shape}"

        max_dlvsq    = self.config.max_dlv**2
        min_dlvsq    = self.config.min_dlv**2
        max_dsqrtVar = torch.diff( self.config.max_fvol * torch.sqrt(expiries), dim=1, prepend=torch.zeros_like(expiries[:,:1]) ).detach()
        min_dsqrtVar = torch.diff( self.config.min_fvol * torch.sqrt(expiries), dim=1, prepend=torch.zeros_like(expiries[:,:1]) ).detach()

        raw       = self.raw_from_state( state )
        assert raw.shape == (S, self.num_output), f"Expected raw output to have shape {(S, self.num_output)}, but found {raw.shape}"
        raw       = torch.sigmoid( raw )
        dlvsqs    = raw[:, :T*n].reshape( S, T, n ) * (max_dlvsq - min_dlvsq) + min_dlvsq
        dsqrtVars = raw[:, T*n:T*n+T].reshape( S, T ) * (max_dsqrtVar - min_dsqrtVar) + min_dsqrtVar

        if self.config.learn_nnms:
            nnms = torch.cumsum( raw[:, T*n+T:].reshape( S, T, n ), dim=2 )
            nnms = nnms - nnms[:,:,:1]   # shift to start at 0
            nnms = nnms / nnms[:,:,-1:]  # normalize to [0,1]
        else:
            nnms = None

        if self.config._debug_dont_learn_sqrtVars:
            market_variances = get(data, "variances", (S,T))
            dsqrtVars = torch.diff( torch.sqrt(market_variances), dim=1, prepend=torch.zeros_like(expiries[:,:1]) ).detach()

        self.verify( lambda : torch.all( torch.isfinite(dlvsqs) & (dlvsqs > 0.) ), "Infinite or non-positive 'dlvsqs'")
        self.verify( lambda : torch.all( torch.isfinite(dsqrtVars) & (dsqrtVars > 0.) ), "Infinite or non-positive 'dsqrtVars'")
        sqrtVars  = torch.cumsum( dsqrtVars, dim=1 )

        if training:
            return dict(dlvsqs=dlvsqs, sqrtVars=sqrtVars,nnms=nnms,)
        
        r = dict(
                dlvsqs=dlvsqs, 
                sqrtVars=sqrtVars,
                nnms=nnms,
                state=state.detach(),
                )
        assert set(r) & set(other) == set(), f"Internal error: expected 'other' to not contain keys {set(r)}, but found {set(r) & set(other)}"
        r.update({k:v.detach() for k,v in other.items() if not v is None})
        return r

    def compute_model( self, model_state : dict[str, torch.Tensor],
                            data      : Pytree, *, 
                            training  : bool = False ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Utility function to compute call prices and densities.
        """
        dlvsqs          = model_state['dlvsqs']
        S, T, n         = dlvsqs.shape
        assert n == self.config.num_nms, f"Expected 'dlvsqs' to have shape {(S, T, self.config.num_nms)}, but found {dlvsqs.shape}"
        sqrtVars        = get(model_state, "sqrtVars", (S, T))
        nnms            = get(model_state, "nnms", (S, T, n)) if self.config.learn_nnms else None
        expiries        = get(data, "expiries", (S, T) )
        market_xstrikes = get(data, "xstrikes")
        Cs              = []
        qs              = [] 
        assert market_xstrikes.shape[:2] == (S, T), f"Expected market_xstrikes to have shape {(S, T)}, but found {market_xstrikes.shape}"
        N               = market_xstrikes.shape[2]-2

        # compute grid geometry
        # ---------------------

        _0              = torch.zeros_like( expiries[:,:1] )
        _1              = torch.ones_like( expiries[:,:1] )
        dt              = torch.diff( expiries, prepend=_0, dim=1 )

        """
        NMS(z,t) = exp( nms * sqrt{ var_e + var_t }/{ e + 1 } )
                 ~ exp( nms * sqrtVar_t * sqrt{ var_e/var_t + 1 }/{ e + 1 } )
        """
        var_e           = ( sqrtVars[:,0]**2 / expiries[:,0] )[:,None] * self.config.nms_eps
        var_t           = sqrtVars**2
        nms_adjusment   = torch.sqrt( ( var_e / var_t + 1 ) / ( self.config.nms_eps + 1. ) )
        assert self.xnms.shape[0] == n+2, f"Expected xnms to have shape {(n+2,)}, but found {self.xnms.shape}"

        if nnms is not None:
            # learn grid
            assert nnms.shape == (S, T, n), f"Expected 'nnms' to have shape {(S, T, n)}, but found {nnms.shape}"
            nms_lo   = self.xnms[1]
            nms_hi   = self.xnms[-2]
            nms      = nnms * (nms_hi - nms_lo) + nms_lo
            zro      = torch.zeros_like(nms[:,:,:1])
            xnms     = torch.cat([zro+self.xnms[0], nms, zro+self.xnms[-1]], dim=2)
            assert xnms.shape == (S, T, n+2), f"Expected 'xnms' to have shape {(S, T, n+2)}, but found {xnms.shape}"
            xstrikes = torch.exp(xnms * sqrtVars[:,:,None] * nms_adjusment[:,:,None])
        else:
            xstrikes = torch.exp(self.xnms[None,None,:] * sqrtVars[:,:,None] * nms_adjusment[:,:,None])

        assert xstrikes.shape == (S,T,n+2), f"Expected xstrikes to have shape {(S,T,n+2)}, but found {xstrikes.shape}"
        self.verify( lambda : torch.all( torch.isfinite(xstrikes) & (xstrikes > 0.) ), "Infinite or non-positive 'xstrikes'" )
        xdensity0       = torch.diff( torch.diff( torch.maximum( _0[0,0], _1[0,0] - xstrikes[:,0,:] ), prepend=_1, dim=1 )
                                      / torch.diff(xstrikes[:,0,:], prepend=_0, dim=1),
                                      append=_0, dim=1 )
        assert xdensity0.shape == (S,n+2), f"Expected xdensity0 to have shape {(S,n+2)}, but found {xdensity0.shape}"
        _0 = None
        _1 = None

        # tridiagonal solver
        box             = xstrikes[:,:,1:-1]**2 * dt[:,:,None] / ( xstrikes[:,:,2:] - xstrikes[:,:,:-2])
        iw_plus         = box / ( xstrikes[:,:,2:] - xstrikes[:,:,1:-1] )
        iw_minus        = box / ( xstrikes[:,:,1:-1] - xstrikes[:,:,:-2] )
        self.verify( torch.min( iw_plus ) >= 0., lambda : f"'iw_plus' must be non-negative, but found min {torch.min(iw_plus)}" )
        self.verify( torch.min( iw_minus ) >= 0., lambda : f"'iw_minus' must be non-negative, but found min {torch.min(iw_minus)}" )
        box = None

        # linear interpolation
        L   = torch.maximum( xstrikes[0,0,0]*0., xstrikes[:,:-1,None,:] - xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
        L1 = torch.diff( L, dim=2, prepend=torch.ones_like( L[:,:,:1,:] ) )
        L2 = torch.diff( xstrikes[:,1:,:,None], dim=2, prepend=torch.zeros_like( xstrikes[:,1:,:1,None] ))
        L = torch.diff( L1/L2, dim=2, append=torch.zeros_like( L[:,:,:1,:] ) )
        assert L.shape == (S, T-1, n+2, n+2), f"Expected L to have shape {(S, T-1, n+2, n+2)}, but found {L.shape}"
        _0 = None
        _1 = None
        L1 = None
        L2 = None


        # model to market
        if self.config.smoothness is None or self.config.smoothness == 0.:
            CX = torch.maximum( xstrikes[0,0,0]*0., xstrikes[:,:,None,:] - market_xstrikes[:,:,1:-1,None] )
        else:
            CX = _bbtorch.bs_price( spot=xstrikes[:,:,None,:],
                                    strikes=market_xstrikes[:,:,1:-1,None],
                                    sqrtVars=math.sqrt(self.config.smoothness) * sqrtVars[:,:,None,None])
        assert CX.shape == (S, T, N, n+2), f"Expected CX to have shape {(S, T, N, n+2)}, but found {CX.shape}"  

        # compute market call prices
        # --------------------------

        for t in range(T): # loop by expiry
            iw_minus_t  = iw_minus[:,t,:]
            iw_plus_t   = iw_plus[:,t,:]
            CX_t        = CX[:,t,:,:]

            # move density forward
            if t>0:
                L_t     = L[:,t-1,:,:] 
                q_tm1   = torch.sum( L_t * q_tm1[:,None,:], dim=-1 ) # TODO: make faster
            else:
                q_tm1   = xdensity0

            self.verify( lambda : torch.min( q_tm1 ) >= -1E-6, lambda : f"Negative density values found at expiry #{t}, before solving DLV equation: min density value is {q_tm1.detach().cpu().numpy().min()}.")

            # solve DLV equation
            dlvsqs_t    = dlvsqs[:,t,:]
            w_minus_t   = dlvsqs_t * iw_minus_t
            w_plus_t    = dlvsqs_t * iw_plus_t
            q_t         = solve_sanos_tridiagonal( w_minus=w_minus_t, w_plus=w_plus_t, rhs=q_tm1, backbone=_bbtorch )

            # compute market call prices
            C_t          = torch.sum( CX_t * q_t[:,None,:], dim=-1 ) # TODO: make faster
            assert C_t.shape == (S,N), f"Expiry #{t}: expected shape {(S,N)} for model call prices C_t, but found {C_t.shape}"
            Cs.append( C_t )
            qs.append( q_t )
            q_tm1 = q_t 

        return dict(
            calls=torch.stack( Cs, axis=1 ),      # (S, T, N)
            xdensities=torch.stack( qs, axis=1 ), # (S, T, n+2)
            xstrikes=xstrikes,                    # (S, T, n+2)
        )

    def __call__(self, data : Pytree, sample_indices : np.array|None = None, training : bool = False ) -> float | Pytree:
        """ Training or valuation """

        # call derived model
        # ------------------
        model_state  = self.translate_state( data, sample_indices=sample_indices, training=training )
        model_output = self.compute_model( model_state=model_state, data=data, training=training )    
        assert set(model_state) & set(model_output) == set(), f"Expected 'model_output' to not contain any keys that are already in 'model_state', but found {sorted(set(model_state) & set(model_output))}"

        S, T, n        = model_state['dlvsqs'].shape
        mids           = get(data, "mids")
        assert mids.shape[:2] == (S, T), f"Expected mids to have shape {(S, T, None)}, but found {mids.shape}"
        N              = mids.shape[2]
        calls          = get(model_output, "calls", (S, T, N))

        # solve for the densities
        # -----------------------

        mids            = get(data, "mids", (S, T, N) )
        weights         = get(data, "weights", (S, T, N))

        loss            = 0.
        raw_error       = 0.
        wsum            = 0.

        raw_errors      = [] if not training else None
        errors          = [] if not training else None

        if self.config.scale_by_weights == "sample":
            sample_scale = 1./torch.sum( weights, dim=(1,2) ) # NB: we always have at least one positive weight per sample, so this should always be finite
            wsum         = S # number of samples
        else:
            sample_scale = None
        
        for t in range(T): # loop by expiry
            # compute market call prices
            weights_t    = weights[:,t,:]
            mids_t       = mids[:,t,:]
            C_t          = calls[:,t,:]
            assert C_t.shape == (S,N), f"Expiry #{t}: expected shape {(S,N)} for model call prices C_t, but found {C_t.shape}"

            # add errors
            raw_error_t   = torch.abs( weights_t * ( C_t - mids_t ) )
            error_t       = torch.where( raw_error_t <= 1., raw_error_t**2, 2.*raw_error_t - 1. )

            if self.config.scale_by_weights == "none":
                # add up all weighted errors; normalize by total non-zero weights
                loss      = loss + torch.sum( error_t )
                wsum      = wsum + torch.sum( weights_t>0 )
            elif self.config.scale_by_weights == "sample":
                # weight here already by sample scale; normalize by total number of smaples
                loss     = loss + torch.sum( error_t * sample_scale[:,None] )
            else:
                assert self.config.scale_by_weights == "total", f"Expected 'scale_by_weights' to be either 'none', 'sample' or 'total', but found {self.config.scale_by_weights}"
                # weight by total weights
                loss      = loss + torch.sum( error_t )
                wsum      = wsum + torch.sum( weights_t )

            if not training:
                raw_errors.append( torch.detach(raw_error_t) )
                errors.append( torch.detach(error_t) )

        loss = loss / wsum
        if training:
            return loss

        r = PrettyObject()
        r.loss        = loss.detach()          # scalar
        r.raw_errors  = torch.stack( raw_errors, axis=1 ) # (S, T, N)
        r.errors      = torch.stack( errors, axis=1 ) # (S, T, N) 

        assert set(model_state) & set(r) == set(), f"Expected 'other' to not contain any keys that are already in the result, but found {sorted(set(model_state) & set(r))}"
        r.update(model_state)
        assert set(model_output) & set(r) == set(), f"Expected 'other' to not contain any keys that are already in the result, but found {sorted(set(model_output) & set(r))}"
        r.update(model_output)

        # from model_statre
        assert r.dlvsqs.shape == (S, T, n), f"Expected dlvsqs to have shape {(S, T, n)}, but found {r.dlvsqs.shape}"
        assert r.sqrtVars.shape == (S, T), f"Expected sqrtVars to have shape {(S, T)}, but found {r.sqrtVars.shape}"

        # ffrom model_output
        assert r.calls.shape == (S, T, N), f"Expected calls to have shape {(S, T, N)}, but found {r.calls.shape}"
        assert r.xdensities.shape == (S, T, n+2), f"Expected xdensities to have shape {(S, T, n+2)}, but found {r.xdensities.shape}"
        assert r.xstrikes.shape == (S, T, n+2), f"Expected xstrikes to have shape {(S, T, n+2)}, but found {r.xstrikes.shape}"

        # from loss
        assert r.raw_errors.shape == (S, T, N), f"Expected raw_errors to have shape {(S, T, N)}, but found {r.raw_errors.shape}"
        assert r.errors.shape == (S, T, N), f"Expected errors to have shape {(S, T, N)}, but found {r.errors.shape}"

        return dict(r)
        





