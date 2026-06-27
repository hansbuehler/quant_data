"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

from collections.abc import Callable
import warnings as warnings

from cdxcore.subdir import SubDir
from cdxcore.uniquehash import named_unique_filename48_8
import numpy as np
from dataclasses import asdict, dataclass, field
from cdxcore.err import error, verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.util import fmt_big_byte_number, fmt_digits, is_filename
import logging
import math

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.torch import BaseEnvironment, BaseModel, torch, get
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.dysanos.grid.torch_market import GridMarketDefinition, PoolConfig
from quantdata.dysanos.grid.torch_pricer import SanosDecoder, SanosDecoderConfig

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
    label            : str = ""           # Unique label for this experiment. If left to "" will be set to the current notebook value.
    group            : str = ""           # Group for this experiment, if any.
    user_version     : str = "0.1"        # Version to trigger updates
    decoder_config   : SanosDecoderConfig = SanosDecoderConfig()  # decoder config
    encoder_config   : FeedForwardConfig = FeedForwardConfig()  # state encoder config
    min_vega_sqrt    : float = 1E-4       # minimum vega/sqrtT; used to compute weights
    scale_by_weights : str = "sample"     # Whether to scale the loss by weights
    seed             : int = 213123123

    @property
    def smoothness(self) -> float|None:
        return self.decoder_config.smoothness
    
    @property
    def num_state(self) -> int:
        return self.decoder_config.num_state

    def __post_init__(self):
        verify_inp( self.min_vega_sqrt > 0., f"Expected 'min_vega_sqrt' to be positive, but found {self.min_vega_sqrt}" )
        verify_inp( self.scale_by_weights in ["none", "sample", "total"], f"Expected 'scale_by_weights' to be either 'none', 'sample' or 'total', but found {self.scale_by_weights}" )
        verify_inp( self.group == "" or is_filename(self.group), f"Expected 'group' to be a valid file name, but found {self.group}" )

        if self.label == "":
            label = _notebook_name()
            verify_inp( label is not None, f"Expected to be able to determine notebook name for 'label', but failed. Please set 'label' explicitly in the config." )
            self.__dict__['label'] = label

    @property
    def qualified_label(self) -> str:
        return f"{self.label}@{f'{self.group}' if self.group else ''} {self.user_version}"
    
    @property
    def label_str(self) -> str:
        return self.qualified_label + ( f" {self.smoothness:.2f}-smooth" if not self.smoothness is None else " linear" )

    @property
    def obj_str(self) -> str:
        return f"{self.scale_by_weights},mvsq={self.min_vega_sqrt:.3g}"

    def __str__(self) -> str:
        return f"{self.qualified_label} {self.decoder_config} {self.obj_str}"

    def create(self, env : BaseEnvironment, *, num_expiries, num_market_strikes ) -> type:
        """ Create Model """
        return SanosModel( env, num_expiries=num_expiries, num_market_strikes=num_market_strikes, config=self )

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
        r.xstrikes  = torch.tensor( xstrikes, device=env.device, dtype=env.fdtype )   # market
        r.nms       = torch.tensor( nms, device=env.device, dtype=env.fdtype )        # market
        r.weights   = torch.tensor( weights, device=env.device, dtype=env.fdtype ) 
        r.variances = torch.tensor( variances, device=env.device, dtype=env.fdtype )

        for k, v in r.items():
            if not v is None:
                assert torch.all( torch.isfinite(v) ), f"{message()}Found non-finite value in {k}"
                assert v.device == r.nms.device, f"{message()}Expected {k} to be on device {r.nms.device}, but found {v.device}"

        return dict(r)

@version("0.2", dependencies=[SanosModelConfig, SanosDecoder])
class SanosModel(BaseModel):

    NUM_FEATURES = 7

    def __init__( self, env : BaseEnvironment, 
                        num_expiries       : int, 
                        num_market_strikes : int, 
                        *, config : SanosModelConfig ):
        super().__init__(env)

        self.config             = config
        self.num_expiries       = num_expiries 
        self.num_market_strikes = num_market_strikes

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(
            recompiles=True,
            graph_breaks=True,
            dynamo=logging.WARNING,
        )

        T                = num_expiries
        N                = num_market_strikes
        self.encoder     = FeedForward( env, num_input=T*N*self.NUM_FEATURES,
                                             num_output=config.decoder_config.num_state,
                                             config=config.encoder_config )
        self.decoder     = SanosDecoder( env, num_expiries=num_expiries, config=config.decoder_config )
 
    @property
    def label(self) -> str:
        return self.config.label
    
    @property
    def qualified_label(self) -> str:
        return self.config.qualified_label

    @property
    def smoothness(self) -> float|None:
        return self.config.smoothness
    
    @property
    def num_state(self) -> int:
        return self.config.num_state

    def build_default_features( self, data : Pytree, *, training : bool = False ) -> torch.Tensor:
        """ Compute feature block """
        data             = dict(data)
        mid_vols         = data.get("mid_vols")
        S, T, N          = mid_vols.shape
        assert (T,N) == (self.num_expiries, self.num_market_strikes), f"Expected (T,N)=({self.num_expiries},{self.num_market_strikes}), but found {(T,N)}"

        market_xstrikes  = get(data, "xstrikes", (S, T, N+2) )
        expiries         = get(data, "expiries", (S, T) )  # must be the same as self.expiries
        variances        = get(data, "variances", (S, T) )
        market_nms       = get(data, "nms", (S, T, N) )
        mid_vega         = get(data, "mid_vega", (S, T, N) )
        mid_delta        = get(data, "mid_delta", (S, T, N) )

        logK      = torch.log( market_xstrikes[:,:,1:-1] )
        dsqrtVars = torch.diff( torch.sqrt(variances), dim=1, prepend=torch.zeros_like(expiries[:,:1]) )
        features  = torch.stack( [ logK, 
                                   logK*0.+expiries[:,:,None], 
                                   logK*0.+dsqrtVars[:,:,None], 
                                   mid_vols, 
                                   market_nms, 
                                   mid_vega, 
                                   mid_delta ], axis=-1 )
        assert features.shape == (S, T, N, self.NUM_FEATURES), f"Expected features to have shape {(S, T, N, self.NUM_FEATURES)}, but found {features.shape}"
        features  = features.reshape( S, -1 )   
        assert features.shape == (S, T*N*self.NUM_FEATURES), f"Expected features to have shape {(S, T*N*self.NUM_FEATURES)}, but found {features.shape}"
        return features

    def __call__(self, data : Pytree, *, 
                       state : Pytree|None = None,
                       training : bool = False ) -> float | Pytree:
        """
        Run Training or valuation
        In valuation mode, this function returns a dictionary with the following keys:
        - loss: the loss value
        - raw_errors: the raw errors (S, T, N)
        - errors: the errors (S, T, N)
        - fits: the fitted prices (S, T, N)
        - fit_vols: the fitted volatilities (S, T, N)
        - state: the state (S, num_state)
        - dlvsqs: the dlvsqs (S, T, N)
        - sqrtVars: the sqrtVars (S, T)
        - calls: the calls (S, T, N)
        - xstrikes: the xstrikes (S, T, N+2)
        - xnms: the xnms (S, N+2)
        - xdensities: the xdensities (S, T, N+2)
        - xdensity0: the xdensity0 (S, N+2)
        - expiries: the expiries (S, T)             
        """
        mids           = get( data, "mids" )
        S, T, N         = mids.shape
        assert (T,N) == (self.num_expiries, self.num_market_strikes), f"Expected (T,N)=({self.num_expiries},{self.num_market_strikes}), but found {(T,N)}"

        # learn map from features to state directly
        features        = self.build_default_features(data, training=training)
        self.verify( lambda : torch.all( torch.isfinite(features) ), "Infinite values found in features" )
        assert state is None or state.shape == (S, self.num_state), f"Expected state to have shape {(S, self.num_state)}, but found {state.shape}"
        state           = self.encoder( features ) if state is None else state

        market_xstrikes = get( data, "xstrikes", (S,T,N+2) )
        mids            = get( data, "mids", (S, T, N) )
        weights         = get( data, "weights", (S, T, N))
        expiries        = get( data, "expiries", (S, T) )
        market_strikes  = market_xstrikes[:,:,1:-1]

        pricer          = self.decoder( state=state, expiries=expiries, training=training )
        fits            = pricer( market_strikes, training=training )
        mids            = get(data, "mids", (S, T, N) )
        weights         = get(data, "weights", (S, T, N))

        loss            = 0.
  
        raw_errors      = torch.abs( weights * ( fits - mids ) )
        errors          = torch.where( raw_errors <= 1., raw_errors**2, 2.*raw_errors - 1. )

        if self.config.scale_by_weights == "none":
            # average error
            loss      = torch.sum( errors ) / torch.sum( weights>0 )
        elif self.config.scale_by_weights == "sample":
            # average error per sample
            loss      = torch.mean( torch.sum( errors / torch.sum( weights, dim=(1,2), keepdims=True ), dim=(1,2) ) )
        else:
            assert self.config.scale_by_weights == "total", f"Expected 'scale_by_weights' to be either 'none', 'sample' or 'total', but found {self.config.scale_by_weights}"
            # weighted
            loss      = torch.sum( errors ) / torch.sum( weights )

        if training:
            return loss

        r = PrettyObject()
        r.loss        = loss.detach()        # scalar
        r.raw_errors  = raw_errors.detach()  # (S, T, N) 
        r.errors      = errors.detach()      # (S, T, N) 
        r.fits        = fits.detach()        # (S, T, N)
        #r.fit_vols    = fit_vols.detach()   # (S, T, N)
        r.state       = state.detach()       # (S, num_state)

        model_output = pricer.asdict()
        assert set(model_output) & set(r) == set(), f"Expected 'model_output' to not contain any keys that are already in the result, but found {sorted(set(model_output) & set(r))}"
        for k,v in model_output.items():
            if isinstance(v, torch.Tensor):
                r[k] = v.detach()

        # from model_statre
        n = r.xstrikes.shape[-1]-2
        assert r.dlvsqs.shape == (S, T, n), f"Expected dlvsqs to have shape {(S, T, n)}, but found {r.dlvsqs.shape}"
        assert r.sqrtVars.shape == (S, T), f"Expected sqrtVars to have shape {(S, T)}, but found {r.sqrtVars.shape}"

        # from model_output
        assert r.fits.shape == (S, T, N), f"Expected fits to have shape {(S, T, N)}, but found {r.fits.shape}"
        assert r.xstrikes.shape == (S, T, n+2), f"Expected xstrikes to have shape {(S, T, n+2)}, but found {r.xstrikes.shape}"
        assert r.xnms.shape == (1, 1, n+2) or r.xnms.shape == (S, T, n+2), f"Expected xnms to have shape (1|{S}, 1|{T}, {n+2}), but found {r.xnms.shape}"
        assert r.xdensities.shape == (S, T, n+2), f"Expected xdensities to have shape {(S, T, n+2)}, but found {r.xdensities.shape}"
        assert r.xdensity0.shape == (S, n+2), f"Expected xdensity0 to have shape {(S, n+2)}, but found {r.xdensity0.shape}"
        assert r.expiries.shape == (S, T), f"Expected expiries to have shape {(S, T)}, but found {r.expiries.shape}"  

        for k,v in r.items():
            self.verify( lambda: torch.all( torch.isfinite(v) ), lambda : f"Found non-finite value in {k}" )
            assert v.device == r.xstrikes.device, f"Expected {k} to be on device {r.xstrikes.device}, but found {v.device}"

        return dict(r)
        
    @version("0.1")
    def create_paths(self,  spots      : torch.Tensor,
                            expiries   : torch.Tensor, 
                            dt         : float,
                            state      : torch.Tensor, *,
                            block_size : int|None = 10000,
                            with_vols  : bool = False,
                            verbose    : Context = Context.all ) -> dict[str, torch.Tensor]:
        """
        Create a universe of calls and vols from a 'state'.
        """
        assert len(spots.shape) == 2, f"Expected spots to have shape (S,) but found {spots.shape}"
        assert dt>0., f"Expected dt to be positive, but found {dt}"
        S, TAU          = spots.shape
        T               = len(expiries)
        assert state.shape == (S,TAU,self.num_state), f"Expected state to have shape {(S,TAU,self.num_state)}, but found {state.shape}"
        N               = self.config.decoder_config.num_nms

        if not block_size is None and S > block_size:
            num_blocks = math.ceil(S / block_size)
            with verbose.write_t(lambda : f"\rCreating paths of {fmt_digits(S)} samples over {TAU} time steps in {num_blocks} blocks of size {block_size} (total {S})...", end='') as tme:
                results = None
                for i in range(num_blocks):
                    verbose.write_t(lambda : f"\rCreating paths of {fmt_digits(S)} samples over {TAU} time steps, block {i+1}/{num_blocks}; so far this took {tme}... ", end='')
                    start = i * block_size
                    end = min((i + 1) * block_size, S)
                    r = self.create_universe(spots[start:end,...], 
                                             expiries, 
                                             dt=dt, 
                                             state=state[start:end,...], 
                                             block_size=None, 
                                             with_vols=with_vols,
                                             verbose=Context.quiet )

                    if results is None:
                        results = { k: [v] for k,v in r.items() }
                    else:
                        for k,v in r.items():
                            results[k].append(v)
                # concatenate results
                results = { k: torch.cat(v, dim=0) for k,v in results.items() }
                verbose.write_t(lambda : f"\rCreated paths of {S} samples over {TAU} time steps in {num_blocks} blocks of size {block_size}: this took {tme}")
                return results

        tme = verbose.timer()

        try:
            pricer          = self.decoder( state=torch.reshape(state,(-1,self.num_state)), 
                                            expiries=expiries[None,:], training=False )

            # calls in t
            strikes_t       = pricer.strikes  
            assert strikes_t.shape == (S*TAU, T, N), f"Expected strikes to have shape {(S*TAU, T, N)}, but found {strikes_t.shape}" 
            strikes_t       = torch.reshape( strikes_t, (S, TAU, T, N) )
            calls_t         = pricer( pricer.strikes, training=False )
            calls_t         = torch.reshape( calls_t, (S, TAU, T, N) )
            calls_t         = calls_t[:,:-1,:,:] # remove last time step
            strikes_t       = strikes_t[:,:-1,:,:] # remove last time step
            cash_calls_t    = calls_t * spots[:,:-1,None,None]
            assert cash_calls_t.shape == (S, TAU-1, T, N), f"Expected cash_calls_t to have shape {(S, TAU-1, T, N)}, but found {cash_calls_t.shape}"

            # calls in t+1
            strikes_tp1     = strikes_t * spots[:,:-1,None,None]  / spots[:,1:,None,None] # strikes at t+1

            strikes_tp1     = torch.cat( [ strikes_t[:,:1,:,:], strikes_tp1 ], dim=1 ) # add last strike to make it the same shape as strikes_t
            assert strikes_tp1.shape == (S, TAU, T, N), f"Expected strikes_tp1 to have shape {(S, TAU, T, N)}, but found {strikes_tp1.shape}"

            calls_tp1       = pricer( torch.reshape(strikes_tp1, (-1, T, N)),
                                      dt=dt,
                                      training=False )
            pricer = None
            assert calls_tp1.shape == (S*TAU, T, N), f"Expected calls_tp1 to have shape {(S*TAU, T, N)}, but found {calls_tp1.shape}"
            calls_tp1       = torch.reshape( calls_tp1, (S, TAU, T, N) )
            calls_tp1       = calls_tp1[:,1:,:,:] # remove first time step
            strikes_tp1     = strikes_tp1[:,1:,:,:] # remove first time step
            cash_calls_tp1  = calls_tp1 * spots[:,1:,None,None] 

            universe = dict(
                C_t       = cash_calls_t, # (S, TAU-1, T, N)
                C_tp1     = cash_calls_tp1, # (S , TAU-1, T, N)
                strikes_t = strikes_t,  # (S, TAU-1, T, N)
                strikes_tp1 = strikes_tp1, # (S, TAU-1, T, N)
                spots     = spots,     # (S, TAU)
                vols_t    = None,      # None or (S, TAU-1, T, N)
                vols_tp1  = None,      # None or (S, TAU-1, T, N)
                delta_t   = None,      # None or (S, TAU-1,  T, N)
                vega_t    = None,      # None or (S, TAU-1,  T, N)
                gamma_t   = None,      # None or (S, TAU-1,  T, N)
                theta_t   = None,      # None or (S, TAU-1,  T, N)
            )

            if with_vols:
                warnings.warn("with_vols: transferring data to CPU")
                np_spots       = spots.detach().cpu().numpy()
                np_expires     = expiries.detach().cpu().numpy()
                np_calls_t     = calls_t.detach().cpu().numpy()
                np_strikes_t   = strikes_t.detach().cpu().numpy()
                np_calls_tp1   = calls_tp1.detach().cpu().numpy()
                np_strikes_tp1 = strikes_tp1.detach().cpu().numpy()
                np_sqrtT       = np.sqrt( np_expires )

                assert np_calls_t.shape == (S, TAU-1, T, N), f"Expected np_calls_t to have shape {(S, TAU-1, T, N)}, but found {np_calls_t.shape}"
                assert np_calls_tp1.shape == (S, TAU-1, T, N), f"Expected np_calls_tp1 to have shape {(S, TAU-1, T, N)}, but found {np_calls_tp1.shape}"
                np_vols_t      = bs.implied( k=np_strikes_t, prices=np_calls_t, sqrtT=np_sqrtT[None,None,:,None], eps=1E-6 )
                np_vols_tp1    = bs.implied( k=np_strikes_tp1, prices=np_calls_tp1, sqrtT=np_sqrtT[None,None,:,None], eps=1E-6 )
                assert np_vols_t.shape == np_calls_t.shape, f"Expected np_vols_t to have shape {np_calls_t.shape}, but found {np_vols_t.shape}"
                assert np_vols_tp1.shape == np_calls_tp1.shape, f"Expected np_vols_tp1 to have shape {np_calls_tp1.shape}, but found {np_vols_tp1.shape}"

                delta_t,\
                vega_t,\
                gamma_t,\
                theta_t        = bs( k=np_strikes_t, vol=np_vols_t, sqrtT=np_sqrtT[None,None,:,None],
                                    what=bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA, on_precision_error="clip" )['delta', 'vega', 'gamma', 'theta']
                vega_t         = np_spots[:,:-1,None,None] * vega_t
                theta_t        = np_spots[:,:-1,None,None] * theta_t
                gamma_t        = gamma_t / np_spots[:,:-1,None,None]

                vols_t         = torch.tensor( np_vols_t, device=spots.device, dtype=spots.dtype )
                vols_tp1       = torch.tensor( np_vols_tp1, device=spots.device, dtype=spots.dtype )
                delta_t        = torch.tensor( delta_t, device=spots.device, dtype=spots.dtype )
                vega_t         = torch.tensor( vega_t, device=spots.device, dtype=spots.dtype )
                gamma_t        = torch.tensor( gamma_t, device=spots.device, dtype=spots.dtype )
                theta_t        = torch.tensor( theta_t, device=spots.device, dtype=spots.dtype )

                universe['vols_t']   = vols_t
                universe['vols_tp1'] = vols_tp1
                universe['delta_t']  = delta_t
                universe['vega_t']   = vega_t
                universe['gamma_t']  = gamma_t
                universe['theta_t']  = theta_t

        except MemoryError as e:
            raise MemoryError(f"MemoryError in create_paths: {e}. Consider reducing 'block_size', currently {block_size}. Error message: {e}") from e

        verbose.write(lambda : f"Created paths of {S} samples over {TAU} time steps: this took {tme}")
        return universe        
