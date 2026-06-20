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
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import Backbone, BackboneNumpy
import logging

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, get, pop
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.dysanos.grid.torch_market import GridMarketDefinition, PoolConfig
from dataclasses import asdict, dataclass, field

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

@version("0.1", dependencies=[GridMarketDefinition, BaseEnvironment])
@dataclass(frozen=True)
class SanosModelConfig:
    label : str = ""                  # Unique label for this experiment. If left to "" will be set to the current notebook value.
    group : str = ""                  # Group for this experiment, if any.
    user_version : str = "0.1"        # Version to trigger updates
    smoothness : float|None = None    # Model smoothness
    min_vega_sqrt : float = 1E-4      # minimum vega/sqrtT; used to compute weights
    scale_by_weights : str = "sample" # Whether to scale the loss by weights

    def __post_init__(self):
        verify_inp( self.smoothness is None or self.smoothness >= 0., f"Expected 'smoothness' to be None or non-negative, but found {self.smoothness}" )
        verify_inp( self.min_vega_sqrt > 0., f"Expected 'min_vega_sqrt' to be positive, but found {self.min_vega_sqrt}" )
        verify_inp( self.scale_by_weights in ["none", "sample", "total"], f"Expected 'scale_by_weights' to be either 'none', 'sample' or 'total', but found {self.scale_by_weights}" )

        if self.label == "":
            label = _notebook_name()
            verify_inp( label is not None, f"Expected to be able to determine notebook name for 'label', but failed. Please set 'label' explicitly in the config." )
            self.__dict__['label'] = label

    @property
    def label_str(self) -> str:
        return f"{self.label}{f'.{self.group}' if self.group else ''} {self.user_version} " + ( f"{self.smoothness:.2f}-smooth" if not self.smoothness is None else "linear" )

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
        variances: np.ndarray    = market['atm_vols']**2 * expiries
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
            vol     = np.sqrt( self.smoothness * variances/expiries )
            assert np.all( np.isfinite(vol) ), f"Internal error: {message()}infinite vol"
            return bs.price( k=strikes/spots, vol=vol, sqrtT=np.sqrt(expiries), eps=1E-6 )*spots

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

        for k, v in r.items():
            if not v is None:
                assert torch.all( torch.isfinite(v) ), f"{message()}Found non-finite value in {k}"
                assert v.device == r.nms.device, f"{message()}Expected {k} to be on device {r.nms.device}, but found {v.device}"

        return dict(r)

class SanosModel(BaseModel):

    def __init__( self, env : BaseEnvironment ):
        super().__init__(env)

        torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(
            recompiles=True,
            graph_breaks=True,
            dynamo=logging.WARNING,
        )

    def compute_dlvsqs( self, data : Pytree, *, sample_indices : np.array|None = None, training : bool = False ) -> tuple[torch.Tensor, dict|None]:
        raise NotImplementedError("This method should be implemented in subclasses to compute the DLVsq grid.")

    def compute_calls( self, data : Pytree, dlvsqs : torch.Tensor, *, sample_indices : np.array|None = None, training : bool = False ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Utility function to compute call prices and densities.
        """
        S, T, N      = dlvsqs.shape
        q_tm1        = get(data, "xdensity0", (S, N+2))
        iw_minus     = get(data, "iw_minus", (S, T, N) )
        iw_plus      = get(data, "iw_plus", (S, T, N) )
        L            = get(data, "L", (S, T-1, N+2, N+2) )
        CX           = get(data, "CX", (S, T, N, N+2) )
        Cs           = []
        qs           = [] 

        for t in range(T): # loop by expiry
            iw_minus_t  = iw_minus[:,t,:]
            iw_plus_t   = iw_plus[:,t,:]
            CX_t        = CX[:,t,:,:]

            # move density forward
            if t>0:
                L_t     = L[:,t-1,:,:] 
                q_tm1   = torch.sum( L_t * q_tm1[:,None,:], dim=-1 ) # TODO: make faster

            self.verify( lambda : torch.min( q_tm1 ) >= -1E-8, lambda : f"Negative density values found at expiry #{t}, before solving DLV equation: min density value is {np.min(q_tm1)}.")

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

        Cs = torch.stack( Cs, axis=1 ) # (S, T, N)
        qs = torch.stack( qs, axis=1 ) # (S, T, N+2)
        return Cs, qs   

    def __call__(self, data : Pytree, sample_indices : np.array|None = None, training : bool = False ) -> float | Pytree:
        """ Training or valuation """

        # call derived model
        # ------------------

        dlvsqs, other     = self.compute_dlvsqs( data, sample_indices=sample_indices, training=training )
        calls, xdensities = self.compute_calls( data, sample_indices=sample_indices, dlvsqs=dlvsqs, training=training )    

        S, T, N      = dlvsqs.shape
        assert calls.shape == (S, T, N), f"{self.__class__.__name__}: Expected calls to have shape {(S, T, N)}, but found {calls.shape}"
        assert not training or other is None, f"{self.__class__.__name__}: Expected 'other' to be None when training"

        # solve for the densities
        # -----------------------

        mids         = get(data, "mids", (S, T, N) )
        xstrikes     = get(data, "xstrikes", (S, T, N+2) )
        expiries     = get(data, "expiries", (S, T) )
        weights      = get(data, "weights", (S, T, N))

        loss         = 0.
        raw_error    = 0.
        wsum         = 0.

        raw_errors   = [] if not training else None
        errors       = [] if not training else None

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

        xdensities = xdensities.detach()  # (S, T, N+2)
        calls      = calls.detach()       # (S, T, N)
        dlvsqs     = dlvsqs.detach()      # (S, T, N)
        raw_errors = torch.stack( raw_errors, axis=1 ) # (S, T, N)
        errors     = torch.stack( errors, axis=1 ) # (S, T, N) 

        assert xdensities.shape == (S, T, N+2), f"Expected xdensities to have shape {(S, T, N+2)}, but found {xdensities.shape}"
        assert calls.shape == (S, T, N), f"Expected calls to have shape {(S, T, N)}, but found {calls.shape}"
        assert dlvsqs.shape == (S, T, N), f"Expected dlvsqs to have shape {(S, T, N)}, but found {dlvsqs.shape}"
        assert raw_errors.shape == (S, T, N), f"Expected raw_errors to have shape {(S, T, N)}, but found {raw_errors.shape}"
        assert errors.shape == (S, T, N), f"Expected errors to have shape {(S, T, N)}, but found {errors.shape}"

        result = dict(loss=loss, xdensities=xdensities, calls=calls, dlvsqs=dlvsqs, raw_errors=raw_errors, raw_weights=weights, errors=errors, weights=weights)
        if not other is None:
            assert set(other) & set(result) == set(), f"Expected 'other' to not contain any keys that are already in the result, but found {sorted(set(other) & set(result))}"
            result.update(other)
        return result
        





