import time

from cdxcore.uniquehash import unique_label48_8
from cdxcore.util import fmt_big_byte_number
from quantdata.torch import BaseModel, BaseEnvironment, get
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.dysanos.grid.torch_train import TrainConfig, MetricTracker
from quantdata.pool import PoolConfig
from dataclasses import dataclass, asdict
from cdxcore.version import version
from cdxcore.subdir import SubDir
from cdxcore.err import verify_inp, verify
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
import os
import numpy as np
import torch
from collections.abc import Callable
import warnings as warnings

_cache_root = SubDir("~/.dysanos_dh/")

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

def torch_utility(utility="entropy", lmbda=1.0) -> tuple[Callable, Callable]:
    if utility == "entropy":
        u     = lambda x : -torch.expm1( -lmbda * x ) / lmbda
        deriv = lambda x : torch.exp(-lmbda * x)
        label = f"entropy@{lmbda:.2g}"
    elif utility == "cvar":
        u     = lambda x : (1.+lmbda) * torch.minimum( x, torch.tensor(0.) )
        deriv = lambda x : torch.where( x<0., 1.+lmbda, 0. )
        label = f"cvar@{lmbda:.2g}"
    elif utility == "powexp":
        """
        u(x) = 1-exp(-lambda * x) / lambda for x > 0
        u(x) = -(-x)^(1+lambda)/(1+lambda) + x for x < 0

        u'(x) = exp(-lambda * x) for x > 0 with u'(0) = 1
        u'(x) = (-x)^lambda + 1 for x < 0 with u'(0) = 1

        u''(x) = -lambda * exp(-lambda * x) for x > 0 with u''(0) = -lambda
        u''(x) = -lambda * (-x)^(lambda-1) for x < 0 with u''(0) = -lambda        
        """

        def u(x):
            x_pos = torch.clamp(x, min=0. )
            pow   = 1.+lmbda
            return torch.where( x>0., -torch.expm1( -lmbda * x_pos ) / lmbda, -(torch.abs(x)**pow)/pow+x )
        def deriv(x):
            x_pos = torch.clamp(x, min=0. )
            return torch.where( x>0., torch.exp(-lmbda * x_pos), 1.+torch.abs(x)**lmbda )
        label = f"powexp@{lmbda:.2g}"
    else:
        raise ValueError(f"Unknown utility: {utility}")

    return u, deriv, label


@version("0.1")
@dataclass(frozen=True)
class DeepHedgingConfig:
    label : str = ""
    group : str = "deephedging"
    user_version : str = "0.1"

    utility : str = "entropy"
    lmnda : float = 1.0
    delta_hedged : bool = True
    action_config : FeedForwardConfig = FeedForwardConfig()

    def __post_init__(self):
        verify_inp( self.utility in ("entropy", "cvar", "powexp"), f"Expected utility to be one of 'entropy', 'cvar', or 'powexp' but got '{self.utility}'." )  
        verify_inp( self.lmnda > 0., f"Expected lmnda to be positive but got {self.lmnda}." )   

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


@version("0.1", dependencies=[DeepHedgingConfig])
class DeepHedging(BaseModel):

    NUM_FEATURES = 6

    def __init__(self, env : BaseEnvironment, *,
                       num_xstate  : int, 
                       num_expiries: int,
                       num_strikes : int,
                       ref_dQdP    : np.ndarray|None = None,    
                       config : DeepHedgingConfig = DeepHedgingConfig() ):
        super().__init__(env)
        self.config       = config
        self.num_xstate   = num_xstate
        self.num_features = num_xstate + num_expiries
        num_instruments   = num_strikes*num_expiries + 1
        self.a            = FeedForward( env, num_input = self.num_features, num_output = num_instruments, config = config.action_config )
        self.y            = torch.nn.Parameter(torch.zeros( (), dtype=env.fdtype, device=env.device ), requires_grad=True)
        self.u,\
        self.u_deriv,\
        self.u_label     = torch_utility(config.utility, config.lmnda)

    @property
    def label(self) -> str:
        return self.config.label
    
    @property
    def qualified_label(self) -> str:
        return self.config.qualified_label

    def __call__(self, data, *, training : bool = False) -> torch.Tensor|dict[str, torch.Tensor]:
        """ Forward pass of the model. """

        dC           = get(data, "dC", 4)
        S, TAU, T, N = dC.shape
        spots        = get(data, "spots", (S, TAU+1))
        delta_t      = get(data, "delta", (S, TAU, T, N))
        dS           = spots[:,1:] - spots[:,:-1]
        dC_h         = ( dC - delta_t * dS[:,:,None,None] ) if self.config.delta_hedged else dC
        expiries     = get(data, "expiries", (1, 1, T))
        ref_dQdP     = get(data, "ref_dQdP", (S,), optional=True)

        # we predict this
        dC_h         = dC_h.reshape(S, TAU, T*N)
        dX           = torch.concatenate( [ dS[:,:,None] + dC_h[:,:,:1]*0, dC_h ], axis=-1 )
        assert dX.shape == (S, TAU, T*N+1), f"Expected dX to have shape {(S, TAU, T*N+1)} but got {dX.shape}"

        # features ... are mostly just the states
        XS           = self.num_xstate
        xstate       = get(data, "xstate", (S, TAU, XS))

        features     = torch.concatenate( [xstate, expiries+0*xstate[:,:,:1]], axis=-1 )
        assert features.shape == (S, TAU, self.num_features), f"Expected features to have shape {(S, TAU, self.num_features)} but got {features.shape}"
        a            = self.a(features)
        assert a.shape == (S, TAU, T*N+1), f"Expected action to have shape {(S, TAU, T*N+1)} but got {a.shape}"

        a_dX         = torch.sum( a * dX, axis=(1,2) )
        assert a_dX.shape == (S,), f"Expected a_dX to have shape {(S,)} but got {a_dX.shape}"
        utility      = self.u( a_dX + self.y ) - self.y   # we wish to maximize this
        loss         = torch.mean( -ref_dQdP * utility ) if ref_dQdP is not None else torch.mean( -utility )

        if training:
            return loss

        a_S          = a[...,0]
        a_C          = a[...,1:].reshape(S, TAU, T, N)
        dQdP         = self.u_deriv( a_dX + self.y )
        dQdP         = dQdP * ref_dQdP if not ref_dQdP is None else dQdP
        assert dQdP.shape == (S,), f"Expected dQdP to have shape {(S,)} but got {dQdP.shape}"
        return dict( loss=loss, utility=utility.detach(), dQdP=dQdP.detach(), a_S=a_S.detach(), a_C=a_C.detach(), y=self.y.detach() )

"""
Static Model Visualization
"""

import numpy as np
from cdxcore.dynaplot import FigStore, colors_tableau, color_tableau, figure, m_o_m, quantile_m_o_m
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir
from dataclasses import dataclass

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_model import SanosModelConfig

@dataclass(frozen=True)
class VizConfig:
    num_viz_dates : int|list[int] = 4       # how many implied vols to render
    seed          : int = 1231231 # seed for chosing the above
    use_nms       : bool = False   # whether use NMS coordinates or K

    def __post_init__(self):
        if isinstance(self.num_viz_dates, int):
            assert self.num_viz_dates > 0, f"Expected 'num_viz_dates' {self.num_viz_dates} to be greater than 0"
        elif isinstance(self.num_viz_dates, list):
            assert all(n >= 0 for n in self.num_viz_dates), f"Expected all 'num_viz_dates' {self.num_viz_dates} to be greater than 0"
        else:
            raise ValueError(f"Expected 'num_viz_dates' to be int or list of int, but found {type(self.num_viz_dates)}")
        assert self.seed >= 0, f"Expected 'seed' {self.seed} to be greater than or equal to 0"

@dataclass(frozen=True)
class VizConfig:
    pass

class Visualizer(object):

    def __init__(self, universe, 
                       total_epochs : int,*,
                       ref_dQdP   : np.ndarray|None = None, 
                       viz_config : VizConfig, 
                       mdl_config : DeepHedgingConfig ):
        self.universe      = universe
        self.total_epochs  = total_epochs
        self.mdl_config    = mdl_config
        self.viz_config    = viz_config
        self.store         = FigStore()
        self.first         = True
        self.dS            = self.universe["spots"][:,1:] - self.universe["spots"][:,:-1]
        self.dC            = self.universe["C_tp1"] - self.universe["C_t"]
        self.days          = np.arange( self.dS.shape[1] )
        self.xdensity      = np.arange( self.dS.shape[0])
        self.ref_dQdP      = ref_dQdP

        # learning & fit
        # --------------

        dev_dS,\
        rdev_dC        = self.compute_devs( None )
        ref_dev_dS,\
        ref_rdev_dC    = self.compute_devs( ref_dQdP ) if ref_dQdP is not None else (None, None)

        self.fig_lrn   = figure(f"Loss", col_size=8, row_size=6, columns=None) #
        self.ax_lrn    = self.fig_lrn.add_subplot("Loss")
        self.ax_lrn.set_xlabel("Epoch")
        self.ax_lrn.set_ylabel("Loss (-Utility)")

        self.ax_dS = self.fig_lrn.add_subplot(r"$E[dS]/\epsilon[dS]$")
        self.ax_dS.plot( self.days, dev_dS, "--", color="blue", label="Market")
        self.ax_dS.plot( self.days, ref_dev_dS, "*:", color="orange", label="Prior") if not ref_dev_dS is None else None
        self.ax_dS.set_xlabel("Days")
        self.ax_dS.set_ylabel("Drift as fraction of MC error")
        self.ref_dS_min_max = np.quantile( ref_dev_dS, q=[0.01, 0.91] ) if not ref_dev_dS is None else None

        self.ax_dC = self.fig_lrn.add_subplot(r"$E[dC]/\epsilon[dC]$")
        self.ax_dC.plot( self.days, rdev_dC.mean, "--", color="blue", label="Market")
        self.ax_dC.fill_between( self.days, rdev_dC.q05, rdev_dC.q95, color="blue", alpha=0.1 )
        self.ax_dC.plot( self.days, ref_rdev_dC.mean, "*:", color="orange", label="Prior") if not ref_rdev_dC is None else None
        self.ax_dC.fill_between( self.days, ref_rdev_dC.q05, ref_rdev_dC.q95, color="orange", alpha=0.1 ) if not ref_rdev_dC is None else None
        self.ax_dC.set_xlabel("Days")
        self.ax_dC.set_ylabel("Drift as fraction of MC error")
        self.ref_dC_min_max = ( np.min(ref_rdev_dC.q05), np.max(ref_rdev_dC.q95) ) if not ref_rdev_dC is None else None

        self.ax_dQdP = self.fig_lrn.add_subplot(r"$dQ/dP$")
        self.ax_dQdP.plot( self.xdensity, self.xdensity*0., "--", color="blue")
        self.ax_dQdP.set_xlabel("Samples, sorted")
        self.ax_dQdP.set_ylabel("Density ratio")

        self.fig_lrn.render()

    def compute_devs( self, dQdS : np.ndarray|None ):
        dQdS = dQdS if dQdS is not None else np.ones( (self.dS.shape[0], ), dtype=self.dS.dtype )
        drift_dS = np.mean( dQdS[:,None]*self.dS, axis=0 )
        drift_dC = np.mean( dQdS[:,None,None,None]*self.dC, axis=0 )
        std_dS   = np.sqrt( np.mean( dQdS[:,None]*(self.dS-drift_dS[None,...])**2, axis=0 ) ) / np.sqrt(self.dS.shape[0])
        std_dC   = np.sqrt( np.mean( dQdS[:,None,None,None]*(self.dC-drift_dC[None,...])**2, axis=0 ) ) / np.sqrt(self.dC.shape[0])
        dev_dS   = np.where( std_dS > 0., drift_dS / np.where( std_dS > 0., std_dS, 1.), 0. )
        dev_dC   = np.where( std_dC > 0., drift_dC / np.where( std_dC > 0., std_dC, 1.), 0. )
        del drift_dS, drift_dC, std_dS, std_dC
        dev_dC   = dev_dC.reshape( dev_dC.shape[0], -1 )
        r = PrettyObject()
        r.mean  = np.mean( dev_dC, axis=1 )    
        r.q05,\
        r.median,\
        r.q95 = np.quantile( dev_dC, q=[0.05, 0.5, 0.95], axis=1 )
        return dev_dS, r

    def __call__(self, tracker, dQdP, *, final : bool = False ):
        """
        Update
        """

        epoch     =   tracker.trn_epochs[-1]
        self.store.clear()

        assert dQdP is None or dQdP.shape[0] == self.dS.shape[0], f"Expected dQdP to have shape {(self.dS.shape[0],)} but got {dQdP.shape}"

        # losses
        # ------

        def losses():
            trn_epochs =  np.asarray(tracker.trn_epochs)
            trn_losses =  np.asarray(tracker.trn_losses) 
            val_epochs =  np.asarray(tracker.val_epochs)
            val_losses =  np.asarray(tracker.val_losses) 
            self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g} [{trn_epochs[-1]}], validation loss {val_losses[-1]:.4g} [{val_epochs[-1]}]. Total epochs {self.total_epochs}." + ("\n(Done)" if final else ""))

            self.store += self.ax_lrn.plot(trn_epochs, trn_losses, color="blue", label="Batch Loss")
            self.store += self.ax_lrn.plot(val_epochs, val_losses, "*:", color="orange", label="Full Loss")
            max_t = np.max(trn_losses) if len(trn_losses)<10 else np.quantile(trn_losses, 0.9)
            max_v = np.max(val_losses) if len(val_losses)<10 else np.quantile(val_losses, 0.9)
            max_  = max( max_t, max_v )
            self.ax_lrn.set_ylim( min( np.min(trn_losses), np.min(val_losses), )*1.1, max_ * ( 1.1 if max_ > 0. else 0.9 ) )
            self.ax_lrn.legend(loc="lower left")
        losses()

        # drift dS
        # --------

        def drift():
            dev_dS, rdev_dC = self.compute_devs( dQdP )

            self.store += self.ax_dS.plot( self.days, dev_dS, "-", color="red", label=f"Epoch {epoch}")
            self.store += self.ax_dC.plot( self.days, rdev_dC.mean, "-", color="red", label=f"Epoch {epoch}")
            self.store += self.ax_dC.fill_between( self.days, rdev_dC.q05, rdev_dC.q95, color="red", alpha=0.1 )
            self.ax_dS.legend(loc="upper left")
            self.ax_dC.legend(loc="upper left")
            min_dS = np.min(dev_dS) if self.ref_dS_min_max is None else min( np.min(dev_dS), self.ref_dS_min_max[0] )
            max_dS = np.max(dev_dS) if self.ref_dS_min_max is None else max( np.max(dev_dS), self.ref_dS_min_max[1] )
            ex_dS  = max( abs(min_dS), abs(max_dS) )

            min_dC = np.min(rdev_dC.q05)# if self.ref_dC_min_max is None else min( np.min(rdev_dC.q05), self.ref_dC_min_max[0] )
            max_dC = np.max(rdev_dC.q95) # if self.ref_dC_min_max is None else max( np.max(rdev_dC.q95), self.ref_dC_min_max[1] )
            ex_dC  = max( abs(min_dC), abs(max_dC) )
            self.ax_dS.set_ylim( -ex_dS*1.5, ex_dS*1.5 )
            self.ax_dC.set_ylim( -ex_dC*1.5, ex_dC*1.5 )
        drift()

        # dQ/dP
        # -----

        def density():
            dQdP_sorted = np.sort(dQdP)
            self.store += self.ax_dQdP.plot( self.xdensity, dQdP_sorted, "-", color="red", label=f"Epoch {epoch}")
            if self.first:
                self.ax_dQdP.legend(loc="upper left")
            self.ax_dQdP.set_ylim( dQdP_sorted[0]*1.1, dQdP_sorted[-1]*1.1 )
        density()

        self.fig_lrn.render() if not final else self.fig_lrn.close()
        self.first = False

def train( universe,
           sampler, *,
           ref_dQdP      : np.ndarray|None = None,
           mdl_config    : DeepHedgingConfig = DeepHedgingConfig(),   
           trn_config    : TrainConfig = TrainConfig(),
           viz_config    : VizConfig = VizConfig(),
           restart       : bool = False,
           just_load     : bool = False,
           env           : BaseEnvironment = BaseEnvironment(),
           verbose       : Context = Context.all ) -> dict:
    """
    Train model
    """
    verify_inp( not (restart and just_load), f"Expected at most one of 'restart' and 'just_load' to be True, but found both True" )

    verbose = verbose if not verbose is None else Context.all
    verbose = verbose if not just_load else Context.quiet

    verbose.write(lambda : f"Training {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' for {trn_config.num_epochs} epochs.")
    verbose.write(lambda : f"Using {'all samples' if trn_config.batch_size is None else f'batches of {trn_config.batch_size} samples'} with learning rate {trn_config.lr:.4g} and weight decay {trn_config.weight_decay:.4g}.")

    # Copy to GPU
    # -----------

    with verbose.write_t(lambda : f"\rCoping data to {env}... ", end='') as tme:
        ndata = dict(
            dC = universe["C_tp1"]-universe["C_t"],
            spots = universe["spots"],
            delta = universe["delta_t"],
            expiries = universe["expiries"][None,None,:],
            xstate = universe['xstate'][:,:-1,...],
        )

        num_samples,\
        num_steps,\
        num_expiries,\
        num_strikes = ndata["dC"].shape
        assert ndata["spots"].shape == (num_samples, num_steps+1), f"Expected spots to have shape {(num_samples, num_steps+1)} but got {ndata['spots'].shape}"
        assert ndata["delta"].shape == (num_samples, num_steps, num_expiries, num_strikes), f"Expected delta to have shape {(num_samples, num_steps, num_expiries, num_strikes)} but got {ndata['delta'].shape}"  
        assert ndata["expiries"].shape == (1, 1, num_expiries), f"Expected expiries to have shape {(1, 1, num_expiries)} but got {ndata['expiries'].shape}"
        assert ndata["xstate"].shape == (num_samples, num_steps, sampler.num_xstate), f"Expected xstate to have shape {(num_samples, num_steps, sampler.num_xstate)} but got {ndata['xstate'].shape}"

        size  = 0
        tdata = {}
        for k, v in ndata.items():
            assert np.isfinite(v).all(), f"Expected all values in {k} to be finite, but found non-finite values."
            size += v.nbytes
            verbose.write(lambda : f"\rCopying {k} of size {fmt_big_byte_number(v.nbytes)} to {env} which brings total GPU memory use to {fmt_big_byte_number(size)} used)... ", end='')
            tdata[k] = torch.tensor(v, dtype=env.fdtype, device=env.device)

        if not ref_dQdP is None: verify_inp(ref_dQdP.shape == (num_samples,), f"Expected ref_dQdP to have shape {(num_samples,)} but got {ref_dQdP.shape}" )

        verbose.write(lambda : f"\rCopied {len(ndata)} arrays with a total size {fmt_big_byte_number(size)} to {env}. This took {tme}.")

    # create model
    # ------------

    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        model     = DeepHedging(  env,
                       num_xstate   = sampler.num_xstate,
                       num_expiries = num_expiries,
                       num_strikes  = num_strikes,
                       ref_dQdP     = ref_dQdP,
                       config       = mdl_config )
        verbose.write(lambda : f"done; this took {tme}.", head=False )
        if model.debug:
            warnings.warn("** WARNING: model is running in (very slow) debug mode. **")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trn_config.lr,
        weight_decay=trn_config.weight_decay
    )

    # load
    # ----

    path          = _cache_root(ext="pt") if mdl_config.group == "" else _cache_root(mdl_config.group,ext="pt")
    file_name     = unique_label48_8(mdl_config.label, as_file_name=True)
    cache_ffn     = path.full_file_name(file_name)
    start_epoch   = 0
    tracker       = MetricTracker()
    model_version = model.version.unique_id48

    if not restart and path.exists(file_name):
        verbose.write(lambda : f"Found existing recovery checkpoint '{cache_ffn}'. Restoring model state from this checkpoint.")

        state = torch.load(cache_ffn, map_location=env.device, weights_only=False)
        user_version = state.pop("version", None)
        mdl_version  = state.pop("mdl_version", None)
        err          = None
        if user_version is None:
            err = f"Checkpoint '{cache_ffn}' does not contain a 'version'. This is likely an invalid checkpoint file."
        elif user_version != mdl_config.user_version:
            err = f"Checkpoint '{cache_ffn}' file user_version {user_version} does not match model user_version {mdl_config.user_version}."
        elif not mdl_version is None and model_version != mdl_version:
            err = f"Checkpoint '{cache_ffn}' file model version {mdl_version} does not match model version {model_version}."
        else:
            model.load_state_dict(state.pop("model_state_dict"))
            if just_load:
                return model

            optimizer.load_state_dict(state.pop("optimizer_state_dict"))
            tracker = MetricTracker(**state.pop("tracker"))
            start_epoch = tracker.trn_epochs[-1] + 1

        if not err is None:
            if just_load:
                raise RuntimeError(err)
            warnings.warn(err + f" -- Restarting training.")

    else:
        verbose.write(lambda : f"Using recovery checkpoint '{cache_ffn}'.")

    # preprocess & copy data into tensors
    # -----------------------------------

    with verbose.write_t(lambda : f"\rComputing loss... ", end='') as tme:        
        ld0 = model( tdata, training=False )
        verbose.write(lambda : f"\n{'Initial' if start_epoch == 0 else 'Current'} loss is {ld0['loss']:.4g}. This took {tme}.", head=False )

    def sample():
        if trn_config.batch_size is None or trn_config.batch_size >= num_samples:
            return tdata, None
        else:
            idx = torch.randint(0, num_samples, (trn_config.batch_size,))
            return { k: v[idx,...] if v.shape[0] > 1 else v for k, v in tdata.items() }

    # compile
    # -------

    def gradient_step( batch ):
        loss = model( batch, training=True ) 
        model.verify( lambda : torch.isfinite(loss).all(), "Internal error: loss infinite" )
        return loss

    def valuation( all_data ):
        with torch.no_grad():
            return model( all_data, training=False )

    if start_epoch < trn_config.num_epochs and trn_config.compile:
        with verbose.write_t(lambda : f"Compiling {mdl_config.qualified_label} for {env.device}: training... ", end='') as tme:
            gradient_step = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( gradient_step )
            _ = gradient_step(sample())
            del _

            verbose.write("valuation... ", head=False, end='')

            valuation = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( valuation )
            _ = valuation(tdata)
            del _
            verbose.write(lambda : f"done; this took {tme}.", head=False ) 

    # visualization & seralization
    # ----------------------------

    disp = Visualizer( universe, ref_dQdP=ref_dQdP, total_epochs=trn_config.num_epochs, viz_config=viz_config, mdl_config=mdl_config )
    del ld0

    def viz_and_store(*, epoch: int, final : bool = False):
        """ call valuation and then visualize and store the model """
        model.eval()
        ld = valuation( tdata )
        assert len(tracker.val_epochs) == 0 or tracker.val_epochs[-1] <= epoch+1, f"Expected last validation epoch {tracker.val_epochs[-1]} to be less than or equal to current epoch {epoch+1}"
        if len(tracker.val_epochs)==0 or tracker.val_epochs[-1] < epoch+1:
            tracker.val_losses.append(ld['loss'].cpu().item())
            tracker.val_epochs.append(epoch+1)
        else:
            tracker.val_losses[-1] = ld['loss'].cpu().item()
            tracker.val_epochs[-1] = epoch+1

        tmp_name = path.full_temp_file_name(file_name, ext="tmp", create_directory=True)
        torch.save(dict(version=mdl_config.user_version,
                        mdl_version=model_version,
                        tracker=asdict(tracker),
                        optimizer_state_dict=optimizer.state_dict(),
                        model_state_dict=model.state_dict(),
                        mdl_config=asdict(mdl_config),
                        mdl_config_class=mdl_config.__class__.__name__,
                        trn_config=asdict(trn_config),
                        viz_config=asdict(viz_config),
                        ), tmp_name)
        os.rename(tmp_name, cache_ffn)
        model_output = dict( 
            dQdP = ld['dQdP'].cpu().numpy(),
            a_S  = ld['a_S'].cpu().numpy(),
            a_C  = ld['a_C'].cpu().numpy(),
            y = ld['y'].cpu().numpy()
        )
        disp(tracker=tracker, dQdP=model_output['dQdP'], final=final)

        return model_output if final else None

    # train
    # -----

    epoch = start_epoch
    if start_epoch < trn_config.num_epochs:
        with verbose.write_t(lambda : f"Training {mdl_config.qualified_label} with {trn_config} on {env}:") as tme:
            last_update = time.time()
            try:
                for epoch in range(start_epoch, trn_config.num_epochs):

                    # -------------------------
                    # Training
                    # -------------------------
                    model.train()

                    train_loss = 0.0

                    optimizer.zero_grad(set_to_none=True)
                    batch = sample()
                    loss  = gradient_step(batch)

                    loss.backward()

                    # Optional but often useful
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    lr = trn_config.set_lr_for_epoch(epoch, optimizer)
                    optimizer.step()
                    train_loss = loss.detach().item() 
                    tracker.trn_losses.append(train_loss)
                    tracker.trn_epochs.append(epoch+1)

                    if epoch in [0,trn_config.num_epochs-1] or (tme.interval_test(1.) and ((epoch+1) % 10 == 0)):
                        verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. So far this took {tme}.", end='')
                        
                    if epoch in [0,trn_config.num_epochs-1] or ((epoch+1) % trn_config.epoch_interval == 0) or ((time.time() - last_update) > trn_config.time_interval):
                        viz_and_store(epoch=epoch)
                        last_update = time.time()
                        verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. Valuation loss: {tracker.val_losses[-1]:.4g}. So far this took {tme}. [Check point saved]", end='')

            except KeyboardInterrupt as e:
                verbose.write(lambda : f"\r*** Training interrupted at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Saving checkpoint and exiting.")
                viz_and_store(epoch=epoch,final=True)
                raise e
        verbose.write(lambda : f"\rModel trained for {trn_config.num_epochs} epochs. This took {tme}.")

    # finalize & return
    # -----------------

    model.eval()
    model_output = viz_and_store(epoch=epoch,final=True)
    mo_path      = path(ext="mo", fmt=SubDir.PYTREE_HDF5)
    verbose1     = verbose(1)
    verbose1.write(lambda : f"\rModel checkpoint file is'{cache_ffn}'.")
    with verbose1.write_t(lambda : f"\rWriting model output to '{mo_path.full_file_name(file_name)}'... ", end='') as tme: 
        mo_path.write(file_name, model_output, raise_on_error=True)
        verbose1.write(lambda : f"done; this took {tme}.", head=False )
    verbose.write(lambda : f"\rModel {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' trained for {trn_config.num_epochs} epochs. Final training loss: {tracker.trn_losses[-1]:.4g}, valuation loss: {tracker.val_losses[-1]:.4g}. This took {tme}.")
    return model, tdata
