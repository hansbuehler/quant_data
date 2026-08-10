from ensurepip import bootstrap
import math
import random
from statistics import quantiles
import time

from cdxcore.uniquehash import unique_label48_8
from cdxcore.util import fmt_big_byte_number
from quantdata.torch import BaseModel, BaseEnvironment, get, pop
from quantdata.torch.ffw import FeedForward, FeedForwardConfig
from quantdata.dysanos.grid.torch_train import TrainConfig, MetricTracker
from quantdata.pool import PoolConfig
from dataclasses import dataclass, asdict, field
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
from matplotlib.ticker import MaxNLocator

if os.environ.get("DYSANONS_USE_LAPTOP") is not None:
    _cache_root = SubDir(r"C:\users\public\.dysanos_dh/")
else:
    _cache_root = SubDir("~/.dysanos_dh")


def _rng_state_dict() -> dict:
    """Capture all RNGs used, directly or indirectly, by training."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _load_rng_state_dict(state: dict|None) -> None:
    """Restore an RNG state; checkpoints predating this field pass ``None``."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        for device, device_state in enumerate(state.get("torch_cuda", [])):
            if device >= torch.cuda.device_count():
                break
            torch.cuda.set_rng_state(device_state.cpu(), device=device)
        

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
import math
from collections.abc import Callable

import torch
from torch import Tensor


TensorFunction = Callable[[Tensor, Tensor], Tensor]


def torch_utility(
    utility: str = "entropy",
    lmbda: float = 1.0,
) -> tuple[TensorFunction, TensorFunction, str]:

    if not math.isfinite(lmbda) or lmbda <= 0.0:
        raise ValueError("'lmbda' must be finite and strictly positive")

    if utility == "entropy":

        def u(x: Tensor, y: Tensor) -> Tensor:
            cap = 30.0 if x.dtype == torch.float32 else 300.0 
            z = -lmbda*(x+y)
            z =  z + (torch.clamp(z, max=cap)-z).detach()
            return -torch.expm1(z) / lmbda - y
        def deriv(x: Tensor, y: Tensor) -> Tensor:
            return torch.exp(-lmbda * (x + y))

        label = f"entropy@{lmbda:.2g}"

    elif utility == "cvar":
        scale = 1.0 + lmbda

        def u(x: Tensor, y: Tensor) -> Tensor:
            # Chooses the right derivative/supergradient 0 at x == 0.
            xbar = x+y
            ux = scale * torch.where(
                xbar < 0,
                xbar,
                torch.zeros_like(xbar),
            )
            return ux - y

        def deriv(x: Tensor, y: Tensor) -> Tensor:
            # This is a selected supergradient at x == 0,
            # not a mathematical derivative there.
            xbar = x+y
            return torch.where(
                xbar < 0,
                torch.full_like(xbar, scale),
                torch.zeros_like(xbar),
            )

        label = f"cvar@{lmbda:.2g}"

    elif utility == "powexp":
        power = 1.0 + lmbda

        def u(x: Tensor, y: Tensor) -> Tensor:
            xbar  = x+y
            x_pos = torch.clamp(xbar, min=0.0)
            x_neg = torch.clamp(-xbar, min=0.0)

            positive = -torch.expm1(-lmbda * x_pos) / lmbda
            negative = xbar - x_neg.pow(power) / power

            return torch.where(xbar > 0, positive, negative) - y

        def deriv(x: Tensor, y: Tensor) -> Tensor:
            xbar  = x+y
            x_pos = torch.clamp(xbar, min=0.0)
            x_neg = torch.clamp(-xbar, min=0.0)

            positive = torch.exp(-lmbda * x_pos)
            negative = 1.0 + x_neg.pow(lmbda)

            return torch.where(xbar > 0, positive, negative)

        label = f"powexp@{lmbda:.2g}"

    else:
        raise ValueError(f"Unknown utility: {utility!r}")

    return u, deriv, label
torch_utility.utilities = [
    "entropy",
    "cvar",
    "powexp"
]


@version("0.1")
@dataclass(frozen=True)
class DualRiskNeutralDeepHedgingConfig:
    label        : str|Callable = ""               # Label; if empty use notebook name
    group        : str = "deephedging"             # Directory
    user_version : str = "0.1"                     # User version; change to invalidate cache.
    utility      : str = "entropy"                 # Utility; see torch_utility.utilities
    lmbda        : float = 1.0                     # Risk aversion 
    delta_hedged : bool = True                     # Whether to consider delta hedging options
    use_fixed    : bool = True
    in_config    : FeedForwardConfig|None = FeedForwardConfig()
    pf_config    : FeedForwardConfig|None = None
    dual_weight  : float = 0.1                    # Dual loss weight or ``None`` for no dual training.
    tert_weight  : float = 0.1                    # Tertiary loss weight or ``None`` for no tertiary training.
    trn_cost_fac : float = 0.1                     # cost as fraction of fair value, during training
    cost         : float = 0.0001                  # cost for dual 
    boot_strap   : int = 1                         # bootstrap primary optimization
    boot_strap_dual : int = 1                      # bootstrap dual optimization
    boot_strap_tert : int = 1                      # bootstrap tertiary optimization
    seed         : int = 1231231                   # seed
    batch_block_size   : int|None = None
    tau_block_size     : int|None = None
    cost_std     : float = 0.

    _floor_P : float = 1E-8

    @property
    def use_in(self) -> bool:
        return not self.in_config is None
    @property
    def use_pf(self) -> bool:
        return not self.pf_config is None
 
    def __post_init__(self):
        verify_inp( self.utility in torch_utility.utilities, f"Expected utility to be one of {torch_utility.utilities} but got '{self.utility}'." ) 
        verify_inp( self.lmbda > 0., f"Expected lmbda to be positive but got {self.lmbda}." )
        verify_inp( self.dual_weight is None or self.dual_weight >= 0., f"Expected dual_weight to be positive or None but got {self.dual_weight}." )
        verify_inp( self.boot_strap >= 1, f"'boot_strap' must be positive, not {self.boot_strap}")
        verify_inp( self.boot_strap_dual >= 1, f"'boot_strap_dual' must be positive, not {self.boot_strap_dual}")
        verify_inp( self.use_fixed or self.use_in or self.use_pf, f"Expected at least one of 'use_fixed', 'use_in' or 'use_pf' to be True." )
        verify_inp( self.trn_cost_fac > 0., f"Expected trn_cost_fac to be positive but got {self.trn_cost_fac}." )
        verify_inp( self.cost >= 0., f"Expected cost to be non-negative but got {self.cost}." )

        if self.dual_weight == 0.:
            self.__dict__['dual_weight'] = None
        if self.tert_weight == 0.:
            self.__dict__['tert_weight'] = None

        verify_inp( not self.dual_weight is None or self.tert_weight is None, "Cannot use tert if dual is off")

        if callable(self.label):
            self.__dict__['label'] = self.label(**asdict(self))
        if self.label == "":
            label = _notebook_name()
            verify_inp( label is not None, f"Expected to be able to determine notebook name for 'label', but failed. Please set 'label' explicitly in the config." )
            self.__dict__['label'] = label

    @property
    def qualified_label(self) -> str:
        return f"{self.label}@{f'{self.group}' if self.group else ''} {self.user_version}"
    
    @property
    def is_dual(self) -> bool:
        return not self.dual_weight is None
    
    @property
    def is_tert(self) -> bool:
        return not self.tert_weight is None
    
    @property
    def quantiles(self) -> np.ndarray:
        return np.array([0.05, 0.1, 0.5, 0.9, 0.95], dtype=np.float32 )

    def create_boot_strap(self, env, num_samples : int):
        """
        Returns a bootstrap density for num_samples.
        The density adds up to num_samples (e.g. the baseline is (1,...,1)) which means to compute e.g. a mean use np.mean( boostrap * x ) not np.sum( bootstrap * x )
        """
        assert num_samples>0
        np.random.seed(self.seed)
        def bootstrap(num_bootstrap):
            Y = np.random.multinomial(
                n=num_samples,
                pvals=np.full(num_samples, 1.0 / num_samples),
                size=num_bootstrap
            ).T   # shape: (num_samples, M)
            assert Y.shape == (num_samples, num_bootstrap)
            assert np.max( np.abs( np.sum( Y, axis=0) - num_samples) )<1E-8, f"Bootstrap Error 1, {np.sum( Y, axis=0 )}"
            Y = Y / np.sum( Y, axis=0, keepdims=True) * float(num_samples)
            assert Y.shape == (num_samples, num_bootstrap)
            assert np.all( np.isfinite(Y) )
            assert np.max( np.abs( np.sum( Y, axis=0) - num_samples ) )<1E-8, f"Bootstrap Error 2, {np.sum( Y, axis=0 )}"
            return Y

        r = PrettyObject()
        if self.boot_strap > 1:
            r.boot_strap_P = torch.tensor( bootstrap(self.boot_strap), dtype=env.fdtype, device=env.device )
        if not self.dual_weight is None and self.boot_strap_dual > 1:
            r.boot_strap_dual_P = torch.tensor( bootstrap(self.boot_strap_dual), dtype=env.fdtype, device=env.device )
        if not self.tert_weight is None and self.boot_strap_tert > 1:
            r.boot_strap_tert_P = torch.tensor( bootstrap(self.boot_strap_tert), dtype=env.fdtype, device=env.device )
        return r


@dataclass
class TrackerContent:
    epochs : list[int] = field(default_factory=list)
    values : list[float] = field(default_factory=list)
    std    : list[float] = field(default_factory=list)
    qtls   : list[np.ndarray] = field(default_factory=list)
    pnl    : list[float] = field(default_factory=list)
    pnl_std: list[float] = field(default_factory=list)

@dataclass
class UtilityTracker:
    util : TrackerContent = field(default_factory=TrackerContent)
    dual : TrackerContent = field(default_factory=TrackerContent)
    tert : TrackerContent = field(default_factory=TrackerContent)

    def __post_init__(self):
        if not self.util is None and not isinstance(self.util, TrackerContent):
            self.util = TrackerContent(**self.util)
        if not self.dual is None and not isinstance(self.dual, TrackerContent):
            self.dual = TrackerContent(**self.dual)
        if not self.tert is None and not isinstance(self.tert, TrackerContent):
            self.tert = TrackerContent(**self.tert)

@version("0.1")
class MeasureChange(BaseModel):
    def __init__(self, env : BaseEnvironment, *,
                       num_steps         : int,
                       num_xstate        : int,
                       num_instruments   : int,
                       num_in_features   : int,
                       num_pf_features   : int,
                       num_bootstrap     : int,
                       cost              : float,
                       pcnt_time         : torch.Tensor,
                       quantiles     : torch.Tensor,  
                       config : DualRiskNeutralDeepHedgingConfig = DualRiskNeutralDeepHedgingConfig() ):
        super().__init__(env)
        self.num_steps       = num_steps
        self.num_xstate      = num_xstate
        self.num_instruments = num_instruments
        self.num_in_features = num_in_features
        self.num_pf_features = num_pf_features
        self.config          = config
        self.cost            = float(cost)
        self.cost_std        = float(config.cost_std)
        self.floor_P         = float(config._floor_P)
        self.num_bootstrap   = num_bootstrap
        self.a_fx            = torch.nn.Parameter(torch.zeros( (num_steps,num_instruments), dtype=env.fdtype, device=env.device ), requires_grad=True) if config.use_fixed else None
        self.a_in            = FeedForward( env, num_input = num_in_features, num_output = 0, config = config.in_config ) if config.use_in else None
        self.a_pf            = FeedForward( env, num_input = num_pf_features, num_output = num_instruments, config = config.pf_config ) if config.use_pf else None
        self.y               = torch.nn.Parameter(torch.zeros( (1,num_bootstrap,) , dtype=env.fdtype, device=env.device ), requires_grad=True)
        self.u,\
        self.u_deriv,\
        self.u_label         = torch_utility(config.utility, config.lmbda)
        self.pcnt_time       = pcnt_time
        self.quantiles       = quantiles

    def compute_features( self,*,
                            S_slc : slice,
                            TAU_slc : slice,
                            iS : int,
                            iTAU : int,
                            dC : torch.Tensor,
                            delta_t : torch.Tensor,
                            dS : torch.Tensor,
                            spots : torch.Tensor,
                            strikes : torch.Tensor,
                            expiries : torch.Tensor,
                            xstate : torch.Tensor ) -> tuple[torch.Tensor|None, torch.Tensor|None]:
        """ Compute features for the model. """

        S, TAU, T, N = dC.shape
        XS           = self.num_xstate
        log_strikes  = torch.log( strikes[S_slc,TAU_slc,:, :] )

        # compute fixed and portfolio action
        # ----------------------------------

        if self.config.use_pf:
            pf_features     = torch.concatenate( [
                                    xstate[S_slc,TAU_slc,:],
                                    expiries.repeat(iS,iTAU,1), # T
                                    log_strikes.reshape((iS,iTAU,T*N)),
                                    self.pcnt_time.reshape(1,iTAU,1).repeat(iS,1,1)
                                    ], dim=-1 )
            assert pf_features.shape == (iS,iTAU,self.num_pf_features), f"Expected pf features to have shape {(iS,iTAU,self.num_pf_features)} but got {pf_features.shape}"
        else:
            pf_features = None

        if self.config.use_in:
            # features
            spot_ztau     = log_strikes[:,:,:1,:1]*0.

            expiries_tau  = expiries.reshape(1,1,T,1).repeat(iS,iTAU,1,N).reshape(iS,iTAU,T*N,1)
            assert expiries_tau.shape == (iS, iTAU, T*N, 1), f"Expected expiries to have shape {(iS, iTAU, T*N,1)} but got {expiries.shape}"
            expiries_tau = torch.cat( [spot_ztau, expiries_tau], dim=2 )
            assert expiries_tau.shape == (iS, iTAU, T*N+1, 1), f"Expected expiries to have shape {(iS, iTAU, T*N+1,1)} but got {expiries_tau.shape}"

            logK_tau      = log_strikes.reshape(iS, iTAU, T*N,1)
            assert logK_tau.shape == (iS, iTAU, T*N, 1), f"Expected logK_tau to have shape {(iS, iTAU, T*N,1)} but got {logK_tau.shape}"
            logK_tau      = torch.cat( [spot_ztau, logK_tau], dim=2 )
            assert logK_tau.shape == (iS, iTAU, T*N+1, 1), f"Expected logK_tau to have shape {(iS, iTAU, T*N+1,1)} but got {logK_tau.shape}"

            assert xstate.shape == (S, TAU, XS), f"Expected xstate to have shape {(S, TAU, XS)} but got {xstate.shape}"
            xstate_tau    = xstate[S_slc, TAU_slc, :].reshape(iS, iTAU,1,XS).repeat(1,1,T*N+1,1)
            assert xstate_tau.shape == (iS, iTAU, T*N+1, XS), f"Expected xstate_tau to have shape {(iS, iTAU, T*N+1,XS)} but got {xstate_tau.shape}"

            time_pcnt_tau = self.pcnt_time[TAU_slc].reshape(1,iTAU,1,1).repeat(iS, 1, T*N+1, 1)
            assert time_pcnt_tau.shape == (iS, iTAU, T*N+1, 1), f"Expected time_pcnt to have shape {(iS, iTAU, T*N+1, 1)} but got {time_pcnt_tau.shape}"

            in_features   = torch.cat( [
                            xstate_tau,   # T*N+1,XS
                            expiries_tau, # T*N+1,1 
                            logK_tau,     # T*N+1,1
                            time_pcnt_tau # T*N+1,1
                            ], dim=-1 )
            assert in_features.shape == (iS, iTAU, T*N+1, self.num_in_features), f"Expected IN features to have shape {(iS, iTAU, T*N+1, self.num_in_features)} but got {in_features.shape}"
        else:
            in_features = None

        return pf_features, in_features

    def compute_action(self, *, 
                       S_slc : slice,
                       TAU_slc : slice,
                       iS : int,
                       iTAU : int,
                       pf_features : torch.Tensor|None, in_features : torch.Tensor|None ) -> torch.Tensor:
        a  = self.a_fx[TAU_slc].reshape((1,iTAU,self.num_instruments)) if not self.a_fx is None else 0.
        a  = ( a + self.a_pf(pf_features) ) if not self.a_pf is None else a
        a  = ( a + self.a_in(in_features) ) if not self.a_in is None else a
        assert a.shape == (iS,iTAU,self.num_instruments) or a.shape == (1,iTAU,self.num_instruments), f"Expected action to have shape (1|{iS}, {iTAU}, {self.num_instruments}) but got {a.shape}"
        return a

    def __call__(self, data, *, dQdP, bootstrap, training : bool = False) -> torch.Tensor|dict[str, torch.Tensor]:
        """ Forward pass of the model. """
        use_cost     = (self.cost > 0.) or (self.cost_std > 0.)

        dC           = get(data, "dC", 4)
        S, TAU, T, N = dC.shape
        XS           = self.num_xstate 
        assert S > 0, f"Expected S to be positive but got {S}"
        assert TAU > 0, f"Expected TAU to be positive but got {TAU}"
        assert T > 0, f"Expected T to be positive but got {T}"
        assert N > 0, f"Expected N to be positive but got {N}"
        assert T*N+1 == self.num_instruments, f"Number of instruments mismatch {T*N+1} != {self.num_instruments}"
        C            = get(data, "C", (S, TAU, T, N)) if use_cost else None
        spots        = get(data, "spots", (S, TAU+1))
        delta_t      = get(data, "delta", (S, TAU, T, N))
        strikes      = get(data, "strikes", (S,TAU,T,N))
        expiries     = get(data, "expiries", (1, 1, T))

        dS           = torch.diff( spots, axis=1 )
        xstate       = get(data, "xstate", (S, TAU, XS)) if (self.config.use_in or self.config.use_pf) else None
        zero4        = (strikes[:1,:1,:1,:1]*0.).detach()
        assert zero4.shape == (1, 1, 1, 1), f"Expected zero4 to have shape {(1, 1, 1, 1)} but got {zero4.shape}"

        if not dQdP is None:
            assert dQdP.shape == (S,), f"Expected dQdP to have shape {(S,)} but got {dQdP.shape}"
        else:
            dQdP         = zero4[:,0,0,0].repeat(S) + 1.

        if bootstrap is None:
            bootstrap    = (zero4[:,:,0,0]+1.)
        else:
            assert bootstrap.shape == (S, self.num_bootstrap), f"Expected bootstrap to have shape {(S, self.num_bootstrap)} but got {bootstrap.shape}"
        dQdP_boot         = bootstrap[:,:]*dQdP[:,None]
        dQdP_boot         = dQdP_boot / torch.mean(dQdP_boot, dim=0, keepdim=True)
        dQdP              = None
        bootstrap         = None

        batch_size        = self.config.batch_block_size if not self.config.batch_block_size is None else S
        tau_size          = self.config.tau_block_size if not self.config.tau_block_size is None else TAU
        num_batch_blocks  = (S + batch_size - 1) // batch_size
        num_tau_blocks    = (TAU + tau_size - 1) // tau_size

        pnl                = None

        # features

        for tb in range(num_tau_blocks):
            TAU_start = tb * tau_size
            TAU_end   = min( (tb+1) * tau_size, TAU )
            TAU_slc   = slice(TAU_start, TAU_end) if TAU_start > 0 or TAU_end < TAU else slice(None)
            iTAU      = TAU_end - TAU_start
            TAU_start = None
            TAU_end   = None

            a  = []
            X  = []
            dX = []

            for iblock in range(num_batch_blocks):
                S_start = iblock * batch_size
                S_end   = min( (iblock+1) * batch_size, S )
                S_slc   = slice(S_start, S_end) if S_start > 0 or S_end < S else slice(None)
                iS      = S_end - S_start
                S_start = None
                S_end   = None

                pf_features, in_features = self.compute_features( S_slc=S_slc, TAU_slc=TAU_slc, iS=iS, iTAU=iTAU, dC=dC, delta_t=delta_t, dS=dS, spots=spots, strikes=strikes, expiries=expiries, xstate=xstate )
                a_s     = self.compute_action( S_slc=S_slc, TAU_slc=TAU_slc, iS=iS, iTAU=iTAU, pf_features=pf_features, in_features=in_features )
                assert len(a_s.shape) == 3 and a_s.shape[0] in [1,iS] and a_s.shape[1:] == (iTAU, self.num_instruments), f"Expected action_a to have shape (1|{iS}, {iTAU}, {self.num_instruments}) but got {a_s.shape}"
                pf_features  = None
                in_features  = None

                # returns & cost
                # --------------
                dC_h         = ( dC[S_slc,TAU_slc,:,:] - delta_t[S_slc,TAU_slc,:,:] * dS[S_slc,TAU_slc,None,None] ) if self.config.delta_hedged else\
                               dC[S_slc,TAU_slc,:,:]
                assert dC_h.shape == (iS, iTAU, T, N), f"Expected dC_h to have shape {(iS, iTAU, T, N)} but got {dC_h.shape}"
                dC_h         = dC_h.reshape(iS, iTAU, T*N)
                dX_s         = torch.concatenate( [ dS[S_slc,TAU_slc,None], dC_h ], dim=-1 ) 
                dC_h         = None
                X_s          = None
                if use_cost:
                    C_h      = ( C[S_slc,TAU_slc,:,:] - delta_t[S_slc,TAU_slc,:,:] * spots[S_slc,TAU_slc,None,None] ) if self.config.delta_hedged else\
                                C[S_slc,TAU_slc,:,:]
                    assert C_h.shape == (iS, iTAU, T, N), f"Expected C_h to have shape {(iS, iTAU, T, N)} but got {C_h.shape}"
                    C_h       = C_h.reshape(iS, iTAU, T*N)
                    X_s       = torch.concatenate( [ spots[:,:-1][S_slc,TAU_slc,None], C_h ], dim=-1 )
                    assert X_s is None or X_s.shape == (iS, iTAU, self.num_instruments), f"Expected X to have shape {(iS, iTAU, self.num_instruments)} but got {X_s.shape}"
                    C_h       = None
                assert dX_s.shape == (iS, iTAU, self.num_instruments), f"Expected dX to have shape {(iS, iTAU, self.num_instruments)} but got {dX_s.shape}"
                if not X_s is None: assert X_s.shape == (iS, iTAU, self.num_instruments), f"Expected X to have shape {(iS, iTAU, self.num_instruments)} but got {X_s.shape}"

                if self.config.use_in or self.config.use_pf:
                    assert a_s.shape == (iS, iTAU, self.num_instruments), f"Expected a_s to have shape {(iS, iTAU, self.num_instruments)} but got {a_s.shape}"
                    a.append(a_s)
                else:
                    assert a_s.shape == (1, iTAU, self.num_instruments), f"Expected a_s to have shape {(1, iTAU, self.num_instruments)} but got {a_s.shape}"
                    a = a_s
                
                dX.append( dX_s )
                X.append( X_s ) if not X_s is None else None

            # collect pnl per sample
            # ----------------------

            a = torch.cat(a, dim=0) if self.config.use_in or self.config.use_pf else a
            dX = torch.cat(dX, dim=0)
            X = torch.cat(X, dim=0) if len(X) > 0 else None

            assert a.shape[0] in [1,S] and a.shape[1:] == (iTAU, self.num_instruments), f"Expected action_a to have shape (1|{S}, {iTAU}, {self.num_instruments}) but got {a.shape}"
            assert dX.shape == (S, iTAU, self.num_instruments), f"Expected dX to have shape {(S, iTAU, self.num_instruments)} but got {dX.shape}"
            if not X is None: assert X.shape == (S, iTAU, self.num_instruments), f"Expected X to have shape {(S, iTAU, self.num_instruments)} but got {X.shape}"

            pnl_tau = a*dX

            if self.cost > 0.:
                pnl_tau = pnl_tau - self.cost*torch.abs(a*X)
            if self.cost_std > 0.:
                std     = torch.mean( dQdP_boot[:,None,None,:]*dX[:,:,:,None], dim=0, keepdim=True)
                std     = torch.mean( std, dim=-1 )
                assert std.shape == (1,iTAU,T*N+1), f"Expected std to have shape {(1,iTAU,T*N+1)} but got {std.shape}"
                pnl_tau = pnl_tau - self.cost_std*torch.abs(a*std)

            assert pnl_tau.shape == (S, iTAU, self.num_instruments), f"Expected pnl_tau to have shape {(S, iTAU, self.num_instruments)} but got {pnl_tau.shape}"

            pnl_tau = torch.sum( pnl_tau, dim=(1,2) ) # sum over instruments
            assert pnl_tau.shape == (S, ), f"Expected pnl_tau to have shape {(S,)} but got {pnl_tau.shape}"

            pnl = (pnl+pnl_tau) if not pnl is None else pnl_tau

        # loss
        # ----

        assert pnl.shape == (S,), f"Expected pnl to have shape {(S,)} but got {pnl.shape}"

        BS = dQdP_boot.shape[1]

        assert self.y.shape == (1,BS), f"Expected y to have shape {(1,BS)} but got {self.y.shape}"
        weighted_utility = dQdP_boot * self.u( pnl[:,None], self.y )     # we wish to maximize this
        assert weighted_utility.shape == (S, BS), f"Expected weighted_utility to have shape {(S, BS)} but got {weighted_utility.shape}"
        utility        = torch.mean( weighted_utility ) # bootstrapped mean
        self.verify( lambda : torch.all(torch.isfinite(utility)), lambda : f"Expected utility to be finite but got {utility}" )

        with torch.no_grad():
            new_dQdP     = dQdP_boot * self.u_deriv( pnl[:,None], self.y ).detach() # sever the gradient here, as we do not want to backprop a and y through the dual action
            assert new_dQdP.shape == (S, BS), f"Expected new_dQdP to have shape {(S,BS)} but got {new_dQdP.shape}"
            new_dQdP     = torch.mean( new_dQdP, dim=1 )   # take mean over the boostraps (recall that \mean boostrap = 1 )
            new_dQdP     = new_dQdP / torch.clip( torch.mean(new_dQdP), min=self.floor_P ) 
            self.verify( lambda : torch.all(torch.isfinite(new_dQdP)), f"dQdP 4 is not finite: {new_dQdP}" )

        if training:
            return dict( utility=utility, dQdP=new_dQdP )
       
        with torch.no_grad():  
            util_uqtl    = torch.quantile( torch.mean( weighted_utility.detach(), dim=0 ) , q=self.quantiles )
            util_std     = torch.mean( torch.std(weighted_utility.detach(), dim=0 ) )
            weighted_pnl = dQdP_boot * pnl[:,None].detach()
            pnl          = torch.mean( weighted_pnl )
            pnl_std      = torch.mean( torch.std(weighted_pnl, dim=0 ) )
            self.verify( lambda : torch.all(torch.isfinite(util_uqtl)), lambda : f"Expected util_uqtl to be finite but got {util_uqtl}" )
            self.verify( lambda : torch.all(torch.isfinite(util_std)), lambda : f"Expected util_std to be finite but got {util_std}" )
            self.verify( lambda : torch.all(torch.isfinite(pnl)), lambda : f"Expected pnl to be finite but got {pnl}" )
            self.verify( lambda : torch.all(torch.isfinite(pnl_std)), lambda : f"Expected pnl_std to be finite but got {pnl_std}" )

        return dict(
            utility=utility,
            utility_std=util_std,
            utility_uqtl=util_uqtl,
            utility_pnl=pnl,
            utility_pnl_std=pnl_std,
            dQdP=new_dQdP
        )

    
@version("0.1", dependencies=[DualRiskNeutralDeepHedgingConfig, MeasureChange])
class DualRiskNeutralDeepHedging(BaseModel):

    def __init__(self, env : BaseEnvironment, *,
                       num_xstate  : int, 
                       num_expiries: int,
                       num_strikes : int,
                       num_steps   : int,
                       config : DualRiskNeutralDeepHedgingConfig = DualRiskNeutralDeepHedgingConfig() ):
        super().__init__(env)
        T, N                    = num_expiries, num_strikes
        self.config             = config
        self.num_xstate         = num_xstate
        self.num_in_features    = ( num_xstate + 1 + 1 + 1  ) 
        self.num_pf_features    = ( num_xstate + num_expiries + num_expiries*num_strikes + 1)
        self.num_instruments    = (num_strikes*num_expiries + 1) 
        self.u,\
        self.u_deriv,\
        self.u_label        = torch_utility(config.utility, config.lmbda)
        np.random.seed(config.seed)

        has_tert         = not config.tert_weight is None
        has_dual         = has_tert or (not config.dual_weight is None)
        self.pcnt_time   = torch.linspace(0, 1, num_steps, dtype=env.fdtype, device=env.device, requires_grad=False)
        self.quantiles   = torch.tensor(config.quantiles, dtype=env.fdtype, device=env.device, requires_grad=False)

        cost_primary     = ( config.trn_cost_fac * config.cost ) if has_dual else config.cost
        cost_dual        = ( config.trn_cost_fac * config.cost ) if has_tert else config.cost
        cost_tert        = config.cost

        self.primary = MeasureChange(env, num_steps=num_steps, num_xstate=num_xstate ,num_instruments=self.num_instruments, num_in_features=self.num_in_features, num_pf_features=self.num_pf_features, num_bootstrap=config.boot_strap,
                                     cost=cost_primary, pcnt_time=self.pcnt_time, quantiles=self.quantiles, config=config )
        self.dual    = MeasureChange(env, num_steps=num_steps, num_xstate=num_xstate ,num_instruments=self.num_instruments, num_in_features=self.num_in_features, num_pf_features=self.num_pf_features, num_bootstrap=config.boot_strap_dual,
                                     cost=cost_dual, pcnt_time=self.pcnt_time, quantiles=self.quantiles, config=config ) if has_dual else None
        self.tert    = MeasureChange(env, num_steps=num_steps, num_xstate=num_xstate ,num_instruments=self.num_instruments, num_in_features=self.num_in_features, num_pf_features=self.num_pf_features, num_bootstrap=config.boot_strap_tert,
                                     cost=cost_tert, pcnt_time=self.pcnt_time, quantiles=self.quantiles, config=config ) if has_tert else None

    @property
    def label(self) -> str:
        return self.config.label
    
    @property
    def qualified_label(self) -> str:
        return self.config.qualified_label

    @property
    def is_dual(self) -> bool:
        return not self.dual is None

    @property
    def is_tert(self) -> bool:
        return not self.tert is None

    def __call__(self, data, *, training : bool = False) -> torch.Tensor|dict[str, torch.Tensor]:
        """ Forward pass of the model. """
        data              = dict(data)
        dC                = get(data, "dC", 4)
        S, TAU, T, N      = dC.shape

        boot_strap_P      = pop(data, "boot_strap_P", (S, self.config.boot_strap)) if self.config.boot_strap > 1 else None
        boot_strap_dual_P = pop(data, "boot_strap_dual_P", (S, self.config.boot_strap_dual)) if (self.config.boot_strap_dual > 1) and (not self.dual is None) else None
        boot_strap_tert_P = pop(data, "boot_strap_tert_P", (S, self.config.boot_strap_tert)) if (self.config.boot_strap_tert > 1) and (not self.tert is None) else None

        ref_dQdP          = pop(data, "ref_dQdP", (S,), optional=True) 

        r = self.primary( data, dQdP=ref_dQdP, bootstrap=boot_strap_P, training=training )
        loss = - r['utility'] 
        dQdP = r['dQdP']
        if training:
            r = None
        if not self.dual is None:
            r_dual = self.dual( data, dQdP=dQdP, bootstrap=boot_strap_dual_P, training=training )
            dQdP   = r_dual['dQdP']
            r.update( { f"dual_{k}":v for k,v in r_dual.items() } ) if not training else None
            loss = loss - self.config.dual_weight * r_dual['utility']
            r_dual = None

        if not self.tert is None:   
            r_tert = self.tert( data, dQdP=dQdP, bootstrap=boot_strap_tert_P, training=training )
            r.update( { f"tert_{k}":v for k,v in r_tert.items() } ) if not training else None
            loss = loss - self.config.tert_weight * r_tert['utility']   
            r_tert = None

        if training:
            return loss
        r['loss'] = loss
        return r
 

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

class Visualizer(object):

    def __init__(self, universe, 
                       total_epochs: int,*,
                       num_trn_samples : int,
                       num_vld_samples : int,
                       col_size    : int = 6,
                       row_size    : int = 4,
                       ref_dQdP    : np.ndarray|None = None, 
                       viz_config  : VizConfig, 
                       mdl_config : DualRiskNeutralDeepHedgingConfig ):
        self.universe      = universe
        self.total_epochs  = total_epochs
        self.mdl_config    = mdl_config
        self.viz_config    = viz_config
        self.store         = FigStore()
        self.first         = True
        self.num_trn_samples = num_trn_samples
        self.num_vld_samples = num_vld_samples
        self.S_T           = self.universe["spots"][:,-1]
        self.S             = self.universe["spots"][:,:-1]
        self.C             = self.universe["C_t"]
        self.dS            = self.universe["spots"][:,1:] - self.universe["spots"][:,:-1]
        self.dC            = self.universe["C_tp1"] - self.universe["C_t"]
        self.days          = np.arange( self.dS.shape[1] )
        self.xdensity      = np.arange( self.dS.shape[0] )
        self.ref_dQdP      = ref_dQdP # full sample ref

        assert ref_dQdP is None or isinstance(ref_dQdP, np.ndarray), f"Expected ref_dQdP to be a numpy array or None, but got {type(ref_dQdP)}"

        # learning & fit
        # --------------

        # loss
        # ----
        self.fig_lrn   = figure(mdl_config.label, col_size=col_size, row_size=row_size, columns=None) #
        self.ax_lrn    = self.fig_lrn.add_subplot("Loss")
        self.ax_lrn.set_xlabel("Epoch")
        self.ax_lrn.set_ylabel("Loss (-Utility)")
        self.ax_lrn2   = self.ax_lrn.twinx()
        self.ax_lrn2.set_ylabel("Learning Rate")

        # util
        # ----

        self.ax_util    = self.fig_lrn.add_subplot("Utilities")
        self.ax_util.set_xlabel("Epoch")
        self.ax_util.set_ylabel("Utility")

        # pnl
        # ---

        self.ax_pnl    = self.fig_lrn.add_subplot("P&L")
        self.ax_pnl.set_xlabel("Epoch")
        self.ax_pnl.set_ylabel("P&L")

        # density
        # -------

        self.ax_dQdP = self.fig_lrn.add_subplot(r"$dQ/dP$")
        self.ax_dQdP.set_xlabel("Samples")
        self.ax_dQdP.set_ylabel("Density ratio")

        # trn
        # ---

        def make_drift(trn:bool):
            rdS, rdC       = self.compute_devs( None, trn=trn )
            ref_rdS,\
            ref_rdC        = (None, None) # turn off prior. self.compute_devs( ref_dQdP, trn=trn ) if ref_dQdP is not None else (None, None)
            cost_str       = f" after cost {self.mdl_config.cost:.4g}" if self.mdl_config.cost > 0. else ""

            self.fig_lrn.next_row()
            r = PrettyObject()
            r.ax_dev_dS = self.fig_lrn.add_subplot(r"$E[dS]/\epsilon[dS]$ " + ('training' if trn else 'validation') + cost_str)
            r.ax_dev_dS.plot( self.days, rdS.dev, "--", color="blue", label="Market")
            r.ax_dev_dS.plot( self.days, rdS.dev*0.+2., ":", color="black")
            r.ax_dev_dS.plot( self.days, rdS.dev*0.-2, ":", color="black")
            r.ax_dev_dS.plot( self.days, ref_rdS.dev, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            r.ax_dev_dS.set_xlabel("Days")
            r.ax_dev_dS.set_ylabel("Drift as fraction of MC error")
            r.ref_dev_dS_min_max = np.quantile( ref_rdS.dev, q=[0.01, 0.91] ) if not ref_rdS is None else None

            r.ax_dev_dC = self.fig_lrn.add_subplot(r"$E[dC]/\epsilon[dC]$ " + ('training' if trn else 'validation') + cost_str)
            r.ax_dev_dC.plot( self.days, rdC.tmean, "--", color="blue", label="Market")
            r.ax_dev_dC.plot( self.days, rdS.dev*0.+2., ":", color="black")
            r.ax_dev_dC.plot( self.days, rdS.dev*0.-2, ":", color="black")
            r.ax_dev_dC.fill_between( self.days, rdC.tq05, rdC.tq95, color="blue", alpha=0.1 )
            r.ax_dev_dC.plot( self.days, ref_rdC.tmean, "*:", color="orange", label="Prior") if not ref_rdC is None else None
#            r.ax_dev_dC.fill_between( self.days, ref_rdC.tq05, ref_rdC.tq95, color="orange", alpha=0.1 ) if not ref_rdC is None else None
            r.ax_dev_dC.set_xlabel("Days")
            r.ax_dev_dC.set_ylabel("Drift as fraction of MC error")
            r.ref_dev_dC_min_max = ( np.min(ref_rdC.tq05), np.max(ref_rdC.tq95) ) if not ref_rdC is None else None

            r.ax_mu_dS = self.fig_lrn.add_subplot(r"$E[dS]$ with $\epsilon[dS]$ Bands " + ('training' if trn else 'validation') + cost_str)
            r.ax_mu_dS.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dS.plot( self.days, rdS.drift, "--", color="blue", label="Market")
            r.ax_mu_dS.fill_between( self.days, rdS.drift-rdS.err, rdS.drift+rdS.err, color="blue", alpha=0.1 )
            r.ax_mu_dS.plot( self.days, ref_rdS.drift, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            #r.ax_mu_dS.fill_between( self.days, ref_rdS.drift-ref_rdS.err, ref_rdS.drift+ref_rdS.err, color="orange", alpha=0.1 ) if not ref_rdS is None else None
            r.ax_mu_dS.set_xlabel("Days")
            r.ax_mu_dS.set_ylabel("Drift as fraction of MC error\n(with standard error)")
            r.ref_mu_dS_min_max = np.quantile( ref_rdS.drift, q=[0.01, 0.91] ) if not ref_rdS is None else None

            r.ax_mu_dC = self.fig_lrn.add_subplot(r"$\mathrm{Mean}\,E[dC]$ with $\mathrm{Mean}\,\epsilon[dC]$ Bands " + ('training' if trn else 'validation') + cost_str)
            dC_mean_drift = np.mean( rdC.drift, axis=(1,2) )
            dC_mean_err   = np.mean( rdC.err, axis=(1,2) )
            r.ax_mu_dC.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dC.plot( self.days, dC_mean_drift, "--", color="blue", label="Market")
            r.ax_mu_dC.fill_between( self.days, dC_mean_drift-dC_mean_err, dC_mean_drift+dC_mean_err, color="blue", alpha=0.1 )
            if not ref_rdC is None:
                ref_dC_mean_drift = np.mean( ref_rdC.drift, axis=(1,2) )
                ref_dC_mean_err   = np.mean( ref_rdC.err, axis=(1,2) )
                r.ax_mu_dC.plot( self.days, ref_dC_mean_drift, "*:", color="orange", label="Prior") 
                r.ax_mu_dC.fill_between( self.days, ref_dC_mean_drift-ref_dC_mean_err, ref_dC_mean_drift+ref_dC_mean_err, color="orange", alpha=0.1 )
                r.ref_mu_dC_min_max = np.quantile( ref_dC_mean_drift, q=[0.01, 0.91] ) 
            else:
                r.ref_mu_dC_min_max = None
            r.ax_mu_dC.set_xlabel("Days")
            r.ax_mu_dC.set_ylabel("Drift as fraction of MC error")
            return r

        self.trn_info = make_drift(trn=True)
        self.vld_info = make_drift(trn=False) if self.num_vld_samples > 0 else None

        self.fig_lrn.render()

    def compute_devs( self, dQdS : np.ndarray|None, trn : bool ) -> tuple[PrettyObject, PrettyObject]:
        dQdS      = dQdS if dQdS is not None else np.ones( (self.dS.shape[0], ), dtype=self.dS.dtype )
        dQdS      = dQdS[:self.num_trn_samples] if trn else dQdS[self.num_trn_samples:]
        dS        = self.dS[:self.num_trn_samples] if trn else self.dS[self.num_trn_samples:]
        dC        = self.dC[:self.num_trn_samples,...] if trn else self.dC[self.num_trn_samples:,...]
        S         = self.S[:self.num_trn_samples] if trn else self.S[self.num_trn_samples:]
        C         = self.C[:self.num_trn_samples,...] if trn else self.C[self.num_trn_samples:,...]

        cost      = self.mdl_config.cost
        cost_std  = self.mdl_config.cost_std

        rdS = PrettyObject()
        rdC = PrettyObject()

        wdS        = dQdS[:,None]*dS
        wdC        = dQdS[:,None,None,None]*dC

        rdS.drift  = np.mean( wdS, axis=0 )
        rdS.cost   = cost * np.mean( dQdS[:,None]*np.abs(S), axis=0 ) + cost_std * np.std( dQdS[:,None]*np.abs(S), axis=0 )
        rdS.Espnl  = np.where( rdS.drift >= 0., np.clip( rdS.drift - rdS.cost, min=0.), np.clip( rdS.drift + rdS.cost, max=0.) )

        rdC.drift  = np.mean( wdC, axis=0 )
        rdC.cost   = cost * np.mean( dQdS[:,None,None,None]*np.abs(C), axis=0 ) 
        rdC.Espnl  = np.where( rdC.drift >= 0., np.clip( rdC.drift - rdC.cost, min=0.), np.clip( rdC.drift + rdC.cost, max=0.) )
 
        rdS.err    = np.sqrt( np.mean( (wdS-rdS.drift[None,...])**2, axis=0 ) ) / np.sqrt(dS.shape[0])
        rdC.err    = np.sqrt( np.mean( (wdC-rdC.drift[None,...])**2, axis=0 ) ) / np.sqrt(dC.shape[0])
        rdS.dev    = np.where( rdS.err > 0., rdS.Espnl / np.where( rdS.err > 0., rdS.err, 1.), 0. )
        rdC.dev    = np.where( rdC.err > 0., rdC.Espnl / np.where( rdC.err > 0., rdC.err, 1.), 0. )
        rdS.raw_dev= np.where( rdS.err > 0., rdS.drift / np.where( rdS.err > 0., rdS.err, 1.), 0. )
        rdC.raw_dev= np.where( rdC.err > 0., rdC.drift / np.where( rdC.err > 0., rdC.err, 1.), 0. )
        rdC.dev    = rdC.dev.reshape( rdC.dev.shape[0], -1 )
        rdC.raw_dev= rdC.raw_dev.reshape( rdC.raw_dev.shape[0], -1 )
        rdC.tmean  = np.mean( rdC.dev, axis=1 )    
        rdC.raw_tmean  = np.mean( rdC.raw_dev, axis=1 )
        rdC.tq05,\
        rdC.tmedian,\
        rdC.tq95  = np.quantile( rdC.dev, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.tmin  = np.min( rdC.dev, axis=1 )
        rdC.tmax  = np.max( rdC.dev, axis=1 )
        rdC.raw_tq05,\
        rdC.raw_tmedian,\
        rdC.raw_tq95  = np.quantile( rdC.raw_dev, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.raw_tmin  = np.min( rdC.raw_dev, axis=1 )
        rdC.raw_tmax  = np.max( rdC.raw_dev, axis=1 )
        return rdS, rdC

    def __call__(self, tracker, trn_tracker, vld_tracker, trn_model_output, vld_model_output, *, final : bool = False ):
        """
        Update
        """

        epoch     = tracker.trn_epochs[-1]
        self.store.clear()

        dQdP      = trn_model_output.get("dQdP", None)
        dual_dQdP = trn_model_output.get("dual_dQdP", None)
        if not dQdP is None and not vld_model_output is None:
            vld_dQdP      = vld_model_output["dQdP"] 
            vld_dual_dQdP = vld_model_output["dual_dQdP"] if not dual_dQdP is None else None
            dQdP          = np.concatenate( [ dQdP, vld_dQdP ], axis=0 ) 
            dual_dQdP     = np.concatenate( [ dual_dQdP, vld_dual_dQdP ], axis=0 ) if not dual_dQdP is None else None
        assert dQdP is None or dQdP.shape[0] == self.dS.shape[0], f"Expected dQdP to have shape {(self.dS.shape[0],)} but got {dQdP.shape}"

        # losses
        # ------
        _CLIP = 1_0000

        def losses():#
            trn_epochs =  np.asarray(tracker.trn_epochs)
            trn_losses =  np.clip( np.asarray(tracker.trn_losses), min=-_CLIP, max=_CLIP ) 
            val_epochs =  np.asarray(tracker.val_epochs)
            val_losses =  np.clip( np.asarray(tracker.val_losses) , min=-_CLIP, max=_CLIP ) 
            vld_epochs =  np.asarray(tracker.vld_epochs) if self.num_vld_samples > 0 else None
            vld_losses =  np.clip( np.asarray(tracker.vld_losses), min=-_CLIP, max=_CLIP ) if self.num_vld_samples > 0 else None

            str_vld_loss = "" if vld_losses is None else f", validation loss {vld_losses[-1]:.4g} [{vld_epochs[-1]}]"

            self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g} [{trn_epochs[-1]}], "+\
                                  f"valuation loss {val_losses[-1]:.4g} [{val_epochs[-1]}]{str_vld_loss}.\n"+\
                                  f"Total epochs {self.total_epochs}{' (Done)' if final else ''}.")

            self.store += self.ax_lrn.plot(trn_epochs, trn_losses, color="blue", label="Batch Loss")
            self.store += self.ax_lrn.plot(val_epochs, val_losses, "*:", color="orange", label="Full Loss")
            if vld_losses is not None:
                self.store += self.ax_lrn.plot(vld_epochs, vld_losses, "*:", color="green", label="Validation Loss")
            max_t   = np.max(trn_losses) if len(trn_losses)<10 else np.quantile(trn_losses, 0.9)
            max_v   = np.max(val_losses) if len(val_losses)<10 else np.quantile(val_losses, 0.9)
            max_vld = ( np.max(vld_losses) if len(vld_losses)<10 else np.quantile(vld_losses, 0.9) ) if vld_losses is not None else max_v
            min_t   = np.min(trn_losses)
            min_v   = np.min(val_losses)
            min_vld = np.min(vld_losses) if vld_losses is not None else min_v
            max_  = min( _CLIP, max( max_t, max_v, max_vld ) )
            min_  = max( -_CLIP, min( min_t, min_v, min_vld ) )
            self.ax_lrn.set_ylim( min_ * 1.1, max_ * ( 1.1 if max_ > 0. else 0.9 ) )
            self.ax_lrn.legend(loc="upper right")

            lr = np.asarray(tracker.lr)
            self.ax_lrn2.plot(tracker.trn_epochs, lr, ":", color="purple")

        losses()

        # utilities
        # --------
        def utilities_pnl():
            def mk(content : TrackerContent):
                content = content if isinstance(content, TrackerContent) else TrackerContent(**content)
                return PrettyObject(
                    epochs =  np.asarray(content.epochs),
                    values =  np.clip( np.asarray(content.values), min=-_CLIP, max=_CLIP ),
                    std    =  np.clip( np.asarray(content.std), min=-_CLIP, max=_CLIP ),
                    pnl    =  np.clip( np.asarray(content.pnl), min=-_CLIP, max=_CLIP ),
                    pnl_std=  np.clip( np.asarray(content.pnl_std), min=-_CLIP, max=_CLIP ),
                    q10    =  np.clip( np.asarray([ uql[1] for uql in content.qtls ]), min=-_CLIP, max=_CLIP ),
                    q90    =  np.clip( np.asarray([ uql[-2] for uql in content.qtls ]), min=-_CLIP, max=_CLIP )
                )

            util = mk(trn_tracker.util)
            dual = mk(trn_tracker.dual) if self.mdl_config.is_dual else None
            tert = mk(trn_tracker.tert) if self.mdl_config.is_tert else None

            trn_sq   = np.sqrt( self.num_trn_samples )
            str_util = f"{util.values[-1]:.4g} ({util.std[-1]/trn_sq:.4g}, {util.q10[-1]:.4g}, {util.q90[-1]:.4g})"
            str_dual = f"{dual.values[-1]:.4g} ({dual.std[-1]/trn_sq:.4g}, {dual.q10[-1]:.4g}, {dual.q90[-1]:.4g})" if not dual is None else None
            str_tert = f"{tert.values[-1]:.4g} ({tert.std[-1]/trn_sq:.4g}, {tert.q10[-1]:.4g}, {tert.q90[-1]:.4g})" if not tert is None else None
            
            self.store += self.ax_util.plot(util.epochs, util.epochs*0., ":", color="black")
            self.store += self.ax_util.plot(util.epochs, util.values, ".-", color="red", label="Trn Primary")
            self.store += self.ax_util.plot(dual.epochs, dual.values, ".-", color="green", label="Trn Dual") if not dual is None else None
            self.store += self.ax_util.plot(tert.epochs, tert.values, ".-", color="blue", label="Trn Tert") if not tert is None else None
            self.store += self.ax_util.fill_between(util.epochs, util.q10, util.q90, alpha=0.1, color="red")
            self.store += self.ax_util.fill_between(dual.epochs, dual.q10, dual.q90, alpha=0.1, color="green") if not dual is None else None
            self.store += self.ax_util.fill_between(tert.epochs, tert.q10, tert.q90, alpha=0.1, color="blue") if not tert is None else None
            
            min_    = min(  np.min(util.values) if len(util.values)<10 else np.quantile(util.values, 0.1),
                            ( np.min(dual.values) if len(dual.values)<10 else np.quantile(dual.values, 0.1)) if not dual is None else np.inf,
                            ( np.min(tert.values) if len(tert.values)<10 else np.quantile(tert.values, 0.1)) if not tert is None else np.inf )
            max_    = max( np.max(util.values),
                           np.max(dual.values) if not dual is None else -np.inf,
                           np.max(tert.values) if not tert is None else -np.inf )

            # pnl
            # ---
            self.store += self.ax_pnl.plot(util.epochs, util.pnl*0., ":", color="black")
            self.store += self.ax_pnl.plot(util.epochs, util.pnl, ".-", color="red", label="Trn Primary")
            self.store += self.ax_pnl.plot(dual.epochs, dual.pnl, ".-", color="green", label="Trn Dual") if not dual is None else ""
            self.store += self.ax_pnl.plot(tert.epochs, tert.pnl, ".-", color="blue", label="Trn Tert") if not tert is None else ""

            if len(util.pnl_std) == len(util.pnl):
                self.store += self.ax_pnl.fill_between(util.epochs, util.pnl-2.*util.pnl_std/trn_sq, util.pnl+2.*util.pnl_std/trn_sq, alpha=0.1, color="red")
                self.store += self.ax_pnl.fill_between(dual.epochs, dual.pnl-2.*dual.pnl_std/trn_sq, dual.pnl+2.*dual.pnl_std/trn_sq, alpha=0.1, color="green") if not dual is None else None
                self.store += self.ax_pnl.fill_between(tert.epochs, tert.pnl-2.*tert.pnl_std/trn_sq, tert.pnl+2.*tert.pnl_std/trn_sq, alpha=0.1, color="blue") if not tert is None else None

            target_pnl = tert.pnl if not tert is None else ( dual.pnl if not dual is None else util.pnl )
            max_target_pnl = np.max(target_pnl) if len(target_pnl)<10 else np.quantile(target_pnl, 0.9)

            min_pnl_ = min(  np.min(util.pnl[-5:]),
                             np.min(util.pnl) if len(util.pnl)<10 else np.quantile(util.pnl, 0.1),
                             (np.min(dual.pnl) if len(dual.pnl)<10 else np.quantile(dual.pnl, 0.1)) if not dual is None else np.inf,
                             (np.min(tert.pnl) if len(tert.pnl)<10 else np.quantile(tert.pnl, 0.1)) if not tert is None else np.inf )
            max_pnl_ = max( np.max(util.pnl) if len(util.pnl)<10 else np.quantile(util.pnl, 0.9),
                            (np.max(dual.pnl) if len(dual.pnl)<10 else np.quantile(dual.pnl, 0.9)) if not dual is None else -np.inf,
                            (np.max(tert.pnl) if len(tert.pnl)<10 else np.quantile(tert.pnl, 0.9)) if not tert is None else -np.inf )
            min_pnl_ = max( min_pnl_, -1 )


            str_util_pnl = f"{util.pnl[-1]:.4g} ({util.pnl_std[-1]/trn_sq:.4g})" if len(util.pnl_std) == len(util.pnl) else f"{util.pnl[-1]:.4g}"
            str_dual_pnl = f"{dual.pnl[-1]:.4g} ({dual.pnl_std[-1]/trn_sq:.4g})" if not dual is None and len(dual.pnl_std) == len(dual.pnl) else (f"{dual.pnl[-1]:.4g}" if not dual is None else "")
            str_tert_pnl = f"{tert.pnl[-1]:.4g} ({tert.pnl_std[-1]/trn_sq:.4g})" if not tert is None and len(tert.pnl_std) == len(tert.pnl) else (f"{tert.pnl[-1]:.4g}" if not tert is None else "")

            if vld_tracker is not None:
                vld_util = mk(vld_tracker.util)
                vld_dual = mk(vld_tracker.dual) if self.mdl_config.is_dual else None
                vld_tert = mk(vld_tracker.tert) if self.mdl_config.is_tert else None

                vld_sq   = np.sqrt( self.num_vld_samples )
                str_util = f"{str_util} ) [{vld_util.values[-1]:.4g} ({vld_util.std[-1]/vld_sq:.4g}, {vld_util.q10[-1]:.4g}, {vld_util.q90[-1]:.4g})]"
                str_dual = f"{str_dual}  [{vld_dual.values[-1]:.4g} ({vld_dual.std[-1]/vld_sq:.4g}, {vld_dual.q10[-1]:.4g}, {vld_dual.q90[-1]:.4g})]" if not vld_dual is None else ""
                str_tert = f"{str_tert} [{vld_tert.values[-1]:.4g} ({vld_tert.std[-1]/vld_sq:.4g}, {vld_tert.q10[-1]:.4g}, {vld_tert.q90[-1]:.4g})]" if not vld_tert is None else ""

                self.store += self.ax_util.plot(vld_util.epochs, vld_util.values, "--", color="orange", label="Vld Primary")
                self.store += self.ax_util.plot(vld_dual.epochs, vld_dual.values, "--", color="lightgreen", label="Vld Dual") if not vld_dual is None else None
                self.store += self.ax_util.plot(vld_tert.epochs, vld_tert.values, "--", color="purple", label="Vld Tert") if not vld_tert is None else None
                self.store += self.ax_util.fill_between(vld_util.epochs, vld_util.q10, vld_util.q90, alpha=0.1, color="orange")
                self.store += self.ax_util.fill_between(vld_dual.epochs, vld_dual.q10, vld_dual.q90, alpha=0.1, color="lightgreen") if not vld_dual is None else None
                self.store += self.ax_util.fill_between(vld_tert.epochs, vld_tert.q10, vld_tert.q90, alpha=0.1, color="purple") if not vld_tert is None else None

                min_ = min( min_,
                            np.min(vld_util.values),
                            ( np.min(vld_dual.values)  if not vld_dual is None else np.inf),
                            ( np.min(vld_tert.values) if not vld_tert is None else np.inf ))
                max_ = max( max_, 
                            np.max(vld_util.values) if len(vld_util.values)<10 else np.quantile(vld_util.values, 0.9),
                            ( np.max(vld_dual.values) if not vld_dual is None and len(vld_dual.values)<10 else np.quantile(vld_dual.values, 0.9) ) if not vld_dual is None else -np.inf,
                            ( np.max(vld_tert.values) if not vld_tert is None and len(vld_tert.values)<10 else np.quantile(vld_tert.values, 0.9) ) if not vld_tert is None else -np.inf )

                # pnl
                # ---
                self.store += self.ax_pnl.plot(vld_util.epochs, vld_util.pnl, "--", color="orange", label="Vld Primary")
                self.store += self.ax_pnl.plot(vld_dual.epochs, vld_dual.pnl, "--", color="lightgreen", label="Vld Dual") if not vld_dual is None else None
                self.store += self.ax_pnl.plot(vld_tert.epochs, vld_tert.pnl, "--", color="purple", label="Vld Tert") if not vld_tert is None else None

                if len(vld_util.pnl_std) == len(vld_util.pnl):
                    self.store += self.ax_pnl.fill_between(vld_util.epochs, vld_util.pnl-2.*vld_util.pnl_std/vld_sq, vld_util.pnl+2.*vld_util.pnl_std/vld_sq, alpha=0.1, color="orange")
                    self.store += self.ax_pnl.fill_between(vld_dual.epochs, vld_dual.pnl-2.*vld_dual.pnl_std/vld_sq, vld_dual.pnl+2.*vld_dual.pnl_std/vld_sq, alpha=0.1, color="lightgreen") if not vld_dual is None else None
                    self.store += self.ax_pnl.fill_between(vld_tert.epochs, vld_tert.pnl-2.*vld_tert.pnl_std/vld_sq, vld_tert.pnl+2.*vld_tert.pnl_std/vld_sq, alpha=0.1, color="purple") if not vld_tert is None else None

                vld_target_pnl = tert.pnl if not tert is None else ( dual.pnl if not dual is None else util.pnl )
                max_target_pnl = max( max_target_pnl, np.max(vld_target_pnl) if len(vld_target_pnl)<10 else np.quantile(vld_target_pnl, 0.9) )

                min_pnl_ = min( min_pnl_,
                                np.min(vld_util.pnl[-5]),
                                np.min(vld_util.pnl) if len(vld_util.pnl)<10 else np.quantile(vld_util.pnl, 0.1),
                                ( np.min(vld_dual.pnl) if len(vld_dual.pnl)<10 else np.quantile(vld_dual.pnl, 0.1)) if not vld_dual is None else np.inf,
                                ( np.min(vld_tert.pnl) if len(vld_tert.pnl)<10 else np.quantile(vld_tert.pnl, 0.1)) if not vld_tert is None else np.inf )
                max_pnl_ = max( max_pnl_, 
                                np.max(vld_util.pnl),
                                np.max(vld_dual.pnl) if not vld_dual is None else -np.inf,
                                np.max(vld_tert.pnl) if not vld_tert is None else -np.inf )

                if len(vld_util.pnl_std) == len(vld_util.pnl):
                    str_util_pnl = f"{str_util_pnl} [{vld_util.pnl[-1]:.4g} ({vld_util.pnl_std[-1]/vld_sq:.4g})]"
                    str_dual_pnl = f"{str_dual_pnl} [{vld_dual.pnl[-1]:.4g} ({vld_dual.pnl_std[-1]/vld_sq:.4g})]" if not vld_dual is None else ""
                    str_tert_pnl = f"{str_tert_pnl} [{vld_tert.pnl[-1]:.4g} ({vld_tert.pnl_std[-1]/vld_sq:.4g})]" if not vld_tert is None else ""
                else:
                    str_util_pnl = f"{str_util_pnl} [{vld_util.pnl[-1]:.4g}]"
                    str_dual_pnl = f"{str_dual_pnl} [{vld_dual.pnl[-1]:.4g}]" if not vld_dual is None else ""
                    str_tert_pnl = f"{str_tert_pnl} [{vld_tert.pnl[-1]:.4g}]" if not vld_tert is None else ""

            self.ax_util.set_ylim( min_ * ( 1.1 if min_ < 0. else 0.9 ), max_ * ( 0.9 if max_ < 0. else 1.1 ) )
            str_dual = "\n" + str_dual if not dual is None else ""
            str_tert = "\n" + str_tert if not tert is None else ""
            self.ax_util.set_title(f"Utilities epoch {epoch}\n{str_util}{str_dual}{str_tert}")
            if self.first:
                self.ax_util.legend(loc="upper left")

            # pnl
            # ---

            max_pnl_ = min( _CLIP, max_target_pnl*3., max_pnl_ )
            min_pnl_ = max( -_CLIP, min_pnl_ )

            self.ax_pnl.set_ylim( min_pnl_ * ( 1.1 if min_pnl_ < 0. else 0.9 ), max_pnl_ * ( 0.9 if max_pnl_ < 0. else 1.1 ) )
            str_dual_pnl = "\n" + str_dual_pnl if not dual is None else ""
            str_tert_pnl = "\n" + str_tert_pnl if not tert is None else ""
            self.ax_pnl.set_title(f"PnL epoch {epoch}\n{str_util_pnl}{str_dual_pnl}{str_tert_pnl}")
            if self.first:
                self.ax_pnl.legend(loc="upper left")

        utilities_pnl()

        # dQ/dP
        # -----

        def density():
            if not dual_dQdP is None:
                self.store += self.ax_dQdP.plot( self.xdensity[:self.num_trn_samples], dual_dQdP[:self.num_trn_samples], "-", color="purple", label=f"Trn Dual Model mean {np.mean(dual_dQdP[:self.num_trn_samples]):.4g}")
                if self.num_vld_samples > 0:
                    self.store += self.ax_dQdP.plot( self.xdensity[self.num_trn_samples:], dual_dQdP[self.num_trn_samples:], "-", color="blue", label=f"Vld Dual Model mean {np.mean(dual_dQdP[self.num_trn_samples:]):.4g}")
                max_ = np.max(dual_dQdP)
            else:
                max_ = 0.

            self.store += self.ax_dQdP.plot( self.xdensity[:self.num_trn_samples], dQdP[:self.num_trn_samples], "-", color="red", label=f"Trn Model mean {np.mean(dQdP[:self.num_trn_samples]):.4g}")
            if self.num_vld_samples > 0:
                self.store += self.ax_dQdP.plot( self.xdensity[self.num_trn_samples:], dQdP[self.num_trn_samples:], "-", color="green", label=f"Vld Model mean {np.mean(dQdP[self.num_trn_samples:]):.4g}")

            if not self.ref_dQdP is None:
                if self.num_vld_samples == 0:
                    self.store += self.ax_dQdP.plot( self.xdensity, self.ref_dQdP, "*:", color="orange", label="Prior")
                else:
                    self.store += self.ax_dQdP.plot( self.xdensity[:self.num_trn_samples], self.ref_dQdP[:self.num_trn_samples], "*:", color="orange", label="Prior (trn)")
                    self.store += self.ax_dQdP.plot( self.xdensity[self.num_trn_samples:], self.ref_dQdP[self.num_trn_samples:], "*:", color="lightgreen", label="Prior (vld)")

                max_ = max( max_, np.max(self.ref_dQdP))
            max_ = min( _CLIP, max_ )
            self.ax_dQdP.legend(loc="upper left")
            self.ax_dQdP.set_ylim( 0., max_ )
        density()


        # dev dS
        # --------

        def dev_drift(trn:bool):
            rdS, rdC = self.compute_devs( dQdP if dual_dQdP is None else dual_dQdP, trn=trn )
            r        = self.trn_info if trn else self.vld_info

            self.store += r.ax_dev_dS.plot( self.days, rdS.raw_dev, "-", color="red", label=f"Drift/Err") 
            self.store += r.ax_dev_dC.plot( self.days, rdC.raw_tmean, ".:", color="red", label=f"Drift/Err")
            self.store += r.ax_dev_dC.fill_between( self.days, rdC.raw_tq05, rdC.raw_tq95, color="red", alpha=0.05 )
            self.store += r.ax_dev_dC.fill_between( self.days, rdC.raw_tmin, rdC.raw_tmax, color="red", alpha=0.05 )

            if self.mdl_config.cost > 0.:
                self.store += r.ax_dev_dS.plot( self.days, rdS.dev, ".:", color="purple", label=f"Cost Adjusted/Err") 
                self.store += r.ax_dev_dC.plot( self.days, rdC.tmean, "-", color="purple", label=f"Cost Adjusted/Err") 
                self.store += r.ax_dev_dC.plot( self.days, rdC.tq05, ":", color="purple") 
                self.store += r.ax_dev_dC.plot( self.days, rdC.tq95, ":", color="purple") 

            r.ax_dev_dS.legend(loc="lower left")
            r.ax_dev_dC.legend(loc="lower left")

            min_dS = np.min(rdS.dev) if r.ref_dev_dS_min_max is None else min( np.min(rdS.dev), r.ref_dev_dS_min_max[0] )
            max_dS = np.max(rdS.dev) if r.ref_dev_dS_min_max is None else max( np.max(rdS.dev), r.ref_dev_dS_min_max[1] )
            ex_dS  = max( abs(min_dS), abs(max_dS) )
            min_dC = np.min(rdC.tq05)# if r.ref_dev_dC_min_max is None else min( np.min(rdC.tq05), r.ref_dev_dC_min_max[0] )
            max_dC = np.max(rdC.tq95) # if r.ref_dev_dC_min_max is None else max( np.max(rdC.tq95), r.ref_dev_dC_min_max[1] )
            ex_dC  = min( max( abs(min_dC), abs(max_dC) ), _CLIP )
            r.ax_dev_dS.set_ylim( -ex_dS*1.5-4, ex_dS*1.5+4 ) #-ex_dS*1.5-3, ex_dS*1.5+4 )
            r.ax_dev_dC.set_ylim( -ex_dC*1.5-4, ex_dC*1.5+4 )

            # drift
            self.store += r.ax_mu_dS.plot( self.days, rdS.Espnl, "-", color="red", label=f"Drift")
            self.store += r.ax_mu_dS.fill_between( self.days, rdS.Espnl-rdS.err, rdS.Espnl+rdS.err, color="red", alpha=0.1 )
            dC_mean_Espnl = np.mean( rdC.Espnl, axis=(1,2) )
            dC_mean_err   = np.mean( rdC.err, axis=(1,2) )
            self.store += r.ax_mu_dC.plot( self.days, dC_mean_Espnl, "-", color="red", label=f"Drift")
            self.store += r.ax_mu_dC.fill_between( self.days, dC_mean_Espnl-dC_mean_err, dC_mean_Espnl+dC_mean_err, color="red", alpha=0.1 )
            r.ax_mu_dS.legend(loc="lower left")
            r.ax_mu_dC.legend(loc="lower left")
            r.ax_mu_dS.set_ylim( -0.001, 0.001 )
            r.ax_mu_dC.set_ylim( -0.001, 0.001 )

        dev_drift(trn=True)
        if self.num_vld_samples > 0:
            dev_drift(trn=False)

        self.fig_lrn.render() if not final else self.fig_lrn.close()
        self.first = False

def train( universe,
           sampler, *,
           ref_dQdP      : np.ndarray|None = None,
           mdl_config    : DualRiskNeutralDeepHedgingConfig = DualRiskNeutralDeepHedgingConfig(),   
           trn_config    : TrainConfig = TrainConfig(),
           viz_config    : VizConfig = VizConfig(),
           restart       : bool = False,
           just_load     : str|bool = False,
           env           : BaseEnvironment = BaseEnvironment(),
           verbose       : Context = Context.all ) -> dict:
    """
    Train model
    """
    if isinstance(just_load, bool):
        just_load = "quiet" if just_load else "no"
    verify_inp( just_load in ["quiet", "plot", "no"], f"Expected 'just_load' to be one of 'quiet', 'no', or a string, but got '{just_load}'" )
    verify_inp( not (restart and just_load != "no"), f"Expected at most one of 'restart' and 'just_load' to be True, but found both True" )
    assert ref_dQdP is None or isinstance(ref_dQdP, np.ndarray), f"Expected 'ref_dQdP' to be a numpy array or None, but got {type(ref_dQdP)}"

    verbose = verbose if not verbose is None else Context.all
    verbose = verbose if just_load == "no" else Context.quiet

    verbose.write(lambda : f"Training {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' for {trn_config.num_epochs} epochs.")
    verbose.write(lambda : f"Using {'all samples' if trn_config.batch_size is None else f'batches of {trn_config.batch_size} samples'} with learning rate {trn_config.lr:.4g} and weight decay {trn_config.weight_decay:.4g}.")

    # create model
    # ------------

    num_total_samples,\
    num_steps,\
    num_expiries,\
    num_strikes = universe["C_t"].shape
    num_trn_samples,\
    num_vld_samples = trn_config.get_trn_vld_samples(num_total_samples)

    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        model     = DualRiskNeutralDeepHedging(  env,
                       num_xstate   = sampler.num_xstate,
                       num_expiries = num_expiries,
                       num_strikes  = num_strikes,
                       num_steps    = num_steps,
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
    tracker          = MetricTracker()
    trn_tracker      = UtilityTracker()
    vld_tracker      = UtilityTracker() if num_vld_samples > 0 else None
    model_version    = model.version.unique_id48
    checkpoint_rng_state = None

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
            if just_load == "quiet":
                return model, None, None

            optimizer.load_state_dict(state.pop("optimizer_state_dict"))
            tracker     = MetricTracker(**state.pop("tracker"))
            trn_tracker = UtilityTracker(**state.pop("trn_tracker"))
            vld_tracker = UtilityTracker(**state.pop("vld_tracker")) if num_vld_samples > 0 else None
            checkpoint_rng_state = state.pop("rng_state", None)
            start_epoch = tracker.trn_epochs[-1] 

        if not err is None:
            if just_load != "no":
                raise RuntimeError("Could not 'just_load': " + err)
            warnings.warn(err + f" -- Restarting training.")

    else:
        verbose.write(lambda : f"Using recovery checkpoint '{cache_ffn}'.")

    # Copy to GPU
    # -----------

    with verbose.write_t(lambda : f"\rCoping data to {env}... ", end='') as tme:
        ndata = dict(
            dC = universe["C_tp1"]-universe["C_t"],
            spots = universe["spots"],
            delta = universe["delta_t"],
            expiries = universe["expiries"][None,None,:],
            strikes = universe["strikes_t"],
            xstate = universe['xstate'][:,:-1,...],
        )

        assert ndata["spots"].shape == (num_total_samples, num_steps+1), f"Expected spots to have shape {(num_total_samples, num_steps+1)} but got {ndata['spots'].shape}"
        assert ndata["delta"].shape == (num_total_samples, num_steps, num_expiries, num_strikes), f"Expected delta to have shape {(num_total_samples, num_steps, num_expiries, num_strikes)} but got {ndata['delta'].shape}"  
        assert ndata["expiries"].shape == (1, 1, num_expiries), f"Expected expiries to have shape {(1, 1, num_expiries)} but got {ndata['expiries'].shape}"
        assert ndata["strikes"].shape == (num_total_samples, num_steps, num_expiries,  num_strikes), f"Expected strikes to have shape {(num_total_samples, num_steps, num_expiries,  num_strikes)} but got {ndata['strikes'].shape}"
        assert ndata["xstate"].shape == (num_total_samples, num_steps, sampler.num_xstate), f"Expected xstate to have shape {(num_total_samples, num_steps, sampler.num_xstate)} but got {ndata['xstate'].shape}"

        if mdl_config.cost > 0. or mdl_config.cost_std > 0.:
            ndata['C'] = universe["C_t"]
            assert ndata["C"].shape == (num_total_samples, num_steps, num_expiries, num_strikes), f"Expected C to have shape {(num_total_samples, num_steps, num_expiries, num_strikes)} but got {ndata['C'].shape}"  

        size  = 0
        tdata = {}
        for k, v in ndata.items():
            assert np.isfinite(v).all(), f"Expected all values in {k} to be finite, but found non-finite values."
            size += v.nbytes
            verbose.write(lambda : f"\rCopying {k} of size {fmt_big_byte_number(v.nbytes)} to {env} which brings total GPU memory use to {fmt_big_byte_number(size)}... ", end='')
            tdata[k] = torch.tensor(v, dtype=env.fdtype, device=env.device, requires_grad=False)
        del ndata

        tdata.update( mdl_config.create_boot_strap(env, num_total_samples))
        if num_vld_samples == 0:
            trn_tdata = tdata
            vld_tdata = None
        else:
            trn_tdata = { k: v[:num_trn_samples,...] if v.shape[0] > 1 else v for k, v in tdata.items() }
            vld_tdata = { k: v[num_trn_samples:,...] if v.shape[0] > 1 else v for k, v in tdata.items() }

        verbose.write(lambda : f"\rCopied {len(tdata)} arrays with a total size {fmt_big_byte_number(size)} to {env}. This took {tme}.")
        del tdata

    # deterministic measure change
    # ----------------------------
    # Is actually a bit fishy!

    if not ref_dQdP is None:
        verify_inp(ref_dQdP.shape == (num_total_samples,), f"Expected ref_dQdP to have shape {(num_total_samples,)} but got {ref_dQdP.shape}" )
        assert abs( np.mean(ref_dQdP) - 1. ) < 1e-8, f"Expected mean of ref_dQdP to be 1, but got {np.mean(ref_dQdP)}"
        if num_vld_samples > 0:
            # normalize samples in both the validation and training set
            ref_dQdP_trn = ref_dQdP[:num_trn_samples] / np.mean(ref_dQdP[:num_trn_samples])
            assert np.isfinite(ref_dQdP_trn).all(), f"Expected all values in ref_dQdP_trn to be finite, but found non-finite values."
            ref_dQdP_vld = ref_dQdP[num_trn_samples:] / np.mean(ref_dQdP[num_trn_samples:]) 
            assert np.isfinite(ref_dQdP_vld).all(), f"Expected all values in ref_dQdP_vld to be finite, but found non-finite values."
            ref_dQdP = np.concatenate([ref_dQdP_trn, ref_dQdP_vld], axis=0)
            torch_ref_dQdP = torch.tensor(ref_dQdP, dtype=env.fdtype, device=env.device)
            trn_tdata['ref_dQdP'] = torch_ref_dQdP[:num_trn_samples]
            vld_tdata['ref_dQdP'] = torch_ref_dQdP[num_trn_samples:] 
            del torch_ref_dQdP
        else:
            trn_tdata['ref_dQdP'] = torch.tensor(ref_dQdP, dtype=env.fdtype, device=env.device)
 
    # preprocess & copy data into tensors
    # -----------------------------------

    with verbose.write_t(lambda : f"\rComputing loss... ", end='') as tme:
        with torch.no_grad():
            trn_ld0 = model( trn_tdata, training=False )
            vld_ld0 = model( vld_tdata, training=False ) if not vld_tdata is None else None
        str_vld = f" and validation loss {vld_ld0['loss']:.4g}" if not vld_tdata is None else ""
        verbose.write(lambda : f"\r{'Initial' if start_epoch == 0 else 'Current'} training loss is {trn_ld0['loss']:.4g}{str_vld}. This took {tme}.", head=False )

    def sample():
        if trn_config.batch_size is None or trn_config.batch_size >= num_trn_samples:
            return trn_tdata
        else:
            idx = torch.randint(0, num_trn_samples, (trn_config.batch_size,))
            return { k: v[idx,...] if v.shape[0] > 1 else v for k, v in trn_tdata.items() }

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

            trn_valuation = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( valuation )
            _ = trn_valuation(trn_tdata)

            if num_vld_samples > 0:
                verbose.write("validation... ", head=False, end='')
                vld_valuation = torch.compile( mode="reduce-overhead",
                                dynamic=False,
                                fullgraph=True,
                                disable=model.debug,
                            )( valuation )
                _ = vld_valuation(vld_tdata)
            del _
            verbose.write(lambda : f"done; this took {tme}.", head=False ) 
    else:
        trn_valuation = valuation
        vld_valuation = valuation if not vld_tdata is None else None

    # Construction, data preparation, and compilation warm-up may consume
    # randomness. Restore afterwards so resumed sampling continues exactly at
    # the checkpoint position.
    _load_rng_state_dict(checkpoint_rng_state)

    # visualization & seralization
    # ----------------------------

    disp = Visualizer( universe, ref_dQdP=ref_dQdP, total_epochs=trn_config.num_epochs, num_trn_samples=num_trn_samples, num_vld_samples=num_vld_samples, viz_config=viz_config, mdl_config=mdl_config )
    del trn_ld0, vld_ld0

    def viz_and_store(*, epoch: int, final : bool = False):
        """ call valuation and then visualize and store the model """
        model.eval()
        trn_ld = trn_valuation( trn_tdata )
        trn_ld = { k: v.detach().cpu() for k, v in trn_ld.items() }
        vld_ld = vld_valuation( vld_tdata ) if not vld_tdata is None else None
        vld_ld = { k: v.detach().cpu() for k, v in vld_ld.items() } if not vld_tdata is None else None

        assert len(tracker.val_epochs) == 0 or tracker.val_epochs[-1] <= epoch+1, f"Expected last validation epoch {tracker.val_epochs[-1]} to be less than or equal to current epoch {epoch+1}"
            
        def append_data( tracker, ld, prefix ):
            try:
                tracker.epochs.append(epoch+1)
                tracker.values.append(ld[f'{prefix}'].item())
                tracker.std.append(ld[f'{prefix}_std'].item())
                tracker.qtls.append(ld[f'{prefix}_uqtl'].numpy())
                tracker.pnl.append(ld[f'{prefix}_pnl'].item())
                tracker.pnl_std.append(ld[f'{prefix}_pnl_std'].item())
            except KeyError as e:
                raise KeyError(f"Original error: {e} Available keys: {sorted(ld)}") from e

        tracker.val_losses.append(trn_ld['loss'].item())
        tracker.val_epochs.append(epoch+1)
        append_data(trn_tracker.util, trn_ld, 'utility')
        append_data(trn_tracker.dual, trn_ld, 'dual_utility') if 'dual_utility' in trn_ld else None
        append_data(trn_tracker.tert, trn_ld, 'tert_utility') if 'tert_utility' in trn_ld else None

        if not vld_ld is None:
            tracker.vld_losses.append(vld_ld['loss'].item())
            tracker.vld_epochs.append(epoch+1)
            append_data(vld_tracker.util, vld_ld, 'utility')
            append_data(vld_tracker.dual, vld_ld, 'dual_utility') if 'dual_utility' in vld_ld else None
            append_data(vld_tracker.tert, vld_ld, 'tert_utility') if 'tert_utility' in vld_ld else None

        if just_load == "no":
            tmp_name = path.full_temp_file_name(file_name, ext="tmp", create_directory=True)
            torch.save(dict(version=mdl_config.user_version,
                            mdl_version=model_version,
                            tracker=asdict(tracker),
                            trn_tracker=asdict(trn_tracker),
                            vld_tracker=asdict(vld_tracker) if not vld_tracker is None else {},
                            optimizer_state_dict=optimizer.state_dict(),
                            model_state_dict=model.state_dict(),
                            rng_state=_rng_state_dict(),
                            mdl_config=asdict(mdl_config),
                            mdl_config_class=mdl_config.__class__.__name__,
                            trn_config=asdict(trn_config),
                            viz_config=asdict(viz_config),
                            ), tmp_name)
            os.remove(cache_ffn) if os.path.exists(cache_ffn) else None
            os.rename(tmp_name, cache_ffn)

        def update(ld):
            model_output = dict( 
                dQdP = ld['dQdP'].cpu().numpy(),
                utility = ld['utility'].cpu().numpy(),
            )
            if 'dual_utility' in ld:
                model_output.update( dual_utility = ld['dual_utility'].cpu().numpy(),
                                     dual_dQdP = ld['dual_dQdP'].cpu().numpy() )
            if 'tert_utility' in ld:
                model_output.update( tert_utility = ld['tert_utility'].cpu().numpy(),
                                     tert_dQdP = ld['tert_dQdP'].cpu().numpy() )
            return model_output

        trn_model_output = update(trn_ld)
        vld_model_output = update(vld_ld) if not vld_ld is None else None

        disp(tracker=tracker, trn_tracker=trn_tracker, vld_tracker=vld_tracker,
                              trn_model_output=trn_model_output, vld_model_output=vld_model_output, 
                              final=final)
        return (trn_model_output, vld_model_output) if final else None

    # train
    # -----

    epoch = start_epoch
    if start_epoch < trn_config.num_epochs and just_load == "no":
        with verbose.write_t(lambda : f"Training {mdl_config.qualified_label} with {trn_config} on {env}:") as tme:
            last_update = time.time()
            try:
                for epoch in range(start_epoch, trn_config.num_epochs):

                    # -------------------------
                    # Training
                    # -------------------------
                    model.train()

                    optimizer.zero_grad(set_to_none=True)
                    batch = sample()
                    loss  = gradient_step(batch)

                    loss.backward()

                    if trn_config.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=trn_config.clip_grad_norm)

                    lr = trn_config.set_lr_for_epoch(epoch, optimizer)
                    optimizer.step()
                    train_loss = loss.detach().item() 
                    if not np.isfinite(train_loss):
                        raise RuntimeError(f"Training loss is not finite at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Stopping training.")
                    tracker.trn_losses.append(train_loss)
                    tracker.trn_epochs.append(epoch+1)
                    tracker.lr.append(lr)

                    print_info = False
                    if epoch in [0,trn_config.num_epochs-1] or ((epoch+1) % trn_config.epoch_interval == 0) or ((time.time() - last_update) > trn_config.time_interval):
                        viz_and_store(epoch=epoch, final=False)
                        last_update = time.time()
                        print_info = True

                    if print_info or ((epoch+1) % trn_config.epoch_interval == 0) or ((time.time() - last_update) > trn_config.time_interval):
                        str_vld = "" if vld_tdata is None else f", {'last ' if not print_info else ''}validation loss: {tracker.vld_losses[-1]:.4g}"
                        verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}: training loss: {train_loss:.4g}, "+\
                                                    f"{'last ' if not print_info else ''}valuation loss: {tracker.val_losses[-1]:.4g}{str_vld}; "+\
                                                    f"lr {lr:.4g}. "+
                                                    f"So far this took {tme}.{' [Check point saved]' if print_info else ''}", end='')
                        
            except KeyboardInterrupt as e:
                verbose.write(lambda : f"\r*** Training interrupted at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Saving checkpoint and exiting.")
                viz_and_store(epoch=epoch,final=True)
                raise e
        verbose.write(lambda : f"\rModel trained for {trn_config.num_epochs} epochs. This took {tme}.")

    # finalize & return
    # -----------------

    model.eval()
    trn_model_output,\
    vld_model_output = viz_and_store(epoch=epoch,final=True)
    mo_path      = path(ext="mo", fmt=SubDir.PYTREE_HDF5)
    verbose1     = verbose(1)
    verbose1.write(lambda : f"\rModel checkpoint file is'{cache_ffn}'.")
    with verbose1.write_t(lambda : f"\rWriting model output to '{mo_path.full_file_name(file_name)}'... ", end='') as tme: 
        mo_path.write(file_name, dict(trn=trn_model_output,vld=vld_model_output) if num_vld_samples > 0 else dict(trn=trn_model_output), raise_on_error=True)
        verbose1.write(lambda : f"done; this took {tme}.", head=False )
    str_vld = "" if vld_model_output is None else f", validation loss: {tracker.vld_losses[-1]:.4g}"
    verbose.write(lambda : f"\rModel {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' trained for {trn_config.num_epochs} epochs. "+\
                           f"Final training loss: {tracker.trn_losses[-1]:.4g}, valuation loss: {tracker.val_losses[-1]:.4g}{str_vld}. This took {tme}.")
    return model, trn_model_output, vld_model_output
