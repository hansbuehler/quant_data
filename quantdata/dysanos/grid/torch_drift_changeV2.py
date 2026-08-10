from multiprocessing import util

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
import importlib
import os
import numpy as np
import torch
import time
from collections.abc import Callable
import warnings as warnings

if os.environ.get("DYSANONS_USE_LAPTOP") is not None:
    _cache_root = SubDir(r"C:\users\public\.dysanos_dhx/")
else:
    _cache_root = SubDir("~/.dysanos_dhx")

torch.set_float32_matmul_precision('high')

def _rng_state_dict() -> dict:
    """Capture all RNGs used, directly or indirectly, by training."""
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _load_rng_state_dict(state: dict|None) -> None:
    """Restore an RNG state; checkpoints predating this field pass ``None``."""
    if not state:
        return
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
        ipynbname = importlib.import_module("ipynbname")
        nb_name = ipynbname.name()
    except Exception as e:
        warnings.warn(f"Failed to get notebook name: {e}. Using 'unknown' instead.")
        return "unknown"
    i = nb_name.rfind("viz_")
    if i >= 0:
        nb_name = nb_name[i+len("viz_"):]
    return nb_name
import math
from collections.abc import Callable

import torch
from torch import Tensor


TensorFunction = Callable[[Tensor, Tensor], Tensor]

@version("0.1")
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
            cap = 30.0 if x.dtype == torch.float32 else 300.0 
            z = -lmbda*(x+y)
            z =  z + (torch.clamp(z, max=cap)-z).detach()
            return torch.exp(z)

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
        """
        u(x) = (1-exp(-lmbda x)) / lmbda for x >= 0 hence u(0+) = 0 for x = 0
             = x - (-x)^2 lmbda /2 for x < 0 hence u(0-) = 0 for x = 0

        u'(x) = exp(-lmbda x) for x >= 0 hence u'(0+) = 1 for x = 0
              = 1 + lmbda -x)^2 for x < 0 hence u'(0-) = 1 for x = 0

        u''(x) = -lmbda*exp(-lmbda*x) for x >= 0 hence u''(0+) = -lmbda for x = 0
               = - lmbda for x < 0         
        """

        def u(x: Tensor, y: Tensor) -> Tensor:
            xbar  = x+y
            x_pos = torch.clamp(xbar, min=0.0)
            x_neg = torch.clamp(-xbar, min=0.0)

            positive = -torch.expm1(-lmbda * x_pos) / lmbda
            negative = xbar - lmbda * (x_neg.pow(2)) / 2.

            return torch.where(xbar > 0, positive, negative) - y

        def deriv(x: Tensor, y: Tensor) -> Tensor:
            xbar  = x+y
            x_pos = torch.clamp(xbar, min=0.0)
            x_neg = torch.clamp(-xbar, min=0.0)

            positive = torch.exp(-lmbda * x_pos)
            negative = 1.0 + lmbda * x_neg

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

def entropy( X : torch.Tensor, lmnda : float = 1., dim : int = 0 ) -> torch.Tensor:
    if lmnda > 0.:
        utility = - ( torch.logsumexp( - lmnda * X, dim=dim ) - math.log(X.shape[dim]) ) / lmnda
    else:
        utility = torch.mean( X, dim=dim )
    return utility


@version("0.2", dependencies=[torch_utility])
@dataclass(frozen=True)
class DRNConfig:
    label           : str|Callable = ""               # Label; if empty use notebook name
    group           : str = "deephedging"             # Directory
    user_version    : str = "0.2"                     # User version; change to invalidate cache.
    utility         : str = "entropy"                 # Utility; see torch_utility.utilities
    lmbda           : float = 1.0                     # Risk aversion 
    delta_hedged    : bool = True                     # Whether to consider delta hedging options
    pf_config       : FeedForwardConfig|None = FeedForwardConfig()
    cost            : float = 0.                      # cost for dual 
    cost_std        : float = 1.
    seed            : int = 1231231                   # seed
    block_size      : int|None = None
    action_cap      : float = 1.
    learn_action_cap: bool = True
    cash_trades     : bool = False
    boot_strap      : int = 1
    boot_mode       : str = "outer" # "outer", "inner", "std"
    boot_lmbda      : float = 2.        
    boot_std_scale  : float = 2.
    boot_lmbda_n    : bool = False
    boot_random     : bool = False      # Draw fresh bootstrap densities per model call

    _floor_P : float = 1E-8

    @property
    def has_some_cost(self) -> bool:
        return (self.cost > 0.) or (self.cost_std > 0.)
 
    def __post_init__(self):
        verify_inp( self.utility in torch_utility.utilities, f"Expected utility to be one of {torch_utility.utilities} but got '{self.utility}'." ) 
        verify_inp( self.lmbda > 0., f"Expected lmbda to be positive but got {self.lmbda}." )
        verify_inp( self.cost >= 0., f"Expected cost to be positive or None but got {self.cost}." )
        verify_inp( self.cost >= 0., f"Expected cost to be non-negative but got {self.cost}." )
        verify_inp( self.cost_std >= 0., f"Expected cost_std to be non-negative but got {self.cost_std}." )
        verify_inp( self.block_size is None or self.block_size > 0, f"Expected block_size to be None or positive but got {self.block_size}." )
        verify_inp( self.action_cap is None or self.action_cap > 0., f"Expected action_cap to be None or positive but got {self.action_cap}." )
        verify_inp( self.boot_strap >= 1, f"Expected boot_strap to be at least 1 but got {self.boot_strap}." )
        verify_inp( math.isfinite(self.boot_lmbda) and self.boot_lmbda > 0.,
                    f"Expected boot_lmbda to be finite and positive but got {self.boot_lmbda}." )

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
    def quantiles(self) -> np.ndarray:
        return np.array([0.05, 0.1, 0.5, 0.9, 0.95], dtype=np.float32 )


    def create_boot_strap(self, env : BaseEnvironment, num_samples : int, num_steps : int) -> tuple[torch.Tensor|None,torch.Tensor|None]:
        """
        Returns a bootstrap density for num_samples.
        The density adds up to num_total_samples (e.g. the baseline is (1,...,1)) which means to compute e.g. a mean use np.mean( boostrap * x ) not np.sum( bootstrap * x )
        """
        assert num_samples>0 and num_steps>0
        if self.boot_strap == 1:
            return (
                torch.ones((num_samples, 1), dtype=env.fdtype, device=env.device, requires_grad=False),
                torch.ones((num_steps, 1), dtype=env.fdtype, device=env.device, requires_grad=False),
            )

        def axis( n ):
            Y = np.random.multinomial(
                n=n,
                pvals=np.full(n, 1.0 / n),
                size=self.boot_strap
            ).T   # shape: (n, M)
            assert Y.shape == (n, self.boot_strap)
            assert np.max( np.abs( np.sum( Y, axis=0) - n) )<1E-8, f"Bootstrap Error 1, {np.sum( Y, axis=0 )}"
            Y = Y / np.sum( Y, axis=0, keepdims=True) * float(n)
            assert Y.shape == (n, self.boot_strap)
            assert np.all( np.isfinite(Y) )
            assert np.max( np.abs( np.sum( Y, axis=0) - n ) )<1E-8, f"Bootstrap Error 2, {np.sum( Y, axis=0 )}"
            return torch.tensor( Y, dtype=env.fdtype, device=env.device, requires_grad=False )
        return axis(num_samples), axis(num_steps)

@dataclass
class OutputTracker:
    epochs     : list[int] = field(default_factory=list)
    loss       : list[float] = field(default_factory=list)

    utility    : list[float] = field(default_factory=list)
    util_mean  : list[float] = field(default_factory=list)
    util_std   : list[float] = field(default_factory=list)

    # by instrument
    inst_mean   : list[np.ndarray] = field(default_factory=list)
    inst_std    : list[np.ndarray] = field(default_factory=list)
    cost_mean   : list[np.ndarray] = field(default_factory=list)
    cost_std    : list[np.ndarray] = field(default_factory=list)
    inst_excess_mean : list[np.ndarray] = field(default_factory=list)
    inst_excess_std  : list[np.ndarray] = field(default_factory=list)
    inst_excess2_mean: list[np.ndarray] = field(default_factory=list)
    inst_excess2_std : list[np.ndarray] = field(default_factory=list)
    mean_dev         : list[np.ndarray] = field(default_factory=list)
    mean_excess_dev  : list[np.ndarray] = field(default_factory=list)
    mean_excess2_dev : list[np.ndarray] = field(default_factory=list)
    pnl_mean   : list[np.ndarray] = field(default_factory=list)
    pnl_std    : list[np.ndarray] = field(default_factory=list)
    raw_pnl_mean: list[np.ndarray] = field(default_factory=list)
    raw_pnl_std: list[np.ndarray] = field(default_factory=list)
    action_mean  : list[np.ndarray] = field(default_factory=list)
    action_std  : list[np.ndarray] = field(default_factory=list)
    action_l1  : list[np.ndarray] = field(default_factory=list)
    action_l2  : list[np.ndarray] = field(default_factory=list)
    action_max : list[np.ndarray] = field(default_factory=list)
    action_cap : list[float] = field(default_factory=list)

@version("0.3")
class DRN(BaseModel):
    def __init__(self, env : BaseEnvironment, *,
                       num_xstate  : int, 
                       num_expiries: int,
                       num_strikes : int,
                       num_steps   : int,
                       cost_num_samples : int|None = None,
                       config : DRNConfig = DRNConfig() ):
        super().__init__(env)
        T, N                    = num_expiries, num_strikes
        self.config             = config
        self.cost_num_samples   = None if cost_num_samples is None else int(cost_num_samples)
        verify_inp(self.cost_num_samples is None or self.cost_num_samples > 0,
                   f"Expected cost_num_samples to be None or positive, got {self.cost_num_samples}")
        self.num_xstate         = int(num_xstate)
        self.num_pf_features    = int(num_xstate) + int(num_expiries) + int(num_expiries)*int(num_strikes)
        self.num_instruments    = int(num_strikes)*int(num_expiries) + 1
        self.boot_strap         = int(config.boot_strap)
        self.num_a_y            = self.num_instruments + 1
        self.u,\
        self.u_deriv,\
        self.u_label            = torch_utility(config.utility, config.lmbda)
        np.random.seed(config.seed)

        self.quantiles          = torch.tensor(config.quantiles, dtype=env.fdtype, device=env.device, requires_grad=False)
        self.cost               = float(config.cost)
        self.cost_std           = float(config.cost_std)
        self.floor_P            = float(config._floor_P)
        self.cash_trades        = bool(config.cash_trades)
        self.boot_lmbda         = float(config.boot_lmbda)
        self.boot_lmbda_n       = bool(config.boot_lmbda_n) 
        self.boot_random        = bool(config.boot_random)
        self.boot_std_scale     = float(config.boot_std_scale)
        self.a_y                = FeedForward( env, num_input = self.num_pf_features, num_output = self.num_a_y, config = config.pf_config )
        self.a_y_cap            = torch.nn.Parameter(torch.full((2,), float(config.action_cap), dtype=env.fdtype, device=env.device ), requires_grad=self.config.learn_action_cap )
        self.torch_zero         = torch.zeros((1,), dtype=env.fdtype, device=env.device, requires_grad=False) 
        self.torch_one          = torch.ones((1,), dtype=env.fdtype, device=env.device, requires_grad=False) 

    def effective_boot_lmbda(self, num_samples: int) -> float:
        """Outer-entropy lambda, optionally scaled by sqrt(number of paths)."""
        assert num_samples > 0, f"Expected num_samples to be positive, got {num_samples}"
        scale = math.sqrt(float(num_samples)) if self.boot_lmbda_n else 1.0
        return self.boot_lmbda * scale

    def compute_a_y(self, *, 
                    S_slc : slice,
                    TAU_slc : slice,
                    iS : int,
                    iTAU : int,
                    log_strikes : torch.Tensor,
                    expiries : torch.Tensor,
                    xstate : torch.Tensor ) -> torch.Tensor:
        """ Compute a,y """
        S, TAU, TN = log_strikes.shape
        pf_features = [ xstate[S_slc,TAU_slc,:],
                        expiries.expand(iS,iTAU,-1), # T
                        log_strikes[S_slc,TAU_slc,:]
                      ]
        pf_features     = torch.cat(pf_features, dim=-1)
        assert pf_features.shape == (iS,iTAU,self.num_pf_features), f"Expected pf features to have shape {(iS,iTAU,self.num_pf_features)} but got {pf_features.shape}"
        a_y             = torch.tanh( self.a_y(pf_features) )
        assert a_y.shape == (iS,iTAU,self.num_a_y), f"Expected a_y to have shape {(iS,iTAU,self.num_a_y)} but got {a_y.shape}"
        a               = a_y[:,:,:self.num_instruments]
        y               = a_y[:,:,self.num_instruments:]
        assert a.shape == (iS,iTAU,self.num_instruments), f"Expected a to have shape {(iS,iTAU,self.num_instruments)} but got {a.shape}"
        assert y.shape == (iS,iTAU,1), f"Expected y to have shape {(iS,iTAU,1)} but got {y.shape}"
        a               = a * torch.nn.functional.softplus( self.a_y_cap[0] )
        y               = y * torch.nn.functional.softplus( self.a_y_cap[1] )
        return a, y

    def __call__(self, data, *, training : bool = False) -> torch.Tensor|dict[str, torch.Tensor]:
        """ Forward pass of the model. """
        data              = dict(data)
        dC                = pop(data, "dC", 4)
        S, TAU, T, N      = dC.shape
        assert S > 0, f"Expected S to be positive but got {S}"
        assert TAU > 0, f"Expected TAU to be positive but got {TAU}"
        assert T > 0, f"Expected T to be positive but got {T}"
        assert N > 0, f"Expected N to be positive but got {N}"
        assert T*N+1 == self.num_instruments, f"Number of instruments mismatch {T*N+1} != {self.num_instruments}"

        use_cost          = (self.cost > 0.) or (self.cost_std > 0.)
        boot_lmbda        = self.effective_boot_lmbda(S)
        XS                = self.num_xstate 
        C                 = pop(data, "C", (S, TAU, T, N)) if use_cost else None
        spots             = pop(data, "spots", (S, TAU+1))
        delta_t           = pop(data, "delta", (S, TAU, T, N))
        log_strikes       = pop(data, "log_strikes", (S,TAU,T,N))
        expiries          = pop(data, "expiries", (1, 1, T))

        dS                = torch.diff( spots, axis=1 )
        spots             = spots[:,:-1]
        xstate            = pop(data, "xstate", (S, TAU, XS)) 

        ref_dQdP           = pop(data, "ref_dQdP", optional=True) 
        boot_strap_samples = pop(data, "boot_strap_samples", (S, self.boot_strap))
        boot_strap_steps   = pop(data, "boot_strap_steps", (TAU, self.boot_strap))

        # densities
        # ---------

        if not ref_dQdP is None:
            assert ref_dQdP.shape == (S,), f"Expected ref_dQdP to have shape {(S,)} but got {ref_dQdP.shape}"
            ref_dQdP = ref_dQdP.reshape(S, 1)
        else:
            ref_dQdP = torch.ones((S, 1), dtype=spots.dtype, device=spots.device, requires_grad=False)
        ref_dQdP = ref_dQdP / torch.mean(ref_dQdP)
        self.verify( lambda : torch.isfinite(ref_dQdP).all(), f"ref_dQdP contains non-finite values: {ref_dQdP}" )

        # consider each (S,TAU) as a sample
        # ---------------------------------

        TN                 = T*N
        dC                 = dC.reshape(S,TAU, TN)        
        dS                 = dS.reshape(S,TAU, 1)
        C                  = C.reshape(S,TAU, TN) if not C is None else None
        spots              = spots.reshape(S,TAU, 1)
        delta_t            = delta_t.reshape(S,TAU, TN)
        log_strikes        = log_strikes.reshape(S,TAU, TN)
        expiries           = expiries.reshape(1, 1, T)
        xstate             = xstate.reshape(S,TAU, XS)
        ref_dQdP           = ref_dQdP.reshape(S,1)

        # Internal batching
        # -----------------

        batch_size        = self.config.block_size if not self.config.block_size is None else S
        num_blocks        = (S + batch_size - 1) // batch_size

        utility_sum       = None
        utility_sqr       = None
        sum_dQdP_S        = [] if not training else None

        # the following are tracked per instrument
        inst_sum          = None
        inst_sqr          = None
        inst_excess_sum   = None
        inst_excess_sqr   = None
        cost_sum          = None
        cost_sqr          = None
        pnl_sum           = None
        pnl_sqr           = None
        raw_pnl_sum       = None
        raw_pnl_sqr       = None
        action_max        = None
        action_sum        = None
        action_sum_abs    = None
        action_sqr        = None

        for tb in range(num_blocks):
            blk_start  = tb * batch_size
            blk_end    = min( (tb+1) * batch_size, S )
            S_slc      = slice(blk_start, blk_end) if blk_start > 0 or blk_end < S else slice(None)
            iS         = blk_end - blk_start
            TAU_slc    = slice(None) # if we need a loop over tau at some point
            iTAU       = TAU         # if we need a loop over tau at some point 
            blk_start  = None
            blk_end    = None

            # actions & intercept
            # -------------------

            a_s, y_s    = self.compute_a_y( S_slc=S_slc, TAU_slc=TAU_slc, iS=iS, iTAU=iTAU, log_strikes=log_strikes, expiries=expiries, xstate=xstate )
            assert a_s.shape == (iS, iTAU, self.num_instruments), f"Expected a_s to have shape {(iS, iTAU, self.num_instruments)} but got {a_s.shape}"
            assert y_s.shape == (iS, iTAU, 1), f"Expected y_s to have shape {(iS, iTAU, 1)} but got {y_s.shape}"

            # returns & cost
            # --------------

            dC_h         = ( dC[S_slc, TAU_slc,:] - delta_t[S_slc,TAU_slc,:] * dS[S_slc,TAU_slc, :] ) if self.config.delta_hedged else dC[S_slc,TAU_slc,:]
            assert dC_h.shape == (iS, iTAU, TN), f"Expected dC_h to have shape {(iS, iTAU, TN)} but got {dC_h.shape}"
            dX_s         = torch.concatenate( [ dS[S_slc,TAU_slc,:], dC_h ], dim=-1 )
            assert dX_s.shape == (iS, iTAU, 1+TN), f"Expected dX to have shape {(iS, iTAU, 1+TN)} but got {dX_s.shape}"
            dC_h         = None
            X_s          = None
            if use_cost or self.cash_trades:
                C_h      = ( C[S_slc, TAU_slc,:] - delta_t[S_slc, TAU_slc,:] * spots[S_slc,TAU_slc,:] ) if self.config.delta_hedged else C[S_slc, TAU_slc,:]
                assert C_h.shape == (iS, iTAU, TN), f"Expected C_h to have shape {(iS, iTAU, TN)} but got {C_h.shape}"
                X_s       = torch.concatenate( [ spots[S_slc, TAU_slc,:], C_h ], dim=-1 )
                assert X_s is None or X_s.shape == (iS, iTAU, 1+TN), f"Expected X to have shape {(iS, iTAU, 1+TN)} but got {X_s.shape}"
                C_h       = None
            assert dX_s.shape == (iS, iTAU, 1+TN), f"Expected dX to have shape {(iS, iTAU, 1+TN)} but got {dX_s.shape}"
            if not X_s is None: assert X_s.shape == (iS, iTAU, 1+TN), f"Expected X to have shape {(iS, iTAU, 1+TN)} but got {X_s.shape}"

            if self.cash_trades:
                assert X_s is not None, f"Expected X_s to be not None when cash_trades is True"
                dX_s = dX_s/X_s
                X_s  = None

            # collect
            # -------

            raw_pnl_s = a_s*dX_s
            cost_s    = self.torch_zero[0]
            if self.cost > 0.:
                cost_s   = self.cost * (torch.abs(X_s) if not X_s is None else torch.ones_like(X_s))
            if self.cost_std > 0.:
                std     = torch.std( ref_dQdP[S_slc, TAU_slc,None]*dX_s, dim=(0,1), keepdim=True)
                assert std.shape == (1,1,self.num_instruments), f"Expected std to have shape {(1,1,self.num_instruments)} but got {std.shape}"
                cost_s  = cost_s + self.cost_std*torch.abs(std)/float(math.sqrt(iS*iTAU)) if not cost_s is None else self.cost_std*torch.abs(std)/float(math.sqrt(iS*iTAU))
                std     = None
            assert raw_pnl_s.shape == (iS, iTAU, self.num_instruments), f"Expected raw_pnl_s to have shape {(iS, iTAU, self.num_instruments)} but got {raw_pnl_s.shape}"
            abs_a = torch.abs(a_s)
            assert abs_a.shape == (iS, iTAU, self.num_instruments), f"Expected abs_a to have shape {(iS, iTAU, self.num_instruments)} but got {abs_a.shape}"

            # utility per sample
            # ------------------
            # we do not provide a full random bootstrap on (S,TAU), but rather two spatial boot straps, one in S and one in TAU.
            # The resulting density is a product.

            boot_strap_s = ( boot_strap_samples[S_slc,None,:,None] * boot_strap_steps[None,TAU_slc,None,:] ).reshape(iS, iTAU, self.boot_strap**2)
            assert boot_strap_s.shape == (iS, iTAU, self.boot_strap**2), f"Expected boot_strap_s to have shape {(iS, iTAU, self.boot_strap**2)} but got {boot_strap_s.shape}"
            boot_dQdP    = ref_dQdP[S_slc,:,None] * boot_strap_s
            assert boot_strap_s.shape == (iS, iTAU, self.boot_strap**2), f"Expected boot_strap_s to have shape {(iS, iTAU, self.boot_strap**2)} but got {boot_strap_s.shape}"
            total_pnl    = torch.sum(raw_pnl_s - abs_a * cost_s, dim=-1, keepdims=True)
            assert total_pnl.shape == (iS, iTAU, 1), f"Expected total_pnl to have shape {(iS, iTAU, 1)} but got {total_pnl.shape}"
            utility_s    = boot_dQdP * self.u( total_pnl, y_s )    # we wish to maximize this
            assert utility_s.shape == (iS, iTAU, self.boot_strap**2), f"Expected utility_s to have shape {(iS, iTAU, self.boot_strap**2)} but got {utility_s.shape}"

            # utility per bootstrap
            utility_sum  = utility_sum + torch.sum( utility_s, dim=(0,1) ) if not utility_sum is None else torch.sum( utility_s, dim=(0,1) )
            utility_sqr  = utility_sqr + torch.sum( utility_s.detach()**2, dim=(0,1) ) if not utility_sqr is None else torch.sum( utility_s.detach()**2, dim=(0,1) )

            if not training:
                # Compute new compound density
                # ----------------------------
                uderiv_s    = self.u_deriv( total_pnl.detach(), y_s.detach() ) if not training else None
                assert uderiv_s.shape == (iS, iTAU, 1) if not uderiv_s is None else True, f"Expected uderiv_s to have shape {(iS, iTAU, 1)} but got {uderiv_s.shape}"
                uderiv_s    = uderiv_s[:,:,0]

                out_dQdP_s  = torch.exp( - boot_lmbda * torch.mean( utility_s.detach(), dim=(0,1), keepdims=True ) )
                assert out_dQdP_s.shape == (1,1,self.boot_strap**2), f"Expected out_dQdP_s to have shape {(1,1,self.boot_strap**2)} but got {out_dQdP_s.shape}"
                out_dQdP_s  = torch.mean( boot_strap_s * out_dQdP_s, dim=-1, keepdims=True )
                assert out_dQdP_s.shape == (iS,iTAU,1), f"Expected out_dQdP_s to have shape {(iS,iTAU,1)} but got {out_dQdP_s.shape}"

                # report mean and std per instrument
                # ----------------------------------

                weighted       = out_dQdP_s * dX_s
                weigthed_cost  = out_dQdP_s * cost_s
                weighted_excess = torch.clip( weighted - weigthed_cost, min=0. ) + torch.clip( weighted + weigthed_cost, max=0. )
                assert weighted.shape == (iS,iTAU,self.num_instruments), f"Expected weighted to have shape {(iS,iTAU,self.num_instruments)} but got {weighted.shape}"
                inst_sum       = inst_sum + torch.mean( weighted, dim=(0,1) ) if inst_sum is not None else torch.mean( weighted, dim=(0,1) )
                inst_sqr       = inst_sqr + torch.mean( weighted**2, dim=(0,1) ) if inst_sqr is not None else torch.mean( weighted**2, dim=(0,1) )
                cost_sum       = cost_sum + torch.mean( weigthed_cost, dim=(0,1) ) if cost_sum is not None else torch.mean( weigthed_cost, dim=(0,1) )
                cost_sqr       = cost_sqr + torch.mean( weigthed_cost**2, dim=(0,1) ) if cost_sqr is not None else torch.mean( weigthed_cost**2, dim=(0,1) )
                inst_excess_sum = inst_excess_sum + torch.mean( weighted_excess, dim=(0,1) ) if inst_excess_sum is not None else torch.mean( weighted_excess, dim=(0,1) )
                inst_excess_sqr = inst_excess_sqr + torch.mean( weighted_excess**2, dim=(0,1) ) if inst_excess_sqr is not None else torch.mean( weighted_excess**2, dim=(0,1) )
                weighted = None

                assert inst_sum.shape == (self.num_instruments,), f"Expected inst_sum to have shape {(self.num_instruments,)} but got {inst_sum.shape}"
                assert inst_sqr.shape == (self.num_instruments,), f"Expected inst_sqr to have shape {(self.num_instruments,)} but got {inst_sqr.shape}"
                assert cost_sum.shape == (self.num_instruments,), f"Expected cost_sum to have shape {(self.num_instruments,)} but got {cost_sum.shape}"
                assert cost_sqr.shape == (self.num_instruments,), f"Expected cost_sqr to have shape {(self.num_instruments,)} but got {cost_sqr.shape}"
                assert inst_excess_sum.shape == (self.num_instruments,), f"Expected inst_excess_sum to have shape {(self.num_instruments,)} but got {inst_excess_sum.shape}"
                assert inst_excess_sqr.shape == (self.num_instruments,), f"Expected inst_excess_sqr to have shape {(self.num_instruments,)} but got {inst_excess_sqr.shape}"

                # density (projected to samples))
                # ------------------------------
                sum_dQdP_S_s = torch.sum( out_dQdP_s[:,:,0] * uderiv_s, dim=1 ) # will divide by TAU later
                assert sum_dQdP_S_s.shape == (iS,), f"Expected sum_dQdP_S_s to have shape {(iS,)} but got {sum_dQdP_S_s.shape}"
                sum_dQdP_S.append( sum_dQdP_S_s )
                uderiv_s     = None
                out_dQdP_s   = None

                # per instrument data
                # --------------------

                pnl_s = ref_dQdP[S_slc,:,None] * ( raw_pnl_s - abs_a * cost_s )
                raw_pnl_s = ref_dQdP[S_slc,:,None] * raw_pnl_s
                assert pnl_s.shape == (iS,iTAU, self.num_instruments), f"Expected pnl_s to have shape {(iS,iTAU, self.num_instruments)} but got {pnl_s.shape}"
                assert raw_pnl_s.shape == (iS,iTAU, self.num_instruments), f"Expected raw_pnl_s to have shape {(iS,iTAU, self.num_instruments)} but got {raw_pnl_s.shape}"  

                # pnl per instrument
                assert pnl_sum is None or pnl_sum.shape == (self.num_instruments,), f"Expected pnl_sum to have shape {(self.num_instruments,)} but got {pnl_sum.shape}"
                pnl_sum = pnl_sum + torch.sum( pnl_s, dim=(0,1) ) if not pnl_sum is None else torch.sum( pnl_s, dim=(0,1) )
                pnl_sqr = pnl_sqr + torch.sum( pnl_s**2, dim=(0,1) ) if not pnl_sqr is None else torch.sum( pnl_s**2, dim=(0,1) )
                raw_pnl_sum = raw_pnl_sum + torch.sum( raw_pnl_s, dim=(0,1) ) if not raw_pnl_sum is None else torch.sum( raw_pnl_s, dim=(0,1) )
                raw_pnl_sqr = raw_pnl_sqr + torch.sum( raw_pnl_s**2, dim=(0,1) ) if not raw_pnl_sqr is None else torch.sum( raw_pnl_s**2, dim=(0,1) )
                assert pnl_sum.shape == (self.num_instruments,), f"Expected pnl_sum to have shape {(self.num_instruments,)} but got {pnl_sum.shape}"   
                assert pnl_sqr.shape == (self.num_instruments,), f"Expected pnl_sqr to have shape {(self.num_instruments,)} but got {pnl_sqr.shape}"
                assert raw_pnl_sum.shape == (self.num_instruments,), f"Expected raw_pnl_sum to have shape {(self.num_instruments,)} but got {raw_pnl_sum.shape}"
                assert raw_pnl_sqr.shape == (self.num_instruments,), f"Expected raw_pnl_sqr to have shape {(self.num_instruments,)} but got {raw_pnl_sqr.shape}"

                # actions stats per instrument
                action_max     = torch.maximum( action_max, torch.amax(abs_a,dim=(0,1)) ) if not action_max is None else torch.amax(abs_a,dim=(0,1))
                action_sum     = action_sum + torch.sum( a_s, dim=(0,1) ) if not action_sum is None else torch.sum( a_s, dim=(0,1) )
                action_sum_abs = action_sum_abs + torch.sum( abs_a,dim=(0,1) ) if not action_sum_abs is None else torch.sum( abs_a,dim=(0,1) )
                action_sqr     = action_sqr + torch.sum( a_s**2,dim=(0,1) ) if not action_sqr is None else torch.sum( a_s**2,dim=(0,1) )
                assert action_max.shape == (self.num_instruments,), f"Expected action_max to have shape {(self.num_instruments,)} but got {action_max.shape}"   
                assert action_sum.shape == (self.num_instruments,), f"Expected action_sum to have shape {(self.num_instruments,)} but got {action_sum.shape}"
                assert action_sum_abs.shape == (self.num_instruments,), f"Expected action_sum_abs to have shape {(self.num_instruments,)} but got {action_sum_abs.shape}"
                assert action_sqr.shape == (self.num_instruments,), f"Expected action_sqr to have shape {(self.num_instruments,)} but got {action_sqr.shape}"
                

            X_s = None
            dX_s = None

            pnl_s = None
            raw_pnl_s = None
            cost_s = None
            total_pnl   = None
            boot_dQdP   = None
            utility_s   = None
            uderiv_s    = None

        # Collect results
        # ---------------

        assert utility_sum.shape == (self.boot_strap**2,), f"Expected utility_sum to have shape {(self.boot_strap**2,)} but got {utility_sum.shape}"
        assert utility_sqr is None or utility_sqr.shape == (self.boot_strap**2,), f"Expected utility_sqr to have shape {(self.boot_strap**2,)} but got {utility_sqr.shape}"

        utility_mean = utility_sum / float(S*TAU)
        utility_sqr  = utility_sqr / float(S*TAU) if not utility_sqr is None else None
        utility      = entropy( utility_mean, lmnda=boot_lmbda, dim=0 ) 
        if training:
            return -utility

        # valuation
        # ---------

        # utility
        utility_std  = torch.mean(torch.sqrt( torch.clip(utility_sqr - utility_mean**2, min=0. ) ))
        utility_mean = torch.mean(utility_mean)
        assert utility_mean.shape == (), f"Expected utility_mean to have shape {()} but got {utility_mean.shape}"
        assert utility_std.shape == (), f"Expected utility_std to have shape {()} but got {utility_std.shape}"

        # density
        sum_dQdP_S    = torch.cat(sum_dQdP_S, dim=0) 
        assert sum_dQdP_S.shape == (S,), f"Expected sum_dQdP_S to have shape {(S,)} but got {sum_dQdP_S.shape}"
        dQdP_S        = sum_dQdP_S / float(TAU) # we summed up over tau
        dQdP_S        = sum_dQdP_S / torch.mean(sum_dQdP_S)

        # instruments
        inst_mean     = inst_sum / float(S*TAU)
        inst_std      = torch.sqrt( torch.clip( inst_sqr / float(S*TAU) - inst_mean**2, min=0. ) )
        cost_mean     = cost_sum / float(S*TAU)
        cost_std      = torch.sqrt( torch.clip( cost_sqr / float(S*TAU) - cost_mean**2, min=0. ) )
        excess_mean  = inst_excess_sum / float(S*TAU)
        excess_std   = torch.sqrt( torch.clip( inst_excess_sqr / float(S*TAU) - excess_mean**2, min=0. ) )

        # pnl
        assert pnl_sum.shape == (self.num_instruments,), f"Expected pnl_sum to have shape {(self.num_instruments,)} but got {pnl_sum.shape}"
        pnl_mean      = pnl_sum / float(S*TAU)
        pnl_std       = torch.sqrt( torch.clip( pnl_sqr / float(S*TAU) - pnl_mean**2, min=0. ) )

        raw_pnl_mean  = raw_pnl_sum / float(S*TAU)
        raw_pnl_std   = torch.sqrt( torch.clip( raw_pnl_sqr / float(S*TAU) - raw_pnl_mean**2, min=0. ) )

        action_max    = torch.max(action_max)
        action_mean   = action_sum / float(S*TAU)
        action_std    = torch.sqrt( torch.clip( action_sqr / float(S*TAU) - action_mean**2, min=0. ) )   
        action_l1     = action_sum_abs / float(S*TAU)
        action_l2     = torch.sqrt(action_sqr / float(S*TAU))

        return dict(
            loss=-utility,           # float
            utility=utility,         # float
            util_mean=utility_mean,  # float
            util_std=utility_std,    # float
            inst_mean=inst_mean,     # this is by instrument now
            inst_std=inst_std,       # this is by instrument now
            cost_mean=cost_mean,     # this is by instrument now
            cost_std=cost_std,       # this is by instrument now
            inst_excess_mean=excess_mean, # this is by instrument now
            inst_excess_std=excess_std,   # this is by instrument now
            pnl_mean=pnl_mean,       # this is by instrument now
            raw_pnl_mean=raw_pnl_mean, # this is by instrument now
            pnl_std=pnl_std,         # this is by instrument now
            raw_pnl_std=raw_pnl_std, # this is by instrument now
            action_mean=action_mean,   # this is by instrument now
            action_std=action_std,     # this is by instrument now
            action_l1=action_l1,     # this is by instrument now
            action_l2=action_l2,     # this is by instrument now
            action_max=action_max,   # this is by instrument now
            dQdP=dQdP_S
        )

    # Outer-bootstrap implementation.  These methods deliberately replace the
    # legacy __call__ above while keeping it in the source for checkpoint/code
    # archaeology; Python dispatches to the last method definition.
    @staticmethod
    def _random_bootstrap_axis(n: int, num_bootstraps: int, *, dtype, device):
        """Draw classical multinomial bootstrap counts on ``device``."""
        assert n > 0 and num_bootstraps > 0
        if num_bootstraps == 1:
            return torch.ones((n, 1), dtype=dtype, device=device)
        draws = torch.randint(n, (num_bootstraps, n), device=device)
        counts = torch.zeros((num_bootstraps, n), dtype=dtype, device=device)
        counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=dtype))
        return counts.T

    def _outer_prepare(self, data):
        d = dict(data)
        dC = pop(d, "dC", 4)
        S, TAU, T, N = dC.shape
        C = pop(d, "C", (S, TAU, T, N)) if self.config.has_some_cost else None
        spots = pop(d, "spots", (S, TAU + 1))
        delta = pop(d, "delta", (S, TAU, T, N))
        log_strikes = pop(d, "log_strikes", (S, TAU, T, N))
        expiries = pop(d, "expiries", (1, 1, T)).reshape(1, 1, T)
        xstate = pop(d, "xstate", (S, TAU, self.num_xstate))
        ref = pop(d, "ref_dQdP", optional=True)
        bs = pop(d, "boot_strap_samples", (S, self.boot_strap))
        bt = pop(d, "boot_strap_steps", (TAU, self.boot_strap))
        if self.boot_random:
            bs = self._random_bootstrap_axis(
                S, self.boot_strap, dtype=dC.dtype, device=dC.device)
            bt = self._random_bootstrap_axis(
                TAU, self.boot_strap, dtype=dC.dtype, device=dC.device)
        ref = (torch.ones(S, dtype=spots.dtype, device=spots.device)
               if ref is None else ref)
        ref = (ref / ref.mean()).reshape(S, 1)
        p = dict(S=S, TAU=TAU, T=T, N=N, dC=dC.reshape(S, TAU, T*N),
                 dS=torch.diff(spots, dim=1).reshape(S, TAU, 1),
                 C=None if C is None else C.reshape(S, TAU, T*N),
                 spots=spots[:, :-1].reshape(S, TAU, 1),
                 delta=delta.reshape(S, TAU, T*N),
                 log_strikes=log_strikes.reshape(S, TAU, T*N),
                 expiries=expiries, xstate=xstate, ref=ref, bs=bs, bt=bt)
        assert bs.shape == (S, self.boot_strap)
        assert bt.shape == (TAU, self.boot_strap)
        return p

    def _outer_slices(self, p):
        n = self.config.block_size or p["S"]
        for start in range(0, p["S"], n):
            yield slice(start, min(start+n, p["S"]))

    def _outer_returns(self, p, slc):
        dC = (p["dC"][slc] - p["delta"][slc] * p["dS"][slc]
              if self.config.delta_hedged else p["dC"][slc])
        dX = torch.cat((p["dS"][slc], dC), dim=-1)
        X = None
        if self.config.has_some_cost or self.cash_trades:
            C = (p["C"][slc] - p["delta"][slc] * p["spots"][slc]
                 if self.config.delta_hedged else p["C"][slc])
            X = torch.cat((p["spots"][slc], C), dim=-1)
        if self.cash_trades:
            dX, X = dX / X, None
        return dX, X

    @torch.no_grad()
    def compute_cost_std_vector(self, p):
        sm = torch.zeros(self.num_instruments, dtype=p["dC"].dtype, device=p["dC"].device)
        sq = torch.zeros_like(sm)
        for slc in self._outer_slices(p):
            dX, _ = self._outer_returns(p, slc)
            z = p["ref"][slc, :, None] * dX
            sm += z.sum((0, 1)); sq += z.square().sum((0, 1))
        n = p["S"] * p["TAU"]
        var = torch.zeros_like(sm) if n <= 1 else ((sq-sm.square()/n)/(n-1)).clamp_min(0)
        # Estimate volatility from the current batch, but scale its standard
        # error by the fixed full training sample size. This keeps minibatch
        # training costs consistent with full-training-set valuation costs.
        cost_n = (self.cost_num_samples or p["S"]) * p["TAU"]
        return (var.sqrt()/math.sqrt(cost_n)).reshape(1, 1, -1)

    @torch.no_grad()
    def compute_bootstrap_normalizers(self, p):
        sm = torch.zeros(self.boot_strap, dtype=p["dC"].dtype, device=p["dC"].device)
        for slc in self._outer_slices(p):
            boot = p["bs"][slc, None, :] * p["bt"][None, :, :]
            sm += (p["ref"][slc, :, None] * boot).sum((0, 1))
        return (sm / float(p["S"]*p["TAU"])).clamp_min(self.floor_P)

    def compute_block(self, p, slc, cost_std, normalizer=None, *, utility=True, deriv=False):
        iS = p["dC"][slc].shape[0]
        a, y = self.compute_a_y(S_slc=slc, TAU_slc=slice(None), iS=iS,
            iTAU=p["TAU"], log_strikes=p["log_strikes"],
            expiries=p["expiries"], xstate=p["xstate"])
        dX, X = self._outer_returns(p, slc)
        cost = torch.zeros((1, 1, self.num_instruments), dtype=dX.dtype, device=dX.device)
        if self.cost > 0:
            cost = self.cost * (X.abs() if X is not None else torch.ones_like(a))
        if self.cost_std > 0:
            cost = cost + self.cost_std * cost_std
        raw = a*dX
        pnl = (raw-a.abs()*cost).sum(-1, keepdim=True)
        out = dict(a=a, y=y, dX=dX, cost=cost, raw=raw, pnl=pnl)
        if utility: out["v"] = self.u(pnl, y)
        if deriv: out["up"] = self.u_deriv(pnl.detach(), y.detach())
        if normalizer is not None:
            boot = p["bs"][slc, None, :] * p["bt"][None, :, :]
            D = p["ref"][slc, :, None]*boot/normalizer[None, None, :]
            assert boot.shape == D.shape == (iS, p["TAU"], self.boot_strap)
            out.update(bootstrap=boot, bootstrap_density=D)
        assert a.shape == dX.shape == (iS, p["TAU"], self.num_instruments)
        assert y.shape == pnl.shape == (iS, p["TAU"], 1)
        return out

    @torch.no_grad()
    def compute_bootstrap_utilities(self, p, normalizer, cost_std, second=False):
        sm = torch.zeros(self.boot_strap, dtype=p["dC"].dtype, device=p["dC"].device)
        sq = torch.zeros_like(sm) if second else None
        for slc in self._outer_slices(p):
            b = self.compute_block(p, slc, cost_std, normalizer)
            sm += (b["bootstrap_density"]*b["v"]).sum((0, 1))
            if second: sq += (b["bootstrap_density"]*b["v"].square()).sum((0, 1))
        n = float(p["S"]*p["TAU"])
        return sm/n, None if sq is None else sq/n

    def backward_outer_objective(self, data):
        p = self._outer_prepare(data)
        with torch.no_grad():
            cost_std = self.compute_cost_std_vector(p)
            norm = self.compute_bootstrap_normalizers(p)
            utility_b, _ = self.compute_bootstrap_utilities(p, norm, cost_std)
            boot_lmbda = self.effective_boot_lmbda(p["S"])
            weights = torch.softmax(-boot_lmbda*utility_b, dim=0)
            robust = entropy(utility_b, lmnda=boot_lmbda, dim=0)
        # Exact identity: grad J = sum_b pi_b grad V_b.
        n = float(p["S"]*p["TAU"])
        for slc in self._outer_slices(p):
            b = self.compute_block(p, slc, cost_std, norm)
            ub = (b["bootstrap_density"]*b["v"]).sum((0, 1))
            (-(weights.detach()*ub).sum()/n).backward()
        return -robust.detach()

    def __call__(self, data, *, training=False, return_density_details=False):
        if training:
            return self.backward_outer_objective(data)
        p = self._outer_prepare(data)
        cost_std = self.compute_cost_std_vector(p)
        norm = self.compute_bootstrap_normalizers(p)
        utility_b, second_b = self.compute_bootstrap_utilities(p, norm, cost_std, True)
        boot_lmbda = self.effective_boot_lmbda(p["S"])
        weights = torch.softmax(-boot_lmbda*utility_b, dim=0)
        utility = entropy(utility_b, lmnda=boot_lmbda, dim=0)
        # ``up`` is the one-step OCE density increment conditional on the
        # current state/history.  The robust bootstrap mixture is a density
        # relative to the reference path measure, while ``mix`` itself already
        # contains that reference density.  Keep these objects separate: the
        # reference path density is applied exactly once when the conditional
        # increments are compounded below.
        q_raw = []
        oce_deriv_st = []
        u_deriv_st = []
        for slc in self._outer_slices(p):
            b = self.compute_block(p, slc, cost_std, norm, deriv=True)
            mix = (b["bootstrap_density"]*weights[None, None, :]).sum(-1)
            up = b["up"][..., 0]
            ref = p["ref"][slc]
            robust_tilt = mix/ref.clamp_min(self.floor_P)
            increment = robust_tilt*up
            q_raw.append(ref*increment)
            oce_deriv_st.append(up)
            u_deriv_st.append(increment)

        q = torch.cat(q_raw, 0)
        oce_deriv_st = torch.cat(oce_deriv_st, 0)
        u_deriv_st = torch.cat(u_deriv_st, 0)
        q = q/q.mean().clamp_min(self.floor_P)
        assert q.shape == (p["S"], p["TAU"])
        assert oce_deriv_st.shape == u_deriv_st.shape == q.shape
        self.verify(lambda: torch.isfinite(q).all() and (q >= 0).all(), "invalid dQdP_st")
        self.verify(lambda: torch.isfinite(oce_deriv_st).all() and (oce_deriv_st >= 0).all(),
                    "invalid OCE utility derivative")
        self.verify(lambda: torch.isfinite(u_deriv_st).all() and (u_deriv_st >= 0).all(),
                    "invalid bootstrap-adjusted conditional density")

        # Chain rule for path densities:
        #   dQ_new/dP = dQ_ref/dP * product_tau dQ_new_tau/dQ_ref_tau.
        # Work in log space because a path can contain many time steps.
        log_ref = torch.log(p["ref"][:, 0].clamp_min(self.floor_P))
        log_increment = torch.log(u_deriv_st.clamp_min(self.floor_P))
        log_dQdP = log_ref + log_increment.sum(dim=1)
        log_dQdP = log_dQdP - torch.logsumexp(log_dQdP, dim=0) + math.log(p["S"])
        dQdP = torch.exp(log_dQdP)
        self.verify(lambda: torch.isfinite(dQdP).all() and (dQdP >= 0).all(),
                    "invalid path density")

        keys = ("inst_sum inst_sqr inst_abs_sum inst_abs_excess_sum inst_abs_excess2_sum "
                "inst_excess_sum inst_excess_sqr "
                "inst_excess2_sum inst_excess2_sqr cost_sum pnl_sum pnl_sqr "
                "raw_pnl_sum raw_pnl_sqr action_sum action_abs action_sqr").split()
        z = torch.zeros(self.num_instruments, dtype=p["dC"].dtype, device=p["dC"].device)
        acc = {k: z.clone() for k in keys}; amax = None
        for slc in self._outer_slices(p):
            b = self.compute_block(p, slc, cost_std, utility=False)
            qb = q[slc, :, None]; wd = qb*b["dX"]
            acc["inst_sum"] += wd.sum((0, 1)); acc["inst_sqr"] += wd.square().sum((0, 1))
            weighted_cost = qb*b["cost"]
            abs_wd = wd.abs()
            acc["inst_abs_sum"] += abs_wd.sum((0, 1))
            acc["inst_abs_excess_sum"] += (abs_wd-weighted_cost).clamp_min(0.).sum((0, 1))
            acc["inst_abs_excess2_sum"] += (abs_wd-2.*weighted_cost).clamp_min(0.).sum((0, 1))
            excess = torch.clip(wd-weighted_cost, min=0.) + torch.clip(wd+weighted_cost, max=0.)
            excess2 = torch.clip(wd-2.*weighted_cost, min=0.) + torch.clip(wd+2.*weighted_cost, max=0.)
            acc["inst_excess_sum"] += excess.sum((0, 1))
            acc["inst_excess_sqr"] += excess.square().sum((0, 1))
            acc["inst_excess2_sum"] += excess2.sum((0, 1))
            acc["inst_excess2_sqr"] += excess2.square().sum((0, 1))
            acc["cost_sum"] += weighted_cost.sum((0, 1))
            aa = b["a"].abs()
            pnl = p["ref"][slc, :, None]*(b["raw"]-aa*b["cost"])
            raw = p["ref"][slc, :, None]*b["raw"]
            for name, value in (("pnl", pnl), ("raw_pnl", raw)):
                acc[name+"_sum"] += value.sum((0, 1)); acc[name+"_sqr"] += value.square().sum((0, 1))
            acc["action_sum"] += b["a"].sum((0, 1)); acc["action_abs"] += aa.sum((0, 1))
            acc["action_sqr"] += b["a"].square().sum((0, 1))
            bm = aa.amax((0, 1)); amax = bm if amax is None else torch.maximum(amax, bm)
        n = float(p["S"]*p["TAU"])
        mean_std = lambda name: (acc[name+"_sum"]/n,
            (acc[name+"_sqr"]/n-(acc[name+"_sum"]/n).square()).clamp_min(0).sqrt())
        inst_mean, inst_std = mean_std("inst")
        inst_excess_mean, inst_excess_std = mean_std("inst_excess")
        inst_excess2_mean, inst_excess2_std = mean_std("inst_excess2")
        inst_err = inst_std/math.sqrt(n)
        dev_denom = torch.where(inst_err > 0., inst_err, torch.ones_like(inst_err))
        mean_dev = torch.where(inst_err > 0., (acc["inst_abs_sum"]/n)/dev_denom, 0.)
        mean_excess_dev = torch.where(inst_err > 0., (acc["inst_abs_excess_sum"]/n)/dev_denom, 0.)
        mean_excess2_dev = torch.where(inst_err > 0., (acc["inst_abs_excess2_sum"]/n)/dev_denom, 0.)
        pnl_mean, pnl_std = mean_std("pnl")
        raw_mean, raw_std = mean_std("raw_pnl"); action_mean, action_std = mean_std("action")
        var_b = (second_b-utility_b.square()).clamp_min(0)
        out = dict(loss=-utility, utility=utility, util_mean=utility_b.mean(),
            util_std=var_b.sqrt().mean(), inst_mean=inst_mean, inst_std=inst_std,
            inst_excess_mean=inst_excess_mean, inst_excess_std=inst_excess_std,
            inst_excess2_mean=inst_excess2_mean, inst_excess2_std=inst_excess2_std,
            mean_dev=mean_dev, mean_excess_dev=mean_excess_dev,
            mean_excess2_dev=mean_excess2_dev,
            cost_mean=acc["cost_sum"]/n, pnl_mean=pnl_mean, pnl_std=pnl_std,
            raw_pnl_mean=raw_mean, raw_pnl_std=raw_std, action_mean=action_mean,
            action_std=action_std,
            action_l1=acc["action_abs"]/n,
            action_l2=(acc["action_sqr"]/n).sqrt(),
            action_max=amax.max(), dQdP=dQdP,
            utility_b=utility_b, outer_weights=weights)
        if return_density_details:
            out.update(dQdP_st=q, oce_deriv_st=oce_deriv_st,
                       u_deriv_st=u_deriv_st)
        return out

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
class VizDRNConfig:
    plot_prior_alpha : float = 0.025
    cost_act_std     : float = 2.
    seed             : int = 1231231 # seed for chosing the above

    def __post_init__(self):
        pass

class DRNVisualizer(object):

    def __init__(self, universe, 
                       total_epochs    : int,*,
                       num_trn_samples : int,
                       num_vld_samples : int,
                       col_size        : int = 6,
                       row_size        : int = 4,
                       ref_dQdP        : np.ndarray|None = None, 
                       viz_config      : VizDRNConfig, 
                       mdl_config      : DRNConfig ):
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
        self.xinst         = np.arange( self.C.shape[2]*self.C.shape[3]+1 )
        self.ref_dQdP      = ref_dQdP # full sample ref
        self.num_steps     = self.dS.shape[1]
        self.ref_trn_dQdP  = ref_dQdP[:self.num_trn_samples] if not ref_dQdP is None else None
        self.ref_vld_dQdP  = ref_dQdP[-self.num_vld_samples:] if not ref_dQdP is None and self.num_vld_samples > 0 else None 

        assert ref_dQdP is None or isinstance(ref_dQdP, np.ndarray), f"Expected ref_dQdP to be a numpy array or None, but got {type(ref_dQdP)}"

        # learning & fit
        # --------------

        # loss
        # ----
        self.fig_lrn   = figure(mdl_config.label, col_size=col_size, row_size=row_size, columns=4) #
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

        self.ax_pnl    = self.fig_lrn.add_subplot("P&L" + (" (with cost)" if self.mdl_config.has_some_cost else ""))
        self.ax_pnl.set_xlabel("Epoch")
        self.ax_pnl.set_ylabel("P&L")

        # actions (magnitude)
        # --------------------

        self.ax_actions = self.fig_lrn.add_subplot("Actions")
        self.ax_actions.set_ylabel("Action Magnitude")
        self.ax_actions.set_xlabel("Epoch")

        # density
        # -------

        self.fig_lrn.next_row()
        self.ax_dQdP = self.fig_lrn.add_subplot(r"$dQ/dP$")
        self.ax_dQdP.set_xlabel("Samples")
        self.ax_dQdP.set_ylabel("Density ratio")

        # pnl per instrument
        # ------------------

        self.ax_pnl_per_inst = self.fig_lrn.add_subplot(r"P&L per instrument")
        self.ax_pnl_per_inst.set_xlabel("P&L")
        self.ax_pnl_per_inst.set_ylabel("Mean and 2 Std P&L per instrument")
        self.ax_pnl_per_inst.plot(self.xinst, self.xinst*0., ":", color="black")

        # actions (inst)
        # --------------

        self.ax_norm_actions_inst = self.fig_lrn.add_subplot(r"Normalized E[a]/$\epsilon[a]$ per instrument")
        self.ax_norm_actions_inst.set_ylabel(r"E[a]/$\epsilon[a]$")
        self.ax_norm_actions_inst.set_xlabel("Instrument")
        self.ax_norm_actions_inst.plot(self.xinst, self.xinst*0.+2., ":", color="black")
        self.ax_norm_actions_inst.plot(self.xinst, self.xinst*0.-2., ":", color="black")
        self.ax_norm_actions_inst.plot(self.xinst, self.xinst*0., "--", color="black")

        self.ax_actions_inst = self.fig_lrn.add_subplot(r"Actions per instrument")
        self.ax_actions_inst.set_ylabel("Action and MC error")
        self.ax_actions_inst.set_xlabel("Instrument")
        self.ax_actions_inst.plot(self.xinst, self.xinst*0., "--", color="black")

        self.fig_lrn.render()

        # inst analysis
        # -------------

        self.fig_lrn_inst   = figure(f"Instrument Analysis {mdl_config.label} using " + r"$\mathbb{Q}$", fig_size=(col_size*4, int(row_size*1.4)), columns=None)

        self.ax_inst_cost = self.fig_lrn_inst.add_subplot(r"Unconditional Drift and Cost per instrument")
        self.ax_inst_cost.set_ylabel("Normalized")
        self.ax_inst_cost.set_xlabel("Instrument")
        self.ax_inst_cost.plot(self.xinst, self.xinst*0.-2, ":", color="black")
        self.ax_inst_cost.plot(self.xinst, self.xinst*0.+2, ":", color="black")

        self.ax_inst_mean_dev = self.fig_lrn_inst.add_subplot(r"Mean absolute drift relative to MC error per instrument")
        self.ax_inst_mean_dev.set_ylabel(r"Mean magnitude / $\epsilon[dH]$")
        self.ax_inst_mean_dev.set_xlabel("Instrument")
        self.ax_inst_mean_dev.plot(self.xinst, self.xinst*0., "--", color="black")

        self.fig_lrn_inst.render()

        # trn
        # ---

        self.fig_lrn_tau   = figure(f"Term Analysis {mdl_config.label} using " + r"$\mathbb{Q}$", col_size=col_size, row_size=row_size, columns=4)

        def make_drift(trn:bool):
            rdS, rdC       = self.compute_devs( None, trn=trn )
            ref_rdS,\
            ref_rdC        = self.compute_devs( self.ref_trn_dQdP if trn else self.ref_vld_dQdP, trn=trn ) if ref_dQdP is not None else (None, None)
            cost_str       = f" after cost {self.mdl_config.cost:.4g}" if self.mdl_config.cost > 0. else ""

            r = PrettyObject()
            # dS/err
            r.ax_dev_dS = self.fig_lrn_tau.add_subplot(r"$E[dS]/\epsilon[dS]$ " + ('(training)' if trn else '(validation)') + cost_str)
            r.ax_dev_dS.plot( self.days, rdS.dev_raw*0.+2., ":", color="black")
            r.ax_dev_dS.plot( self.days, rdS.dev_raw*0.-2, ":", color="black")
            r.ax_dev_dS.plot( self.days, rdS.dev_raw, "--", color="blue", label=r"Market E[dS]/$\epsilon[dS]$")
            r.ax_dev_dS.plot( self.days, rdS.dev_wcost, "--", color="lightblue", label=r"Market E[[dS-Cost]]/$\epsilon[dS]$")
            r.ax_dev_dS.plot( self.days, ref_rdS.dev_raw, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            r.ax_dev_dS.set_xlabel("Days")
            r.ax_dev_dS.set_ylabel("Drift/MC error")
            r.ref_dev_dS_min_max = np.quantile( ref_rdS.dev_raw, q=[0.01, 0.99] ) if not ref_rdS is None else None

            # dC/err
            r.ax_dev_dC = self.fig_lrn_tau.add_subplot(r"$E[dC]/\epsilon[dC]$ with quantile bands across options" + ('(training)' if trn else '(validation)') + cost_str)
            r.ax_dev_dC.plot( self.days, rdC.dev_raw.mean*0.+2., ":", color="black")
            r.ax_dev_dC.plot( self.days, rdC.dev_raw.mean*0.-2, ":", color="black")

            r.ax_dev_dC.plot( self.days, rdC.dev_raw.mean, "--", color="blue", label=r"Market E[dC]/$\epsilon[dC]$")
            r.ax_dev_dC.fill_between( self.days, rdC.dev_raw.q05, rdC.dev_raw.q95, color="blue", alpha=0.1 )
            r.ax_dev_dC.plot( self.days, rdC.dev_wcost.mean, "--", color="lightblue", label=r"Market E[<dC-Cost>]/$\epsilon[dC]$ ")
            r.ax_dev_dC.fill_between( self.days, rdC.dev_wcost.q05, rdC.dev_wcost.q95, color="lightblue", alpha=0.1 )

            r.ax_dev_dC.plot( self.days, ref_rdC.dev_raw.mean, "*:", color="orange", label="Prior") if not ref_rdC is None else None
            if self.viz_config.plot_prior_alpha > 0.:
                r.ax_dev_dC.fill_between( self.days, ref_rdC.dev_raw.q05, ref_rdC.dev_raw.q95, color="orange", alpha=self.viz_config.plot_prior_alpha ) if not ref_rdC is None else None
            r.ax_dev_dC.set_xlabel("Days")
            r.ax_dev_dC.set_ylabel("Drift/MC error (5%/95% quantiles across options)")
            r.ref_dev_dC_min_max = ( np.min(ref_rdC.dev_raw.q05), np.max(ref_rdC.dev_raw.q95) ) if not ref_rdC is None else None

            # dS
            r.ax_mu_dS = self.fig_lrn_tau.add_subplot(r"$E[dS]$ with $\epsilon[dS]$ bands " + ('(training)' if trn else '(validation)') + cost_str)
            r.ax_mu_dS.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dS.plot( self.days, rdS.drift, "--", color="blue", label="Market E[dS]")
            r.ax_mu_dS.plot( self.days, rdS.excess, ":", color="blue", label="Market E[[dS-cost]]")
            r.ax_mu_dS.fill_between( self.days, rdS.drift-2.*rdS.err, rdS.drift+2.*rdS.err, color="blue", alpha=0.1 )

            r.ax_mu_dS.plot( self.days, ref_rdS.drift, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            if self.viz_config.plot_prior_alpha > 0.:
                r.ax_mu_dS.fill_between( self.days, ref_rdS.drift-2.*ref_rdS.err, ref_rdS.drift+2.*ref_rdS.err, color="orange", alpha=self.viz_config.plot_prior_alpha ) if not ref_rdS is None else None
            r.ax_mu_dS.set_xlabel("Days")
            r.ax_mu_dS.set_ylabel(r"Drift and $\pm$2 MC error bands")
            r.ref_mu_dS_min_max = np.quantile( ref_rdS.drift, q=[0.01, 0.99] ) if not ref_rdS is None else None

            r.ax_mu_dC = self.fig_lrn_tau.add_subplot(r"$\mathrm{Mean}\,E[dC]$ with $\mathrm{Mean}\,\epsilon[dC]$ bands " + ('(training)' if trn else '(validation)') + cost_str)
            r.ax_mu_dC.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dC.plot( self.days, rdC.mean_drift, "--", color="blue", label="Market E[dC]")
            r.ax_mu_dC.plot( self.days, rdC.mean_excess, ":", color="blue", label="Market E[[dC-cost]]")
            r.ax_mu_dC.fill_between( self.days, rdC.mean_drift-2.*rdC.mean_err, rdC.mean_drift+2.*rdC.mean_err, color="blue", alpha=0.1 )
            if not ref_rdC is None:
                r.ax_mu_dC.plot( self.days, ref_rdC.mean_drift, "*:", color="orange", label="Prior") 
                if self.viz_config.plot_prior_alpha > 0.:
                    r.ax_mu_dC.fill_between( self.days, ref_rdC.mean_drift-2.*ref_rdC.mean_err, ref_rdC.mean_drift+2.*ref_rdC.mean_err, color="orange", alpha=self.viz_config.plot_prior_alpha )
                r.ref_mu_dC_min_max = np.quantile( ref_rdC.mean_drift, q=[0.01, 0.91] ) 
            else:
                r.ref_mu_dC_min_max = None
            r.ax_mu_dC.set_xlabel("Days")
            r.ax_mu_dC.set_ylabel(r"Drift and $\pm$2 MC error bands")
            return r

        self.trn_info = make_drift(trn=True)
        self.vld_info = make_drift(trn=False) if self.num_vld_samples > 0 else None

        self.fig_lrn_tau.render()

    def compute_devs( self, dQdS : np.ndarray|None, trn : bool ) -> tuple[PrettyObject, PrettyObject]:
        """
        TODO: implement relative and delta hedged
        """
        S   = self.num_trn_samples if trn else self.num_vld_samples
        TAU = self.num_steps
        if not dQdS is None:
            assert dQdS.shape == (S,), f"Expected dQdS to have shape {(S,)} but got {dQdS.shape}"
        else:
            dQdS = np.ones( (S,), dtype=self.dS.dtype )

        dS        = self.dS[:self.num_trn_samples] if trn else self.dS[self.num_trn_samples:]
        dC        = self.dC[:self.num_trn_samples,...] if trn else self.dC[self.num_trn_samples:,...]
        spot      = self.S[:self.num_trn_samples] if trn else self.S[self.num_trn_samples:]
        C         = self.C[:self.num_trn_samples,...] if trn else self.C[self.num_trn_samples:,...]

        assert dS.shape == (S, TAU), f"Expected dS to have shape {(S, TAU)} but got {dS.shape}"
        assert len(dC.shape) == 4 and dC.shape[:2] == (S, TAU), f"Expected dC to have shape ({S}, {TAU}, ?, ?) but got {dC.shape}"
        T, N      = dC.shape[2:]
        assert spot.shape == (S, TAU), f"Expected spot to have shape {(S, TAU)} but got {spot.shape}"
        assert C.shape == (S, TAU, T, N), f"Expected C to have shape {(S, TAU, T, N)} but got {C.shape}"

        cost      = self.mdl_config.cost
        cost_std  = self.mdl_config.cost_std

        wdS        = dQdS[:,None]*dS
        wdC        = dQdS[:,None,None,None]*dC
        cost_n     = self.num_trn_samples * TAU
        S_cost     = cost * dQdS[:,None]*np.abs(spot) + cost_std * np.std( wdS ) / np.sqrt(cost_n)
        wSExcess   = np.clip( wdS - S_cost, min=0.) + np.clip( wdS + S_cost, max=0.)
        C_cost     = cost * dQdS[:,None,None,None]*np.abs(C) + cost_std * np.std( wdC, axis=(0,1) ) / np.sqrt(cost_n) 
        wCExcess   = np.clip( wdC - C_cost, min=0.) + np.clip( wdC + C_cost, max=0.)

        rdS_drift    = np.mean( wdS, axis=0 ) # (TAU,)

        rdS_cost     = np.mean( S_cost, axis=0 ) # (TAU,)
        rdS_excess   = np.mean( wSExcess, axis=0 ) # (TAU,)

        rdC_drift    = np.mean( wdC, axis=0 ) # (TAU,T,N)
        rdC_cost     = np.mean( C_cost, axis=0 ) # (TAU,T,N)
        rdC_excess   = np.mean( wCExcess, axis=0 ) # (TAU,T,N)
 
        rdS_err      = np.std(wdS, axis=0 ) / np.sqrt(dS.shape[0])
        rdC_err      = np.std(wdC, axis=0 ) / np.sqrt(dC.shape[0])
        rdS_xerr     = np.std(wSExcess, axis=0 ) / np.sqrt(dS.shape[0])
        rdC_xerr     = np.std(wCExcess, axis=0 ) / np.sqrt(dC.shape[0])

        # normalize

        eps_ = min(0.01/math.sqrt(255), 1E-7)
        rdS_dev_wcost = rdS_excess / (rdS_err + eps_) # with cost (TAU,)
        rdC_dev_wcost = rdC_excess / (rdC_err + eps_) # with cost (TAU,T,N)
        rdS_dev_raw   = rdS_drift / (rdS_err + eps_) # without cost (TAU,)
        rdC_dev_raw   = rdC_drift / (rdC_err + eps_) # without cost (TAU,T,N)

        rdS = PrettyObject()
        rdS.drift      = rdS_drift
        rdS.err        = rdS_err
        rdS.cost       = rdS_cost
        rdS.excess     = rdS_excess
        rdS.excess_std = rdS_xerr
        rdS.dev_wcost =  rdS_dev_wcost
        rdS.dev_raw   =  rdS_dev_raw
                                                                        
        rdC_dev_wcost = rdC_dev_wcost.reshape( rdC_dev_wcost.shape[0], -1 ) # (TAU,TN)
        rdC = PrettyObject()
        # non-dev
        rdC.mean_drift = np.mean( rdC_drift, axis=(1,2))
        rdC.mean_err   = np.mean( rdC_err, axis=(1,2))
        rdC.mean_cost  = np.mean( rdC_cost, axis=(1,2))
        rdC.mean_excess = np.mean( rdC_excess, axis=(1,2))
        rdC.mean_excess_err = np.mean( rdC_xerr, axis=(1,2))

        rdC.dev_wcost = PrettyObject()
        rdC.dev_wcost.mean    = np.mean( rdC_dev_wcost, axis=1 )
        rdC.dev_wcost.q05,\
        rdC.dev_wcost.median,\
        rdC.dev_wcost.q95      = np.quantile( rdC_dev_wcost, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.dev_wcost.min      = np.min( rdC_dev_wcost, axis=1 )
        rdC.dev_wcost.max      = np.max( rdC_dev_wcost, axis=1 )

        rdC_dev_raw   = rdC_dev_raw.reshape( rdC_dev_raw.shape[0], -1 )
        rdC.dev_raw = PrettyObject()
        rdC.dev_raw.mean    = np.mean( rdC_dev_raw, axis=1 )
        rdC.dev_raw.q05,\
        rdC.dev_raw.median,\
        rdC.dev_raw.q95  = np.quantile( rdC_dev_raw, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.dev_raw.min  = np.min( rdC_dev_raw, axis=1 )
        rdC.dev_raw.max  = np.max( rdC_dev_raw, axis=1 )

        return rdS, rdC

    def __call__(self, tracker : MetricTracker, 
                       trn_tracker : OutputTracker,
                       vld_tracker : OutputTracker, 
                       trn_model_output, 
                       vld_model_output, 
                       *, final : bool = False ):
        """
        Update
        """

        epoch     = tracker.trn_epochs[-1]
        self.store.clear()

        dQdP      = trn_model_output.get("dQdP", None)
        vld_dQdP  = vld_model_output["dQdP"]  if not dQdP is None and not vld_model_output is None else None
        assert dQdP is None or dQdP.shape[0] == self.num_trn_samples  , f"Expected dQdP to have shape {(self.num_trn_samples,)} but got {dQdP.shape}"
        assert vld_dQdP is None or vld_dQdP.shape[0] == self.num_vld_samples, f"Expected vld_dQdP to have shape {(self.num_vld_samples,)} but got {vld_dQdP.shape}"

        # losses
        # ------
        _CLIP = 1_000_000

        def mk(content : OutputTracker, num_samples : int) -> PrettyObject :
            content = content if isinstance(content, OutputTracker) else OutputTracker(**content)
            sqrt_total       = np.sqrt(num_samples * self.num_steps)

            r = PrettyObject()
            r.epochs     =  np.asarray(content.epochs)
            r.utility    =  np.clip( np.asarray(content.utility), min=-_CLIP, max=_CLIP )
            r.util_mean  =  np.clip( np.asarray(content.util_mean), min=-_CLIP, max=_CLIP )
            r.util_std   =  np.clip( np.asarray(content.util_std), min=-_CLIP, max=_CLIP )

            # pnl, aggregate
            r.pnl_mean     =  np.clip( np.asarray([np.sum(x) for x in np.asarray(content.pnl_mean)]), min=-_CLIP, max=_CLIP )
            r.pnl_std      =  np.clip( np.asarray([np.sum(x) for x in np.asarray(content.pnl_std)]), min=-_CLIP, max=_CLIP )
            r.raw_pnl_mean =  np.clip( np.asarray([np.sum(x) for x in np.asarray(content.raw_pnl_mean)]), min=-_CLIP, max=_CLIP )
            r.raw_pnl_std  =  np.clip( np.asarray([np.sum(x) for x in np.asarray(content.raw_pnl_std)]), min=-_CLIP, max=_CLIP )
            # Model outputs retain per-instrument action metrics.  Aggregate
            # only the portfolio-level time series displayed in this panel.
            r.action_l1    =  np.clip( np.asarray([
                np.mean(x) if np.ndim(x) else x for x in content.action_l1
            ]), min=-_CLIP, max=_CLIP )
            r.action_l2    =  np.clip( np.asarray([
                np.sqrt(np.mean(np.square(x))) if np.ndim(x) else x
                for x in content.action_l2
            ]), min=-_CLIP, max=_CLIP )
            r.action_max   =  np.clip( np.asarray([np.sum(x) for x in content.action_max]), min=-_CLIP, max=_CLIP )

            # last pnl, by inst
            r.inst_mean         =  np.clip( np.asarray(content.inst_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_std          =  np.clip( np.asarray(content.inst_std[-1]), min=-_CLIP, max=_CLIP )
            r.inst_cost         =  np.clip( np.asarray(content.cost_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_excess_mean  = np.clip( np.asarray(content.inst_excess_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_excess_std   = np.clip( np.asarray(content.inst_excess_std[-1]), min=-_CLIP, max=_CLIP )
            r.inst_excess2_mean = np.clip( np.asarray(content.inst_excess2_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_excess2_std  = np.clip( np.asarray(content.inst_excess2_std[-1]), min=-_CLIP, max=_CLIP )
            r.mean_dev          = np.clip( np.asarray(content.mean_dev[-1]), min=-_CLIP, max=_CLIP )
            r.mean_excess_dev   = np.clip( np.asarray(content.mean_excess_dev[-1]), min=-_CLIP, max=_CLIP )
            r.mean_excess2_dev  = np.clip( np.asarray(content.mean_excess2_dev[-1]), min=-_CLIP, max=_CLIP )

            r.inst_pnl_mean     =  np.clip( np.asarray(content.pnl_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_pnl_std      =  np.clip( np.asarray(content.pnl_std[-1]), min=-_CLIP, max=_CLIP )
            r.inst_raw_pnl_mean =  np.clip( np.asarray(content.raw_pnl_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_raw_pnl_std  =  np.clip( np.asarray(content.raw_pnl_std[-1]), min=-_CLIP, max=_CLIP )
            r.inst_action_mean  =  np.clip( np.asarray(content.action_mean[-1]), min=-_CLIP, max=_CLIP )
            r.inst_action_std   =  np.clip( np.asarray(content.action_std[-1]), min=-_CLIP, max=_CLIP )
            r.inst_action_l1    =  np.clip( np.asarray(content.action_l1[-1]), min=-_CLIP, max=_CLIP )
            r.inst_action_l2    =  np.clip( np.asarray(content.action_l2[-1]), min=-_CLIP, max=_CLIP )
            r.inst_action_max   =  np.clip( np.asarray(content.action_max[-1]), min=-_CLIP, max=_CLIP )

            # actions
            r.action_cap   =  np.clip( np.asarray(content.action_cap), min=-_CLIP, max=_CLIP ) 

            r.util_err         = r.util_std / sqrt_total
            r.pnl_err          = r.pnl_std / sqrt_total
            r.raw_pnl_err      = r.raw_pnl_std / sqrt_total
            r.inst_action_err  = r.inst_action_std / sqrt_total
            r.inst_err         = r.inst_std / sqrt_total
            r.inst_excess_err   = r.inst_excess_std / sqrt_total
            r.inst_excess2_err  = r.inst_excess2_std / sqrt_total
            r.inst_pnl_err     = r.inst_pnl_std / sqrt_total
            r.inst_raw_pnl_err = r.inst_raw_pnl_std / sqrt_total
            return r

        def minmax( value, err=0., mom = None ):
            err = float(err)
            if len(value) < 10:
                mom_ = np.min(value)-err, np.max(value)+err
            else:
                mom_ = min( np.quantile(value, 0.1), np.min(value[-5:])-err ), max( np.quantile(value, 0.9), np.max( value[-5:] )+err )
            if not mom is None:
                mom_ = min(mom_[0], mom[0]), max(mom_[1], mom[1])
            return mom_

        util          = mk(trn_tracker, self.num_trn_samples)
        vld_util      = mk(vld_tracker, self.num_vld_samples) if vld_tracker is not None else None
        has_some_cost = self.mdl_config.has_some_cost

        def losses():#
            trn_epochs =  np.asarray(tracker.trn_epochs)
            trn_losses =  np.clip( np.asarray(tracker.trn_losses), min=-_CLIP, max=_CLIP ) 
            val_epochs =  np.asarray(tracker.val_epochs)
            val_losses =  np.clip( np.asarray(tracker.val_losses) , min=-_CLIP, max=_CLIP ) 
            vld_epochs =  np.asarray(tracker.vld_epochs) if self.num_vld_samples > 0 else None
            vld_losses =  np.clip( np.asarray(tracker.vld_losses), min=-_CLIP, max=_CLIP ) if self.num_vld_samples > 0 else None

            str_vld_loss = "" if vld_losses is None else f", validation loss {vld_losses[-1]:.4g} [{vld_epochs[-1]}]"

            self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g}, "+\
                                  f"valuation loss {val_losses[-1]:.4g}{str_vld_loss}.\n"+\
                                  f"Epoch {trn_epochs[-1]}/{self.total_epochs}{' (Done)' if final else ''}.")

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

        def utilities():
            str_util = f"{util.utility[-1]:.3g} (mean {util.util_mean[-1]:.3g}, err {util.util_err[-1]:.3g})"
            
            self.store += self.ax_util.plot(util.epochs, util.epochs*0., ":", color="black")
            self.store += self.ax_util.plot(util.epochs, util.utility, ".-", color="red", label="Trn Rbst Utility")
            self.store += self.ax_util.plot(util.epochs, util.util_mean, "-", color="orange", label="Trn Mean Utility")
            self.store += self.ax_util.fill_between(util.epochs, util.util_mean-2.*util.util_err, util.util_mean+2.*util.util_err, alpha=0.1, color="orange")
            
            mom = minmax(util.utility, util.util_err[-1])

            if vld_util is not None:
                str_util = f"{str_util}\n{vld_util.utility[-1]:.3g} (mean {vld_util.util_mean[-1]:.3g}, err {vld_util.util_err[-1]:.3g})"

                self.store += self.ax_util.plot(vld_util.epochs, vld_util.utility, "--", color="green", label="Vld Rbst Utility")
                self.store += self.ax_util.plot(vld_util.epochs, vld_util.util_mean, "--", color="lightgreen", label="Vld Mean Utility")
                self.store += self.ax_util.fill_between(vld_util.epochs, vld_util.util_mean-2.*vld_util.util_err[-1], vld_util.util_mean+2.*vld_util.util_err[-1], alpha=0.1, color="lightgreen")
                mom = minmax(vld_util.utility, vld_util.util_err[-1], mom)

            self.ax_util.set_ylim( *mom )
            self.ax_util.set_title(f"Utilities\n{str_util}")
            if self.first:
                self.ax_util.legend(loc="upper left")
        utilities()

        # pnl
        # ---

        def pnl():
            self.store += self.ax_pnl.plot(util.epochs, util.pnl_mean*0., ":", color="black")
            self.store += self.ax_pnl.plot(util.epochs, util.raw_pnl_mean, ".-", color="orange" if has_some_cost else "red", label="Trn P&L" if not has_some_cost else "Trn without Cost")
            self.store += self.ax_pnl.fill_between(util.epochs, util.raw_pnl_mean-2.*util.raw_pnl_err, util.raw_pnl_mean+2.*util.raw_pnl_err, alpha=0.1, color="orange" if has_some_cost else "red")
            mom_pnl = minmax(util.raw_pnl_mean, util.raw_pnl_err[-1])
            str_util_pnl = f"{util.raw_pnl_mean[-1]:.3g} (err {util.raw_pnl_err[-1]:.3g})"

            if self.mdl_config.has_some_cost:
                self.store += self.ax_pnl.plot(util.epochs, util.pnl_mean, ".-", color="red", label="Trn with Cost")
                self.store += self.ax_pnl.fill_between(util.epochs, util.pnl_mean-2.*util.pnl_err, util.pnl_mean+2.*util.pnl_err, alpha=0.1, color="red")
                mom_pnl = minmax(util.pnl_mean, util.pnl_err[-1])
                str_util_pnl = f"{util.raw_pnl_mean[-1]:.3g} (err {util.raw_pnl_err[-1]:.3g}) [raw: {str_util_pnl}]"

            if vld_util is not None:
                # pnl
                # ---
                self.store += self.ax_pnl.plot(vld_util.epochs, vld_util.raw_pnl_mean, "--", color="lightgreen" if has_some_cost else "green",  label="Vld P&L" if not has_some_cost else "Vld without Cost")
                self.store += self.ax_pnl.fill_between(vld_util.epochs, vld_util.raw_pnl_mean-2.*vld_util.raw_pnl_err[-1], vld_util.raw_pnl_mean+2.*vld_util.raw_pnl_err[-1], alpha=0.1, color="lightgreen" if has_some_cost else "green")
                mom_pnl = minmax(vld_util.raw_pnl_mean, vld_util.raw_pnl_err[-1], mom_pnl)

                vld_str_util_pnl = f"{vld_util.raw_pnl_mean[-1]:.3g} (err {vld_util.raw_pnl_err[-1]:.3g})" 

                if self.mdl_config.has_some_cost:
                    self.store += self.ax_pnl.plot(vld_util.epochs, vld_util.pnl_mean, ".-", color="green", label="Vld with Cost")
                    self.store += self.ax_pnl.fill_between(vld_util.epochs, vld_util.pnl_mean-2.*vld_util.pnl_err[-1], vld_util.pnl_mean+2.*vld_util.pnl_err[-1], alpha=0.1, color="green")
                    mom_pnl = minmax(vld_util.pnl_mean, vld_util.pnl_err[-1], mom_pnl)  
                    vld_str_util_pnl = f"{vld_util.pnl_mean[-1]:.3g} (err {vld_util.pnl_err[-1]:.3g}) [raw: {vld_str_util_pnl}]"

                str_util_pnl = f"{str_util_pnl}\n{vld_str_util_pnl}"
                del vld_str_util_pnl

            self.ax_pnl.set_ylim( *mom_pnl )
            self.ax_pnl.set_title(f"P&L\n{str_util_pnl}")
            if self.first:
                self.ax_pnl.legend(loc="upper left")
        pnl()

        # action (aggregate)
        # ------------------

        def epoch_actions():

            self.store += self.ax_actions.plot(util.epochs, util.action_l1*0., ":", color="black")
            self.store += self.ax_actions.plot(util.epochs, util.action_l1, ".-", color="blue", label="Trn Action L1")
            self.store += self.ax_actions.plot(util.epochs, util.action_l2, ".-", color="green", label="Trn Action L2")
            self.store += self.ax_actions.plot(util.epochs, util.action_max, ".-", color="red", label="Trn Action Max")
            self.store += self.ax_actions.plot(util.epochs, util.action_cap, "*-", color="black", label="Trn Action Cap")
            mom_action = minmax(util.action_l1, 0.)
            mom_action = minmax(util.action_l2, 0., mom_action)
            mom_action = minmax(util.action_max, 0., mom_action)
            mom_action = minmax(util.action_cap, 0., mom_action)

            if vld_util is not None:
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_l1, "--", color="lightblue", label="Vld Action L1")
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_l2, "--", color="lightgreen", label="Vld Action L2")
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_max, "--", color="orange", label="Vld Action Max")
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_cap, "--", color="grey", label="Vld Action Cap")
                mom_action = minmax(vld_util.action_l1, 0., mom_action)
                mom_action = minmax(vld_util.action_l2, 0., mom_action)
                mom_action = minmax(vld_util.action_max, 0., mom_action)
                mom_action = minmax(vld_util.action_cap, 0., mom_action)

            self.ax_actions.set_ylim( *mom_action )
            self.ax_actions.set_title(f"Action Metrics")
            if self.first:
                self.ax_actions.legend(loc="upper left")
        epoch_actions()
        
        # per instrument pnl
        # ------------------

        def pnl_per_inst():
            self.store += self.ax_pnl_per_inst.plot(self.xinst, util.inst_raw_pnl_mean, ".-", color="orange", label="Trn P&L" if not has_some_cost else "Trn without Cost")
            self.store += self.ax_pnl_per_inst.fill_between(self.xinst, util.inst_raw_pnl_mean-2.*util.inst_raw_pnl_err, util.inst_raw_pnl_mean+2.*util.inst_raw_pnl_err, alpha=0.1, color="orange")
            self.store += self.ax_pnl_per_inst.plot(self.xinst, util.inst_pnl_mean, ".-", color="red", label="Trn P&L" if not has_some_cost else "Trn with Cost")
            self.store += self.ax_pnl_per_inst.fill_between(self.xinst, util.inst_pnl_mean-2.*util.inst_pnl_err, util.inst_pnl_mean+2.*util.inst_pnl_err, alpha=0.1, color="red")

            min_ = np.min(util.inst_raw_pnl_mean-2.*util.inst_raw_pnl_err)
            max_ = np.max(util.inst_raw_pnl_mean+2.*util.inst_raw_pnl_err)
            min_ = min(min_, np.min(util.inst_pnl_mean-2.*util.inst_pnl_err))
            max_ = max(max_, np.max(util.inst_pnl_mean+2.*util.inst_pnl_err))

            if vld_util is not None:
                self.store += self.ax_pnl_per_inst.plot(self.xinst, vld_util.inst_raw_pnl_mean, ".-", color="lightgreen", label="Vld P&L" if not has_some_cost else "Vld without Cost")
                self.store += self.ax_pnl_per_inst.fill_between(self.xinst, vld_util.inst_raw_pnl_mean-2.*vld_util.inst_raw_pnl_err, vld_util.inst_raw_pnl_mean+2.*vld_util.inst_raw_pnl_err, alpha=0.1, color="lightgreen")
                self.store += self.ax_pnl_per_inst.plot(self.xinst, vld_util.inst_pnl_mean, ".-", color="green", label="Vld P&L" if not has_some_cost else "Vld with Cost")
                self.store += self.ax_pnl_per_inst.fill_between(self.xinst, vld_util.inst_pnl_mean-2.*vld_util.inst_pnl_err, vld_util.inst_pnl_mean+2.*vld_util.inst_pnl_err, alpha=0.1, color="green")
                min_ = min(min_, np.min(vld_util.inst_raw_pnl_mean-2.*vld_util.inst_raw_pnl_err))
                max_ = max(max_, np.max(vld_util.inst_raw_pnl_mean+2.*vld_util.inst_raw_pnl_err))
                min_ = min(min_, np.min(vld_util.inst_pnl_mean-2.*vld_util.inst_pnl_err))
                max_ = max(max_, np.max(vld_util.inst_pnl_mean+2.*vld_util.inst_pnl_err))

            self.ax_pnl_per_inst.set_ylim( min_, max_ )
            if self.first:
                self.ax_pnl_per_inst.legend(loc="upper left")
        pnl_per_inst()

        # per instrument pnl
        # ------------------

        def action_per_inst():
            # relative
            action_by_err = util.inst_action_mean / np.where(util.inst_action_err > 0., util.inst_action_err, 1.)
            self.store += self.ax_norm_actions_inst.plot(self.xinst, action_by_err, ".-", color="red", label="Training" if vld_util is not None else None)

            min_ = min( np.min(action_by_err), -3 )
            max_ = max( np.max(action_by_err), 3 )

            if vld_util is not None:
                action_by_err = vld_util.inst_action_mean / np.where(vld_util.inst_action_err > 0., vld_util.inst_action_err, 1.)
                self.store += self.ax_norm_actions_inst.plot(self.xinst, action_by_err, ".-", color="green", label="Validation")
                min_ = min( np.min(action_by_err), min_ )
                max_ = max( np.max(action_by_err), max_ )

            min_ = max( min_, -10, )
            max_ = min( max_, 10 )

            self.ax_norm_actions_inst.set_ylim( min_, max_ )
            if self.first and vld_util is not None:
                self.ax_norm_actions_inst.legend(loc="upper left")

            # ax_inst_cost

            rel_mean = util.inst_mean / ( util.inst_err + 1E-7 )
            rel_cost = util.inst_cost / ( util.inst_err + 1E-7 )
            rel_excess = util.inst_excess_mean / ( util.inst_err + 1E-7 )
            rel_excess2 = util.inst_excess2_mean / ( util.inst_err + 1E-7 )

            trlb = "Trn " if not vld_util is None else ""
            self.store += self.ax_inst_cost.plot(self.xinst, rel_mean, "o:", color="red", label=rf"{trlb}E[dH]/$\epsilon[dH]$")
            self.store += self.ax_inst_cost.plot(self.xinst, rel_cost, ".:", color="pink", label=rf"{trlb}E[cost]/$\epsilon[dH]$")
            self.store += self.ax_inst_cost.plot(self.xinst, rel_excess, "o:", color="purple", label=rf"{trlb}E[[dH-cost]]/$\epsilon[dH]$")
            self.store += self.ax_inst_cost.plot(self.xinst, rel_excess2, "o:", color="orange", label=rf"{trlb}E[[dH-2cost]]/$\epsilon[dH]$")

            min_ = min( np.min(rel_mean), np.min(rel_excess), np.min(rel_excess2), -3 )
            max_ = max( np.max(rel_mean), np.max(rel_excess), np.max(rel_excess2), 3 )

            if vld_util is not None:
                rel_mean = vld_util.inst_mean / ( vld_util.inst_err + 1E-7 )
                rel_cost = vld_util.inst_cost / ( vld_util.inst_err + 1E-7 )
                rel_excess = vld_util.inst_excess_mean / ( vld_util.inst_err + 1E-7 )
                rel_excess2 = vld_util.inst_excess2_mean / ( vld_util.inst_err + 1E-7 )
                self.store += self.ax_inst_cost.plot(self.xinst, rel_mean, "-", color="blue", label=rf"Vld E[dH]/$\epsilon[dH]$")
                self.store += self.ax_inst_cost.plot(self.xinst, rel_cost, ":", color="lightgreen", label=rf"Vld E[cost]/$\epsilon[dH]$")
                self.store += self.ax_inst_cost.plot(self.xinst, rel_excess, "-", color="green", label=rf"Vld E[[dH-cost]]/$\epsilon[dH]$")
                self.store += self.ax_inst_cost.plot(self.xinst, rel_excess2, "-", color="lightblue", label=rf"Vld E[[dH-2cost]]/$\epsilon[dH]$")
                min_ = min( np.min(rel_mean), np.min(rel_excess), np.min(rel_excess2), min_ )
                max_ = max( np.max(rel_mean), np.max(rel_excess), np.max(rel_excess2), max_ )

            self.ax_inst_cost.set_ylim( min_, max_ )
            if self.first:
                self.ax_inst_cost.legend(loc="upper left")

            self.store += self.ax_inst_mean_dev.plot(self.xinst, util.mean_dev, "o:", color="red", label=rf"{trlb}Mean |dH|/$\epsilon[dH]$")
            self.store += self.ax_inst_mean_dev.plot(self.xinst, util.mean_excess_dev, "o:", color="purple", label=rf"{trlb}Mean max(|dH|-cost,0)/$\epsilon[dH]$")
            self.store += self.ax_inst_mean_dev.plot(self.xinst, util.mean_excess2_dev, "o:", color="orange", label=rf"{trlb}Mean max(|dH|-2cost,0)/$\epsilon[dH]$")
            min_dev = min(np.min(util.mean_dev), np.min(util.mean_excess_dev), np.min(util.mean_excess2_dev), 0.)
            max_dev = max(np.max(util.mean_dev), np.max(util.mean_excess_dev), np.max(util.mean_excess2_dev))
            if vld_util is not None:
                self.store += self.ax_inst_mean_dev.plot(self.xinst, vld_util.mean_dev, "-", color="blue", label=rf"Vld Mean |dH|/$\epsilon[dH]$")
                self.store += self.ax_inst_mean_dev.plot(self.xinst, vld_util.mean_excess_dev, "-", color="green", label=rf"Vld Mean max(|dH|-cost,0)/$\epsilon[dH]$")
                self.store += self.ax_inst_mean_dev.plot(self.xinst, vld_util.mean_excess2_dev, "-", color="lightblue", label=rf"Vld Mean max(|dH|-2cost,0)/$\epsilon[dH]$")
                min_dev = min(min_dev, np.min(vld_util.mean_dev), np.min(vld_util.mean_excess_dev), np.min(vld_util.mean_excess2_dev))
                max_dev = max(max_dev, np.max(vld_util.mean_dev), np.max(vld_util.mean_excess_dev), np.max(vld_util.mean_excess2_dev))
            self.ax_inst_mean_dev.set_ylim(min_dev, max_dev*1.05 if max_dev > 0. else 1.)
            if self.first:
                self.ax_inst_mean_dev.legend(loc="upper left")

            # relative
            self.store += self.ax_actions_inst.plot(self.xinst, util.inst_action_mean, ".-", color="red", label="Training" if vld_util is not None else None)
            self.store += self.ax_actions_inst.fill_between(self.xinst, util.inst_action_mean-2.*util.inst_action_err, util.inst_action_mean+2.*util.inst_action_err, alpha=0.1, color="red")

            min_ = min( np.min(util.inst_action_mean-2.*util.inst_action_err), -3 )
            max_ = max( np.max(util.inst_action_mean+2.*util.inst_action_err), 3 )

            if vld_util is not None:
                self.store += self.ax_actions_inst.plot(self.xinst, vld_util.inst_action_mean, ".-", color="green", label="Validation")
                self.store += self.ax_actions_inst.fill_between(self.xinst, vld_util.inst_action_mean-2.*vld_util.inst_action_err, vld_util.inst_action_mean+2.*vld_util.inst_action_err, alpha=0.1, color="green")
                min_ = min( np.min(vld_util.inst_action_mean-2.*vld_util.inst_action_err), min_ )
                max_ = max( np.max(vld_util.inst_action_mean+2.*vld_util.inst_action_err), max_ )

            self.ax_actions_inst.set_ylim( min_, max_ )
            if self.first and vld_util is not None:
                self.ax_actions_inst.legend(loc="upper left")
        action_per_inst()

        # dQ/dP
        # -----

        def density():
            self.store += self.ax_dQdP.plot( self.xdensity[:self.num_trn_samples], dQdP, "-", color="red", label=f"Trn Model mean {np.mean(dQdP):.4g}")
            if self.num_vld_samples > 0:
                self.store += self.ax_dQdP.plot( self.xdensity[self.num_trn_samples:], vld_dQdP, "-", color="green", label=f"Vld Model mean {np.mean(vld_dQdP):.4g}")
            max_ = max( np.max(dQdP), np.max(vld_dQdP) if not vld_dQdP is None else 0. )

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
            has_cost = self.mdl_config.cost > 0. or self.mdl_config.cost_std > 0.
            rdS, rdC = self.compute_devs( dQdP if trn else vld_dQdP, trn=trn )
            r        = self.trn_info if trn else self.vld_info

            # dev = drift/std
            # ---------------
            # no cost 'raw'
            self.store += r.ax_dev_dS.plot( self.days, rdS.dev_raw, "-", color="red", label=rf"E[dS]/$\epsilon[dS]$")

            self.store += r.ax_dev_dC.plot( self.days, rdC.dev_raw.mean, ".:", color="red", label=rf"E[dC]/$\epsilon[dC]$")
            self.store += r.ax_dev_dC.fill_between( self.days, rdC.dev_raw.q05, rdC.dev_raw.q95, color="red", alpha=0.05 )
            self.store += r.ax_dev_dC.fill_between( self.days, rdC.dev_raw.min, rdC.dev_raw.max, color="red", alpha=0.05 )

            if has_cost:
                # cost 'wcost'
                self.store += r.ax_dev_dS.plot( self.days, rdS.dev_wcost, ".:", color="purple", label=rf"E[[dS-cost]]/$\epsilon[dS]$") 

                self.store += r.ax_dev_dC.plot( self.days, rdC.dev_wcost.mean, "-", color="purple", label=rf"E[[dC-cost]]/$\epsilon[dC]$") 
                self.store += r.ax_dev_dC.plot( self.days, rdC.dev_wcost.q05, ":", color="purple") 
                self.store += r.ax_dev_dC.plot( self.days, rdC.dev_wcost.q95, ":", color="purple") 

            r.ax_dev_dS.legend(loc="lower left")
            r.ax_dev_dC.legend(loc="lower left")
            min_dS = np.min(rdS.dev_raw) if r.ref_dev_dS_min_max is None else min( np.min(rdS.dev_raw), r.ref_dev_dS_min_max[0] )
            max_dS = np.max(rdS.dev_raw) if r.ref_dev_dS_min_max is None else max( np.max(rdS.dev_raw), r.ref_dev_dS_min_max[1] )
            ex_dS  = max( abs(min_dS), abs(max_dS) )
            min_dC = np.min(rdC.dev_raw.q05)
            max_dC = np.max(rdC.dev_raw.q95)
            ex_dC  = min( max( abs(min_dC), abs(max_dC) ), _CLIP )
            r.ax_dev_dS.set_ylim( -ex_dS*1.5-4, ex_dS*1.5+4 ) #-ex_dS*1.5-3, ex_dS*1.5+4 )
            r.ax_dev_dC.set_ylim( -ex_dC*1.5-4, ex_dC*1.5+4 )

            # drft
            # -----
            # drift

            self.store += r.ax_mu_dS.plot( self.days, rdS.drift, "-", color="red", label=f"E[dS]")
            self.store += r.ax_mu_dS.plot( self.days, rdS.excess, ".-", color="orange", label=f"E[[dS-Cost]]")
            self.store += r.ax_mu_dS.plot( self.days, rdS.cost, ":", color="green", label=f"E[Cost]")
            self.store += r.ax_mu_dS.fill_between( self.days, rdS.drift-2*rdS.err, rdS.drift+2*rdS.err, color="red", alpha=0.1 )
            self.store += r.ax_mu_dS.fill_between( self.days, rdS.excess-2*rdS.err, rdS.excess+2*rdS.err, color="orange", alpha=0.1 )

            self.store += r.ax_mu_dC.plot( self.days, rdC.mean_drift, "-", color="red", label=f"E[dC]")
            self.store += r.ax_mu_dC.plot( self.days, rdC.mean_excess, ".:", color="orange", label=f"E[[dS-Cost]]")
            self.store += r.ax_mu_dC.plot( self.days, rdC.mean_cost, ":", color="green", label=f"E[Cost]")
            self.store += r.ax_mu_dC.fill_between( self.days, rdC.mean_drift-2.*rdC.mean_err, rdC.mean_drift+2.*rdC.mean_err, color="red", alpha=0.1 )
            self.store += r.ax_mu_dC.fill_between( self.days, rdC.mean_excess-2.*rdC.mean_err, rdC.mean_excess+2.*rdC.mean_err, color="orange", alpha=0.1 )
            r.ax_mu_dS.legend(loc="lower left")
            r.ax_mu_dC.legend(loc="lower left")
            r.ax_mu_dS.set_ylim( -0.001, 0.001 )
            r.ax_mu_dC.set_ylim( -0.001, 0.001 )

        dev_drift(trn=True)
        if self.num_vld_samples > 0:
            dev_drift(trn=False)

        self.fig_lrn.render() if not final else self.fig_lrn.close()
        self.fig_lrn_tau.render() if not final else self.fig_lrn_tau.close()
        self.fig_lrn_inst.render() if not final else self.fig_lrn_inst.close()
        self.first = False

def train( universe, *,
           ref_dQdP      : np.ndarray|None = None,
           return_density_details : bool = False,
           mdl_config    : DRNConfig = DRNConfig(),   
           trn_config    : TrainConfig = TrainConfig(),
           viz_config    : VizDRNConfig = VizDRNConfig(),
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

    torch.manual_seed(mdl_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mdl_config.seed)
    # create model
    # ------------

    num_total_samples,\
    num_steps,\
    num_expiries,\
    num_strikes = universe["C_t"].shape
    num_trn_samples,\
    num_vld_samples = trn_config.get_trn_vld_samples(num_total_samples)
    num_xstate = universe['xstate'].shape[-1]
 
    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        model     = DRN(  env,
                       num_xstate   = num_xstate,
                       num_expiries = num_expiries,
                       num_strikes  = num_strikes,
                       num_steps    = num_steps,
                       cost_num_samples = num_trn_samples,
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

    path           = _cache_root(ext="pt") if mdl_config.group == "" else _cache_root(mdl_config.group,ext="pt")
    file_name      = unique_label48_8(mdl_config.label, as_file_name=True)
    cache_ffn      = path.full_file_name(file_name)
    start_epoch    = 0
    tracker        = MetricTracker()
    trn_tracker    = OutputTracker()
    vld_tracker    = OutputTracker() if num_vld_samples > 0 else None
    model_version  = model.version.unique_id48
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
            trn_tracker = OutputTracker(**state.pop("trn_tracker"))
            vld_tracker = OutputTracker(**state.pop("vld_tracker")) if num_vld_samples > 0 else None
            checkpoint_rng_state = state.pop("rng_state", None)
            start_epoch = tracker.trn_epochs[-1] + 1

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
            log_strikes = np.log( universe["strikes_t"] ),
            xstate = universe['xstate'][:,:-1,...],
        )

        assert ndata["spots"].shape == (num_total_samples, num_steps+1), f"Expected spots to have shape {(num_total_samples, num_steps+1)} but got {ndata['spots'].shape}"
        assert ndata["delta"].shape == (num_total_samples, num_steps, num_expiries, num_strikes), f"Expected delta to have shape {(num_total_samples, num_steps, num_expiries, num_strikes)} but got {ndata['delta'].shape}"  
        assert ndata["expiries"].shape == (1, 1, num_expiries), f"Expected expiries to have shape {(1, 1, num_expiries)} but got {ndata['expiries'].shape}"
        assert ndata["log_strikes"].shape == (num_total_samples, num_steps, num_expiries,  num_strikes), f"Expected log_strikes to have shape {(num_total_samples, num_steps, num_expiries,  num_strikes)} but got {ndata['log_strikes'].shape}"
        assert ndata["xstate"].shape == (num_total_samples, num_steps, num_xstate), f"Expected xstate to have shape {(num_total_samples, num_steps, num_xstate)} but got {ndata['xstate'].shape}"

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

        if num_vld_samples == 0:
            vld_tdata = None
            trn_tdata = tdata
        else:
            trn_tdata = { k: v[:num_trn_samples,...] if v.shape[0] > 1 else v for k, v in tdata.items() }
            vld_tdata = { k: v[num_trn_samples:,...] if v.shape[0] > 1 else v for k, v in tdata.items() }

        boot_strap_samples, boot_strap_steps = mdl_config.create_boot_strap( env, num_samples=num_trn_samples, num_steps=num_steps )
        assert boot_strap_samples.shape == (num_trn_samples,mdl_config.boot_strap), f"Expected boot_strap to have shape {(num_trn_samples,num_steps,mdl_config.boot_strap)} but got {boot_strap_samples.shape}"
        assert boot_strap_steps.shape == (num_steps,mdl_config.boot_strap), f"Expected boot_strap to have shape {(num_steps,mdl_config.boot_strap)} but got {boot_strap_steps.shape}"
        trn_tdata['boot_strap_samples'] = boot_strap_samples
        trn_tdata['boot_strap_steps'] = boot_strap_steps
        del boot_strap_samples, boot_strap_steps

        if num_vld_samples > 0:
            vld_boot_strap_samples, vld_boot_strap_steps = mdl_config.create_boot_strap( env, num_samples=num_vld_samples, num_steps=num_steps )
            vld_tdata['boot_strap_samples'] = vld_boot_strap_samples
            vld_tdata['boot_strap_steps'] = vld_boot_strap_steps
            del vld_boot_strap_samples, vld_boot_strap_steps
            
        verbose.write(lambda : f"\rCopied {len(tdata)} arrays with a total size {fmt_big_byte_number(size)} to {env}. This took {tme}.")
        del tdata

    # deterministic measure change
    # ----------------------------
    # Is actually a bit fishy!

    if not ref_dQdP is None:
        verify_inp(ref_dQdP.shape == (num_total_samples,), f"Expected ref_dQdP to have shape {(num_total_samples,)} but got {ref_dQdP.shape}" )
        assert abs( np.mean(ref_dQdP) - 1. ) < 1e-5, f"Expected mean of ref_dQdP to be 1, but got {np.mean(ref_dQdP)}"
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
            batch = {k: v[idx, ...] if (v.shape[0] > 1 and k not in {"boot_strap_steps", "expiries"}) else v for k, v in trn_tdata.items()}
            batch["boot_strap_samples"] = batch["boot_strap_samples"] / batch["boot_strap_samples"].mean(
                dim=0, keepdim=True).clamp_min(model.floor_P)
            return batch

    # compile
    # -------

    def gradient_step( batch ):
        loss = model( batch, training=True ) 
        model.verify( lambda : torch.isfinite(loss).all(), "Internal error: loss infinite" )
        return loss

    def valuation( all_data ):
        with torch.no_grad():
            return model(all_data, training=False,
                         return_density_details=return_density_details)

    def valuation_batches( src, num_src_samples ):
        """ Sampling valuation batches if necessary """
        if mdl_config.boot_mode == "outer" or trn_config.val_batch_size is None or trn_config.val_batch_size >= num_src_samples:
            r = valuation(src)
            r = { k: v.detach().cpu().numpy() for k, v in r.items() }
            return r

        with torch.no_grad():
            num_batches = (num_src_samples + trn_config.val_batch_size - 1) // trn_config.val_batch_size
            for ib in range(num_batches):
                start = ib * trn_config.val_batch_size
                end   = min((ib+1)*trn_config.val_batch_size, num_src_samples)  
                idx   = slice(start, end)
                ilen  = end - start
                batch = { k: v[idx,...] if v.shape[0] > 1 else v for k, v in src.items() }
                if ib == 0:
                    out = valuation(batch)
                    out = { k: v.detach().cpu().numpy() for k, v in out.items() }

                    for k in out.keys():
                        assert np.all( np.isfinite(out[k]) ), f"Expected all values in output '{k}' to be finite, but found non-finite values."
                        if k == "dQdP":
                            continue
                        if k == "action_max":
                            continue
                        out[k] = out[k] * float( ilen ) / float(num_src_samples)

                else:
                    out_ = valuation(batch)
                    out_ = { k: v.detach().cpu().numpy() for k, v in out_.items() }

                    for k in out.keys():
                        assert np.all( np.isfinite(out_[k]) ), f"Expected all values in output '{k}' to be finite, but found non-finite values."
                        if k == "dQdP":
                            out[k] = np.concatenate([out[k], out_[k]], axis=0)
                            continue
                        if k == "action_max":
                            out[k] = np.maximum( out[k], out_[k] )
                            continue
                        assert len(out[k].shape) in (0,1), f"Expected output '{k}' to have shape 0 or 1, but got {out[k].shape}"
                        out[k] = out[k] + out_[k] * float(ilen) / float(num_src_samples)

            return out

    if start_epoch < trn_config.num_epochs and trn_config.compile and mdl_config.boot_mode != "outer":
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
                            dynamic=True,
                            fullgraph=True,
                            disable=model.debug,
                        )( valuation )
            _ = valuation(trn_tdata) # triggers annoying recompilation warning

            if num_vld_samples > 0:
                verbose.write("validation... ", head=False, end='')
                _ = valuation(vld_tdata) # triggers annoying recompilation warning
            del _
            verbose.write(lambda : f"done; this took {tme}.", head=False ) 

    # Construction, data preparation, and compilation warm-up may consume
    # randomness. Restore afterwards so resumed sampling continues exactly at
    # the checkpoint position.
    _load_rng_state_dict(checkpoint_rng_state)

    # visualization & seralization
    # ----------------------------

    disp = DRNVisualizer( universe, ref_dQdP=ref_dQdP, total_epochs=trn_config.num_epochs, num_trn_samples=num_trn_samples, num_vld_samples=num_vld_samples, viz_config=viz_config, mdl_config=mdl_config )
    del trn_ld0, vld_ld0

    def viz_and_store(*, epoch: int, final : bool = False):
        """ call valuation and then visualize and store the model """
        model.eval()
        trn_model_output = valuation_batches( trn_tdata, num_trn_samples )
        vld_model_output = valuation_batches( vld_tdata, num_vld_samples ) if not vld_tdata is None else None

        assert len(tracker.val_epochs) == 0 or tracker.val_epochs[-1] <= epoch+1, f"Expected last validation epoch {tracker.val_epochs[-1]} to be less than or equal to current epoch {epoch+1}"
            
        def append_data( tracker, ld ):
            tracker.epochs.append(epoch+1)
            for k, v in ld.items():
                if k in ['dQdP', 'dQdP_st', 'oce_deriv_st', 'u_deriv_st',
                         'utility_b', 'outer_weights']:
                    continue
                v = float(v) if isinstance(v, (torch.Tensor, float)) and v.numel() in (0, 1) else v
                klst = getattr(tracker, k, None)
                if klst is None:
                    raise KeyError(f"Expected tracker to have attribute '{k}', but it does not. Available attributes: {sorted(vars(tracker))}")
                klst.append(v)

            if model.a_y_cap is not None:
                inst_cap = torch.nn.functional.softplus(model.a_y_cap).detach().cpu().numpy()[0]
                tracker.action_cap.append( inst_cap )


        tracker.val_losses.append(trn_model_output['loss'].item())
        tracker.val_epochs.append(epoch+1)
        append_data(trn_tracker, trn_model_output)

        if not vld_model_output is None:
            tracker.vld_losses.append(vld_model_output['loss'].item())
            tracker.vld_epochs.append(epoch+1)
            append_data(vld_tracker, vld_model_output)

        utility_b = trn_model_output["utility_b"]
        sigma_boot = utility_b.std()

        effective_boot_lmbda = model.effective_boot_lmbda(num_trn_samples)
        rho = effective_boot_lmbda * sigma_boot
        weights = trn_model_output["outer_weights"]
        ess = 1.0 / np.sum(weights**2)
        relative_ess = ess / len(weights)

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
            os.replace(tmp_name, cache_ffn)

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
