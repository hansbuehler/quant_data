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


@version("0.2", dependencies=[torch_utility])
@dataclass(frozen=True)
class DRNConfig:
    label           : str|Callable = ""               # Label; if empty use notebook name
    group           : str = "deephedging"             # Directory
    user_version    : str = "0.1"                     # User version; change to invalidate cache.
    utility         : str = "entropy"                 # Utility; see torch_utility.utilities
    lmbda           : float = 1.0                     # Risk aversion 
    delta_hedged    : bool = True                     # Whether to consider delta hedging options
    use_fixed       : bool = True
    in_config       : FeedForwardConfig|None = FeedForwardConfig()
    pf_config       : FeedForwardConfig|None = None
    cost            : float = 0.0001                  # cost for dual 
    cost_std        : float = 0.
    seed            : int = 1231231                   # seed
    batch_block_size: int|None = None
    tau_block_size  : int|None = None
    action_cap      : float|None = None
    learn_action_cap: str = "off"  # "off", "flat", "time", "inst", "all"
    cash_trades     : bool = False

    _floor_P : float = 1E-8
    _use_pcnt_time : bool = False

    @property
    def use_in(self) -> bool:
        return not self.in_config is None
    @property
    def use_pf(self) -> bool:
        return not self.pf_config is None
    @property
    def has_some_cost(self) -> bool:
        return (self.cost > 0.) or (self.cost_std > 0.)
 
    def __post_init__(self):
        verify_inp( self.utility in torch_utility.utilities, f"Expected utility to be one of {torch_utility.utilities} but got '{self.utility}'." ) 
        verify_inp( self.lmbda > 0., f"Expected lmbda to be positive but got {self.lmbda}." )
        verify_inp( self.cost >= 0., f"Expected cost to be positive or None but got {self.cost}." )
        verify_inp( self.use_fixed or self.use_in or self.use_pf, f"Expected at least one of 'use_fixed', 'use_in' or 'use_pf' to be True." )
        verify_inp( self.cost >= 0., f"Expected cost to be non-negative but got {self.cost}." )
        verify_inp( self.cost_std >= 0., f"Expected cost_std to be non-negative but got {self.cost_std}." )
        verify_inp( self.batch_block_size is None or self.batch_block_size > 0, f"Expected batch_block_size to be None or positive but got {self.batch_block_size}." )
        verify_inp( self.tau_block_size is None or self.tau_block_size > 0, f"Expected tau_block_size to be None or positive but got {self.tau_block_size}." )
        verify_inp( self.learn_action_cap in ["off", "flat", "time", "inst", "all"], f"Expected learn_action_cap to be one of ['off', 'flat', 'time', 'inst', 'all'] but got '{self.learn_action_cap}'." )
        verify_inp( self.action_cap is None or self.action_cap > 0., f"Expected action_cap to be None or positive but got {self.action_cap}." )

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


@dataclass
class OutputTracker:
    epochs   : list[int] = field(default_factory=list)
    loss  :list[float] = field(default_factory=list)
    utility  :list[float] = field(default_factory=list)
    util_std : list[float] = field(default_factory=list)
    pnl      : list[float] = field(default_factory=list)
    pnl_std  : list[float] = field(default_factory=list)
    raw_pnl    : list[float] = field(default_factory=list)
    raw_pnl_std: list[float] = field(default_factory=list)
    action_l1 : list[float] = field(default_factory=list)
    action_l2 : list[float] = field(default_factory=list)
    action_max: list[float] = field(default_factory=list)
    cap_max : list[float] = field(default_factory=list)
    cap_l1 : list[float] = field(default_factory=list)
    cap_l2 : list[float] = field(default_factory=list)

@version("0.1")
class DRN(BaseModel):
    def __init__(self, env : BaseEnvironment, *,
                       num_xstate  : int, 
                       num_expiries: int,
                       num_strikes : int,
                       num_steps   : int,
                       config : DRNConfig = DRNConfig() ):
        super().__init__(env)
        T, N                    = num_expiries, num_strikes
        self.config             = config
        self.num_xstate         = num_xstate
        self.num_in_features    = num_xstate + 1 + 1 + ( 1 if config._use_pcnt_time else 0 ) # xstate + expiry + logK + time_pcnt
        self.num_pf_features    = num_xstate + num_expiries + num_expiries*num_strikes + ( 1 if config._use_pcnt_time else 0 )
        self.num_instruments    = num_strikes*num_expiries + 1
        self.num
        self.u,\
        self.u_deriv,\
        self.u_label            = torch_utility(config.utility, config.lmbda)
        np.random.seed(config.seed)

        self.pcnt_time          = torch.linspace(0, 1, num_steps, dtype=env.fdtype, device=env.device, requires_grad=False)
        self.quantiles          = torch.tensor(config.quantiles, dtype=env.fdtype, device=env.device, requires_grad=False)
        self.cost               = float(config.cost)
        self.cost_std           = float(config.cost_std)
        self.floor_P            = float(config._floor_P)
        self.use_pcnt_time      = bool(config._use_pcnt_time)
        self.cash_trades        = bool(config.cash_trades)

        self.a_fx            = torch.nn.Parameter(torch.zeros( (num_steps,self.num_instruments), dtype=env.fdtype, device=env.device ), requires_grad=True) if config.use_fixed else None
        self.a_in            = FeedForward( env, num_input = self.num_in_features, num_output = 0, config = config.in_config ) if config.use_in else None
        self.a_pf            = FeedForward( env, num_input = self.num_pf_features, num_output = self.num_instruments, config = config.pf_config ) if config.use_pf else None
        self.y               = torch.nn.Parameter(torch.zeros( (1,) , dtype=env.fdtype, device=env.device ), requires_grad=True) 
        

        if self.config.learn_action_cap == "flat":
            learn_max_action_shape = (1,1)
        elif self.config.learn_action_cap == "time":
            learn_max_action_shape = (num_steps,1)
        elif self.config.learn_action_cap == "inst":
            learn_max_action_shape = (1,self.num_instruments)
        elif self.config.learn_action_cap == "all":
            learn_max_action_shape = (num_steps,self.num_instruments)
        else:
            verify_inp( self.config.learn_action_cap == "off", f"Expected learn_action_cap to be one of ['off', 'flat', 'time', 'inst', 'all'] but got '{self.config.learn_action_cap}'." )
            learn_max_action_shape = (1,1) if not self.config.action_cap is None else None

        action_cap             = float(config.action_cap) if not config.action_cap is None else 1.0
        self.lrn_action_cap    = torch.nn.Parameter(torch.full(learn_max_action_shape, action_cap, dtype=env.fdtype, device=env.device ),
                                                    requires_grad=self.config.learn_action_cap != "off") if not learn_max_action_shape is None else None

    @property
    def use_in(self) -> bool:
        return not self.a_in is None
    @property
    def use_pf(self) -> bool:
        return not self.a_pf is None

    def compute_action(self, *, 
                        S_slc : slice,
                        TAU_slc : slice,
                        iS : int,
                        iTAU : int,
                        log_strikes : torch.Tensor,
                        expiries : torch.Tensor,
                        xstate : torch.Tensor ) -> torch.Tensor:
        a  = self.a_fx[TAU_slc].reshape((1,iTAU,self.num_instruments)) if not self.a_fx is None else 0.
        S, TAU, T, N = log_strikes.shape#
        XS           = self.num_xstate

        if self.use_pf:
            pf_features = [ xstate[S_slc,TAU_slc,:],
                            expiries.repeat(iS,iTAU,1), # T
                            log_strikes[S_slc,TAU_slc,:,:].reshape((iS,iTAU,T*N)),
                    ]
            pf_features.append(self.pcnt_time[TAU_slc].reshape(1,iTAU,1).repeat(iS,1,1)) if self.use_pcnt_time else None
            pf_features     = torch.cat(pf_features, dim=-1)
            assert pf_features.shape == (iS,iTAU,self.num_pf_features), f"Expected pf features to have shape {(iS,iTAU,self.num_pf_features)} but got {pf_features.shape}"
            a  = ( a + self.a_pf(pf_features) ) if not self.a_pf is None else a

        if self.use_in:
            spot_ztau     = log_strikes[:,:,:1,:1]*0.

            expiries_tau  = expiries.reshape(1,1,T,1).repeat(iS,iTAU,1,N).reshape(iS,iTAU,T*N,1)
            assert expiries_tau.shape == (iS, iTAU, T*N, 1), f"Expected expiries to have shape {(iS, iTAU, T*N,1)} but got {expiries.shape}"
            expiries_tau = torch.cat( [spot_ztau, expiries_tau], dim=2 )
            assert expiries_tau.shape == (iS, iTAU, T*N+1, 1), f"Expected expiries to have shape {(iS, iTAU, T*N+1,1)} but got {expiries_tau.shape}"

            logK_tau      = log_strikes[S_slc,TAU_slc,:,:].reshape(iS, iTAU, T*N,1)
            assert logK_tau.shape == (iS, iTAU, T*N, 1), f"Expected logK_tau to have shape {(iS, iTAU, T*N,1)} but got {logK_tau.shape}"
            logK_tau      = torch.cat( [spot_ztau, logK_tau], dim=2 )
            assert logK_tau.shape == (iS, iTAU, T*N+1, 1), f"Expected logK_tau to have shape {(iS, iTAU, T*N+1,1)} but got {logK_tau.shape}"

            assert xstate.shape == (S, TAU, XS), f"Expected xstate to have shape {(S, TAU, XS)} but got {xstate.shape}"
            xstate_tau    = xstate[S_slc, TAU_slc, :].reshape(iS, iTAU,1,XS).repeat(1,1,T*N+1,1)
            assert xstate_tau.shape == (iS, iTAU, T*N+1, XS), f"Expected xstate_tau to have shape {(iS, iTAU, T*N+1,XS)} but got {xstate_tau.shape}"

            in_features = [ xstate_tau,   # T*N+1,XS
                            expiries_tau, # T*N+1,1 
                            logK_tau,     # T*N+1,1
                          ]
            in_features.append(self.pcnt_time[TAU_slc].reshape(1,iTAU,1,1).repeat(iS, 1, T*N+1, 1)) if self.use_pcnt_time else None
            in_features   = torch.cat( in_features, dim=-1 )    
            assert in_features.shape == (iS, iTAU, T*N+1, self.num_in_features), f"Expected IN features to have shape {(iS, iTAU, T*N+1, self.num_in_features)} but got {in_features.shape}"
            a  = ( a + self.a_in(in_features) ) if not self.a_in is None else a

        assert a.shape == (iS,iTAU,self.num_instruments) or a.shape == (1,iTAU,self.num_instruments), f"Expected action to have shape (1|{iS}, {iTAU}, {self.num_instruments}) but got {a.shape}"

        if not self.lrn_action_cap is None:
            max_action = self.lrn_action_cap[None,TAU_slc, :] if self.lrn_action_cap.shape[1] == TAU else self.lrn_action_cap[None, :, :] 
            a          = torch.tanh( a ) * max_action

        return a

    def __call__(self, data, *, training : bool = False) -> torch.Tensor|dict[str, torch.Tensor]:
        """ Forward pass of the model. """
        data              = dict(data)
        dC                = get(data, "dC", 4)
        S, TAU, T, N      = dC.shape
        assert S > 0, f"Expected S to be positive but got {S}"
        assert TAU > 0, f"Expected TAU to be positive but got {TAU}"
        assert T > 0, f"Expected T to be positive but got {T}"
        assert N > 0, f"Expected N to be positive but got {N}"
        assert T*N+1 == self.num_instruments, f"Number of instruments mismatch {T*N+1} != {self.num_instruments}"

        ref_dQdP          = pop(data, "ref_dQdP", (S,), optional=True) 
        use_cost          = (self.cost > 0.) or (self.cost_std > 0.)

        dC                = get(data, "dC", 4)
        XS                = self.num_xstate 
        C                 = get(data, "C", (S, TAU, T, N)) if use_cost else None
        spots             = get(data, "spots", (S, TAU+1))
        delta_t           = get(data, "delta", (S, TAU, T, N))
        log_strikes       = get(data, "log_strikes", (S,TAU,T,N))
        expiries          = get(data, "expiries", (1, 1, T))

        dS                = torch.diff( spots, axis=1 )
        xstate            = get(data, "xstate", (S, TAU, XS)) if (self.use_in or self.use_pf) else None
        zero4             = (log_strikes[:1,:1,:1,:1]*0.).detach()
        assert zero4.shape == (1, 1, 1, 1), f"Expected zero4 to have shape {(1, 1, 1, 1)} but got {zero4.shape}"

        # densities
        # ---------

        if not ref_dQdP is None:
            assert ref_dQdP.shape == (S,), f"Expected ref_dQdP to have shape {(S,)} but got {ref_dQdP.shape}"
            ref_dQdP         = ref_dQdP / torch.mean(ref_dQdP, dim=0, keepdim=True)
        else:
            ref_dQdP         = zero4[:,0,0,0].repeat(S) + 1.

        # Internal batching
        # -----------------

        batch_size        = self.config.batch_block_size if not self.config.batch_block_size is None else S
        tau_size          = self.config.tau_block_size if not self.config.tau_block_size is None else TAU
        num_batch_blocks  = (S + batch_size - 1) // batch_size
        num_tau_blocks    = (TAU + tau_size - 1) // tau_size
        pnl               = None
        raw_pnl           = None
        sum_abs_action    = None
        sum_sqr_action    = None
        action_max        = None

        # features

        for tb in range(num_tau_blocks):
            TAU_start = tb * tau_size
            TAU_end   = min( (tb+1) * tau_size, TAU )
            TAU_slc   = slice(TAU_start, TAU_end) if TAU_start > 0 or TAU_end < TAU else slice(None)
            iTAU      = TAU_end - TAU_start
            TAU_start = None
            TAU_end   = None

            a     = []
            X     = []
            dX    = []

            for iblock in range(num_batch_blocks):
                S_start = iblock * batch_size
                S_end   = min( (iblock+1) * batch_size, S )
                S_slc   = slice(S_start, S_end) if S_start > 0 or S_end < S else slice(None)
                iS      = S_end - S_start
                S_start = None
                S_end   = None

                a_s          = self.compute_action( S_slc=S_slc, TAU_slc=TAU_slc, iS=iS, iTAU=iTAU, log_strikes=log_strikes, expiries=expiries, xstate=xstate )
                assert len(a_s.shape) == 3 and a_s.shape[0] in [1,iS] and a_s.shape[1:] == (iTAU, self.num_instruments), f"Expected action_a to have shape (1|{iS}, {iTAU}, {self.num_instruments}) but got {a_s.shape}"

                # returns & cost
                # --------------

                dC_h         = ( dC[S_slc,TAU_slc,:,:] - delta_t[S_slc,TAU_slc,:,:] * dS[S_slc,TAU_slc,None,None] ) if self.config.delta_hedged else\
                               dC[S_slc,TAU_slc,:,:]
                assert dC_h.shape == (iS, iTAU, T, N), f"Expected dC_h to have shape {(iS, iTAU, T, N)} but got {dC_h.shape}"
                dC_h         = dC_h.reshape(iS, iTAU, T*N)
                dX_s         = torch.concatenate( [ dS[S_slc,TAU_slc,None], dC_h ], dim=-1 ) 
                dC_h         = None
                X_s          = None
                if use_cost or self.cash_trades:
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

                if self.cash_trades:
                    assert X_s is not None, f"Expected X_s to be not None when cash_trades is True"
                    dX_s = dX_s/X_s
                    X_s  = None

                dX.append( dX_s )
                X.append( X_s ) if not X_s is None else None
                a_s = None
                dX_s = None
                X_s = None

            # collect pnl per sample
            # ----------------------

            a  = torch.cat(a, dim=0) if self.config.use_in or self.config.use_pf else a
            dX = torch.cat(dX, dim=0)
            X  = torch.cat(X, dim=0) if len(X) > 0 else None

            assert a.shape[0] in [1,S] and a.shape[1:] == (iTAU, self.num_instruments), f"Expected action_a to have shape (1|{S}, {iTAU}, {self.num_instruments}) but got {a.shape}"
            assert dX.shape == (S, iTAU, self.num_instruments), f"Expected dX to have shape {(S, iTAU, self.num_instruments)} but got {dX.shape}"
            if not X is None: assert X.shape == (S, iTAU, self.num_instruments), f"Expected X to have shape {(S, iTAU, self.num_instruments)} but got {X.shape}"

            if not training:
                abs_a             = torch.abs(a)
                action_max        = torch.maximum( action_max, torch.max(abs_a) ) if not action_max is None else torch.max(abs_a)
                sum_abs_action    = (sum_abs_action + torch.sum( abs_a, dim=(1,2) )) if not sum_abs_action is None else torch.sum( abs_a, dim=(1,2) )
                sum_sqr_action    = (sum_sqr_action + torch.sum( a**2, dim=(1,2) )) if not sum_sqr_action is None else torch.sum( a**2, dim=(1,2) )
                abs_a             = None

            raw_pnl_tau       = a*dX
            pnl_tau           = raw_pnl_tau

            if self.cost > 0.:
                pnl_tau = pnl_tau - self.cost*( torch.abs(a*X if not X is None else a) ) # cost on action * price 
            if self.cost_std > 0.:
                std     = torch.std( ref_dQdP[:,None,None]*dX[:,:,:], dim=0, keepdim=True)
                assert std.shape == (1,iTAU,T*N+1), f"Expected std to have shape {(1,iTAU,T*N+1)} but got {std.shape}"
                pnl_tau = pnl_tau - self.cost_std*torch.abs(a*std)

            assert pnl_tau.shape == (S, iTAU, self.num_instruments), f"Expected pnl_tau to have shape {(S, iTAU, self.num_instruments)} but got {pnl_tau.shape}"
            pnl_tau     = torch.sum( pnl_tau, dim=(1,2) ) # sum over instruments
            assert pnl_tau.shape == (S, ), f"Expected pnl_tau to have shape {(S,)} but got {pnl_tau.shape}"
            pnl = (pnl+pnl_tau) if not pnl is None else pnl_tau

            if not training:
                raw_pnl_tau = torch.sum( raw_pnl_tau.detach(), dim=(1,2) ) # sum over instruments
                raw_pnl = (raw_pnl+raw_pnl_tau) if not raw_pnl is None else raw_pnl_tau

        # loss
        # ----

        assert pnl.shape == (S,), f"Expected pnl to have shape {(S,)} but got {pnl.shape}"
        if not training: assert raw_pnl.shape == (S,), f"Expected raw_pnl to have shape {(S,)} but got {raw_pnl.shape}"

        weighted_utility = ref_dQdP * self.u( pnl, self.y )     # we wish to maximize this
        assert weighted_utility.shape == (S,), f"Expected weighted_utility to have shape {(S,)} but got {weighted_utility.shape}"
        utility          = torch.mean( weighted_utility ) 
        self.verify( lambda : torch.all(torch.isfinite(utility)), lambda : f"Expected utility to be finite but got {utility}" )

        if training:
            return - utility # loss

        with torch.no_grad():
            # new density
            new_dQdP     = ref_dQdP * self.u_deriv( pnl, self.y ).detach() # sever the gradient here, as we do not want to backprop a and y through the dual action
            assert new_dQdP.shape == (S,), f"Expected new_dQdP to have shape {(S,)} but got {new_dQdP.shape}"
            new_dQdP     = new_dQdP / torch.clip( torch.mean(new_dQdP), min=self.floor_P ) 
            self.verify( lambda : torch.all(torch.isfinite(new_dQdP)), f"dQdP 4 is not finite: {new_dQdP}" )

            util_std         = torch.std( weighted_utility.detach() )
            # simple & direct
            weighted_pnl     = ( ref_dQdP * pnl ).detach()
            pnl              = torch.mean( weighted_pnl )
            pnl_std          = torch.std( weighted_pnl )
            weighted_raw_pnl = ( ref_dQdP * raw_pnl ).detach()
            raw_pnl          = torch.mean( weighted_raw_pnl )
            raw_pnl_std      = torch.std( weighted_raw_pnl )
            weighted_pnl     = None

            self.verify( lambda : torch.all(torch.isfinite(util_std)), lambda : f"Expected util_std to be finite but got {util_std}" )
            self.verify( lambda : torch.all(torch.isfinite(pnl)), lambda : f"Expected pnl to be finite but got {pnl}" )
            self.verify( lambda : torch.all(torch.isfinite(pnl_std)), lambda : f"Expected pnl_std to be finite but got {pnl_std}" )
            self.verify( lambda : torch.all(torch.isfinite(raw_pnl)), lambda : f"Expected raw_pnl to be finite but got {raw_pnl}" )
            self.verify( lambda : torch.all(torch.isfinite(raw_pnl_std)), lambda : f"Expected raw_pnl_std to be finite but got {raw_pnl_std}" )
            
            # action metrics
            action_l1  = ( sum_abs_action / (TAU* self.num_instruments) ).mean()
            action_l2  = torch.sqrt(( sum_sqr_action / (TAU* self.num_instruments) ).mean())

        return dict(
            loss=-utility,
            utility=utility,
            util_std=util_std,
            pnl=pnl,
            raw_pnl=raw_pnl,
            pnl_std=pnl_std,
            raw_pnl_std=raw_pnl_std,
            action_l1=action_l1,
            action_l2=action_l2,
            action_max=action_max,
            dQdP=new_dQdP
        )

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

class Visualizer(object):

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

        self.ax_pnl    = self.fig_lrn.add_subplot("P&L" + (" (with cost)" if self.mdl_config.has_some_cost else ""))
        self.ax_pnl.set_xlabel("Epoch")
        self.ax_pnl.set_ylabel("P&L")

        # density
        # -------

        self.fig_lrn.next_row()
        self.ax_dQdP = self.fig_lrn.add_subplot(r"$dQ/dP$")
        self.ax_dQdP.set_xlabel("Samples")
        self.ax_dQdP.set_ylabel("Density ratio")

        # actions
        # -------

        self.ax_actions = self.fig_lrn.add_subplot("Actions")
        self.ax_actions.set_xlabel("Action Magnitude")
        self.ax_actions.set_ylabel("Epoch")

        # trn
        # ---

        def make_drift(trn:bool):
            rdS, rdC       = self.compute_devs( None, trn=trn )
            ref_rdS,\
            ref_rdC        = (None, None) # turn off prior. self.compute_devs( ref_dQdP, trn=trn ) if ref_dQdP is not None else (None, None)
            cost_str       = f" after cost {self.mdl_config.cost:.4g}" if self.mdl_config.cost > 0. else ""

            self.fig_lrn.next_row()
            r = PrettyObject()
            # dS/err
            r.ax_dev_dS = self.fig_lrn.add_subplot(r"$E[dS]/\epsilon[dS]$ " + ('training' if trn else 'validation') + cost_str)
            r.ax_dev_dS.plot( self.days, rdS.dev, "--", color="blue", label="Market")
            r.ax_dev_dS.plot( self.days, rdS.dev*0.+2., ":", color="black")
            r.ax_dev_dS.plot( self.days, rdS.dev*0.-2, ":", color="black")
            r.ax_dev_dS.plot( self.days, ref_rdS.dev, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            r.ax_dev_dS.set_xlabel("Days")
            r.ax_dev_dS.set_ylabel("Drift as fraction of MC error")
            r.ref_dev_dS_min_max = np.quantile( ref_rdS.dev, q=[0.01, 0.91] ) if not ref_rdS is None else None

            # dC/err
            r.ax_dev_dC = self.fig_lrn.add_subplot(r"$E[dC]/\epsilon[dC]$ " + ('training' if trn else 'validation') + cost_str)
            r.ax_dev_dC.plot( self.days, rdC.tmean, "--", color="blue", label="Market")
            r.ax_dev_dC.plot( self.days, rdS.dev*0.+2., ":", color="black")
            r.ax_dev_dC.plot( self.days, rdS.dev*0.-2, ":", color="black")
            r.ax_dev_dC.fill_between( self.days, rdC.tq05, rdC.tq95, color="blue", alpha=0.1 )
            r.ax_dev_dC.plot( self.days, ref_rdC.tmean, "*:", color="orange", label="Prior") if not ref_rdC is None else None
            if self.viz_config.plot_prior_alpha > 0.:
                r.ax_dev_dC.fill_between( self.days, ref_rdC.tq05, ref_rdC.tq95, color="orange", alpha=self.viz_config.plot_prior_alpha ) if not ref_rdC is None else None
            r.ax_dev_dC.set_xlabel("Days")
            r.ax_dev_dC.set_ylabel("Drift as fraction of MC error")
            r.ref_dev_dC_min_max = ( np.min(ref_rdC.tq05), np.max(ref_rdC.tq95) ) if not ref_rdC is None else None

            # dS
            r.ax_mu_dS = self.fig_lrn.add_subplot(r"$E[dS]$ with $\epsilon[dS]$ Bands " + ('training' if trn else 'validation') + cost_str)
            r.ax_mu_dS.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dS.plot( self.days, rdS.drift, "--", color="blue", label="Market")
            r.ax_mu_dS.fill_between( self.days, rdS.drift-2.*rdS.err, rdS.drift+2.*rdS.err, color="blue", alpha=0.1 )
            r.ax_mu_dS.plot( self.days, ref_rdS.drift, "*:", color="orange", label="Prior") if not ref_rdS is None else None
            if self.viz_config.plot_prior_alpha > 0.:
                r.ax_mu_dS.fill_between( self.days, ref_rdS.drift-2.*ref_rdS.err, ref_rdS.drift+2.*ref_rdS.err, color="orange", alpha=self.viz_config.plot_prior_alpha ) if not ref_rdS is None else None
            r.ax_mu_dS.set_xlabel("Days")
            r.ax_mu_dS.set_ylabel(r"Drift and $\pm$2 MC error bands")
            r.ref_mu_dS_min_max = np.quantile( ref_rdS.drift, q=[0.01, 0.91] ) if not ref_rdS is None else None

            r.ax_mu_dC = self.fig_lrn.add_subplot(r"$\mathrm{Mean}\,E[dC]$ with $\mathrm{Mean}\,\epsilon[dC]$ Bands " + ('training' if trn else 'validation') + cost_str)
            dC_mean_drift = np.mean( rdC.drift, axis=(1,2) )
            dC_mean_err   = np.mean( rdC.err, axis=(1,2) )
            r.ax_mu_dC.plot( self.days,  self.days*0., ":", color="black")
            r.ax_mu_dC.plot( self.days, dC_mean_drift, "--", color="blue", label="Market")
            r.ax_mu_dC.fill_between( self.days, dC_mean_drift-2.*dC_mean_err, dC_mean_drift+2.*dC_mean_err, color="blue", alpha=0.1 )
            if not ref_rdC is None:
                ref_dC_mean_drift = np.mean( ref_rdC.drift, axis=(1,2) )
                ref_dC_mean_err   = np.mean( ref_rdC.err, axis=(1,2) )
                r.ax_mu_dC.plot( self.days, ref_dC_mean_drift, "*:", color="orange", label="Prior") 
                if self.viz_config.plot_prior_alpha > 0.:
                    r.ax_mu_dC.fill_between( self.days, ref_dC_mean_drift-2.*ref_dC_mean_err, ref_dC_mean_drift+2.*ref_dC_mean_err, color="orange", alpha=self.viz_config.plot_prior_alpha )
                r.ref_mu_dC_min_max = np.quantile( ref_dC_mean_drift, q=[0.01, 0.91] ) 
            else:
                r.ref_mu_dC_min_max = None
            r.ax_mu_dC.set_xlabel("Days")
            r.ax_mu_dC.set_ylabel(r"Drift and $\pm$2 MC error bands")
            return r

        self.trn_info = make_drift(trn=True)
        self.vld_info = make_drift(trn=False) if self.num_vld_samples > 0 else None

        self.fig_lrn.render()

    def compute_devs( self, dQdS : np.ndarray|None, trn : bool ) -> tuple[PrettyObject, PrettyObject]:
        if not dQdS is None:
            assert dQdS.shape == (self.num_trn_samples if trn else self.num_vld_samples,), f"Expected dQdS to have shape {(self.num_trn_samples if trn else self.num_vld_samples,)} but got {dQdS.shape}"
        else:
            dQdS = np.ones( (self.num_trn_samples if trn else self.num_vld_samples,), dtype=self.dS.dtype )

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
        rdS.cost   = cost * np.mean( dQdS[:,None]*np.abs(S), axis=0 ) + cost_std * np.std( wdS, axis=0 ) # note that 'cost_std' is already scaled by 1/sqrt{N}
        rdS.Espnl  = np.where( rdS.drift >= 0., np.clip( rdS.drift - rdS.cost, min=0.), np.clip( rdS.drift + rdS.cost, max=0.) )

        rdC.drift  = np.mean( wdC, axis=0 )
        rdC.cost   = cost * np.mean( dQdS[:,None,None,None]*np.abs(C), axis=0 ) + cost_std * np.std( wdC, axis=0 ) # note that 'cost_std' is already scaled by 1/sqrt{N}
        rdC.Espnl  = np.where( rdC.drift >= 0., np.clip( rdC.drift - rdC.cost, min=0.), np.clip( rdC.drift + rdC.cost, max=0.) )
 
        rdS.err    = np.std(wdS, axis=0 ) / np.sqrt(dS.shape[0])
        rdC.err    = np.std(wdC-rdC.drift[None,...], axis=0 ) / np.sqrt(dC.shape[0])
        rdS.dev    = np.where( rdS.err > 0., rdS.Espnl / np.where( rdS.err > 0., rdS.err, 1.), 0. ) # with cost
        rdC.dev    = np.where( rdC.err > 0., rdC.Espnl / np.where( rdC.err > 0., rdC.err, 1.), 0. ) # with cost
        rdS.raw_dev= np.where( rdS.err > 0., rdS.drift / np.where( rdS.err > 0., rdS.err, 1.), 0. ) # without cost
        rdC.raw_dev= np.where( rdC.err > 0., rdC.drift / np.where( rdC.err > 0., rdC.err, 1.), 0. ) # without cost

        rdC.dev       = rdC.dev.reshape( rdC.dev.shape[0], -1 ) # (TAU,TN)
        rdC.tmean     = np.mean( rdC.dev, axis=1 )    
        rdC.tq05,\
        rdC.tmedian,\
        rdC.tq95      = np.quantile( rdC.dev, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.tmin      = np.min( rdC.dev, axis=1 )
        rdC.tmax      = np.max( rdC.dev, axis=1 )

        rdC.raw_dev   = rdC.raw_dev.reshape( rdC.raw_dev.shape[0], -1 )
        rdC.raw_tmean = np.mean( rdC.raw_dev, axis=1 )
        rdC.raw_tq05,\
        rdC.raw_tmedian,\
        rdC.raw_tq95  = np.quantile( rdC.raw_dev, q=[0.05, 0.5, 0.95], axis=1 )
        rdC.raw_tmin  = np.min( rdC.raw_dev, axis=1 )
        rdC.raw_tmax  = np.max( rdC.raw_dev, axis=1 )
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
        def utilities_pnl():
            def mk(content : OutputTracker, num_samples : int) -> PrettyObject :
                content = content if isinstance(content, OutputTracker) else OutputTracker(**content)
                r = PrettyObject(
                    epochs     =  np.asarray(content.epochs),
                    values     =  np.clip( np.asarray(content.utility), min=-_CLIP, max=_CLIP ),
                    std        =  np.clip( np.asarray(content.util_std), min=-_CLIP, max=_CLIP ),
                    pnl        =  np.clip( np.asarray(content.pnl), min=-_CLIP, max=_CLIP ),
                    pnl_std    =  np.clip( np.asarray(content.pnl_std), min=-_CLIP, max=_CLIP ),
                    raw_pnl    =  np.clip( np.asarray(content.raw_pnl), min=-_CLIP, max=_CLIP ),
                    raw_pnl_std=  np.clip( np.asarray(content.raw_pnl_std), min=-_CLIP, max=_CLIP ),
                    action_l1  =  np.clip( np.asarray(content.action_l1), min=-_CLIP, max=_CLIP ),
                    action_l2  =  np.clip( np.asarray(content.action_l2), min=-_CLIP, max=_CLIP ),
                    action_max =  np.clip( np.asarray(content.action_max), min=-_CLIP, max=_CLIP ),
                    cap_l1     =  np.clip( np.asarray(content.cap_l1), min=-_CLIP, max=_CLIP ) if len(content.cap_l1) > 0 else None,
                    cap_l2     =  np.clip( np.asarray(content.cap_l2), min=-_CLIP, max=_CLIP ) if len(content.cap_l2) > 0 else None,
                    cap_max    =  np.clip( np.asarray(content.cap_max), min=-_CLIP, max=_CLIP ) if len(content.cap_max) > 0 else None,
                )
                sqrt_num_samples = np.sqrt(num_samples)
                r.err = r.std / sqrt_num_samples
                r.pnl_err = r.pnl_std / sqrt_num_samples
                r.raw_pnl_err = r.raw_pnl_std / sqrt_num_samples
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


            util     = mk(trn_tracker, self.num_trn_samples)
            str_util = f"{util.values[-1]:.3g} (err {util.err[-1]:.3g})"
            
            self.store += self.ax_util.plot(util.epochs, util.epochs*0., ":", color="black")
            self.store += self.ax_util.plot(util.epochs, util.values, ".-", color="red", label="Training" if self.num_vld_samples > 0 else None)
            self.store += self.ax_util.fill_between(util.epochs, util.values-2.*util.err, util.values+2.*util.err, alpha=0.1, color="red")
            
            mom = minmax(util.values, util.err[-1])

            # pnl
            # ---

            has_some_cost = self.mdl_config.has_some_cost

            self.store += self.ax_pnl.plot(util.epochs, util.pnl*0., ":", color="black")
            self.store += self.ax_pnl.plot(util.epochs, util.pnl, ".-", color="red", label=("Training" if not has_some_cost else "Training (with Cost)") if self.num_vld_samples > 0 else (None if not has_some_cost else "With Cost"))
            self.store += self.ax_pnl.fill_between(util.epochs, util.pnl-2.*util.pnl_err, util.pnl+2.*util.pnl_err, alpha=0.1, color="red")
            mom_pnl = minmax(util.pnl, util.pnl_err[-1])
            str_util_pnl = f"{util.pnl[-1]:.3g} (err {util.pnl_err[-1]:.3g})" 

            if self.mdl_config.has_some_cost:
                self.store += self.ax_pnl.plot(util.epochs, util.raw_pnl, ".-", color="blue", label="Training (without Cost)" if self.num_vld_samples > 0 else "Without Cost")
                self.store += self.ax_pnl.fill_between(util.epochs, util.raw_pnl-2.*util.raw_pnl_err, util.raw_pnl+2.*util.raw_pnl_err, alpha=0.1, color="blue")
                mom_pnl = minmax(util.raw_pnl, util.raw_pnl_err[-1])
                str_util_pnl = str_util_pnl + f" [ raw {util.raw_pnl[-1]:.3g} (err {util.raw_pnl_err[-1]:.3g}) ]"

            # action
            # ------

            self.store += self.ax_actions.plot(util.epochs, util.pnl*0., ":", color="black")
            self.store += self.ax_actions.plot(util.epochs, util.action_l1, ".-", color="blue", label="Action L1")
            self.store += self.ax_actions.plot(util.epochs, util.action_l2, ".-", color="green", label="Action L2")
            self.store += self.ax_actions.plot(util.epochs, util.action_max, ".-", color="red", label="Action Max")

            mom_action = minmax(util.action_max, 0., None) # (max > l1 > l2)

            if not util.cap_max is None:
                self.store += self.ax_actions.plot(util.epochs, util.cap_max, "--", color="black", label="Cap Max")
                self.store += self.ax_actions.plot(util.epochs, util.cap_l1, ":", color="black", label="Cap L1")
                self.store += self.ax_actions.plot(util.epochs, util.cap_l2, "-.", color="black", label="Cap L2")
                mom_action = minmax(util.cap_max, 0., mom_action)


            if vld_tracker is not None:
                assert self.num_vld_samples > 0, f"Expected num_vld_samples to be > 0 but got {self.num_vld_samples}"
                vld_util = mk(vld_tracker, self.num_vld_samples)

                str_util = f"{str_util}\n{vld_util.values[-1]:.3g} (err {vld_util.err[-1]:.3g})"

                self.store += self.ax_util.plot(vld_util.epochs, vld_util.values, "--", color="orange", label="Validation")
                self.store += self.ax_util.fill_between(vld_util.epochs, vld_util.values-2.*vld_util.err[-1], vld_util.values+2.*vld_util.err[-1], alpha=0.1, color="orange")
                mom = minmax(vld_util.values, vld_util.err[-1], mom)

                # pnl
                # ---
                self.store += self.ax_pnl.plot(vld_util.epochs, vld_util.pnl, "--", color="orange",  label="Validation" if not has_some_cost else "Validation (with Cost)") 
                self.store += self.ax_pnl.fill_between(vld_util.epochs, vld_util.pnl-2.*vld_util.pnl_err[-1], vld_util.pnl+2.*vld_util.pnl_err[-1], alpha=0.1, color="orange")
                mom_pnl = minmax(vld_util.pnl, vld_util.pnl_err[-1], mom_pnl)

                str_util_pnl = f"{str_util_pnl}\n{vld_util.pnl[-1]:.3g} (err {vld_util.pnl_err[-1]:.3g})" 

                if self.mdl_config.has_some_cost:
                    self.store += self.ax_pnl.plot(vld_util.epochs, vld_util.raw_pnl, ".-", color="blue", label="Validation (without Cost)")
                    self.store += self.ax_pnl.fill_between(vld_util.epochs, vld_util.raw_pnl-2.*vld_util.raw_pnl_err, vld_util.raw_pnl+2.*vld_util.raw_pnl_err, alpha=0.1, color="blue")
                    mom_pnl = minmax(vld_util.raw_pnl, vld_util.raw_pnl_err[-1], mom_pnl)
                    str_util_pnl = str_util_pnl + f" [ raw {vld_util.raw_pnl[-1]:.3g} (err {vld_util.raw_pnl_err[-1]:.3g}) ]"

                # actions
                # -------

                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_l1, "--", color="lightblue", label="Validation Action L1")
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_l2, "--", color="lightgreen", label="Validation Action L2")
                self.store += self.ax_actions.plot(vld_util.epochs, vld_util.action_max, "--", color="orange", label="Validation Action Max")
                mom_action = minmax(vld_util.action_l1, 0., mom_action)
                mom_action = minmax(vld_util.action_l2, 0., mom_action)
                mom_action = minmax(vld_util.action_max, 0., mom_action)


            # legend & co - utilities
            # -----------------------

            self.ax_util.set_ylim( mom[0], mom[1] )
            self.ax_util.set_title(f"Utilities\n{str_util}")
            if self.first and self.num_vld_samples > 0:
                self.ax_util.legend(loc="upper left")

            # pnl
            # ---

            self.ax_pnl.set_ylim( mom_pnl[0], mom_pnl[1] )
            self.ax_pnl.set_title(f"P&L\n{str_util_pnl}")
            if self.first and ( self.num_vld_samples > 0 or has_some_cost ):
                self.ax_pnl.legend(loc="upper left")

            # actions
            # -------

            self.ax_actions.set_ylim( mom_action[0], mom_action[1] )
            self.ax_actions.set_title(f"Action Metrics")
            if self.first:
                self.ax_actions.legend(loc="upper left")

        utilities_pnl()

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
            rdS, rdC = self.compute_devs( dQdP if trn else vld_dQdP, trn=trn )
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

def train( universe, *,
           ref_dQdP      : np.ndarray|None = None,
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

    def valuation_batches( src, num_src_samples ):
        """ Sampling valuation batches if necessary """
        if trn_config.val_batch_size is None or trn_config.val_batch_size >= num_src_samples:
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

    disp = Visualizer( universe, ref_dQdP=ref_dQdP, total_epochs=trn_config.num_epochs, num_trn_samples=num_trn_samples, num_vld_samples=num_vld_samples, viz_config=viz_config, mdl_config=mdl_config )
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
                if k in ['dQdP']:
                    continue
                v = float(v) if isinstance(v, (torch.Tensor, float)) and v.numel() in (0, 1) else v
                klst = getattr(tracker, k, None)
                if klst is None:
                    raise KeyError(f"Expected tracker to have attribute '{k}', but it does not. Available attributes: {sorted(vars(tracker))}")
                klst.append(v)

            if not model.lrn_action_cap is None:
                inst_cap = model.lrn_action_cap.detach().cpu().numpy()
                tracker.cap_l1.append( np.mean(inst_cap) )
                tracker.cap_l2.append( np.std(inst_cap) )
                tracker.cap_max.append( np.max(inst_cap) )

        tracker.val_losses.append(trn_model_output['loss'].item())
        tracker.val_epochs.append(epoch+1)
        append_data(trn_tracker, trn_model_output)

        if not vld_model_output is None:
            tracker.vld_losses.append(vld_model_output['loss'].item())
            tracker.vld_epochs.append(epoch+1)
            append_data(vld_tracker, vld_model_output)

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
