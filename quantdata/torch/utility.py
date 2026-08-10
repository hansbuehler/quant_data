import math
from typing import Callable

import torch
from cdxcore.version import version
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
