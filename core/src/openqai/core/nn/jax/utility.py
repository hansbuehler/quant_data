import jax
import jax.numpy as jnp
from cdxcore.err import verify_inp
from collections.abc import Callable
from typing import Tuple

def jax_utility(utility : str ="entropy", lmbda : float = 1.0) -> Tuple[Callable, Callable, str]:
    """
    Return utility, derivative and name.
    """

    verify_inp( lmbda >= 0., lambda: f"Lambda must be non-negative,found {lmbda}" )

    if utility == "entropy":
        u     = lambda x : -jnp.expm1( -lmbda * x ) / lmbda
        deriv = lambda x : jnp.exp(-lmbda * x)
        label = f"entropy@{lmbda:.2g}"
    elif utility == "cvar":
        u     = lambda x : (1.+lmbda) * jnp.minimum( x, 0. )
        deriv = lambda x : jnp.where( x<0., 1.+lmbda, 0. )
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
            x_pos = jnp.clip(x, min=0. )
            pow   = 1.+lmbda
            return jnp.where( x>0., -jnp.expm1( -lmbda * x_pos ) / lmbda, -(jnp.abs(x)**pow)/pow+x )
        def deriv(x):
            x_pos = jnp.clip(x, min=0. )
            return jnp.where( x>0., jnp.exp(-lmbda * x_pos), 1.+jnp.abs(x)**lmbda )
        label = f"powexp@{lmbda:.2g}"
    else:
        verify_inp( utility is None or utility == "id", lambda:  f"Unknown utility '{utility}'" )
        u     = lambda x : x
        deriv = lambda x : jnp.ones_like(x)
        label = "id"
    return u, deriv, label
