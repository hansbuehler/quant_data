"""
Basic utilities for SANOS
"""

import numpy as np
from cdxcore.err import verify_inp
import math

def vec_max_dx( strikes : np.ndarray, *, min_k : float, max_k : float, max_dx : float|None = None, min_dx : float = 0. ) -> tuple:
    """
    Generate model strikes.
    
    This function returns a vector of strikes such that:
        
    * Lowest strike is ``min_k``
    * Highest strike is ``max_k``
    * Strikes are at least ``min_dx`` apart from each other. Strikes which violate this are dropped iteratively from the front.
    * Strikes in the original strike range are at most ``max_dx`` apart from each other. If None, do not add strikes except min_k or max_k. Outside the extreme strikes of ``strikes``` 
      the function adds strikes with ``10 max_dx``.
    
    The function returns a tuple, the first element is the new strike vector.
    The second element represents the index positions of the ``strikes`` in the new vector. An index is -1 if an element was dropped due to ``min_k``.
    The function returns ``slice(1,-1)`` if
    only min_k, max_k were added ie the returned strikes are [min_k, strike[0], ..., strike[-1], max_k].    
    """
    verify_inp( min_k > 0. and min_k < 1. and max_k > 1., "'min_k' must be in (0,1) and 'max_k' must be above 1" )
    verify_inp( min_k < strikes[0] and max_k > strikes[-1], "'min_k' and 'max_k' must be outside 'strikes'")
    verify_inp( max_dx is None or max_dx >= 0., "'max_dx' must be None or non-negative")
    verify_inp( min_dx >= 0., "'min_dx' must be non-negative")

    if max_dx is None:
        if np.max( strikes[1:]-strikes[:-1] ) >= min_dx:
            out = np.empty( (len(strikes)+2,), dtype=strikes.dtype )
            out[1:-1] = strikes
            out[0]    = min_k
            out[-1]   = max_k
            return out, slice(1,-1)
        
        out = [ min_k ]
        six = []
        for x in strikes[1:]:
            dx = x - out[-1]
            if dx <= min_dx:
                six.append(-1)
                continue
            else:
                six.append( len(out ))
                out.append( x )
        return np.asarray( out, dtype=strikes.dtype ), np.asarray( six, dtype=np.int32 )        
            
    # max_dx
    # ------

    outer_max_dx = min(0.25,max_dx*10.) 
    inner_max_dx = max_dx

    left    = strikes[0]-inner_max_dx
    out     = np.linspace( min_k, left, max(2,1+math.floor( (left - min_k)/outer_max_dx )),dtype=strikes.dtype).tolist()\
              if left > min_k else\
              [ min_k ]    
    six     = [ len(out) ]
    out.append( strikes[0] )
    
    for x in strikes[1:]:
        dx = x - out[-1]
        if dx <= min_dx:
            six.append(-1)
            continue
        if dx > inner_max_dx:
            n = max(1,int(dx/inner_max_dx))+2
            r = np.linspace( out[-1], x, n )
            for x in r[1:]:
                out.append(x)
            six.append( len(out)-1 )
        else:
            six.append( len(out ))
            out.append( x )

    right = strikes[-1]+inner_max_dx
    rout  = np.linspace( right, max_k, max(2,1+math.floor( (max_k-right)/outer_max_dx )),dtype=strikes.dtype).tolist()\
            if right < max_k else\
            [ max_k ]
   
    out += rout
    out  = np.array( out, dtype=strikes.dtype )
    six  = np.asarray( six, dtype=np.int32 ) if len(out) > len(strikes)+2 else slice(1,-1)
    assert np.all( out[six] == strikes ), ("Internal error", out[six][:20], strikes[:20] )
    return np.asarray( out, dtype=strikes.dtype ), six     