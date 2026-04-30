"""
To be added to cdxcore
"""

from cdxcore.dynaplot import m_o_m
from cdxcore.err import verify, verify_inp
import numpy as np

def quantile_m_o_m( *args, lo : float = 0.1, hi : float = 0.9, pos_floor : float|None = None, buf : float = 0.05, min_dx : float = 0.01 ):
    """
    Computes a lower and higher quantile of all elements of ``args``.
    This operation creates a copy of all data and sorts it, therefore it is expensive.

    This function is useful to compute arguments for :meth:`matplotlib.axes.Axes.set_xlim`
    or :meth:`matplotlib.axes.Axes.set_ylim`.
    
    If ``pos_floor`` is ``None`` then the function returns::
        
        dx = max( min_dx, max_ - min_ )
        return min_ - dx*buf, max_ + dx*buf
    
    If ``pos_floor`` is not ``None``, then the function floors ``min_ - dx*buf`` at ``min_*pos_floor``.
    This is used for positive values by using a positive ``pos_floor`` (e.g. 0.1).
    
    *Usage*::

        from cdxcore.dynaplot import figure, m_o_m
        import numpy as np
            
        x = np.random.normal(size=(10,))
        y = np.random.normal(size=(8,2))
        z = [ np.random.normal(size=(3,2)), 0.1, None ]
        
        with figure() as fig:
            ax = fig.add_subplot()
            ax.plot( x )
            ax.plot( y )
            ax.plot( z[1] )
            ax.set_ylim( *quantile_m_o_m(x,y,z, lo=0.01, hi=0.99, min_dx=0.01) )
    
    Returns
    -------
        qlo, qhi : float | None, float | None
            The adjusted minimum and maximum values as discussed above.
            The function returns ``None, None`` if ``args`` does not contain any numbers.
            Such result can be passed directly to :meth:`matplotlib.axes.Axes.set_xlim` or
            :meth:`matplotlib.axes.Axes.set_ylim` as both accept ``None`` as default values.
    
    """
    verify_inp( 0. <= lo < hi <= 1., "'lo' and 'hi' must be within [0,1] and 'lo' must be smaller than 'hi'")

    if lo==0. and hi==1.:
        return m_o_m( *args, pos_floor=pos_floor, buf=buf, min_dx=min_dx )

    args = [ np.ravel( np.asarray(a) ) for a in args if not a is None ]
    if len(args) == 0:
        return None, None
    if len(args) > 1:
        size = sum( a.size for a in args )
        tar  = np.empty( (size,), dtype=np.float32 )
        i = 0
        for a in args:
            tar[i:i+a.size] = a
            i += a.size
        assert i == size, ("Internal error", i, size)
        args = tar
        del tar
        del size

    args = np.sort(args)
    ixlo = min(args.size-2, max(0, int( np.floor( lo*args.size ) ) ) ) if lo > 0. else 0
    ixhi = max(ixlo+1, min(args.size-1, int( np.ceil( hi*args.size ) ) ) ) if hi < 1. else args.size-1

    min_ = args[ixlo]
    max_ = args[ixhi]

    if buf is None or buf == 0.:
        return min_, max_
    
    dx   = max( min_dx, max_ - min_ )
    if pos_floor is None:
        return min_ - dx*buf, max_ + dx*buf
    else:
        verify( min_ > 0., lambda : f"Cannot use 'buf' in 'pos' mode: 'min_o_min' of the inputs is not positive, but {min_:.4g}")
        verify_inp( pos_floor >= 0., "'pos_floor' cannot be negative")
        rmin = max( min_ - dx*buf, min_*pos_floor )
        return rmin, max_+dx*buf