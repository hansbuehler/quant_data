"""
Grid Sanos tridiagonal solver for SANOS-type transition operators, c.f. http://arxiv.org/abs/2211.13227
"""

from __future__ import annotations
from quantdata.utils.backbone import Backbone, BackboneNumpy
from cdxcore.err import verify_inp

def solve_sanos_tridiagonal(*, w_minus: Backbone.Array,
                            w_plus: Backbone.Array,
                            rhs: Backbone.Array,
                            backbone : Backbone = BackboneNumpy(),
                            ) -> Backbone.Array:
    """Solve SANOS tridiagonal system with boundary structure from w_minus/w_plus.

    The system size is ``N = (N-2) + 2`` where:
    - ``w_minus`` and ``w_plus`` have shape ``(..., N-2)``
    - ``rhs`` has shape ``(..., N)``

    This avoids materializing full tridiagonal bands and is more memory efficient
    for the SANOS parameterization.
    """
    assert w_minus.shape == w_plus.shape, f"w_minus and w_plus must have the same shape, got {w_minus.shape} and {w_plus.shape}"

    n_inner = w_minus.shape[-1]
    n       = n_inner + 2
    lead    = w_minus.shape[:-1]

    assert rhs.shape == lead + (n,), f"rhs must have shape {lead + (n,)}, got {tuple(rhs.shape)}"

    # classic form
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    #
    # b_0 c_0                      x_0      rhs_0
    # a_1 b_1 c_1                  x_1      rhs_1
    #     a_2 b_2 c_2              x_2   =  rhs_2
    #         ...                  ...      ...  
    #             a_{N-1} b_{N-1}  x_{N-1}  rhs_{N-1}
    #

    # Row 0:
    # b_0 = 1
    # c_0 = -w_minus[0]
    # 
    #  1 * x0 - w_minus[0] * x1 = rhs0
    cp_terms = [-w_minus[..., 0]]  # = c_0/b_0
    y_terms = [rhs[..., 0]]        # = rhs_0/b_0

    # Rows 1..N-2
    # a_i =   - w_plus[i-1] (if i > 1 else 0)
    # b_i = 1 + w_minus[i] + w_plus[i]
    # c_i =   - w_minus[i+1] (if i < N-2 else 0)
    #
    # =>
    #
    # cp_i = c_i / (b_i - a_i * cp_{i-1})
    # y_i = (rhs_i - a_i * y_{i-1}) / (b_i - a_i * cp_{i-1})

    for i in range(1, n - 1):
        a_i = -w_plus[..., i - 2] if i > 1 else 0.
        b_i = 1.0 + w_minus[..., i - 1] + w_plus[..., i - 1]
        c_i = -w_minus[..., i] if i < (n - 2) else 0.

        denom = b_i - a_i * cp_terms[i - 1]
        cp_terms.append(c_i / denom)
        y_terms.append((rhs[..., i] - a_i * y_terms[i - 1]) / denom)

    # Last row:
    # a_{N-1} = -w_plus[N-3]
    # b_{N-1} = 1
    #
    # -w_plus[N-3] * x[N-2] + 1 * x[N-1] = rhs[N-1]
    a_last = -w_plus[..., -1]
    b_last = 1.
    denom = b_last - a_last * cp_terms[-1]
    y_terms.append((rhs[..., -1] - a_last * y_terms[-1]) / denom)

    # forward pass
    x_terms = [None] * n
    x_terms[-1] = y_terms[-1]
    for i in range(n - 2, -1, -1):
        x_terms[i] = y_terms[i] - cp_terms[i] * x_terms[i + 1]
    return backbone.stack(x_terms, axis=-1, dtype=rhs.dtype)