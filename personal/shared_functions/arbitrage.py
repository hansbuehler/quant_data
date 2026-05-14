"""
Arbitrage-Free Volatility Smile Checker
========================================
Based on: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2013)

Two checks are implemented:
  1. Butterfly arbitrage  – checks g(k) >= 0 for all strikes (single slice)
  2. Calendar spread arb  – checks total variance is non-decreasing across maturities

Usage
-----
    from svi_arbitrage_check import check_butterfly, check_calendar_spread, check_surface

Each function returns a dict with a boolean `is_arbitrage_free` flag and diagnostics.
A plotting helper is also included.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Core formula: g(k) from equation (2.1) of Gatheral & Jacquier (2013)
# ---------------------------------------------------------------------------


def compute_g(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute the function g(k) from Eq. (2.1) of Gatheral & Jacquier (2013).

    A slice is free of butterfly arbitrage iff g(k) >= 0 for all k,
    AND lim_{k->+inf} d+(k) = -inf  (handled separately via Lee bound check).

    Parameters
    ----------
    k : array of log-moneyness values  (k = log(K / F))
    w : array of total implied variance (w = sigma_BS^2 * T)
        Must be the same length as k and correspond to the same strikes.

    Returns
    -------
    g : array, same shape as k
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    # Numerical first and second derivatives
    w1 = np.gradient(w, k)  # dw/dk
    w2 = np.gradient(w1, k)  # d^2w/dk^2

    term1 = (1.0 - k * w1 / (2.0 * w)) ** 2
    term2 = (w1**2 / 4.0) * (1.0 / w + 0.25)
    term3 = w2 / 2.0

    g = term1 - term2 + term3
    return g


def compute_d_plus(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    d+(k) = -k / sqrt(w) + sqrt(w) / 2
    Must tend to -inf as k -> +inf for no-arbitrage (call prices -> 0).
    """
    return -k / np.sqrt(w) + np.sqrt(w) / 2.0


# ---------------------------------------------------------------------------
# Check 1: Butterfly arbitrage (single slice)
# ---------------------------------------------------------------------------


def check_butterfly(
    k: Union[list, np.ndarray],
    iv: Union[list, np.ndarray],
    T: float,
    tol: float = -1e-6,
    plot: bool = False,
    title: str = "Butterfly Arbitrage Check",
) -> dict:
    """
    Check a single vol smile slice for butterfly arbitrage.

    Parameters
    ----------
    k  : log-moneyness array  (k = log(K / F), ATM = 0)
    iv : implied volatility array (in decimal, e.g. 0.20 for 20%)
    T  : time to expiry in years
    tol: numerical tolerance for g >= 0 check (small negatives allowed)
    plot: if True, plot w(k) and g(k)
    title: plot title

    Returns
    -------
    dict with keys:
        is_arbitrage_free : bool
        min_g             : float  – minimum value of g(k) over the grid
        violations        : array  – k values where g < tol
        g                 : array  – full g(k) array
        w                 : array  – total implied variance array
        lee_bound_ok      : bool   – whether d+(k) -> -inf at right tail
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)

    w = iv**2 * T  # total implied variance
    g = compute_g(k, w)

    min_g = float(g.min())
    violations = k[g < tol]

    # Lee moment bound: d+(k) should be very negative at the far right strike
    d_plus = compute_d_plus(k, w)
    lee_bound_ok = bool(d_plus[-1] < 0)  # at largest k, d+ must be negative

    is_arb_free = (min_g >= tol) and lee_bound_ok

    result = {
        "is_arbitrage_free": is_arb_free,
        "min_g": min_g,
        "violations": violations,
        "g": g,
        "w": w,
        "k": k,
        "lee_bound_ok": lee_bound_ok,
    }

    if plot:
        _plot_butterfly(k, w, g, title, tol)

    return result


def _plot_butterfly(k, w, g, title, tol):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    # Left: total variance smile
    axes[0].plot(k, w, color="steelblue", lw=2)
    axes[0].set_xlabel("Log-moneyness k")
    axes[0].set_ylabel("Total implied variance w(k)")
    axes[0].set_title("Total Variance Smile")
    axes[0].grid(True, alpha=0.3)

    # Right: g(k) – must be >= 0
    axes[1].plot(k, g, color="darkorange", lw=2, label="g(k)")
    axes[1].axhline(0, color="red", lw=1.2, linestyle="--", label="g = 0")
    axes[1].fill_between(k, g, 0, where=(g < tol), color="red", alpha=0.3, label="Arbitrage region")
    axes[1].set_xlabel("Log-moneyness k")
    axes[1].set_ylabel("g(k)")
    axes[1].set_title("Butterfly Arbitrage Indicator g(k)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Check 2: Calendar spread arbitrage (multiple slices)
# ---------------------------------------------------------------------------


def check_calendar_spread(
    k_grid: Union[list, np.ndarray],
    slices: list,  # list of (T, iv_array) tuples, sorted ascending by T
    tol: float = -1e-8,
    plot: bool = False,
    title: str = "Calendar Spread Arbitrage Check",
) -> dict:
    """
    Check a set of vol smile slices for calendar spread arbitrage.

    Calendar spread arbitrage is absent iff  dw/dT >= 0 at every strike,
    i.e. total variance is non-decreasing in maturity at each log-strike.

    Parameters
    ----------
    k_grid : common log-moneyness grid
    slices : list of (T, iv_array) tuples, sorted by T ascending.
             Each iv_array must correspond to k_grid.
    tol    : numerical tolerance
    plot   : if True, plot the total-variance curves
    title  : plot title

    Returns
    -------
    dict with keys:
        is_arbitrage_free   : bool
        crossing_pairs      : list of (T1, T2) pairs that cross
        min_dw_dT           : float  – most negative dw/dT found
        w_surfaces          : list of (T, w_array) tuples
    """
    k_grid = np.asarray(k_grid, dtype=float)

    # Sort slices by maturity
    slices_sorted = sorted(slices, key=lambda x: x[0])
    Ts = [s[0] for s in slices_sorted]
    ws = [np.asarray(iv, dtype=float) ** 2 * T for T, iv in slices_sorted]

    crossing_pairs = []
    min_dw_dT = np.inf

    for i in range(len(ws) - 1):
        T1, w1 = Ts[i], ws[i]
        T2, w2 = Ts[i + 1], ws[i + 1]
        dw = w2 - w1
        local_min = float(dw.min())
        if local_min < min_dw_dT:
            min_dw_dT = local_min
        if local_min < tol:
            crossing_pairs.append((T1, T2))

    is_arb_free = len(crossing_pairs) == 0

    result = {
        "is_arbitrage_free": is_arb_free,
        "crossing_pairs": crossing_pairs,
        "min_dw_dT": float(min_dw_dT),
        "w_surfaces": list(zip(Ts, ws)),
        "k_grid": k_grid,
    }

    if plot:
        _plot_calendar(k_grid, Ts, ws, title)

    return result


def _plot_calendar(k_grid, Ts, ws, title):
    plt.figure(figsize=(10, 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(Ts) - 1, 1)) for i in range(len(Ts))]

    for T, w, color in zip(Ts, ws, colors):
        plt.plot(k_grid, w, lw=2, color=color, label=f"T = {T:.3f}")

    plt.xlabel("Log-moneyness k")
    plt.ylabel("Total implied variance w(k)")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Combined surface check
# ---------------------------------------------------------------------------


def check_surface(
    k_grid: Union[list, np.ndarray],
    slices: list,
    tol_butterfly: float = -1e-6,
    tol_calendar: float = -1e-8,
    plot: bool = False,
) -> dict:
    """
    Full static arbitrage check for a volatility surface.

    Runs both butterfly and calendar spread checks on all slices.

    Parameters
    ----------
    k_grid          : common log-moneyness grid
    slices          : list of (T, iv_array) tuples
    tol_butterfly   : tolerance for g(k) >= 0
    tol_calendar    : tolerance for dw/dT >= 0
    plot            : if True, show all diagnostic plots

    Returns
    -------
    dict with keys:
        is_arbitrage_free     : bool  – True only if BOTH checks pass for ALL slices
        butterfly_results     : list of per-slice dicts from check_butterfly()
        calendar_result       : dict from check_calendar_spread()
        summary               : str  – human-readable summary
    """
    k_grid = np.asarray(k_grid, dtype=float)
    slices_sorted = sorted(slices, key=lambda x: x[0])

    butterfly_results = []
    for T, iv in slices_sorted:
        res = check_butterfly(
            k_grid,
            iv,
            T,
            tol=tol_butterfly,
            plot=plot,
            title=f"Butterfly Check – T = {T:.4f}",
        )
        res["T"] = T
        butterfly_results.append(res)

    calendar_result = check_calendar_spread(
        k_grid,
        slices_sorted,
        tol=tol_calendar,
        plot=plot,
        title="Calendar Spread Check",
    )

    all_butterfly_ok = all(r["is_arbitrage_free"] for r in butterfly_results)
    calendar_ok = calendar_result["is_arbitrage_free"]
    is_arb_free = all_butterfly_ok and calendar_ok

    # Build a readable summary
    lines = ["=" * 55, "  STATIC ARBITRAGE CHECK SUMMARY", "=" * 55]

    lines.append(f"\n{'Maturity':<10} {'Butterfly':<18} {'min g(k)':<14} {'Lee bound'}")
    lines.append("-" * 55)
    for r in butterfly_results:
        status = "OK ✓" if r["is_arbitrage_free"] else "ARBITRAGE ✗"
        lines.append(
            f"  T={r['T']:<7.4f} {status:<18} {r['min_g']:<14.6f} {'OK' if r['lee_bound_ok'] else 'FAIL'}"
        )

    lines.append("")
    cal_status = "OK ✓" if calendar_ok else "ARBITRAGE ✗"
    lines.append(f"Calendar spread: {cal_status}  (min dw/dT = {calendar_result['min_dw_dT']:.2e})")
    if not calendar_ok:
        for T1, T2 in calendar_result["crossing_pairs"]:
            lines.append(f"  → Crossing between T={T1:.4f} and T={T2:.4f}")

    lines.append("")
    lines.append(f"OVERALL: {'ARBITRAGE-FREE ✓' if is_arb_free else 'ARBITRAGE DETECTED ✗'}")
    lines.append("=" * 55)
    summary = "\n".join(lines)

    return {
        "is_arbitrage_free": is_arb_free,
        "butterfly_results": butterfly_results,
        "calendar_result": calendar_result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# SVI raw parameterization helper
# ---------------------------------------------------------------------------


def svi_raw(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    Raw SVI total implied variance: Eq. (3.1) of Gatheral & Jacquier (2013).

    w(k) = a + b * { rho*(k-m) + sqrt((k-m)^2 + sigma^2) }

    Parameters
    ----------
    k     : log-moneyness array
    a, b, rho, m, sigma : raw SVI parameters

    Returns
    -------
    w : total implied variance array
    """
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def iv_from_svi(k: np.ndarray, T: float, a, b, rho, m, sigma) -> np.ndarray:
    """Convert raw SVI parameters to implied volatility (decimal)."""
    w = svi_raw(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0.0) / T)


# # ---------------------------------------------------------------------------
# # Quick demo / example
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     print("=" * 55)
#     print("  EXAMPLE 1: Arbitrageable SVI from Example 3.1")
#     print("  (Axel Vogt's parameters from the paper)")
#     print("=" * 55)

#     k = np.linspace(-1.5, 1.5, 300)
#     T = 1.0

#     # Vogt parameters from Example 3.1 — known to violate butterfly arb
#     a, b, m, rho, sigma = -0.0410, 0.1331, 0.3586, 0.3060, 0.4153
#     iv_vogt = iv_from_svi(k, T, a, b, rho, m, sigma)

#     res_vogt = check_butterfly(k, iv_vogt, T, plot=True, title="Example 3.1: Vogt (arbitrageable)")
#     print(f"\nButterfly arbitrage free: {res_vogt['is_arbitrage_free']}")
#     print(f"min g(k) = {res_vogt['min_g']:.6f}  (must be >= 0 for no arb)")

#     print("\n" + "=" * 55)
#     print("  EXAMPLE 2: Clean SSVI power-law surface (no arb)")
#     print("=" * 55)

#     # Power-law SSVI: phi(theta) = eta * theta^{-gamma}
#     # Conditions: gamma in (0,1), eta*(1+|rho|) <= 2
#     eta = 0.5
#     gamma = 0.4
#     rho_ssvi = -0.7
#     Ts = [0.1, 0.25, 0.5, 1.0, 2.0]

#     def ssvi_iv(k_arr, T_val):
#         theta = 0.04 * T_val          # simple flat ATM variance: sigma^2 * T
#         phi = eta * theta ** (-gamma)
#         w = (theta / 2) * (
#             1 + rho_ssvi * phi * k_arr + np.sqrt((phi * k_arr + rho_ssvi) ** 2 + (1 - rho_ssvi**2))
#         )
#         return np.sqrt(np.maximum(w, 0.0) / T_val)

#     slices = [(T_val, ssvi_iv(k, T_val)) for T_val in Ts]

#     result = check_surface(k, slices, plot=True)
#     print(result["summary"])
