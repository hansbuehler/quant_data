"""Price short VIX options from a frozen daily GROMM v1 Q calibration.

The underlying is the full-strike log-contract variance proxy at settlement,
not a simulation of Cboe's opening auction or its future discrete strike strip.
No VIX quotes are used to fit or rescale the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from cdxcore.err import verify_inp

from quantdata.biz_calendar import AM_EXIRY_SUBTRACT, DAYS_PER_YEAR, get_business_days
from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    advance_front_density,
    decode_density_tower_from_front,
    one_day_risk_neutral_substepped,
    risk_neutral_dlv_drift,
    stabilized_relative_drift_increment,
)
from quantdata.widb.client import WIDBClient


def log_variance_level(support: np.ndarray, density: np.ndarray, *, business_years: float,
                       calendar_years: float) -> np.ndarray:
    """Full-strike variance in VIX points, normalized by each path's forward."""
    verify_inp(business_years > 0 and calendar_years > 0, lambda: "Maturities must be positive")
    if not np.all(np.isfinite(support)) or np.any(support <= 0) or not np.all(np.isfinite(density)) or np.min(density) < -1e-12:
        raise ValueError("Invalid support or conditional probabilities")
    if not np.allclose(density.sum(axis=-1), 1, atol=1e-9, rtol=0):
        raise ValueError("Conditional density is not normalized")
    mean = density @ support
    total = -2 * (density @ np.log(support) - np.log(mean)) + BACKGROUND_ANNUAL_VOL**2 * business_years
    if np.any(total < -1e-10):
        raise ValueError("Negative conditional log variance")
    return 100 * np.sqrt(np.maximum(total, 0) / calendar_years)


def payoff_statistics(levels: np.ndarray, strikes: np.ndarray, calls: np.ndarray,
                      *, discount: float) -> tuple[np.ndarray, np.ndarray]:
    """Discounted Q means and standard errors over independent antithetic pairs."""
    verify_inp(len(levels) >= 4 and len(levels) % 2 == 0, lambda: "Need at least two complete pairs")
    payoffs = discount * np.maximum((levels[:, None] - strikes) * np.where(calls, 1, -1), 0)
    pairs = (payoffs[::2] + payoffs[1::2]) / 2
    return pairs.mean(axis=0), pairs.std(axis=0, ddof=1) / np.sqrt(len(pairs))


def _partial_day(support, state, loadings, normals, *, fraction: float):
    dt = 1 / DAYS_PER_YEAR
    size = fraction * dt / len(normals)
    def step(carry, inputs):
        front, dlv = carry
        index, z = inputs
        remaining = dt - index * size
        _, _, drift = risk_neutral_dlv_drift(support, front, dlv, loadings, dt=dt, front_horizon=remaining)
        dlv = dlv * jnp.exp(stabilized_relative_drift_increment(drift, size)
                           - .5 * jnp.sum(loadings**2, axis=-1) * size
                           + jnp.sqrt(size) * jnp.einsum('jnf,pf->pjn', loadings, z))
        front = advance_front_density(front, z[:, 0], remaining_time=remaining, step_size=size, final_step=False)
        return (front, dlv), None
    (front, dlv), _ = jax.lax.scan(step, (state.front_density, state.future_dlvsq), (jnp.arange(len(normals)), normals))
    # Future SANOS intervals remain one day; the first future expiry is now
    # (2-fraction) days away, following the partially revealed front expiry.
    return decode_density_tower_from_front(support, front, dlv, dt=dt)


def run(*, diffusion: Path, quotes: Path, output: Path, paths: int = 8192,
        batch_size: int = 128, substeps: int = 8, seed: int = 20261027) -> dict:
    """Evaluate only expiries supported by the original 25-day surface."""
    verify_inp(batch_size >= 4 and paths >= 4 and paths % batch_size == 0 and batch_size % 2 == 0,
               lambda: "Paths must be divisible by an even batch size")
    verify_inp(substeps >= 1, lambda: "Substeps must be positive")
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1")
    output.mkdir(parents=True, exist_ok=False)
    origin = date(2025, 8, 29)
    raw = pd.read_parquet(quotes)
    curve = WIDBClient().zero_curve(origin)
    curve.to_parquet(output / 'zero_curve.parquet', index=False)
    sessions = get_business_days(origin, date(2026, 8, 1))
    with np.load(diffusion, allow_pickle=False) as saved:
        support = saved['support'].copy()
        tower = saved['density_tower'].copy()
        dlv = saved['dlvsq'].copy()
        loadings = saved['annualized_loadings'].copy()
    if tower.shape != (25, len(support)) or dlv.shape != (25, len(support)-2) or loadings.shape != (24, len(support)-2, 8):
        raise ValueError('Expected the frozen 25-expiry, rank-eight v1 calibration')
    coverage = []
    for expiry in sorted(set(raw.exdate)):
        expiry = pd.Timestamp(expiry).date()
        target = expiry + timedelta(days=30)
        settlement = sessions.index(expiry) - AM_EXIRY_SUBTRACT
        target_business = sessions.index(target)  # All downloaded expiries +30 are trading days.
        coverage.append(dict(expiry=str(expiry), settlement_business_days=settlement,
                             spx_expiry=str(target), spx_business_days=target_business,
                             supported=target_business <= len(tower)))
    pd.DataFrame(coverage).to_csv(output / 'expiry_coverage.csv', index=False)
    reports = []
    for entry in coverage:
        if not entry['supported']:
            continue
        supported_book = raw[raw.exdate.astype(str) == entry['expiry']]
        if not supported_book.symbol.str.startswith('VIXW ').all():
            raise ValueError('Only weekly PM-SPX settlement is implemented')
        elapsed_days = entry['settlement_business_days']
        whole_days = int(np.floor(elapsed_days))
        fraction = elapsed_days - whole_days
        if not 0 < fraction < 1:
            raise ValueError('This settlement path requires a fractional final day')
        target_index = entry['spx_business_days'] - whole_days - 2
        business_years = (entry['spx_business_days'] - elapsed_days) / DAYS_PER_YEAR
        # Weekly VIX settles to PM SPX options 30 days and 390 minutes later.
        calendar_years = (30 + 390 / 1440) / 365
        daily_step = jax.jit(one_day_risk_neutral_substepped, static_argnames=('expiry_dt', 'time_power'))
        partial = jax.jit(_partial_day, static_argnames=('fraction',))
        generator = np.random.default_rng(seed)
        levels = []
        started = perf_counter()
        for start in range(0, paths, batch_size):
            z = generator.standard_normal((whole_days + 1, substeps, batch_size // 2, loadings.shape[-1]))
            z = np.stack((z, -z), axis=3).reshape(whole_days + 1, substeps, batch_size, loadings.shape[-1])
            state = RiskNeutralState(jnp.broadcast_to(tower[0], (batch_size, len(support))),
                                     jnp.broadcast_to(dlv[1:], (batch_size,) + dlv[1:].shape), jnp.ones(batch_size))
            for day in range(whole_days):
                state = daily_step(jnp.asarray(support), state, jnp.asarray(loadings[:24-day]),
                                   jnp.asarray(z[day]), jnp.zeros(batch_size)).state
            terminal = partial(jnp.asarray(support), state, jnp.asarray(loadings[:24-whole_days]),
                               jnp.asarray(z[-1]), fraction=fraction)
            density = np.asarray(terminal[:, target_index])
            levels.append(log_variance_level(support, density, business_years=business_years, calendar_years=calendar_years))
            elapsed = perf_counter() - started
            print(json.dumps(dict(expiry=entry['expiry'], completed=start+batch_size, total=paths,
                                  elapsed_seconds=elapsed, remaining_seconds=elapsed*(paths-start-batch_size)/(start+batch_size))), flush=True)
        levels = np.concatenate(levels)
        book = raw[raw.exdate.astype(str) == entry['expiry']].copy()
        days = (date.fromisoformat(entry['expiry']) - origin).days - 405 / 1440
        rate = np.interp(days, curve.days.to_numpy(dtype=float), curve.rate.to_numpy(dtype=float)) / 100
        discount = float(np.exp(-rate * days / 365))
        book['model_price'], book['mc_se'] = payoff_statistics(levels, book.strike_price.to_numpy(), book.cp_flag.to_numpy() == 'C', discount=discount)
        book['mid'] = (book.best_bid + book.best_offer) / 2
        book['valid_quote'] = (book.best_bid >= 0) & (book.best_offer >= book.best_bid) & (book.best_offer > 0)
        book['inside_spread'] = (book.model_price >= book.best_bid) & (book.model_price <= book.best_offer)
        book['model_minus_mid'] = book.model_price - book.mid
        book.to_csv(output / f"comparison_{entry['expiry']}.csv", index=False)
        np.save(output / f"levels_{entry['expiry']}.npy", levels)
        valid = book[book.valid_quote]
        reports.append(dict(**entry, paths=paths, substeps=substeps, discount=discount,
                            expected_vix_proxy=float(levels.mean()), quotes=len(valid),
                            inside_spread=int(valid.inside_spread.sum()),
                            mae_mid=float(valid.model_minus_mid.abs().mean()),
                            maximum_mc_se=float(valid.mc_se.max())))
    report = dict(model='Gromm v1 (4,3)', origin=str(origin), seed=seed, results=reports,
                  diffusion=str(diffusion), diffusion_sha256=hashlib.sha256(diffusion.read_bytes()).hexdigest(),
                  quotes=str(quotes), quotes_sha256=hashlib.sha256(quotes.read_bytes()).hexdigest(),
                  limitations=['Continuous full-strike log-contract proxy, not discrete VRO opening auction.',
                               'AM timing uses the project 0.8-business-day adjustment; weekly SPX tenor is 30 days plus 390 minutes.',
                               'No extrapolation, VIX fitting, or rescaling; independent pair-based sample standard errors.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def plot_comparison(output: Path) -> Path:
    """Plot quoted spreads and Q estimates over the informative strike range."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    report = json.loads((output / 'report.json').read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, side, title in zip(axes, ('C', 'P'), ('Calls', 'Puts'), strict=True):
        for result in report['results']:
            book = pd.read_csv(output / f"comparison_{result['expiry']}.csv")
            book = book[(book.cp_flag == side) & book.strike_price.between(12, 30)].sort_values('strike_price')
            axis.fill_between(book.strike_price, book.best_bid, book.best_offer,
                              color='gray', alpha=.3, label='WRDS bid–ask')
            axis.plot(book.strike_price, book.mid, '.-', color='black', label='WRDS midpoint')
            axis.errorbar(book.strike_price, book.model_price, yerr=1.96*book.mc_se,
                          color='tab:blue', marker='.', label='v1 Q proxy ± 1.96 MC SE')
        axis.set(title=title, xlabel='VIX strike', ylim=(0, 15))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel('Option premium (VIX points; $100 per point)')
    figure.suptitle('Gromm v1 (4,3) — 29 August 2025 quotes, 3 September expiry\n'
                    'Continuous log-contract settlement proxy; no VIX calibration')
    figure.tight_layout()
    path = output / 'comparison.png'
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diffusion', 'quotes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--paths', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--substeps', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))
