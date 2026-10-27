"""Market 14-calendar-day variance index and sampled GROMM v1 P/Q levels.

Market levels use raw SPX option mids and the Cboe discrete-strike formula,
with total variance interpolated between Friday expiries to 14 days. Simulated
levels use the full-strike log contract at the same expiry clocks. They are
conditional September 3 AM samples, not historical fitted-model observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from cdxcore.err import verify_inp

from quantdata import SubDir, env
from quantdata.biz_calendar import AM_EXIRY_SUBTRACT, DAYS_PER_YEAR, get_business_days
from quantdata.sdlv.daily.front_spot.calibration import estimate_historical_risk_neutral_drift
from quantdata.sdlv.daily.front_spot.estimation import (
    BACKGROUND_ANNUAL_VOL,
    align_fixed_contract_history,
    fit_achievable_physical_drift,
    fit_joint_innovations,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    advance_front_density,
    decode_density_tower_from_front,
    one_day_physical_substepped,
    risk_neutral_dlv_drift,
    stabilized_relative_drift_increment,
)
from quantdata.sdlv.daily.vix_options import log_variance_level
from quantdata.vixdata.formula import (
    constant_maturity_vix,
    discrete_total_variance,
    select_vix_strip,
)

ORIGIN = date(2025, 8, 29)
SAMPLE_DATE = date(2025, 9, 3)
TRAINING_RUN = Path('/home/hans/quantdata_data/experiments/quant_data/sdlv/Gromm-v1-4,3/20261023-kVyePV')
FROZEN = TRAINING_RUN / 'rolling252-full-q-20261026-wpxWhA/full'
ROLLING = TRAINING_RUN / 'rolling252-unconditional-20261024-po3Jf1/2025-08-29-cpu-XYDacI'


def emit(stage: str, completed: int, total: int, started: float) -> None:
    elapsed = perf_counter() - started
    print(json.dumps(dict(stage=stage, completed=completed, total=total, elapsed_seconds=elapsed,
                          remaining_seconds=elapsed*(total-completed)/completed if completed else None)), flush=True)


def eligible_pair(day: date, raw: pd.DataFrame, sessions: list[date]) -> list[tuple[date, bool, float, float]]:
    """Select actual Friday books bracketing 14 days, retaining settlement clocks."""
    available = set(zip(raw.exdate.astype(str), raw.am_settlement.astype(float)))
    candidates = []
    for offset in range(1, 25):
        nominal = day + timedelta(days=offset)
        if nominal.weekday() != 4:
            continue
        settlement = max(x for x in sessions if x <= nominal)
        is_am = 15 <= nominal.day <= 21
        if (settlement.isoformat(), float(is_am)) not in available:
            continue
        calendar = (settlement-day).days - (405 if is_am else 15)/1440
        business = sessions.index(settlement) - sessions.index(day) - (AM_EXIRY_SUBTRACT if is_am else 0)
        if calendar > 0 and business > 0:
            candidates.append((settlement, is_am, calendar, business))
    near = [x for x in candidates if x[2] <= 14]
    far = [x for x in candidates if x[2] > 14]
    if not near or not far:
        raise ValueError('Raw market books lack Friday expiries bracketing 14 calendar days')
    return [max(near, key=lambda x: x[2]), min(far, key=lambda x: x[2])]


def market_term(raw: pd.DataFrame, zero: pd.DataFrame, expiry: tuple) -> tuple[float, dict]:
    """Compute discrete total variance from an observed, unfiltered option book."""
    settlement, is_am, days, business = expiry
    rows = raw[(raw.exdate.astype(str) == str(settlement)) & (raw.am_settlement.astype(float) == float(is_am))]
    rows = rows[(rows.exercise_style == 'E') & (rows.ss_flag.astype(str) == '0')]
    if rows.duplicated(['strike_price', 'cp_flag']).any():
        raise ValueError('Duplicate strike/side in market book')
    book = rows.pivot(index='strike_price', columns='cp_flag', values=['best_bid', 'best_offer']).sort_index()
    curve = zero.sort_values('days')
    rate = np.interp(days, curve.days.to_numpy(dtype=float), curve.rate.to_numpy(dtype=float))/100
    discount = float(np.exp(-rate*days/365))
    def values(field, side):
        return book[(field, side)].to_numpy(dtype=float, na_value=np.nan)
    strip = select_vix_strip(strikes=book.index.to_numpy(dtype=float), discount_factor=discount,
                             call_bids=values('best_bid', 'C'), call_asks=values('best_offer', 'C'),
                             put_bids=values('best_bid', 'P'), put_asks=values('best_offer', 'P'))
    total = discrete_total_variance(strip, discount_factor=discount)
    return total, dict(expiry=str(settlement), am=is_am, calendar_days=days, business_days=business,
                       forward=strip.forward, discount=discount, total_variance=total, strikes=len(strip.strikes))


def construct_market(output: Path) -> pd.DataFrame:
    """Read the 891 historical raw quote dates; retain every unavailable-date reason."""
    with np.load(TRAINING_RUN / 'market.npz', allow_pickle=False) as data:
        dates = [date.fromisoformat(x) for x in data['dates'].astype(str)]
    sessions = get_business_days(dates[0]-timedelta(days=10), dates[-1]+timedelta(days=50))
    cache = env.locations('widb', default_fmt=SubDir.PANDAS_PARQUET).raw
    records, audits = [], []
    started = perf_counter()
    for i, day in enumerate(dates):
        record = dict(date=str(day), market_vix_2w=None, reason=None)
        raw, zero = cache.read(f'options/108105/{day}'), cache.read(f'zero_curve/{day}')
        try:
            if raw is None or zero is None:
                raise ValueError('Missing cached raw market options or zero curve')
            pair = eligible_pair(day, raw, sessions)
            terms = [market_term(raw, zero, term) for term in pair]
            record['market_vix_2w'] = constant_maturity_vix(near_days=pair[0][2], next_days=pair[1][2],
                near_total_variance=terms[0][0], next_total_variance=terms[1][0], target_days=14)
            audits.extend([dict(date=str(day), **term[1]) for term in terms])
        except (ValueError, KeyError) as error:
            record['reason'] = str(error)
        records.append(record)
        if i == 0 or (i+1) % 25 == 0 or i+1 == len(dates):
            emit('raw-market-2w', i+1, len(dates), started)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'historical_market_2w.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'market_expiry_audit.csv', index=False)
    return frame


def fit_physical(output: Path) -> np.ndarray:
    """Fit only the P measure change on the frozen calibration's prior 252 dates."""
    started = perf_counter()
    with np.load(FROZEN / 'initial-fit.npz', allow_pickle=False) as data:
        raw = data['raw'].copy()
    history = align_fixed_contract_history(dataset_path=TRAINING_RUN/'market.npz', raw_grids=raw,
        training_start=date(2024, 8, 20), training_end=date(2025, 8, 28))
    innovations = fit_joint_innovations(history, factor_rank=8, smoothing_sigma=(1., 2.))
    with np.load(FROZEN / 'diffusion.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(innovations.annualized_loadings, saved['annualized_loadings'], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(history.origin_indices, saved['origin_indices'])
        np.testing.assert_array_equal(history.endpoint_indices, saved['endpoint_indices'])
        innovations = replace(innovations, annualized_loadings=saved['annualized_loadings'].copy())
    emit('validate-frozen-diffusion', 1, 1, started)
    risk_neutral = estimate_historical_risk_neutral_drift(history, innovations,
        dataset_path=TRAINING_RUN/'market.npz', raw_grids=raw, batch_size=32)
    physical = fit_achievable_physical_drift(history, innovations, risk_neutral.log_dlvsq_drift,
                                           risk_neutral.coordinate_weights)
    lam = physical.constant_market_price_of_risk
    if not np.all(np.isfinite(lam)):
        raise FloatingPointError('Physical market price of risk is non-finite')
    np.savez(output/'physical_fit.npz', constant_market_price_of_risk=lam,
             conditional_intercept=innovations.conditional_intercept, conditional_slope=innovations.conditional_slope,
             origin_indices=history.origin_indices, endpoint_indices=history.endpoint_indices)
    report = dict(training_start='2024-08-20', training_end='2025-08-28', pair_count=len(history.origin_indices),
                  diffusion_matches_frozen=True, constant_market_price_of_risk=lam.tolist(),
                  constant_achievable_fraction=physical.constant_achievable_fraction,
                  statewise_achievable_fraction=physical.achievable_fraction,
                  convention='dW_Q = dW_P + lambda dt; constant density-weighted projection of historical AR drift',
                  source='New P projection on the same strictly prior rolling history; existing Q surface and diffusion unchanged.')
    (output/'physical_fit.json').write_text(json.dumps(report, indent=2)+'\n')
    emit('fit-P-measure-change', 1, 1, started)
    return lam


def partial_tower(support, state, loadings, normals, risk_price, *, fraction: float):
    """Advance the front bridge and DLV dynamics over a partial day under P or Q."""
    dt = 1/DAYS_PER_YEAR
    size = fraction*dt/len(normals)
    premium = jnp.einsum('jnf,f->jn', loadings, risk_price)
    def step(carry, inputs):
        front, dlv = carry
        i, z = inputs
        remaining = dt-i*size
        _, _, drift = risk_neutral_dlv_drift(support, front, dlv, loadings, dt=dt, front_horizon=remaining)
        dlv = dlv*jnp.exp(stabilized_relative_drift_increment(drift+premium, size)
                          -.5*jnp.sum(loadings**2, axis=-1)*size
                          +jnp.sqrt(size)*jnp.einsum('jnf,pf->pjn', loadings, z))
        front = advance_front_density(front, z[:, 0]+risk_price[0]*jnp.sqrt(size),
            remaining_time=remaining, step_size=size, final_step=False)
        return (front, dlv), None
    (front, dlv), _ = jax.lax.scan(step, (state.front_density, state.future_dlvsq), (jnp.arange(len(normals)), normals))
    return decode_density_tower_from_front(support, front, dlv, dt=dt)


def conditional_total(support: np.ndarray, tower: np.ndarray, *, absolute_business: float,
                      calendar_days: float, elapsed_days: float = 1.2) -> np.ndarray:
    """Interpolate remaining fixed-expiry densities; include independent background variance."""
    # After one whole day, tower[0] is the original third business-day expiry.
    index = absolute_business-3
    lower, upper = int(np.floor(index)), int(np.ceil(index))
    if lower < 0 or upper >= tower.shape[-2]:
        raise ValueError('Required conditional density is outside the remaining tower')
    density = (1-(index-lower))*tower[:, lower] + (index-lower)*tower[:, upper]
    levels = log_variance_level(support, density,
        business_years=(absolute_business-elapsed_days)/DAYS_PER_YEAR, calendar_years=calendar_days/365)
    return (levels/100)**2*(calendar_days/365)


def sample_models(output: Path, lam: np.ndarray, *, paths: int, batch_size: int = 128) -> dict:
    """Draw independent antithetic pairs, with common Brownian draws for P and Q."""
    verify_inp(paths >= 4 and paths % batch_size == 0 and batch_size % 2 == 0,
               lambda: 'Paths must be a multiple of the even batch size')
    with np.load(FROZEN/'diffusion.npz', allow_pickle=False) as saved:
        support, dlv, tower, loadings = (saved[x].copy() for x in ('support', 'dlvsq', 'density_tower', 'annualized_loadings'))
    sessions = get_business_days(ORIGIN, date(2025, 10, 10))
    # Sample at September 3 AM. Near PM and next AM Friday expiries bracket 14 days.
    terms = [(date(2025, 9, 12), False), (date(2025, 9, 19), True)]
    expiries = [(sessions.index(day)-(AM_EXIRY_SUBTRACT if am else 0),
                 (day-SAMPLE_DATE).days+(0 if am else 390/1440)) for day, am in terms]
    alpha = (14-expiries[0][1])/(expiries[1][1]-expiries[0][1])
    daily = jax.jit(one_day_physical_substepped, static_argnames=('expiry_dt', 'time_power'))
    partial = jax.jit(partial_tower, static_argnames=('fraction',))
    samples = {}
    for measure, risk_price in [('Q', np.zeros_like(lam)), ('P', lam)]:
        rng = np.random.default_rng(20261028)
        blocks = []
        started = perf_counter()
        for start in range(0, paths, batch_size):
            z = rng.standard_normal((2, 8, batch_size//2, 8))
            z = np.stack((z, -z), axis=3).reshape(2, 8, batch_size, 8)
            state = RiskNeutralState(jnp.broadcast_to(tower[0], (batch_size, len(support))),
                jnp.broadcast_to(dlv[1:], (batch_size,)+dlv[1:].shape), jnp.ones(batch_size))
            state = daily(jnp.asarray(support), state, jnp.asarray(loadings), jnp.asarray(risk_price),
                          jnp.asarray(z[0]), jnp.zeros(batch_size)).state
            sampled = np.asarray(partial(jnp.asarray(support), state, jnp.asarray(loadings[:23]),
                jnp.asarray(z[1]), jnp.asarray(risk_price), fraction=.2))
            total = [conditional_total(support, sampled, absolute_business=b, calendar_days=c) for b,c in expiries]
            levels = 100*np.sqrt(((1-alpha)*total[0]+alpha*total[1])*365/14)
            if not np.all(np.isfinite(levels)):
                raise FloatingPointError(f'Non-finite {measure} levels')
            blocks.append(levels)
            emit(f'sample-{measure}', start+batch_size, paths, started)
        samples[measure] = np.concatenate(blocks)
        np.save(output/f'sampled_{measure}_2w.npy', samples[measure])
    return dict(expiry_business_calendar_days=expiries, alpha=alpha, paths_per_measure=paths,
                substeps_per_day=8, seed=20261028, sample_business_days=1.2)


def plot_results(output: Path, market: pd.DataFrame) -> dict:
    """Use equal bins and percentages; preserve and report the complete sample tails."""
    os.environ.setdefault('MPLCONFIGDIR', str(output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    values = {'Market history': market.market_vix_2w.dropna().to_numpy(dtype=float),
              'GROMM Q': np.load(output/'sampled_Q_2w.npy'), 'GROMM P': np.load(output/'sampled_P_2w.npy')}
    maximum = max(x.max() for x in values.values())
    bins = np.arange(0, 2*np.ceil(maximum/2)+2, 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, layout='constrained')
    stats = {}
    for ax, (label, v), color in zip(axes, values.items(), ['#30343B','#0072B2','#D55E00'], strict=True):
        counts, _ = np.histogram(v, bins=bins)
        assert counts.sum() == len(v)
        ax.hist(v, bins=bins, weights=np.full(len(v), 100/len(v)), color=color, alpha=.85, edgecolor='white')
        ax.axvline(v.mean(), color='#CC79A7', ls='--', label=f'Mean {v.mean():.2f}')
        ax.set(title=f'{label} · {len(v):,} observations', xlabel='14-day variance index (volatility points)')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(frameon=False)
        stats[label] = dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), population_std=float(v.std(ddof=0)),
                           p05=float(np.quantile(v,.05)), p95=float(np.quantile(v,.95)), maximum=float(v.max()))
        if label != 'Market history':
            pairs=(v[::2]+v[1::2])/2
            stats[label]['mean_mc_se'] = float(pairs.std(ddof=1)/np.sqrt(len(pairs)))
    axes[0].set_ylabel('Observations (% per 2-point bin)')
    fig.suptitle('Market 2-week VIX-style index versus sampled GROMM v1 (4,3)\n'
                 'Market: Jan 2022–Aug 2025 | GROMM: 3 Sep 2025 AM, conditional on 29 Aug 2025', fontsize=14)
    fig.supxlabel('Market: raw SPX option mids, discrete OTM strips. GROMM: full-strike variance proxy at matching expiry clocks.\n'
                  'Historical distribution across dates versus conditional P/Q distributions across simulated paths.', fontsize=9)
    for extension in ('png','pdf'):
        fig.savefig(output/f'histograms_2w_full_range.{extension}', dpi=180)
    # A clearly marked zoom leaves all observations in the normalization and
    # records the displayed-tail probabilities instead of silently dropping them.
    for ax, (label,v) in zip(axes, values.items(), strict=True):
        ax.set_xlim(0,60)
        ax.text(.97,.8,f'Above 60: {100*np.mean(v>60):.2f}%',transform=ax.transAxes,ha='right',fontsize=9)
    for extension in ('png','pdf'):
        fig.savefig(output/f'histograms_2w.{extension}', dpi=180)
    plt.close(fig)
    return stats


def run(output: Path, paths: int) -> None:
    """Build fresh external artifacts without refitting any historical SANOS surface."""
    output.mkdir(parents=True, exist_ok=False)
    if not jax.config.x64_enabled:
        raise RuntimeError('Set JAX_ENABLE_X64=1')
    market = construct_market(output)
    lam = fit_physical(output)
    simulation = sample_models(output, lam, paths=paths)
    stats = plot_results(output, market)
    report = dict(model='Gromm-v1-4,3', tenor_calendar_days=14, origin=str(ORIGIN), sample_date=str(SAMPLE_DATE),
                  market_unavailable=market.loc[market.market_vix_2w.isna(), 'reason'].value_counts().to_dict(),
                  simulation=simulation, statistics=stats,
                  frozen_diffusion=str(FROZEN/'diffusion.npz'),
                  frozen_diffusion_sha256=hashlib.sha256((FROZEN/'diffusion.npz').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=['Custom 14-day index, not published Cboe 30-day VIX.',
                    'Market observations use raw quote mids; sampled index uses continuous full-strike log variance.',
                    'Historic unconditional distribution is not the same estimand as conditional P/Q distributions.',
                    'P uses the existing constant risk-price projection method, newly estimated on the same causal rolling window.'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paths',type=int,default=8192)
    run(**vars(parser.parse_args()))
