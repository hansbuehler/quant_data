"""
Out-of-sample evaluation of the IV Kalman filter on matched trades.

Usage
-----
    from iv_evaluation import evaluate_kalman_iv
    metrics = evaluate_kalman_iv(df_quotes_iv, df_trades_iv)

Expected columns
----------------
df_quotes_iv : ts_ns, strike, is_call, observed_iv, bid_iv, ask_iv, kalman_iv
df_trades_iv : ts_ns, strike, is_call, observed_iv, bid_iv, ask_iv,
               kalman_matched, trade_residual
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def evaluate_kalman_iv(
    df_quotes_iv: pl.DataFrame,
    df_trades_iv: pl.DataFrame,
) -> dict:
    """
    Compute metrics and plot diagnostics for Kalman IV filter on held-out trades.

    Returns a dict with scalar metrics and the Ljung-Box result table.
    """
    df_t = df_trades_iv.to_pandas()
    df_q = df_quotes_iv.to_pandas()

    res = df_t["trade_residual"].dropna()

    # ── 1. Summary scalars ────────────────────────────────────────────────────
    rmse = np.sqrt((res**2).mean())
    mae = res.abs().mean()
    bias = res.mean()

    # ── 2. Spread-normalized residuals + hit rate ─────────────────────────────
    spread = (df_t["ask_iv"] - df_t["bid_iv"]).replace(0, np.nan)
    norm_res = (2 * df_t["trade_residual"] / spread).dropna()
    hit_rate = (
        (df_t["kalman_matched"] >= df_t["bid_iv"]) & (df_t["kalman_matched"] <= df_t["ask_iv"])
    ).mean()

    # ── 3. Naive baseline: last quote mid-IV before each trade ───────────────
    df_t_s = df_t.sort_values("ts_ns")
    df_q_s = df_q.sort_values("ts_ns")
    naive_parts = []
    for (strike, is_call), grp_t in df_t_s.groupby(["strike", "is_call"], sort=False):
        grp_q = df_q_s[(df_q_s["strike"] == strike) & (df_q_s["is_call"] == is_call)]
        if grp_q.empty:
            continue
        merged = pd.merge_asof(
            grp_t[["ts_ns", "observed_iv"]].rename(columns={"observed_iv": "trade_iv"}),
            grp_q[["ts_ns", "observed_iv"]].rename(columns={"observed_iv": "quote_mid_iv"}),
            on="ts_ns",
            direction="backward",
        )
        merged["naive_residual"] = merged["trade_iv"] - merged["quote_mid_iv"]
        naive_parts.append(merged)

    naive_res = (
        pd.concat(naive_parts)["naive_residual"].dropna() if naive_parts else pd.Series(dtype=float)
    )
    naive_rmse = np.sqrt((naive_res**2).mean()) if len(naive_res) else np.nan
    rmse_ratio = rmse / naive_rmse if naive_rmse and naive_rmse > 0 else np.nan

    # ── 4. Autocorrelation / Ljung-Box ───────────────────────────────────────
    lb_result = acorr_ljungbox(res.values, lags=[10, 20], return_df=True)

    # ── 5. Time-of-day (ET) ──────────────────────────────────────────────────
    df_t["time_et"] = pd.to_datetime(df_t["ts_ns"], unit="ns", utc=True).dt.tz_convert(
        "America/New_York"
    )
    df_t["minute"] = df_t["time_et"].dt.hour * 60 + df_t["time_et"].dt.minute
    tod_rmse = (
        df_t.groupby("minute")["trade_residual"]
        .apply(lambda x: np.sqrt((x.dropna() ** 2).mean()))
        .sort_index()
    )

    # ── 6. Residual by strike bucket (proxy for moneyness) ───────────────────
    labels, bins = pd.qcut(df_t["strike"], q=5, retbins=True, duplicates="drop")
    df_t["strike_bucket"] = labels
    mono_stats = df_t.groupby("strike_bucket", observed=True)["trade_residual"].agg(
        rmse=lambda x: np.sqrt((x.dropna() ** 2).mean()),
        count="count",
    )
    bucket_labels = [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(bins[:-1], bins[1:])]

    # ── 7. Cumulative residuals ───────────────────────────────────────────────
    cumres = res.reset_index(drop=True).cumsum()

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # Row 0: residual time series + cumulative residuals + histogram
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(
        df_t["ts_ns"].iloc[res.index],
        res.values,
        alpha=0.35,
        lw=0.5,
        color="steelblue",
        label="residual",
    )
    ax0_r = ax0.twinx()
    ax0_r.plot(
        range(len(cumres)), cumres.values, color="firebrick", lw=1.2, alpha=0.7, label="cumulative"
    )
    ax0_r.set_ylabel("cumulative residual", color="firebrick", fontsize=8)
    ax0.axhline(0, color="k", lw=0.8)
    ax0.set_title("Trade residuals + cumulative drift")
    ax0.set_ylabel("residual (IV)")
    ax0.set_xlabel("timestamp (ns)")

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.hist(res, bins=60, color="steelblue", alpha=0.7, density=True)
    x_grid = np.linspace(res.min(), res.max(), 300)
    ax1.plot(
        x_grid,
        stats.norm.pdf(x_grid, res.mean(), res.std()),
        color="crimson",
        lw=1.5,
        label="N(μ,σ)",
    )
    ax1.set_title("Residual distribution")
    ax1.set_xlabel("residual (IV)")
    ax1.legend(fontsize=8)

    # Row 1: QQ + ACF + spread-normalized histogram
    ax2 = fig.add_subplot(gs[1, 0])
    stats.probplot(res, dist="norm", plot=ax2)
    ax2.set_title("QQ plot (vs Normal)")

    ax3 = fig.add_subplot(gs[1, 1])
    plot_acf(res, lags=40, ax=ax3, zero=False, alpha=0.05)
    ax3.set_title("ACF of residuals")
    ax3.set_xlabel("lag")

    ax4 = fig.add_subplot(gs[1, 2])
    clipped = norm_res.clip(-3, 3)
    ax4.hist(clipped, bins=60, color="darkorange", alpha=0.7, density=True)
    ax4.axvline(-1, color="k", lw=1, ls="--")
    ax4.axvline(1, color="k", lw=1, ls="--", label="±half-spread")
    ax4.set_title(f"Spread-normalised residual\nhit rate = {hit_rate:.1%}")
    ax4.set_xlabel("residual / half-spread")
    ax4.legend(fontsize=8)

    # Row 2: time-of-day + moneyness + Kalman vs naive
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(tod_rmse.index, tod_rmse.values, color="teal", lw=1.2)
    ax5.set_title("RMSE by time-of-day (ET)")
    ax5.set_xlabel("minute of day")
    ax5.set_ylabel("RMSE")

    ax6 = fig.add_subplot(gs[2, 1])
    x_pos = range(len(mono_stats))
    ax6.bar(x_pos, mono_stats["rmse"].values, color="mediumslateblue", alpha=0.8)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(bucket_labels[: len(mono_stats)], fontsize=8, rotation=20)
    ax6.set_title("RMSE by strike bucket\n(proxy for moneyness)")
    ax6.set_ylabel("RMSE")

    ax7 = fig.add_subplot(gs[2, 2])
    bars = ax7.bar(
        ["Kalman", "Naive\n(last mid-IV)"],
        [rmse, naive_rmse],
        color=["steelblue", "salmon"],
        alpha=0.85,
    )
    for bar, val in zip(bars, [rmse, naive_rmse]):
        ax7.text(
            bar.get_x() + bar.get_width() / 2, val + 0.0002, f"{val:.4f}", ha="center", fontsize=8
        )
    ax7.set_title(f"RMSE: Kalman vs naive\nratio = {rmse_ratio:.3f}  (<1 → Kalman wins)")
    ax7.set_ylabel("RMSE")

    plt.suptitle("Kalman IV Surface — Out-of-Sample Evaluation on Trades", fontsize=13, y=1.01)
    plt.show()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"{'RMSE':<20} {rmse:.5f}")
    print(f"{'MAE':<20} {mae:.5f}")
    print(f"{'Bias':<20} {bias:.5f}")
    print(f"{'Hit rate':<20} {hit_rate:.1%}")
    print(f"{'Naive RMSE':<20} {naive_rmse:.5f}")
    print(f"{'RMSE ratio':<20} {rmse_ratio:.3f}  (<1 = Kalman beats naive)")
    print(f"\nLjung-Box (H0: no autocorrelation in residuals):")
    print(lb_result.to_string())

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "hit_rate": hit_rate,
        "naive_rmse": naive_rmse,
        "rmse_ratio": rmse_ratio,
        "ljung_box": lb_result,
    }
