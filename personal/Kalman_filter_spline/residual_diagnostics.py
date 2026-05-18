"""
Diagnostic plots for SVI + RBF family Kalman filter residuals.

Expects df_trade_fam with columns:
  ts_ns, quote_ts_ns, strike, is_call, log_moneyness, T,
  observed_iv, bid_iv, ask_iv, bid_price, ask_price, forward,
  ts_ns_right, family_matched, svi_component, delta_iv, trade_residual
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_residual_diagnostics(df_trade_fam: pl.DataFrame):
    # Clean: drop rows with null/nan residuals
    df = df_trade_fam.filter(
        pl.col("trade_residual").is_not_null()
        & pl.col("trade_residual").is_not_nan()
        & pl.col("log_moneyness").is_not_null()
        & pl.col("log_moneyness").is_not_nan()
    ).sort("ts_ns")

    ts = df["ts_ns"].to_numpy()
    k = df["log_moneyness"].to_numpy()
    is_call = df["is_call"].to_numpy()
    res = df["trade_residual"].to_numpy()
    obs_iv = df["observed_iv"].to_numpy()
    fam_iv = df["family_matched"].to_numpy() if "family_matched" in df.columns else None

    # Bucket labels: OTM puts (k<0 & put), OTM calls (k>=0 & call),
    #                ITM puts (k>=0 & put), ITM calls (k<0 & call)
    otm_call = is_call & (k >= 0)
    otm_put = (~is_call) & (k < 0)
    itm_call = is_call & (k < 0)
    itm_put = (~is_call) & (k >= 0)

    buckets = {
        "OTM calls (k≥0, call)": otm_call,
        "OTM puts  (k<0, put)": otm_put,
        "ITM calls (k<0, call)": itm_call,
        "ITM puts  (k≥0, put)": itm_put,
    }
    colors = {
        "OTM calls (k≥0, call)": "tab:blue",
        "OTM puts  (k<0, put)": "tab:red",
        "ITM calls (k<0, call)": "tab:cyan",
        "ITM puts  (k≥0, put)": "tab:orange",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ─────────────────────────────────────────────────────────────────
    # Plot 1: cumulative residual split by ITM/OTM × call/put
    # ─────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for label, mask in buckets.items():
        if mask.sum() == 0:
            continue
        cum = np.cumsum(res[mask])
        ax.plot(ts[mask], cum, label=f"{label}  (n={mask.sum():,})", color=colors[label], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("cumulative residual")
    ax.set_title("Cumulative residual by bucket\n(flat = unbiased, sloped = systematic bias)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 2: mean residual vs log-moneyness, per bucket
    # ─────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    bin_w = 0.005
    for label, mask in buckets.items():
        if mask.sum() < 50:
            continue
        k_b = k[mask]
        r_b = res[mask]
        # Bin by k
        bins = np.round(k_b / bin_w) * bin_w
        df_bin = (
            pl.DataFrame({"bin": bins, "r": r_b})
            .group_by("bin")
            .agg(
                pl.col("r").mean().alias("mean"),
                pl.col("r").std().alias("std"),
                pl.col("r").count().alias("n"),
            )
            .filter(pl.col("n") >= 20)
            .sort("bin")
        )

        kk = df_bin["bin"].to_numpy()
        mm = df_bin["mean"].to_numpy()
        se = df_bin["std"].to_numpy() / np.sqrt(df_bin["n"].to_numpy())
        ax.errorbar(
            kk,
            mm,
            yerr=se,
            fmt="o-",
            ms=4,
            lw=1.2,
            capsize=2,
            color=colors[label],
            label=label,
            alpha=0.8,
        )

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("mean residual  (obs − fitted)")
    ax.set_title("Mean residual vs k, per bucket\n(structured shape = basis can't reach it)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────────
    # Plot 3: fitted ATM vs rolling-mean observed ATM IV over time
    # (detects level-tracking lag)
    # ─────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    # Pick a narrow ATM band
    atm_band = np.abs(k) < 0.01
    if atm_band.sum() > 50 and fam_iv is not None:
        ts_atm = ts[atm_band]
        obs_atm = obs_iv[atm_band]
        fam_atm = fam_iv[atm_band]

        # Rolling means via simple convolution-style smoothing
        order = np.argsort(ts_atm)
        ts_atm = ts_atm[order]
        obs_atm = obs_atm[order]
        fam_atm = fam_atm[order]

        win = max(50, len(ts_atm) // 100)
        kernel = np.ones(win) / win
        obs_smooth = np.convolve(obs_atm, kernel, mode="valid")
        fam_smooth = np.convolve(fam_atm, kernel, mode="valid")
        ts_smooth = ts_atm[win - 1 :]

        ax.plot(ts_smooth, obs_smooth, color="black", lw=1.5, label="observed ATM IV (rolling)")
        ax.plot(ts_smooth, fam_smooth, color="tab:red", lw=1.5, label="filter ATM IV (rolling)")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("ATM IV")
        ax.set_title(
            f"ATM IV: market vs filter (|k|<0.01, n={atm_band.sum():,})\n"
            "(filter consistently below market = level lag)"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough ATM trades", ha="center", transform=ax.transAxes)

    # ─────────────────────────────────────────────────────────────────
    # Plot 4: put–call IV asymmetry at matched strikes
    # (detects forward / dividend misspecification)
    # ─────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    # Pair calls and puts at the same strike & time bucket
    df_pc = df.with_columns(
        ((pl.col("ts_ns").cast(pl.Int64) // 10_000_000_000) * 10_000_000_000).alias(
            "ts_bucket"
        )  # 10-second buckets
    )
    calls = (
        df_pc.filter(pl.col("is_call"))
        .group_by(["ts_bucket", "strike"])
        .agg(
            pl.col("observed_iv").mean().alias("iv_call"),
            pl.col("log_moneyness").first().alias("k"),
        )
    )
    puts = (
        df_pc.filter(~pl.col("is_call"))
        .group_by(["ts_bucket", "strike"])
        .agg(
            pl.col("observed_iv").mean().alias("iv_put"),
        )
    )
    pairs = calls.join(puts, on=["ts_bucket", "strike"], how="inner")

    if pairs.shape[0] > 0:
        # Bin by k
        bin_w_pc = 0.005
        pairs = pairs.with_columns(
            ((pl.col("k") / bin_w_pc).round(0) * bin_w_pc).alias("k_bin"),
            (pl.col("iv_call") - pl.col("iv_put")).alias("diff"),
        )
        df_pc_agg = (
            pairs.group_by("k_bin")
            .agg(
                pl.col("diff").mean().alias("mean"),
                pl.col("diff").std().alias("std"),
                pl.col("diff").count().alias("n"),
            )
            .filter(pl.col("n") >= 10)
            .sort("k_bin")
        )

        kk = df_pc_agg["k_bin"].to_numpy()
        mm = df_pc_agg["mean"].to_numpy()
        se = df_pc_agg["std"].to_numpy() / np.sqrt(df_pc_agg["n"].to_numpy())

        ax.errorbar(kk, mm, yerr=se, fmt="o-", ms=4, lw=1.2, capsize=2, color="purple")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("log-moneyness  k = log(K/F)")
        ax.set_ylabel("IV(call) − IV(put)  at matched strike")
        ax.set_title(
            f"Put–call IV parity check (n_pairs={pairs.shape[0]:,})\n"
            "(systematic offset from 0 = forward misspec)"
        )
        ax.grid(alpha=0.3)

        mean_diff = pairs["diff"].mean()
        ax.text(
            0.02,
            0.98,
            f"global mean diff: {mean_diff:+.5f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No matched call/put pairs", ha="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────
    # Print summary stats
    # ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Residual summary by bucket")
    print("=" * 70)
    print(f"{'Bucket':<28}{'n':>10}{'mean':>12}{'std':>12}{'RMSE':>12}")
    print("-" * 70)
    for label, mask in buckets.items():
        if mask.sum() == 0:
            continue
        r_b = res[mask]
        print(
            f"{label:<28}{mask.sum():>10,}{r_b.mean():>12.5f}"
            f"{r_b.std():>12.5f}{np.sqrt(np.mean(r_b**2)):>12.5f}"
        )
    print("-" * 70)
    print(
        f"{'ALL':<28}{len(res):>10,}{res.mean():>12.5f}"
        f"{res.std():>12.5f}{np.sqrt(np.mean(res**2)):>12.5f}"
    )


# # Run it
# plot_residual_diagnostics(df_trade_fam)
