import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl


def plot_residual_vs_lag(df_trades_result: pl.DataFrame, n_bins=50, lag="all"):
    """ """

    # Filter positive lag only
    if lag == "positive":
        df = df_trades_result.filter(pl.col("matched_lag_ns") > 0)
    elif lag == "negative":
        df = df_trades_result.filter(pl.col("matched_lag_ns") < 0)
    else:
        df = df_trades_result

    lag_min = df["matched_lag_ns"].min()
    lag_max = df["matched_lag_ns"].max()
    bins = np.linspace(lag_min, lag_max, n_bins + 1)

    # Assign bin index
    df = df.with_columns(pl.col("matched_lag_ns").cut(breaks=bins[1:-1].tolist()).alias("bin"))

    grouped = (
        df.group_by("bin", maintain_order=False)
        .agg(
            [
                pl.col("trade_residual").mean().alias("mean"),
                pl.col("trade_residual").std().alias("std"),
                pl.col("matched_lag_ns").mean().alias("bin_center"),
                pl.col("trade_residual").count().alias("count"),
            ]
        )
        .sort("bin_center")
    )

    bin_centers = grouped["bin_center"].to_numpy()
    means = grouped["mean"].to_numpy()
    stds = grouped["std"].to_numpy()
    counts = grouped["count"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    bar_width = (bins[1] - bins[0]) * 0.9

    ax1.bar(
        bin_centers, means, width=bar_width, alpha=0.6, label="Mean residual", color="steelblue"
    )
    ax1.errorbar(
        bin_centers, means, yerr=stds, fmt="none", color="black", capsize=3, label="±1 std"
    )
    ax1.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("matched_lag_ns")
    ax1.set_ylabel("trade_residual")

    ax2 = ax1.twinx()
    ax2.plot(bin_centers, counts, color="orange", linewidth=1.5, label="Count")
    ax2.set_ylabel("Count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.title("Trade residual vs matched lag (positive lags, uniform bins)")
    plt.tight_layout()
    plt.show()


def scatter_residual_vs_lag(df_trades_result: pl.DataFrame):
    df_plot = df_trades_result.with_columns(
        [
            (df_trades_result["trade_residual"] ** 2).alias("squared_residual"),
            pl.when(pl.col("matched_lag_ns") >= 0)
            .then(pl.lit("forward (≤250ms)"))
            .otherwise(pl.lit("backward fallback"))
            .alias("match_type"),
        ]
    )

    fig = px.scatter(
        df_plot.to_pandas(),
        x="matched_lag_ns",
        y="squared_residual",
        color="match_type",
        labels={"matched_lag_ns": "Matched lag (ms)", "squared_residual": "Squared residual"},
        title="Squared trade residual vs matched quote lag",
        opacity=0.3,
        color_discrete_map={"forward (≤250ms)": "steelblue", "backward fallback": "tomato"},
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()
