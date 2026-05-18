"""AAPL next-15-minute volume baselines.

This script loads the pre-processed AAPL NCDE VoluVol HDF5 files, constructs 
leakage-safe forward-volume labels, splits by trading date, evaluates four 
simple baselines, and writes the headline metrics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NY_TZ = "America/New_York"
DEFAULT_AAPL_H5_PATH = Path("quantdata_data/training/ncde/voluvol/experiment 15.0/AAPL")
DEFAULT_LAG_WINDOWS_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True)
class VolumeBaselineConfig:
    """Configuration for the AAPL baseline experiment."""

    input_path: Path = DEFAULT_AAPL_H5_PATH
    output_dir: Path = Path("outputs/volume_baselines_aapl")
    horizon_minutes: int = 15
    bar_minutes: int = 1
    train_frac: float = 0.70
    val_frac: float = 0.15
    max_files: int | None = None
    mape_epsilon: float = 1.0
    seasonality_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.bar_minutes <= 0:
            raise ValueError("bar_minutes must be positive")
        if self.horizon_minutes % self.bar_minutes:
            raise ValueError("horizon_minutes must be a multiple of bar_minutes")
        if self.train_frac <= 0 or self.val_frac <= 0 or self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac and val_frac must be positive and leave a test split")


@dataclass
class H5BarData:
    """Trade and quote bars loaded from the AAPL HDF5 files."""

    bars: pd.DataFrame
    quote_bars: pd.DataFrame | None


@dataclass
class IntradayContext:
    """Training-only intraday curves used by the simple baselines."""

    avg_next_by_bucket: pd.Series
    avg_prev_by_bucket: pd.Series
    global_next_mean: float
    global_prev_mean: float


def _h5_files(path: Path, max_files: int | None = None) -> list[Path]:
    """Return HDF5 files in chronological filename order."""

    if path.is_file() and path.suffix.lower() == ".h5":
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.h5"))
    else:
        files = []
    return files[:max_files] if max_files is not None else files


def _bar_times(day: Any, bar_idx: pd.Series, bar_minutes: int) -> pd.Series:
    """Convert integer intraday bar indices to New York bar-end timestamps."""

    open_ts = pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30)
    return open_ts + pd.to_timedelta((bar_idx.astype("int64") + 1) * bar_minutes, unit="min")


def _empty_quote_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "quote_spread",
            "quote_update_count",
            "mid_close",
            "mid_return_sq",
        ]
    )


def _load_one_h5(path: Path, bar_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one AAPL daily HDF5 file into trade bars and quote bars."""

    import h5py

    day = pd.Timestamp(path.stem).date()
    bar_ms = bar_minutes * 60_000
    with h5py.File(path, "r") as h5:
        tr = h5["root/trades"]
        tt = tr["time_of_day_ms"][:].astype("int64")
        size = tr["size"][:].astype("float64")
        price = tr["price"][:].astype("float64")
        qu = h5["root/quotes"]
        qt = qu["time_of_day_ms"][:].astype("int64")
        bid = qu["bid_price"][:].astype("float64")
        ask = qu["ask_price"][:].astype("float64")
    mask = (
        (tt >= 0)
        & np.isfinite(size)
        & np.isfinite(price)
        & (size >= 0)
        & (price > 0)
    )
    trades = pd.DataFrame(
        {
            "bar": tt[mask] // bar_ms,
            "volume": size[mask],
            "trade_count": 1.0,
            "close": price[mask],
        }
    )
    if trades.empty:
        bars = pd.DataFrame(columns=["timestamp", "volume", "trade_count", "close"])
    else:
        bars = (
            trades.groupby("bar", sort=True)
            .agg(
                volume=("volume", "sum"),
                trade_count=("trade_count", "sum"),
                close=("close", "last"),
            )
            .reset_index()
        )
        bars["timestamp"] = _bar_times(day, bars["bar"], bar_minutes)
        bars = bars[["timestamp", "volume", "trade_count", "close"]]

    qmask = (
        (qt >= 0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid > 0)
        & (ask > bid)
    )
    if not qmask.any():
        return bars, _empty_quote_bars()

    mid = (bid[qmask] + ask[qmask]) / 2.0
    prev = np.roll(mid, 1)
    ret = np.zeros_like(mid)
    good = (mid > 0) & (prev > 0)
    good[0] = False
    ret[good] = np.log(mid[good] / prev[good])
    quotes = pd.DataFrame(
        {
            "bar": qt[qmask] // bar_ms,
            "quote_spread": ask[qmask] - bid[qmask],
            "quote_update_count": 1.0,
            "mid_close": mid,
            "mid_return_sq": ret**2,
        }
    )
    quote_bars = (
        quotes.groupby("bar", sort=True)
        .agg(
            quote_spread=("quote_spread", "mean"),
            quote_update_count=("quote_update_count", "sum"),
            mid_close=("mid_close", "last"),
            mid_return_sq=("mid_return_sq", "sum"),
        )
        .reset_index()
    )
    quote_bars["timestamp"] = _bar_times(day, quote_bars["bar"], bar_minutes)
    return bars, quote_bars[list(_empty_quote_bars().columns)]


def load_aapl_ncde_voluvol_bars(
    path: Path,
    *,
    bar_minutes: int,
    max_files: int | None = None,
) -> H5BarData:
    """Load all requested AAPL HDF5 files into concatenated bar frames."""

    files = _h5_files(path, max_files)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {path}")

    bars, quotes = zip(*(_load_one_h5(p, bar_minutes) for p in files))
    bars = [b for b in bars if not b.empty]
    quotes = [q for q in quotes if not q.empty]
    if not bars:
        raise ValueError("No trade bars found")

    return H5BarData(
        bars=(
            pd.concat(bars, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        ),
        quote_bars=(
            pd.concat(quotes, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
            if quotes
            else None
        ),
    )


def regularise_bars(bars: pd.DataFrame, *, bar_minutes: int) -> pd.DataFrame:
    """Reindex each observed trading day to a regular bar grid."""

    if bars.empty:
        raise ValueError("cannot regularise an empty bar frame")

    bars = bars.copy().sort_values("timestamp")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce").fillna(0).clip(lower=0)
    if "trade_count" in bars:
        bars["trade_count"] = (
            pd.to_numeric(bars["trade_count"], errors="coerce").fillna(0).clip(lower=0)
        )
    bars["session_date"] = bars["timestamp"].dt.date

    pieces: list[pd.DataFrame] = []
    for _, day in bars.groupby("session_date", sort=True):
        day = day.drop_duplicates("timestamp", keep="last")
        grid = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    day["timestamp"].min(),
                    day["timestamp"].max(),
                    freq=f"{bar_minutes}min",
                )
            }
        )
        day = grid.merge(day.drop(columns="session_date"), on="timestamp", how="left")
        day["volume"] = day["volume"].fillna(0)
        if "trade_count" in day:
            day["trade_count"] = day["trade_count"].fillna(0)
        if "close" in day:
            day["close"] = day["close"].ffill()
        pieces.append(day)

    df = pd.concat(pieces, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["session_date"] = df["timestamp"].dt.date
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def attach_quote_features(bars: pd.DataFrame, quote_bars: pd.DataFrame | None) -> pd.DataFrame:
    """Merge quote-derived features onto regular trade bars."""

    if quote_bars is None or quote_bars.empty:
        return bars
    q = quote_bars.copy()
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    out = bars.merge(q, on="timestamp", how="left")
    for col in ("quote_update_count", "mid_return_sq"):
        if col in out:
            out[col] = out[col].fillna(0)
    return out


def _future_sum(x: np.ndarray, h: int) -> np.ndarray:
    """Sum the next h bars, excluding the current bar."""

    c = np.r_[0.0, np.cumsum(np.asarray(x, float))]
    y = np.full(len(x), np.nan)
    for i in range(max(0, len(x) - h)):
        y[i] = c[i + h + 1] - c[i + 1]
    return y


def construct_supervised_dataset(
    bars: pd.DataFrame,
    *,
    horizon_minutes: int = 15,
    bar_minutes: int = 1,
    lag_windows_minutes: tuple[int, ...] = DEFAULT_LAG_WINDOWS_MINUTES,
) -> pd.DataFrame:
    """Create leakage-safe supervised rows indexed by prediction time t.

    The target is total traded volume over ``(t, t + horizon]``. All lagged
    features are rolling windows ending at ``t`` within the same trading day.
    Rows whose target horizon crosses the observed session end are dropped.
    """

    h = horizon_minutes // bar_minutes
    windows = tuple(sorted({*lag_windows_minutes, horizon_minutes, 15}))
    df = bars.sort_values("timestamp").copy()
    dt = pd.Timedelta(minutes=horizon_minutes)
    df["prediction_time"] = df["timestamp"]
    df["features_end_time"] = df["timestamp"]
    df["target_start_time"] = df["timestamp"]
    df["target_end_time"] = df["timestamp"] + dt
    pieces = []
    for _, day in df.groupby("session_date", sort=True):
        day = day.copy().sort_values("timestamp")
        day["session_end_time"] = day["timestamp"].max()
        day["target_volume"] = _future_sum(day["volume"].to_numpy(float), h)
        for w in windows:
            n = max(1, w // bar_minutes)
            day[f"volume_lag_{w}m"] = day["volume"].rolling(n, min_periods=1).sum()
            for base in ("trade_count", "quote_update_count"):
                if base in day:
                    day[f"{base}_lag_{w}m"] = day[base].rolling(n, min_periods=1).sum()
            if "quote_spread" in day:
                day[f"quote_spread_mean_{w}m"] = (
                    day["quote_spread"].rolling(n, min_periods=1).mean()
                )
            if "mid_return_sq" in day:
                day[f"mid_return_vol_{w}m"] = np.sqrt(
                    day["mid_return_sq"].rolling(n, min_periods=1).sum()
                )
        pieces.append(day)

    df = pd.concat(pieces, ignore_index=True).dropna(subset=["target_volume"])
    df = df[df["target_end_time"] <= df["session_end_time"]].copy()
    df["tod_bucket"] = (df["minute_of_day"] // bar_minutes).astype(int)
    df["tod_sin"] = np.sin(2 * math.pi * df["minute_of_day"] / 1440)
    df["tod_cos"] = np.cos(2 * math.pi * df["minute_of_day"] / 1440)
    df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7)
    return df.sort_values("timestamp").reset_index(drop=True)


def _split_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    if n < 3:
        raise ValueError("need at least three rows/days")
    n_train = max(1, int(math.floor(n * train_frac)))
    n_val = max(1, int(math.floor(n * val_frac)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return min(n_train, n - n_val - 1), n_val


def split_by_trading_date(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique trading days so no day appears in more than one split."""

    df = df.sort_values("timestamp").reset_index(drop=True)
    dates = list(pd.Index(df["session_date"].drop_duplicates()))
    n_train, n_val = _split_counts(len(dates), train_frac, val_frac)
    date_sets = [
        set(dates[:n_train]),
        set(dates[n_train : n_train + n_val]),
        set(dates[n_train + n_val :]),
    ]
    out = tuple(df[df["session_date"].isin(s)].copy() for s in date_sets)
    if date_sets[0] & date_sets[1] or date_sets[0] & date_sets[2] or date_sets[1] & date_sets[2]:
        raise ValueError("day appears in multiple splits")
    return out


def fit_intraday_context(
    train: pd.DataFrame,
    *,
    horizon_minutes: int,
    seasonality_epsilon: float = 1e-9,
) -> IntradayContext:
    """Fit train-only intraday volume curves."""

    prev = f"volume_lag_{horizon_minutes}m"
    return IntradayContext(
        avg_next_by_bucket=train.groupby("tod_bucket")["target_volume"].mean(),
        avg_prev_by_bucket=train.groupby("tod_bucket")[prev].mean(),
        global_next_mean=float(train["target_volume"].mean()),
        global_prev_mean=float(max(train[prev].mean(), seasonality_epsilon)),
    )


def apply_intraday_context(
    df: pd.DataFrame,
    context: IntradayContext,
    *,
    horizon_minutes: int,
    seasonality_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Attach train-only intraday averages to a split."""

    prev = f"volume_lag_{horizon_minutes}m"
    out = df.copy()
    out["hist_avg_next_volume"] = (
        out["tod_bucket"].map(context.avg_next_by_bucket).fillna(context.global_next_mean)
    )
    out["hist_avg_prev_volume"] = (
        out["tod_bucket"].map(context.avg_prev_by_bucket).fillna(context.global_prev_mean)
    )
    out["hist_avg_prev_volume"] = out["hist_avg_prev_volume"].clip(lower=seasonality_epsilon)
    out["volume_vs_intraday_avg"] = out[prev] / out["hist_avg_prev_volume"]
    return out


def predict_historical_intraday(df: pd.DataFrame) -> np.ndarray:
    return df["hist_avg_next_volume"].to_numpy(float).clip(min=0)


def predict_naive_persistence(df: pd.DataFrame, *, horizon_minutes: int) -> np.ndarray:
    return df[f"volume_lag_{horizon_minutes}m"].to_numpy(float).clip(min=0)


def predict_seasonality_adjusted_persistence(
    df: pd.DataFrame,
    *,
    horizon_minutes: int,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Scale previous-horizon volume by the train-only intraday curve ratio."""

    prev = df[f"volume_lag_{horizon_minutes}m"].to_numpy(float)
    ratio = df["hist_avg_next_volume"].to_numpy(float) / np.maximum(
        df["hist_avg_prev_volume"].to_numpy(float),
        epsilon,
    )
    return (prev * ratio).clip(min=0)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Ridge feature matrix from past-looking columns only."""

    x = pd.DataFrame(index=df.index)
    for w in DEFAULT_LAG_WINDOWS_MINUTES:
        for col in (f"volume_lag_{w}m", f"trade_count_lag_{w}m", f"quote_update_count_lag_{w}m"):
            if col in df:
                x[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
        for col in (f"quote_spread_mean_{w}m", f"mid_return_vol_{w}m"):
            if col in df:
                x[col] = df[col]
    for col in ("tod_sin", "tod_cos", "dow_sin", "dow_cos"):
        x[col] = df[col]
    x["log1p_volume_vs_intraday_avg"] = np.log1p(df["volume_vs_intraday_avg"].clip(0, 1000))
    return x


def fit_predict_linear_regression(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Ridge on log1p volume and return level-volume predictions."""

    xs = [_features(x) for x in (train, val, test)]
    names = list(xs[0].columns)
    if len(train) < 3:
        mean = float(train["target_volume"].mean())
        return (
            np.full(len(train), mean),
            np.full(len(val), mean),
            np.full(len(test), mean),
        )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    y = np.log1p(train["target_volume"].to_numpy(float))
    model.fit(xs[0], y)
    max_log = float(y.max())

    def predict(x: pd.DataFrame) -> np.ndarray:
        pred_log = np.clip(model.predict(x[names]), 0.0, max_log)
        return np.expm1(pred_log).clip(min=0)

    return predict(xs[0]), predict(xs[1]), predict(xs[2])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_epsilon: float = 1.0,
) -> dict[str, float]:
    """Compute level, normalised, percentage, log, and R-squared metrics."""

    y = np.asarray(y_true, float).clip(min=0)
    p = np.asarray(y_pred, float).clip(min=0)
    err = p - y
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    std = float(y.std())
    mean = float(y.mean())
    log_err = np.log1p(p) - np.log1p(y)
    denom = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": mae,
        "rmse": rmse,
        "rmse_over_mean": np.nan if mean <= 0 else rmse / mean,
        "mae_over_mean": np.nan if mean <= 0 else mae / mean,
        "rmse_over_std": np.nan if std <= 0 else rmse / std,
        "mae_over_std": np.nan if std <= 0 else mae / std,
        "mape": float((np.abs(err) / np.maximum(np.abs(y), mape_epsilon)).mean() * 100),
        "rmsle": float(np.sqrt((log_err**2).mean())),
        "r2": np.nan if denom <= 0 else float(1 - (err**2).sum() / denom),
        "log_mae": float(np.abs(log_err).mean()),
        "log_rmse": float(np.sqrt((log_err**2).mean())),
        "n": int(len(y)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    mape_epsilon: float = 1.0,
) -> pd.DataFrame:
    """Evaluate every prediction column over validation and test."""

    rows: list[dict[str, Any]] = []
    prediction_columns = [c for c in predictions if c.startswith("pred_")]
    for col in prediction_columns:
        row: dict[str, Any] = {"model": col.removeprefix("pred_")}
        for split in ("validation", "test"):
            part = predictions[predictions["split"] == split]
            metrics = compute_metrics(part["target_volume"], part[col], mape_epsilon=mape_epsilon)
            row.update({f"{split}_{k}": v for k, v in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)


def _predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: VolumeBaselineConfig,
) -> pd.DataFrame:
    """Create all baseline predictions and return a compact prediction frame."""

    lr_train, lr_val, lr_test = fit_predict_linear_regression(train, val, test)
    parts: list[pd.DataFrame] = []
    split_triplets = (
        ("train", train, lr_train),
        ("validation", val, lr_val),
        ("test", test, lr_test),
    )
    for name, df, lr in split_triplets:
        df = df.copy()
        df["split"] = name
        df["previous_15m_volume"] = df["volume_lag_15m"]
        df["pred_historical_intraday_average"] = predict_historical_intraday(df)
        df["pred_naive_persistence"] = predict_naive_persistence(
            df,
            horizon_minutes=config.horizon_minutes,
        )
        df["pred_seasonality_adjusted_persistence"] = (
            predict_seasonality_adjusted_persistence(
                df,
                horizon_minutes=config.horizon_minutes,
                epsilon=config.seasonality_epsilon,
            )
        )
        df["pred_ridge_regression"] = lr
        parts.append(df)

    keep = [
        "timestamp",
        "prediction_time",
        "features_end_time",
        "target_start_time",
        "target_end_time",
        "session_end_time",
        "session_date",
        "minute_of_day",
        "tod_bucket",
        "split",
        "target_volume",
        "previous_15m_volume",
    ]
    pred_cols = [c for c in parts[0] if c.startswith("pred_")]
    return pd.concat(parts, ignore_index=True)[keep + pred_cols]


def run_experiment(
    config: VolumeBaselineConfig,
) -> pd.DataFrame:
    """Run the minimal baseline experiment and save metrics/predictions."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_aapl_ncde_voluvol_bars(
        config.input_path,
        bar_minutes=config.bar_minutes,
        max_files=config.max_files,
    )
    bars = regularise_bars(loaded.bars, bar_minutes=config.bar_minutes)
    bars = attach_quote_features(bars, loaded.quote_bars)
    dataset = construct_supervised_dataset(
        bars,
        horizon_minutes=config.horizon_minutes,
        bar_minutes=config.bar_minutes,
    )

    train, val, test = split_by_trading_date(
        dataset,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
    )
    context = fit_intraday_context(
        train,
        horizon_minutes=config.horizon_minutes,
        seasonality_epsilon=config.seasonality_epsilon,
    )
    train, val, test = (
        apply_intraday_context(
            split,
            context,
            horizon_minutes=config.horizon_minutes,
            seasonality_epsilon=config.seasonality_epsilon,
        )
        for split in (train, val, test)
    )

    predictions = _predictions(train, val, test, config)
    metrics = evaluate_predictions(predictions, mape_epsilon=config.mape_epsilon)

    predictions.to_csv(config.output_dir / "predictions.csv", index=False)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    return metrics


def parse_args(argv: list[str] | None = None) -> VolumeBaselineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_AAPL_H5_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/volume_baselines_aapl"))
    parser.add_argument("--horizon-minutes", type=int, default=15)
    parser.add_argument("--bar-minutes", type=int, default=1)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args(argv)
    return VolumeBaselineConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        horizon_minutes=args.horizon_minutes,
        bar_minutes=args.bar_minutes,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        max_files=args.max_files,
    )


def main(argv: list[str] | None = None) -> None:
    metrics = run_experiment(parse_args(argv))
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
