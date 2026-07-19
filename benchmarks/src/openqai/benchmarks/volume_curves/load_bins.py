"""
Load curve bin files from a directory, add ticker column, fill half-day trailing buckets with 0,
and verify there are no other nulls in bucket columns.
"""

from __future__ import annotations


import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.subdir import version
from cdxcore.util import fmt_date

from openqai.core import Context, SubDir, env
from openqai.benchmarks.volume_curves.curve_bins import curve_def, date, ticker_curve_bins

_curves_cache = env.locations.processing("volume_curves")

BUCKET_PREFIXES = ("vol_bucket_", "cash_vol_bucket_", "n_trades_bucket_", "return_bucket_")


def _is_bucket_col(col: str) -> bool:
    return col.startswith(BUCKET_PREFIXES)


def _bucket_index(col: str) -> int | None:
    """
    Parse trailing integer bucket index from names like:
      "vol_bucket_0", "cash_vol_bucket_25", ...
    Returns None if not parseable.
    """
    try:
        return int(col.rsplit("_", 1)[-1])
    except Exception:
        return None


@version("1.0", dependencies=[curve_def])
def load_curve_definition(
    directory: SubDir,
    *,
    verbose: Context = Context.quiet,
) -> dict:
    """
    Load the curve definition file from a directory.
    """
    curve_def_data = directory.read(
        "curve_def", version="*", raise_on_error=True, fmt=SubDir.PICKLE
    )
    curve_def_data["files"] = directory.files()
    return curve_def_data


@_curves_cache.cache(
    "0.1",
    dependencies=[ticker_curve_bins, load_curve_definition],
    label=lambda start, end, bucket_minutes: f"{fmt_date(start)} {fmt_date(end)}@{bucket_minutes}",
    exclude_args=["verbose"],
)
def merge_curve_files(
    directory: SubDir,
    tickers: list[str],
    start: date,
    end: date,
    bucket_minutes: int,
    *,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Load all bin files from `directory`, add a ticker column, fill half-day trailing buckets with 0,
    and verify bucket columns have no unexpected nulls.

    Assumptions:
      - Each file contains a 'date' column (date type) and an 'is_half_day' boolean column.
      - Bucket columns are named like '{prefix}{i}' with prefixes in BUCKET_PREFIXES.
      - Full day: 390 minutes; Half day: 210 minutes.
      - With bucket_minutes, compute:
          full_n = 390 / bucket_minutes
          half_n = 210 / bucket_minutes
        Then indices [half_n, ..., full_n-1] are the "trailing buckets" that should be 0 on half days.

    Returns:
      Polars DataFrame sorted by (date, ticker).
    """
    verify_inp(bucket_minutes > 0, lambda: f"bucket_minutes must be > 0, found {bucket_minutes}")
    verify(len(tickers) > 0, lambda: f"No files found in directory {directory}")

    full_buckets = 390 // bucket_minutes
    half_buckets = 210 // bucket_minutes
    verify_inp(
        390 % bucket_minutes == 0, lambda: f"390 not divisible by bucket_minutes {bucket_minutes}"
    )
    verify_inp(
        210 % bucket_minutes == 0, lambda: f"210 not divisible by bucket_minutes {bucket_minutes}"
    )
    verify(full_buckets > half_buckets, lambda: "Expected full_buckets > half_buckets")

    trailing_idx = set(range(half_buckets, full_buckets))

    dfs: list[pl.DataFrame] = []
    tmx = verbose.timer()

    for ticker in tickers:
        verbose.write(lambda: f"\rLoading {directory.full_file_name(ticker)}... ", end="")
        df = directory.read(ticker, version="*", raise_on_error=True)

        verify("ticker" not in df.columns, lambda: f"'ticker' already exists in file {ticker}")
        verify("date" in df.columns, lambda: f"'date' column missing in file {ticker}")
        verify(
            "is_half_day" in df.columns, lambda: f"'is_half_day' column missing in file {ticker}"
        )

        df = df.with_columns(pl.lit(ticker).alias("ticker"))
        dfs.append(df)

    verbose.write(lambda: f"\rLoaded {len(dfs)} files from {directory}; this took {tmx}.")
    df_all = pl.concat(dfs, how="vertical", rechunk=True)

    bucket_cols = [c for c in df_all.columns if _is_bucket_col(c)]
    trailing_bucket_cols = [
        c for c in bucket_cols if (idx := _bucket_index(c)) is not None and idx in trailing_idx
    ]

    # Fill trailing half-day bucket nulls with 0 (ONLY on half-days)
    if trailing_bucket_cols:
        with verbose.write_t(
            lambda: "Filling half-day trailing bucket nulls with 0... ", end=""
        ) as tme:
            fill_exprs = []
            for c in trailing_bucket_cols:
                dt = df_all.schema[c]
                zero = pl.lit(0).cast(dt)

                fill_exprs.append(
                    pl.when(pl.col("is_half_day") == True)
                    .then(pl.col(c).fill_null(zero))
                    .otherwise(pl.col(c))
                    .alias(c)
                )
            df_all = df_all.with_columns(fill_exprs)
            verbose.write(lambda: f"done ({len(fill_exprs)} cols); this took {tme}.", head=False)

    # Verify: no nulls in any bucket columns
    if bucket_cols:
        with verbose.write_t(lambda: "Checking bucket columns for nulls... ", end="") as tme:
            null_counts = df_all.select([pl.col(c).null_count().alias(c) for c in bucket_cols]).row(
                0
            )
            offenders = [
                (c, null_counts[i]) for i, c in enumerate(bucket_cols) if null_counts[i] != 0
            ]
            verify(
                len(offenders) == 0,
                lambda: f"Found nulls in bucket columns: {offenders[:20]} (showing up to 20)",
            )
            verbose.write(lambda: f"ok; this took {tme}.", head=False)

    verify(
        df_all.select(pl.len()).item() == df_all.unique(subset=["date", "ticker"]).height,
        lambda: "Duplicate (date,ticker) rows found",
    )

    return df_all.sort(("date", "ticker"))


@version("1.0", dependencies=[load_curve_definition, merge_curve_files])
def load_curve_files(
    directory: SubDir,
    *,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """Load curve definition and merge all curve bins into a single DataFrame."""
    curve_def_data = load_curve_definition(directory, verbose=verbose)
    return merge_curve_files(
        directory,
        tickers=curve_def_data["files"],
        start=curve_def_data["start"],
        end=curve_def_data["end"],
        bucket_minutes=curve_def_data["bucket_minutes"],
        verbose=verbose,
    )
