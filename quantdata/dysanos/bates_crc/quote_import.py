r"""A schema-validated import adapter for real option-quote data that does
not come through the existing WRDS/Massive connectors (``quantdata.widb``,
``quantdata.massive``).

Converts an arbitrary quote table into this repository's own "pure" unit
convention (``C/(DF*F)``, ``K/F`` -- see ``MODEL_CONVENTIONS.md`` Section 1),
so downstream fitting code does not need to know the data's origin.

Deliberately does NOT:
  - accept a table with only a mid or implied-volatility column and
    reconstruct bid/ask from it (explicitly disallowed by the task);
  - fabricate missing fields (raises, with the exact missing column named);
  - assume zero carry for raw quotes (carry normalisation is a modelling
    choice applied on top of this adapter's output, not baked into import).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "underlying",
    "valuation_date",
    "expiry",
    "strike",
    "is_call",
    "bid",
    "ask",
)

# At least one of these must be present to establish carry -- either the
# discount factor/forward pair directly, or a zero-rate/dividend-yield pair
# deterministic enough to build them (per MODEL_CONVENTIONS.md Section 2).
CARRY_COLUMN_SETS: tuple[tuple[str, ...], ...] = (
    ("discount_factor", "forward"),
    ("zero_rate", "dividend_yield", "spot"),
)


@dataclass(frozen=True)
class QuoteImportReport:
    """What was imported, what was dropped, and why -- never silent."""

    n_input_rows: int
    n_output_rows: int
    dropped_missing_bid_ask: int
    dropped_non_positive_spread: int
    dropped_non_positive_time_to_expiry: int
    dropped_non_positive_forward: int
    carry_source: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return dict(
            n_input_rows=self.n_input_rows,
            n_output_rows=self.n_output_rows,
            dropped_missing_bid_ask=self.dropped_missing_bid_ask,
            dropped_non_positive_spread=self.dropped_non_positive_spread,
            dropped_non_positive_time_to_expiry=self.dropped_non_positive_time_to_expiry,
            dropped_non_positive_forward=self.dropped_non_positive_forward,
            carry_source=self.carry_source,
            warnings=list(self.warnings),
        )


def _validate_schema(df: pd.DataFrame) -> tuple[str, ...]:
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            "quote import is missing required column(s) "
            f"{missing_required}; refusing to reconstruct bid/ask or carry "
            "from an incomplete table (see DATA_STATUS.md Section 4)"
        )
    for candidate_set in CARRY_COLUMN_SETS:
        if all(c in df.columns for c in candidate_set):
            return candidate_set
    raise ValueError(
        "quote import has no usable carry information: need either "
        f"{CARRY_COLUMN_SETS[0]} or {CARRY_COLUMN_SETS[1]}; zero carry is "
        "never assumed for raw real quotes (see MODEL_CONVENTIONS.md "
        "Section 2) -- it must be applied explicitly by the caller after "
        "import, not silently defaulted here"
    )


def _discount_and_forward(df: pd.DataFrame, carry_source: tuple[str, ...],
                           time_to_expiry_y: pd.Series) -> tuple[pd.Series, pd.Series]:
    if carry_source == ("discount_factor", "forward"):
        return df["discount_factor"].astype(float), df["forward"].astype(float)
    # ("zero_rate", "dividend_yield", "spot"): deterministic continuous-rate carry.
    zero_rate = df["zero_rate"].astype(float)
    dividend_yield = df["dividend_yield"].astype(float)
    spot = df["spot"].astype(float)
    discount_factor = np.exp(-zero_rate * time_to_expiry_y)
    forward = spot * np.exp((zero_rate - dividend_yield) * time_to_expiry_y)
    return discount_factor, forward


def import_quotes(df: pd.DataFrame, *, ticker_filter: str | None = None) -> tuple[pd.DataFrame, QuoteImportReport]:
    """Import a real (never synthetic) option-quote table into the "pure"
    unit convention. Every row surviving to the output has a positive
    bid-ask spread, a strictly positive discounted forward, and a strictly
    positive remaining time to expiry."""
    n_input = len(df)
    warnings: list[str] = []
    carry_source = _validate_schema(df)

    work = df.copy()
    if ticker_filter is not None:
        work = work[work["underlying"] == ticker_filter]

    valuation_date = pd.to_datetime(work["valuation_date"])
    expiry = pd.to_datetime(work["expiry"])
    time_to_expiry_y = (expiry - valuation_date).dt.days.astype(float) / 365.25

    dropped_missing = int(work["bid"].isna().sum() + work["ask"].isna().sum())
    work = work.dropna(subset=["bid", "ask"])
    time_to_expiry_y = time_to_expiry_y.loc[work.index]

    spread_ok = (work["ask"].astype(float) - work["bid"].astype(float)) > 0.0
    dropped_spread = int((~spread_ok).sum())
    work = work.loc[spread_ok]
    time_to_expiry_y = time_to_expiry_y.loc[work.index]

    tte_ok = time_to_expiry_y > 0.0
    dropped_tte = int((~tte_ok).sum())
    work = work.loc[tte_ok]
    time_to_expiry_y = time_to_expiry_y.loc[tte_ok]

    discount_factor, forward = _discount_and_forward(work, carry_source, time_to_expiry_y)
    discounted_forward = discount_factor * forward
    forward_ok = discounted_forward > 0.0
    dropped_forward = int((~forward_ok).sum())
    work = work.loc[forward_ok]
    discounted_forward = discounted_forward.loc[forward_ok]
    time_to_expiry_y = time_to_expiry_y.loc[forward_ok]

    if len(work) == 0:
        warnings.append("no rows survived import filtering")

    out = pd.DataFrame({
        "underlying": work["underlying"].to_numpy(),
        "valuation_date": pd.to_datetime(work["valuation_date"]).to_numpy(),
        "expiry": pd.to_datetime(work["expiry"]).to_numpy(),
        "tte_y": time_to_expiry_y.to_numpy(),
        "is_call": work["is_call"].astype(bool).to_numpy(),
        "pure_strike": (work["strike"].astype(float) / forward.loc[work.index]).to_numpy(),
        "pure_bid_price": (work["bid"].astype(float) / discounted_forward).to_numpy(),
        "pure_ask_price": (work["ask"].astype(float) / discounted_forward).to_numpy(),
        "discount_factor": discount_factor.loc[work.index].to_numpy(),
        "forward": forward.loc[work.index].to_numpy(),
    })

    report = QuoteImportReport(
        n_input_rows=n_input, n_output_rows=len(out),
        dropped_missing_bid_ask=dropped_missing, dropped_non_positive_spread=dropped_spread,
        dropped_non_positive_time_to_expiry=dropped_tte, dropped_non_positive_forward=dropped_forward,
        carry_source="+".join(carry_source), warnings=warnings,
    )
    return out, report
