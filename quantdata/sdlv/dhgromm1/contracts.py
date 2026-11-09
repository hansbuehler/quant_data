"""One gain convention for every GROMM v1 hedge: floating payoff minus price."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RETURN_CONVENTION = "floating-payoff-minus-price-v1"
SCENARIO_SCHEMA = "dhgromm1-expiry-options-v4"
LEGACY_PREMIUM_SCHEMAS = {"dhgromm1-expiry-options-v2", "dhgromm1-expiry-options-v3"}


def floating_gains(prices: np.ndarray, premium_returns: np.ndarray) -> np.ndarray:
    """Explicit legacy conversion: r_premium * opening_mark / S_open.

    This adapter is ONLY for identified old datasets, not a second production
    return convention. New generators directly compute (close-open)/S_open.
    """
    if (prices.ndim != 3 or prices.shape[1] != 21 or prices.shape[2] not in (17, 21)
            or premium_returns.shape != (len(prices), 20, prices.shape[2])):
        raise ValueError("expected the existing 21-date option-expiry trade geometry")
    if (not np.isfinite(prices).all() or not np.isfinite(premium_returns).all()
            or np.any(prices[:, :20, 0] <= 0)):
        raise ValueError("floating contracts require finite marks and strictly positive opening spots")
    return premium_returns.astype(np.float64) * (prices[:, :20] / prices[:, :20, :1])


def canonical_gains(prices: np.ndarray, returns: np.ndarray, *, identity: dict) -> np.ndarray:
    """Convert known legacy schemas once; reject unknown/mislabeled conventions."""
    schema = identity.get("schema")
    convention = identity.get("return_convention")
    if schema in LEGACY_PREMIUM_SCHEMAS and convention is None:
        return floating_gains(prices, returns)
    if schema != SCENARIO_SCHEMA or convention != RETURN_CONVENTION:
        raise ValueError(f"unsupported or ambiguous option-return convention: schema={schema!r}, convention={convention!r}")
    if (returns.shape != (len(prices), 20, prices.shape[2]) or not np.isfinite(returns).all()
            or not np.isfinite(prices).all() or np.any(prices[:, :20, 0] <= 0)):
        raise ValueError("invalid floating-contract scenario data")
    return returns.astype(np.float64, copy=False)


def scenario_identity(directory: Path) -> dict:
    """A sample directory's parent must explicitly identify its stored units."""
    return json.loads((directory.parent / "identity.json").read_text())
