"""Availability-aware Greek performance, always paired to the same reference dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdata.hedge2.metrics import lower_tail_mean

CORE_KEYS = ("origin", "family", "claim", "side")
CONFIG_KEYS = ("variant", "family", "claim", "side", "greek_budget")
PORTFOLIO_KEYS = ("variant", *CORE_KEYS, "greek_budget")


def _validate_keys(frame: pd.DataFrame, keys: tuple[str, ...], *, label: str) -> None:
    if (not set(keys).issubset(frame.columns) or frame[list(keys)].isna().any().any()
            or frame.duplicated(list(keys)).any()):
        raise ValueError(f"{label} needs unique, fully specified portfolio identities")
    if not frame.side.isin((-1, 1)).all():
        raise ValueError(f"{label} has an invalid claim side")


def _statistics(values: np.ndarray, costs: np.ndarray, *, prefix: str) -> dict[str, float]:
    empty = {f"{prefix}{name}": np.nan for name in
        ("mean_pnl", "median_pnl", "std_pnl", "scaled_mad", "mean_entry_cost", "es95_loss")}
    if not len(values):
        return empty
    return {f"{prefix}mean_pnl": float(values.mean()), f"{prefix}median_pnl": float(np.median(values)),
        f"{prefix}std_pnl": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        f"{prefix}scaled_mad": float(np.median(np.abs(values - np.median(values))) / .6744897501960817),
        f"{prefix}mean_entry_cost": float(costs.mean()), f"{prefix}es95_loss": -lower_tail_mean(values)}


def matched_frontier_metrics(*, requested: pd.DataFrame, available: pd.DataFrame,
                             unavailable: pd.DataFrame, reference: pd.DataFrame,
                             comparison_sample: str = "common",
                             allow_reference_gaps: bool = False,
                             pending: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep every requested setting, including settings with no valid performance.

    Common sampling retains an origin/target/side only when every requested
    variant/budget has a completed result AND the reference exists.
    Per-configuration sampling uses each setting's own available dates;
    its reference metrics use exactly those same dates, not a full-sample
    baseline. Neither mode assigns zero returns to unavailable portfolios.
    All P&L/cost inputs must already use the same original payoff units.
    Statistics are per product across origins; strikes are never pooled as
    independent observations. No winner selection or bootstrap occurs here.
    Pending computation is a separate outcome, never economic infeasibility.
    Keep the full requested population when reporting a running experiment.
    """
    if comparison_sample not in ("common", "per_configuration") or not isinstance(allow_reference_gaps, bool):
        raise ValueError("Choose common or per_configuration sampling and explicit reference-gap handling")
    pending = pd.DataFrame(columns=[*PORTFOLIO_KEYS, "costed_pnl"]) if pending is None else pending
    for label, frame in (("Requested", requested), ("Available", available),
                         ("Unavailable", unavailable), ("Pending", pending)):
        _validate_keys(frame, PORTFOLIO_KEYS, label=label)
        fractions = frame.greek_budget.to_numpy(dtype=float)
        if not np.isfinite(fractions).all() or np.any((fractions < 0.) | (fractions > 1.)):
            raise ValueError(f"{label} contains an invalid Greek budget")
    _validate_keys(reference, CORE_KEYS, label="Reference")
    if requested.empty:
        raise ValueError("A performance report needs its explicit nonempty requested population")
    outcomes = pd.concat([frame[list(PORTFOLIO_KEYS)] for frame in (available, unavailable, pending)], ignore_index=True)
    if len(outcomes) != len(requested) or outcomes.duplicated(list(PORTFOLIO_KEYS)).any():
        raise ValueError("Every requested portfolio must be completed, unavailable or pending exactly once")
    reconciled = requested[list(PORTFOLIO_KEYS)].merge(outcomes, on=list(PORTFOLIO_KEYS),
        how="outer", indicator=True, validate="one_to_one")
    if not reconciled._merge.eq("both").all():
        raise ValueError("Requested portfolios were omitted or substituted")
    if "costed_pnl" not in unavailable or not unavailable.costed_pnl.isna().all():
        raise ValueError("Unavailable portfolios must not have fabricated terminal P&L")
    if "costed_pnl" not in pending or not pending.costed_pnl.isna().all():
        raise ValueError("Pending portfolios must not have fabricated terminal P&L")
    for label, frame in (("Available", available), ("Reference", reference)):
        columns = ["costed_pnl", "entry_cost", "initial_price"]
        if not set(columns).issubset(frame.columns) or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{label} performance needs finite P&L, actual entry costs and inception prices")
        if np.any(frame.entry_cost.to_numpy(dtype=float) < 0.):
            raise ValueError(f"{label} has negative trading costs")
    binding = "contract_definition_sha256"
    bound = binding in available or binding in reference
    if bound and any(binding not in frame or not frame[binding].map(
            lambda value: isinstance(value, str) and len(value) == 64).all() for frame in (available, reference)):
        raise ValueError("Both strategies need complete claim-definition and numeraire bindings")
    baseline = reference[[*CORE_KEYS, "costed_pnl", "entry_cost", "initial_price"]].rename(columns={
        "costed_pnl": "reference_pnl", "entry_cost": "reference_cost", "initial_price": "reference_price"})
    if bound:
        baseline["reference_contract_binding"] = reference[binding].to_numpy()
    paired = available.merge(baseline, on=list(CORE_KEYS), how="left", indicator=True, validate="many_to_one")
    paired["has_reference"] = paired._merge.eq("both")
    if not allow_reference_gaps and not paired.has_reference.all():
        raise ValueError("Some completed portfolios lack the matched reference; disclose reference gaps explicitly")
    matched = paired.loc[paired.has_reference].copy()
    if np.max(np.abs((matched.initial_price - matched.reference_price).to_numpy(dtype=float)), initial=0.) > 1e-12:
        raise ValueError("Strategies do not share the same inception target price")
    if bound and not matched[binding].eq(matched.reference_contract_binding).all():
        raise ValueError("Strategies do not share the same claim definition and numeraire")
    matched["selected_for_performance"] = True
    if comparison_sample == "common":
        required = requested.groupby(list(CORE_KEYS), sort=False).size().rename("required_settings")
        completed = matched.groupby(list(CORE_KEYS), sort=False).size().rename("available_settings")
        counts = required.to_frame().join(completed).fillna({"available_settings": 0}).reset_index()
        product_keys = ["family", "claim", "side"]
        expected = requested[list(CONFIG_KEYS)].drop_duplicates().groupby(product_keys).size().rename("expected_settings")
        counts = counts.merge(expected.reset_index(), on=product_keys, validate="many_to_one")
        eligible = counts.loc[counts.expected_settings == counts.available_settings, list(CORE_KEYS)]
        eligible["common_sample"] = True
        matched = matched.merge(eligible, on=list(CORE_KEYS), how="left", validate="many_to_one")
        matched["selected_for_performance"] = matched.common_sample.eq(True)
    counts = requested.groupby(list(CONFIG_KEYS), sort=False).size().rename("requested_origins").to_frame()
    for name, frame in (("completed_origins", available), ("unavailable_origins", unavailable),
                        ("pending_origins", pending),
                        ("paired_origins_before_common_filter", paired.loc[paired.has_reference]),
                        ("missing_reference_origins", paired.loc[~paired.has_reference])):
        counts = counts.join(frame.groupby(list(CONFIG_KEYS), sort=False).size().rename(name)).fillna({name: 0})
    rows = []
    selected = matched.loc[matched.selected_for_performance]
    selected_groups = {key: frame for key, frame in selected.groupby(list(CONFIG_KEYS), sort=False)}
    for key, coverage in counts.iterrows():
        group = selected_groups.get(key)
        pnl = np.array([]) if group is None else group.costed_pnl.to_numpy(dtype=float)
        costs = np.array([]) if group is None else group.entry_cost.to_numpy(dtype=float)
        reference_pnl = np.array([]) if group is None else group.reference_pnl.to_numpy(dtype=float)
        reference_cost = np.array([]) if group is None else group.reference_cost.to_numpy(dtype=float)
        stats = _statistics(pnl, costs, prefix="")
        stats.update(_statistics(reference_pnl, reference_cost, prefix="reference_"))
        rows.append({**dict(zip(CONFIG_KEYS, key)), **{name: int(value) for name, value in coverage.items()},
            "paired_origins": len(pnl), "availability_fraction": coverage.completed_origins / coverage.requested_origins,
            "comparison_sample": comparison_sample, "performance_available": len(pnl) > 1,
            **stats, "std_difference": stats["std_pnl"] - stats["reference_std_pnl"],
            "mean_difference": stats["mean_pnl"] - stats["reference_mean_pnl"]})
    return pd.DataFrame(rows)
