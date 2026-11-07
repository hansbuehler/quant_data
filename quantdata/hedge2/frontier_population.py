"""Read independently checked Greek outcomes without hiding unfinished computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from cdxcore.util import is_filename

from quantdata.hedge2.audit_greek_frontier import audit_isolated_frontier
from quantdata.hedge2.episode_artifacts import verify_episode_artifacts
from quantdata.hedge2.frontier_metrics import PORTFOLIO_KEYS

EPISODE_KEYS = ("variant", "origin", "family")


@dataclass(frozen=True, kw_only=True)
class FrontierPopulation:
    requested: pd.DataFrame
    available: pd.DataFrame
    unavailable: pd.DataFrame
    pending: pd.DataFrame
    episodes: pd.DataFrame
    evidence: dict[str, object]


def external_output_directory(directory: Path) -> None:
    """Reject generated reports and audits anywhere inside a Git working tree."""
    resolved = directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Generated frontier evidence must remain outside Git working trees")


def _validate_requested(requested: pd.DataFrame) -> None:
    if (requested.empty or not set(PORTFOLIO_KEYS).issubset(requested.columns)
            or requested[list(PORTFOLIO_KEYS)].isna().any().any()
            or requested.duplicated(list(PORTFOLIO_KEYS)).any()
            or not requested.side.isin((-1, 1)).all()):
        raise ValueError("The explicit requested population needs unique complete portfolio identities")
    if not requested.origin.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Requested origins must be ISO date strings")
    if (not requested.variant.map(lambda value: isinstance(value, str) and is_filename(value)).all()
            or not requested.family.isin(("digital-1D", "digital-3D", "digital-5D", "ko", "ratio_straddle")).all()):
        raise ValueError("Requested variants and families must be safe supported directory names")
    parsed = pd.to_datetime(requested.origin, format="%Y-%m-%d", errors="raise")
    if not parsed.dt.strftime("%Y-%m-%d").eq(requested.origin).all():
        raise ValueError("Requested origins must be canonical ISO dates")
    budgets = requested.greek_budget.to_numpy(dtype=float)
    if not np.isfinite(budgets).all() or np.any((budgets < 0.) | (budgets > 1.)):
        raise ValueError("Requested Greek budgets must lie in [0,1]")


def _empty_outcomes() -> pd.DataFrame:
    return pd.DataFrame(columns=[*PORTFOLIO_KEYS, "costed_pnl", "entry_cost", "initial_price"])


def _contract_binding(*, configuration: dict[str, object], checkpoint: Path,
                      shared: dict[str, str]) -> str:
    """Bind source-defined claims, original numeraire, observations and frozen book."""
    with np.load(checkpoint, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity_json"]))
    if identity["input_hashes"] != shared:
        raise ValueError("The episode checkpoint and sealed source-input hashes disagree")
    if (not np.isfinite(identity["initial_spot"]) or identity["initial_spot"] <= 0.
            or not shared or any(not isinstance(value, str) or len(value) != 64 for value in shared.values())):
        raise ValueError("Claim binding requires a positive original spot and sealed source hashes")
    model = configuration.get("identity", configuration)
    dynamics = Path(model["dynamics_directory"]).resolve()
    economic = {name: digest for name, digest in shared.items()
                if not Path(name).is_absolute() or not Path(name).resolve().is_relative_to(dynamics)}
    if not {"frozen_book", "observed_marks_spreads_and_carry"}.issubset(economic):
        raise ValueError("Claim binding lacks the frozen book or original market observations")
    definition = {"origin": identity["origin"], "family": identity["family"],
        "initial_spot": identity["initial_spot"], "option_side": identity["option_side"],
        "include_co_terminal_forwards": identity["include_co_terminal_forwards"],
        "claims_implementation": configuration["implementation_sha256"]["claims.py"],
        "economic_inputs": economic}
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def load_frontier_population(*, requested: pd.DataFrame,
                             sources: dict[str, tuple[Path, ...]],
                             audit_directory: Path,
                             expected_identities: dict[str, dict[str, object]] | None = None,
                             progress: Callable[[str], None] | None = None) -> FrontierPopulation:
    """Audit closed frontiers, keeping every other requested chain explicitly pending.

    A completion summary is required before any frontier is read. The
    independent auditor checks immutable accounting artifacts, risk replay,
    unavailable geometry and actual executed prefixes. Source directories
    are read-only. Multiple completed copies of one episode are rejected,
    not silently selected. Pending does not assert that a worker is live.
    """
    _validate_requested(requested)
    external_output_directory(audit_directory)
    if set(sources) != set(requested.variant):
        raise ValueError("Every requested variant needs its explicit source directories")
    if expected_identities is not None and set(expected_identities) != set(sources):
        raise ValueError("Every source variant needs its expected economic/model identity")
    for directories in sources.values():
        if not directories or len({directory.resolve() for directory in directories}) != len(directories):
            raise ValueError("Frontier source directories must be nonempty and distinct")
    available, unavailable, pending, episodes, evidence = [], [], [], [], {}
    groups = requested.groupby(list(EPISODE_KEYS), sort=True)
    for index, (key, selection) in enumerate(groups):
        variant, origin, family = key
        candidates = [directory / origin / family for directory in sources[variant]
                      if (directory / origin / family / "summary.json").exists()]
        if len(candidates) > 1:
            raise ValueError(f"Multiple completed frontier sources for {key}")
        if not candidates:
            pending.append(selection.assign(costed_pnl=np.nan))
            episodes.append(dict(zip(EPISODE_KEYS, key)) | {"status": "pending",
                "requested_portfolios": len(selection), "completed_portfolios": 0,
                "unavailable_portfolios": 0, "pending_portfolios": len(selection)})
            continue
        directory = candidates[0]
        configuration = json.loads((directory / "config.json").read_text())
        if configuration["origin"] != origin or configuration["family"] != family:
            raise ValueError("Frontier path and saved episode identity disagree")
        if expected_identities is not None and any(configuration["identity"].get(name) != value
                for name, value in expected_identities[variant].items()):
            raise ValueError("Frontier economic/model identity differs from the report specification")
        rules = configuration["rules"]
        if any(rule["method"] != "DLV Greek" or rule["objective"] != "none"
               or rule["risk_aversion"] != 0. for rule in rules):
            raise ValueError("A Greek frontier cannot substitute another optimization method")
        expected = {(label, rule["side"], rule["greek_budget"])
                    for label in configuration["claim_labels"] for rule in rules}
        actual = set(selection[["claim", "side", "greek_budget"]].itertuples(index=False, name=None))
        if len(selection) != len(expected) or actual != expected:
            raise ValueError("Completed frontier differs from the explicitly requested target/rule population")
        audit = audit_isolated_frontier(input_directory=directory,
            output_directory=audit_directory / variant / origin / family)
        completed = pd.read_parquet(directory / "results.parquet")
        missing = pd.read_parquet(directory / "unavailable.parquet")
        shared = json.loads((directory / "shared_input_hashes.json").read_text())
        bindings = {_contract_binding(configuration=configuration,
            checkpoint=path.parent / "steps/episode_checkpoint.npz", shared=shared)
            for path in sorted((directory / "batches").glob("*/*/*/completed.json"))}
        if len(completed) and len(bindings) != 1:
            raise ValueError("Completed Greek batches do not share one source-defined claim/numeraire identity")
        for frame in (completed, missing):
            if len(frame) and (not frame.origin.eq(origin).all()
                              or "family" in frame and not frame.family.eq(family).all()):
                raise ValueError("A frontier result contains another episode's outcomes")
        if len(completed):
            available.append(completed.assign(variant=variant, family=family,
                contract_definition_sha256=next(iter(bindings))))
        if len(missing):
            unavailable.append(missing.assign(variant=variant, family=family))
        episodes.append(dict(zip(EPISODE_KEYS, key)) | {"status": "audited",
            "requested_portfolios": len(selection), "completed_portfolios": len(completed),
            "unavailable_portfolios": len(missing), "pending_portfolios": 0})
        evidence[str(directory)] = {"audit": audit,
            "artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("config.json", "summary.json", "results.parquet", "unavailable.parquet", "shared_input_hashes.json")}}
        if progress is not None and (index % 10 == 0 or index + 1 == len(groups)):
            progress(f"Frontier audit episodes={index + 1}/{len(groups)} last={variant}/{origin}/{family}")
    result = FrontierPopulation(requested=requested.copy(),
        available=pd.concat(available, ignore_index=True) if available else _empty_outcomes(),
        unavailable=pd.concat(unavailable, ignore_index=True) if unavailable else _empty_outcomes(),
        pending=pd.concat(pending, ignore_index=True) if pending else _empty_outcomes(),
        episodes=pd.DataFrame(episodes), evidence=evidence)
    outcomes = pd.concat([frame[list(PORTFOLIO_KEYS)] for frame in
        (result.available, result.unavailable, result.pending)], ignore_index=True)
    check = requested[list(PORTFOLIO_KEYS)].merge(outcomes, on=list(PORTFOLIO_KEYS),
        how="outer", indicator=True, validate="one_to_one")
    if not check._merge.eq("both").all():
        raise ValueError("Frontier sources omitted or substituted requested portfolios")
    return result


def load_reference_population(*, directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a complete analytic reference with its retained cash audit and sealed files."""
    summary = json.loads((directory / "summary.json").read_text())
    configuration = json.loads((directory / "config.json").read_text())
    audit = json.loads((directory / "independent_cash_audit" / "summary.json").read_text())
    completions = sorted(directory.glob("*/*/completed.json"))
    maximum_error = float(audit["maximum_cash_error"])
    if (summary["status"] != "complete" or len(completions) != summary["episodes"]
            or audit["completed_episodes"] != summary["episodes"] or audit["portfolios"] != summary["cells"]
            or not np.isfinite(maximum_error) or maximum_error > 1e-10):
        raise ValueError("An incomplete or unaudited reference cannot enter frontier comparisons")
    frames, evidence = [], {}
    configuration_sha256 = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    for completion in completions:
        verify_episode_artifacts(directory=completion.parent, configuration_sha256=configuration_sha256)
        frame = pd.read_parquet(completion.with_name("summary.parquet"))
        if not frame.origin.eq(completion.parent.parent.name).all():
            raise ValueError("A reference result contains a different origin")
        shared = json.loads((completion.parent / "input_hashes.json").read_text())
        binding = _contract_binding(configuration=configuration,
            checkpoint=completion.parent / "steps/episode_checkpoint.npz", shared=shared)
        frames.append(frame.assign(family=completion.parent.name, contract_definition_sha256=binding))
        evidence[str(completion)] = hashlib.sha256(completion.read_bytes()).hexdigest()
    data = pd.concat(frames, ignore_index=True)
    if len(data) != summary["cells"]:
        raise ValueError("Reference completion counts do not match its retained results")
    return data, {"completion": summary, "configuration": configuration,
        "configuration_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
        "cash_audit": audit, "completion_sha256": evidence,
        "audit_sha256": hashlib.sha256((directory / "independent_cash_audit" / "summary.json").read_bytes()).hexdigest()}
