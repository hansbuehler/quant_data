"""Isolate proven unavailable Greek chains without changing other portfolios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.hedge2.episodes import EpisodeResult, HedgeRule
from quantdata.hedge2.greek_hedge import GreekBoxInfeasible
from quantdata.hedge2.premium import GreekBudgetInfeasible

RULE_COLUMNS = ("method", "side", "objective", "risk_aversion", "greek_budget")


@dataclass(frozen=True, kw_only=True)
class IsolatedGreekFrontier:
    results: pd.DataFrame
    unavailable: pd.DataFrame
    completed_batches: tuple[Path, ...]
    summary: dict[str, object]


def partition_failed_portfolio(*, claim_labels: tuple[str, ...], rules: tuple[HedgeRule, ...],
                               failed_claim: str, failed_rule: HedgeRule
                               ) -> tuple[tuple[tuple[str, ...], tuple[HedgeRule, ...]], ...]:
    """Partition a rectangular batch into disjoint rectangles excluding one chain."""
    if failed_claim not in claim_labels or failed_rule not in rules:
        raise ValueError("The reported unavailable portfolio is outside its current batch")
    other_claims = tuple(label for label in claim_labels if label != failed_claim)
    other_rules = tuple(rule for rule in rules if rule != failed_rule)
    parts = []
    if other_claims:
        parts.append((other_claims, rules))
    if other_rules:
        parts.append(((failed_claim,), other_rules))
    return tuple(parts)


def _hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "completed.json"}


def _verify_hashes(directory: Path, hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Immutable Greek-frontier evidence changed: {path}")


def _save_result(result: EpisodeResult, episode: Path) -> None:
    result.summary.to_parquet(episode / "summary.parquet", index=False)
    result.trades.to_parquet(episode / "trades.parquet", index=False)
    result.initial_book.to_parquet(episode / "initial_book.parquet", index=False)
    np.savez_compressed(episode / "positions.npz",
        **{f"step_{step}": values for step, values in enumerate(result.positions)},
        **{f"active_{step}": values for step, values in enumerate(result.active_indices)})
    (episode / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")
    (episode / "input_hashes.json").write_text(json.dumps(result.input_hashes, indent=2) + "\n")
    temporary = episode / "completed.partial.json"
    hashes = _hashes(episode)
    hashes.pop("completed.partial.json", None)
    temporary.write_text(json.dumps({"artifacts_sha256": hashes}, indent=2) + "\n")
    temporary.replace(episode / "completed.json")


def _bind_shared_inputs(output_directory: Path, hashes: dict[str, str] | None) -> None:
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("Greek batch lacks captured market/model input hashes")
    path = output_directory / "shared_input_hashes.json"
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError("Greek batches used different market/model inputs")
    path.write_text(json.dumps(hashes, indent=2) + "\n")


def run_isolated_greek_frontier(*, origin: date, family: str, claim_labels: tuple[str, ...],
                                rules: tuple[HedgeRule, ...], identity: dict[str, object],
                                run_batch: Callable[[tuple[str, ...], tuple[HedgeRule, ...], Path], EpisodeResult],
                                output_directory: Path) -> IsolatedGreekFrontier:
    """Run unchanged independent chains, retaining failed prefixes without scoring them.

    A failed batch is immutable diagnostic evidence. Only the exactly
    identified, independently proven unavailable portfolio is excluded;
    the remaining disjoint rectangles are replayed from inception with
    unchanged inputs and rules. A replay never liquidates the unavailable
    inventory or fabricates its later trades, settlement or terminal P&L.
    Generic numerical failures still propagate. Completed batches and
    unavailable evidence are hash-verified on a same-identity restart.
    """
    if (not claim_labels or len(set(claim_labels)) != len(claim_labels) or not rules
            or len(set(rules)) != len(rules) or any(rule.method != "DLV Greek" for rule in rules)
            or family not in ("digital-1D", "digital-3D", "digital-5D", "ko", "ratio_straddle")):
        raise ValueError("Isolation requires distinct Greek rules and claim labels in one supported family")
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Greek-frontier evidence must be outside Git working trees")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        configuration = {"origin": origin.isoformat(), "family": family, "claim_labels": list(claim_labels),
            "rules": [asdict(rule) for rule in rules], "identity": identity,
            "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))}}
        config_path = output_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError("Greek frontier inputs, rules or implementation changed on restart")
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
        queue = [(claim_labels, rules)]
        completed, unavailable, batch_directories = [], [], []
        while queue:
            labels, selected_rules = queue.pop(0)
            selection = {"claim_labels": list(labels), "rules": [asdict(rule) for rule in selected_rules]}
            batch_name = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:24]
            batch = output_directory / "batches" / batch_name
            episode = batch / origin.isoformat() / family
            episode.mkdir(parents=True, exist_ok=True)
            batch_configuration = {**identity, "origin": origin.isoformat(), "family": family, **selection}
            batch_config_path = batch / "config.json"
            if batch_config_path.exists() and json.loads(batch_config_path.read_text()) != batch_configuration:
                raise ValueError("A cached Greek batch has a different input identity")
            batch_config_path.write_text(json.dumps(batch_configuration, indent=2) + "\n")
            manifest_path = episode / "completed.json"
            unavailable_path = batch / "unavailable.json"
            failure = None
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                _verify_hashes(episode, manifest["artifacts_sha256"])
                frame = pd.read_parquet(episode / "summary.parquet")
                _bind_shared_inputs(output_directory, json.loads((episode / "input_hashes.json").read_text()))
            elif unavailable_path.exists():
                failure = json.loads(unavailable_path.read_text())
                _verify_hashes(episode, failure["artifacts_sha256"])
                if json.loads((episode / "unavailable_context.json").read_text()) != failure["context"]:
                    raise ValueError("Unavailable context differs from its sealed failure evidence")
                _bind_shared_inputs(output_directory, failure["context"].get("input_hashes"))
            else:
                try:
                    result = run_batch(labels, selected_rules, episode / "steps")
                except (GreekBudgetInfeasible, GreekBoxInfeasible) as error:
                    if error.portfolio_context is None or error.portfolio_context.get("executed_at_failure") is not False:
                        raise
                    _bind_shared_inputs(output_directory, error.portfolio_context.get("input_hashes"))
                    (episode / "unavailable_context.json").write_text(json.dumps(error.portfolio_context, indent=2) + "\n")
                    failure = {"context": error.portfolio_context, "artifacts_sha256": _hashes(episode)}
                    temporary = unavailable_path.with_suffix(".partial.json")
                    temporary.write_text(json.dumps(failure, indent=2) + "\n")
                    temporary.replace(unavailable_path)
                else:
                    _bind_shared_inputs(output_directory, result.input_hashes)
                    _save_result(result, episode)
                    frame = result.summary
            if failure is not None:
                context = failure["context"]
                if context["origin"] != origin.isoformat() or context["family"] != family:
                    raise ValueError("Unavailable evidence belongs to another episode")
                failed_rule = HedgeRule(**{name: context[name] for name in RULE_COLUMNS})
                parts = partition_failed_portfolio(claim_labels=labels, rules=selected_rules,
                    failed_claim=context["claim"], failed_rule=failed_rule)
                unavailable.append({"origin": origin.isoformat(), "family": family, "claim": context["claim"],
                    **asdict(failed_rule), "failure_date": context["date"], "error_type": context["error_type"],
                    "exact_error": context["exact_error"], "evidence_directory": str(batch), "costed_pnl": np.nan})
                queue.extend(parts)
                print(f"Greek frontier unavailable claim={context['claim']} side={failed_rule.side} "
                    f"budget={failed_rule.greek_budget:g}; unchanged remaining batches={len(queue)}", flush=True)
                continue
            expected = {(label, *[getattr(rule, name) for name in RULE_COLUMNS])
                        for label in labels for rule in selected_rules}
            actual = set(frame[["claim", *RULE_COLUMNS]].itertuples(index=False, name=None))
            if len(frame) != len(expected) or actual != expected or not np.isfinite(frame.costed_pnl).all():
                raise ValueError("A completed Greek batch lost, duplicated or fabricated portfolio outcomes")
            (batch / "summary.json").write_text(json.dumps({"status": "complete", "episodes": 1,
                "cells": len(frame), "frontier_subset_only": True}, indent=2) + "\n")
            completed.append(frame)
            batch_directories.append(batch)
        results = pd.concat(completed, ignore_index=True) if completed else pd.DataFrame()
        missing = pd.DataFrame(unavailable)
        identifiers = ["claim", *RULE_COLUMNS]
        outcomes = pd.concat([frame[identifiers] for frame in (results, missing) if len(frame)], ignore_index=True)
        expected_count = len(claim_labels) * len(rules)
        if len(outcomes) != expected_count or outcomes.duplicated(identifiers).any():
            raise RuntimeError("Greek frontier did not account exactly once for every requested portfolio")
        results.to_parquet(output_directory / "results.parquet", index=False)
        missing.to_parquet(output_directory / "unavailable.parquet", index=False)
        summary = {"status": "processing_complete_with_unavailable" if len(missing) else "complete",
            "requested_portfolios": expected_count, "completed_portfolios": len(results),
            "unavailable_portfolios": len(missing), "completed_batches": len(batch_directories),
            "complete_feasible_frontier": not len(missing), "substitute_trades": False,
            "unavailable_terminal_pnl_fabricated": False}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return IsolatedGreekFrontier(results=results, unavailable=missing,
            completed_batches=tuple(batch_directories), summary=summary)
