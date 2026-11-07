"""Pilot groups and cash replay must not change the benchmark's target or book definitions."""

from dataclasses import replace

import numpy as np
import pytest

from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.terminal_market import TerminalSimulationConfig, generate_terminal_paths
from quantdata.hedge2.terminal_pilot import audit_terminal_cash, claim_groups
from tests.quantdata.hedge2.test_terminal_market import inputs


def test_claim_groups_cover_all_targets_without_mixing_digital_and_exotic_books() -> None:
    claims = make_claims(initial_spot=100., atm_vols_by_day={tenor: .2 for tenor in (1, 2, 3, 5)})
    groups = claim_groups(claims)
    assert [len(group) for group in groups.values()] == [10, 10, 10, 18]
    assert set(claim.label for group in groups.values() for claim in group) == set(claim.label for claim in claims)
    assert {claim.terminal_day for claim in groups["digital-3D"]} == {3}
    assert all(claim.kind != "digital" for claim in groups["exotic-3D-5D"])


def test_cash_audit_detects_corrupt_raw_accounting_independently_of_centered_gains() -> None:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=6))
    audit = audit_terminal_cash(paths=result, book=arguments["book"], curve=arguments["curve"],
        claims=arguments["claims"], current_marks=arguments["current_claim_marks"])
    assert max(audit.values()) < 1e-12
    corrupted = result.raw_instrument_gains.copy()
    corrupted[0, 0, 1] += .001
    with pytest.raises(ValueError, match="cash replay"):
        audit_terminal_cash(paths=replace(result, raw_instrument_gains=corrupted), book=arguments["book"],
            curve=arguments["curve"], claims=arguments["claims"], current_marks=arguments["current_claim_marks"])


def test_cash_audit_catches_claim_payoff_changes() -> None:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=6))
    corrupted = result.claim_payoffs.copy()
    corrupted[0, 2] += .001
    with pytest.raises(ValueError, match="cash replay"):
        audit_terminal_cash(paths=replace(result, claim_payoffs=corrupted), book=arguments["book"],
            curve=arguments["curve"], claims=arguments["claims"], current_marks=arguments["current_claim_marks"])
