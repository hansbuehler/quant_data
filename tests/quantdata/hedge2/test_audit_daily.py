"""Partial cash audits must not count unfinished or changed evidence as completed."""

import hashlib
import json

import pytest

from quantdata.hedge2.audit_daily import completed_audit_dates


def test_partial_audit_captures_only_hashed_completed_dates(tmp_path) -> None:
    directory = tmp_path / "2024-04-19"
    directory.mkdir()
    evidence = directory / "results.parquet"
    evidence.write_bytes(b"immutable evidence")
    marker = {"artifacts_sha256": {evidence.name: hashlib.sha256(evidence.read_bytes()).hexdigest()}}
    (directory / "completed.json").write_text(json.dumps(marker))
    configuration = {"method": "parametric", "dates": ["2024-04-19", "2024-04-22"]}
    assert completed_audit_dates(input_directory=tmp_path, configuration=configuration,
        completed_only=True) == ("2024-04-19",)
    assert completed_audit_dates(input_directory=tmp_path, configuration=configuration,
        completed_only=False) == ("2024-04-19", "2024-04-22")
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        completed_audit_dates(input_directory=tmp_path, configuration=configuration, completed_only=True)


def test_empty_partial_run_cannot_report_a_successful_audit(tmp_path) -> None:
    with pytest.raises(ValueError, match="No complete"):
        completed_audit_dates(input_directory=tmp_path,
            configuration={"method": "historical", "dates": ["2024-04-19"]}, completed_only=True)


def test_legacy_flat_completed_markers_remain_readable(tmp_path) -> None:
    evidence = tmp_path / "2024-04-19.parquet"
    evidence.write_bytes(b"completed reference")
    (tmp_path / "2024-04-19.complete.json").write_text(json.dumps({
        evidence.name: hashlib.sha256(evidence.read_bytes()).hexdigest()}))
    assert completed_audit_dates(input_directory=tmp_path, configuration={"dates": ["2024-04-19"]},
        completed_only=True) == ("2024-04-19",)
