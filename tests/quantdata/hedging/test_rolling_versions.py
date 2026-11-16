"""Only the agreed versions/measures may enter the corrected rolling report."""

import json
import sys
from contextlib import nullcontext
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from quantdata.hedging.model_audit import rolling_versions
from quantdata.hedging.model_audit.rolling_versions import (
    INITIALIZATION_POLICY,
    REPORT_CASES,
    origins,
)


def test_report_case_inventory_has_no_dysanos_q_or_extra_variants() -> None:
    assert REPORT_CASES == ("DYSANOS v1-P", "DYSANOS v2-P", "GROMM v1-P", "GROMM v1-Q", "GROMM v2-P", "GROMM v2-Q")


def test_initialization_policy_requires_independent_windows() -> None:
    assert INITIALIZATION_POLICY == "independent rolling windows; no preceding-origin fitted weights"


@pytest.mark.parametrize("family", ["gromm1", "gromm2"])
def test_later_origin_does_not_load_or_pass_previous_origin(tmp_path, monkeypatch, family) -> None:
    selected = ("2024-04-01", "2024-04-02")
    monkeypatch.setattr(rolling_versions, "origins", lambda reference: selected)
    monkeypatch.setattr(rolling_versions.signal, "signal", lambda *args: None)
    calls = {}
    origin = date(2024, 4, 2)
    model = SimpleNamespace(origin=origin, dates=[date(2024, 4, 1), origin],
        endpoints=SimpleNamespace(pairs=[(date(2024, 3, 28), date(2024, 4, 1))]))

    def fit_origin(dataset, day, config, training, directory, *, previous, **kwargs):
        calls["representation_previous"] = previous
        assert directory == tmp_path / "fits"
        (directory / family / str(day) / "representation.npz").write_bytes(b"independent representation")
        return model

    def estimate_origin(representation, dataset, sessions, destination, *, previous, **kwargs):
        calls["dynamics_previous"] = previous
        destination.write_bytes(b"independent dynamics")

    def probability_refit(representation, dynamics, directory, *, previous_path, **kwargs):
        calls["probability_previous"] = previous_path
        directory.mkdir()
        (directory / "dynamics.npz").write_bytes(b"independent weighted dynamics")

    def forbidden_prior_load(path):
        raise AssertionError(f"Unexpected fitted representation load: {path}")

    namespace = "openqdata.models.gromm"
    modules = {
        "openqdata.biz_calendar": SimpleNamespace(get_business_days=lambda *args: []),
        namespace: SimpleNamespace(gromm1=SimpleNamespace(Config=object), gromm2=SimpleNamespace(Config=object)),
        f"{namespace}.config": SimpleNamespace(TrainingConfig=lambda **kwargs: kwargs, FitConfig=lambda **kwargs: kwargs),
        f"{namespace}.data": SimpleNamespace(LicensedDataset=lambda *args, **kwargs: object()),
        f"{namespace}.dynamics": SimpleNamespace(estimate_origin=estimate_origin,
            load_dynamics=lambda path: SimpleNamespace(identity="independent-fit")),
        f"{namespace}.rolling": SimpleNamespace(fit_origin=fit_origin, load_representation=forbidden_prior_load),
        f"{namespace}.progress": SimpleNamespace(Heartbeat=lambda *args, **kwargs:
            nullcontext(SimpleNamespace(update=lambda event: None))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    from quantdata.hedging.model_audit import gromm_probability

    monkeypatch.setattr(gromm_probability, "run", probability_refit)
    rolling_versions.gromm_origin(tmp_path, tmp_path, family=family, day=selected[1], source=tmp_path)
    assert calls == ({"representation_previous": None, "dynamics_previous": None, "probability_previous": None}
                     if family == "gromm1" else {"representation_previous": None, "dynamics_previous": None})
    completed = json.loads((tmp_path / f"fits/{family}/{selected[1]}/COMPLETE.json").read_text())
    assert completed["initialization_origin"] is None
    assert completed["initialization_policy"] == INITIALIZATION_POLICY


def test_rolling_origin_manifest_is_matched_not_a_single_origin(tmp_path) -> None:
    (tmp_path / "observable_reference.json").write_text(json.dumps({"dates": ["2025-08-29"]}))
    with pytest.raises(ValueError, match="44 April-May"):
        origins(tmp_path)


def test_rolling_origin_manifest_keeps_exact_order(tmp_path) -> None:
    first = date(2024, 4, 1)
    dates = [str(first + timedelta(days=offset)) for offset in range(61)
             if (first + timedelta(days=offset)).weekday() < 5]
    dates.remove("2024-05-27")
    (tmp_path / "observable_reference.json").write_text(json.dumps({"dates": dates}))
    assert origins(tmp_path) == tuple(dates)
    (tmp_path / "observable_reference.json").write_text(json.dumps({"dates": dates[::-1]}))
    with pytest.raises(ValueError, match="44 April-May"):
        origins(tmp_path)
