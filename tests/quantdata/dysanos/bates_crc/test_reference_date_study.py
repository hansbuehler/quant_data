from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.bates_crc.reference_date_study import (
    MIN_HALF_SPREAD,
    QuoteBook,
    _spread_residual,
    crc_dynamic_audit_blocker,
    fit_metrics,
    load_dataset,
    model_provenance,
)


def _book() -> QuoteBook:
    return QuoteBook(
        date="2025-08-29",
        strikes=np.asarray((0.9, 1.1)),
        is_call=np.asarray((True, False)),
        bids=np.asarray((0.11, 0.19)),
        asks=np.asarray((0.13, 0.21)),
        mids=np.asarray((0.12, 0.20)),
        business_dtes=np.asarray((5.0, 5.0)),
        spot_ref=1.0,
    )


def test_put_quotes_are_converted_to_call_equivalents() -> None:
    bid, ask, mid = _book().call_equivalents()
    np.testing.assert_allclose(bid, (0.11, 0.09))
    np.testing.assert_allclose(ask, (0.13, 0.11))
    np.testing.assert_allclose(mid, (0.12, 0.10))


def test_excess_is_zero_inside_and_distance_to_nearest_boundary_outside() -> None:
    book = _book()
    residual = _spread_residual(np.asarray((0.12, 0.13)), book)
    np.testing.assert_allclose(residual, (0.0, 2.0))
    metrics = fit_metrics(np.asarray((0.12, 0.13)), book)
    assert metrics["quote_count"] == 2
    assert metrics["percent_inside_bid_ask"] == 50.0
    assert metrics["conditional_mean_excess_half_spreads"] == pytest.approx(2.0)
    assert MIN_HALF_SPREAD > 0.0


def test_crc_blocker_prevents_mislabeling_no_reset_bates_as_crc() -> None:
    blocker = crc_dynamic_audit_blocker()
    assert blocker["status"] == "blocked-not-run"
    assert "not a test of the reserve-CRC model" in blocker["invalid_shortcut"]
    assert blocker["requested_geometry"]["antithetic_pair_count"] == 50_000


def test_reference_dataset_rejects_non_strict_pre_target_history(tmp_path) -> None:
    path = tmp_path / "bad_dates.npz"
    np.savez(path, dates=np.asarray(("2025-08-29", "2025-08-28", "2025-08-29")))
    with pytest.raises(RuntimeError, match="exactly one"):
        load_dataset(path)

    path = tmp_path / "out_of_order_dates.npz"
    np.savez(path, dates=np.asarray(("2025-08-30", "2025-08-29")))
    with pytest.raises(RuntimeError, match="strictly before"):
        load_dataset(path)


def test_model_provenance_is_explicit_and_deterministic() -> None:
    first = model_provenance()
    second = model_provenance()
    assert first == second
    assert first["model_family"] == (
        "DLV x ordinary-Bates background calibration for the in-repo reserve CRC"
    )
    assert first["implementation_scope"] == "static initial fit of the no-reset background"
    assert first["literal_gambara_teichmann_algorithm_1"] is False
