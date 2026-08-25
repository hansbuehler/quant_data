from __future__ import annotations

import numpy as np

from quantdata.dysanos.flex.powerexp_oce import evaluate_powerexp_oce_grid


def test_powerexp_oce_of_deterministic_pnl_is_the_pnl() -> None:
    pnl = np.full((128, 2), (0.02, -0.03), dtype=np.float64)
    report = evaluate_powerexp_oce_grid(pnl, (0.0, 2.0, 8.0))

    for item in report:
        notional = float(item["notional"])
        np.testing.assert_allclose(item["oce"], notional * np.array((0.02, -0.03)), atol=1.0e-12)
        np.testing.assert_allclose(item["notional_derivative"], pnl[0], atol=1.0e-12)
        assert item["maximum_translation_first_order_residual"] < 1.0e-12


def test_powerexp_oce_penalizes_a_two_sided_zero_mean_direction() -> None:
    pnl = np.tile(np.array((-0.05, 0.05), dtype=np.float64), 256)[:, None]
    report = evaluate_powerexp_oce_grid(pnl, (0.0, 1.0, 4.0, 16.0))

    assert abs(float(report[0]["oce"][0])) < 1.0e-12
    assert all(float(item["oce"][0]) < 0.0 for item in report[1:])
    assert all(float(item["notional_derivative"][0]) < 0.0 for item in report[1:])
    assert all(item["maximum_translation_first_order_residual"] < 1.0e-12 for item in report)
