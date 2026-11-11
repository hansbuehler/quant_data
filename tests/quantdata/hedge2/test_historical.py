"""Historical instruments float across pairs but keep cash strike within a pair."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.historical import (
    build_historical_returns,
    historical_pair_returns,
    historical_pairs,
    sanos_europeans,
)
from quantdata.hedge2.source import surface_digitals
from quantdata.sanos.arbfree import ArbFreePricer


def source() -> ArbFreePricer:
    pricer = ArbFreePricer(smoothness=.1)
    pricer.expiries = np.array([1., 3., 5., 25.]) / 252.
    pricer.variances = .04 * pricer.expiries
    pricer.fitsX = [np.zeros(1) for _ in pricer.expiries]
    pricer.mstrikes = [np.array([.95, 1., 1.05]) for _ in pricer.expiries]
    pricer.mdensities = [np.array([.25, .5, .25]) for _ in pricer.expiries]
    return pricer


class DigitalPricer:
    """DLV stand-in whose digitals equal the saved test surface, so expected values stay explicit."""

    def __init__(self, surface: ArbFreePricer | None = None, shift: float = 0.) -> None:
        self.surface, self.shift = surface or source(), shift

    def european_at(self, *, expiry_y: float, strikes: np.ndarray, is_call: bool = True,
                    digital: bool = False) -> np.ndarray:
        assert digital
        return surface_digitals(self.surface, expiry_y=expiry_y, strikes=strikes, is_call=is_call) + self.shift


def curve(spot: float, *, forward: float = 1., reference: float = 1., discount: float = 1.) -> CashCurve:
    times = np.array([1., 3., 5., 25.])
    return CashCurve(spot=spot, business_dtes=times, forwards=spot * forward**times,
        reference_forwards=spot * reference**times, discounts=discount**times)


def test_historical_cash_strike_does_not_move_with_next_spot() -> None:
    old, following = curve(100., forward=1.005, reference=1.01, discount=.99), curve(102.)
    pricer = source()
    book = pd.DataFrame({"strike": [5000.], "bdte": [1.], "option_type": ["call"]})
    claim = Claim(label="atm", kind="digital", terminal_day=1, strike=5000., initial_spot=5000.)
    gains, target = historical_pair_returns(before_source=pricer, after_source=pricer,
        before_pricer=DigitalPricer(pricer), after_pricer=DigitalPricer(pricer),
        before_curve=old, after_curve=following, current_spot=5000., episode_initial_spot=5000.,
        book=book, claims=(claim,))
    old_option = sanos_europeans(source=pricer, curve=old, business_dtes=np.array([1.]),
        cash_strikes=np.array([100.]), is_call=np.array([True]))[0]
    expected_digital = .99 * surface_digitals(pricer, expiry_y=1. / 252., strikes=np.array([100. / 100.5]))[0]
    assert gains[0] == pytest.approx(.99 * (102. - 101.) / 100.)
    assert gains[1] == pytest.approx((.99 * 2. - old_option) / 100.)
    assert target[0] == pytest.approx(.99 - expected_digital)
    # The hedged digital is marked by the DLV pricer, never by the SANOS surface itself.
    _, shifted = historical_pair_returns(before_source=pricer, after_source=pricer,
        before_pricer=DigitalPricer(pricer, shift=.01), after_pricer=DigitalPricer(pricer),
        before_curve=old, after_curve=following, current_spot=5000., episode_initial_spot=5000.,
        book=book, claims=(claim,))
    assert shifted[0] == pytest.approx(target[0] - .99 * .01)


def test_unit_digital_has_no_spot_notional_multiplier_and_tie_is_put() -> None:
    book = pd.DataFrame({"strike": [5000.], "bdte": [1.], "option_type": ["put"]})
    claims = tuple(Claim(label=str(side), kind="digital", terminal_day=1,
        strike=5000., initial_spot=10000., is_call=side) for side in (False, True))
    arguments = dict(before_source=source(), after_source=source(), before_pricer=DigitalPricer(),
        after_pricer=DigitalPricer(), before_curve=curve(100.),
        after_curve=curve(100.), current_spot=5000., book=book, claims=claims)
    gains, target = historical_pair_returns(**arguments, episode_initial_spot=5000.)
    smaller, same_target = historical_pair_returns(**arguments, episode_initial_spot=10000.)
    np.testing.assert_allclose(smaller, .5 * gains)
    np.testing.assert_allclose(same_target, target)
    assert target[0] > 0
    assert target[1] < 0
    assert target.sum() == pytest.approx(0., abs=1e-12)


def test_source_forward_prices_and_reference_forward_finances_spot_separately() -> None:
    pricer, old = source(), curve(100., forward=1.005, reference=.999, discount=.998)
    book = pd.DataFrame({"strike": [4950., 5050.], "bdte": [3., 3.], "option_type": ["put", "call"]})
    claim = Claim(label="3D", kind="digital", terminal_day=3, strike=5000., initial_spot=5000.)
    arguments = dict(before_source=pricer, after_source=pricer, before_pricer=DigitalPricer(pricer),
        after_pricer=DigitalPricer(pricer), before_curve=old, after_curve=curve(101.),
        current_spot=5000., episode_initial_spot=5000., book=book, claims=(claim,))
    gains, target = historical_pair_returns(**arguments)
    assert gains[0] == pytest.approx(.998 * (101. - 99.9) / 100.)
    before = .998**3 * surface_digitals(pricer, expiry_y=3. / 252., strikes=np.array([100. / (100. * 1.005**3)]))[0]
    after = surface_digitals(pricer, expiry_y=2. / 252., strikes=np.array([100. / 101.]))[0]
    assert target[0] == pytest.approx(.998 * after - before)
    discounted, discounted_target = historical_pair_returns(**arguments, discount_to_current=.9)
    np.testing.assert_allclose(discounted, .9 * gains)
    np.testing.assert_allclose(discounted_target, .9 * target)


def test_history_window_excludes_future_dates_and_nonadjacent_sessions() -> None:
    days = tuple(date(2023, 1, day) for day in (9, 10, 12, 13, 17))
    result = historical_pairs(dates=days, as_of_date=date(2023, 1, 13), window_transitions=2,
        history_start=date(2023, 1, 9))
    assert result == ((days[0], days[1]), (days[2], days[3]))
    with pytest.raises(ValueError, match="found 2"):
        historical_pairs(dates=days, as_of_date=date(2023, 1, 13), window_transitions=3,
            history_start=date(2023, 1, 9))


def test_exotic_joint_law_is_not_invented_from_saved_european_marginals() -> None:
    book = pd.DataFrame({"strike": [5000.], "bdte": [5.], "option_type": ["call"]})
    claim = Claim(label="ko", kind="ko", terminal_day=5, observation_day=3,
        barrier=5000., strike=4950., initial_spot=5000.)
    with pytest.raises(ValueError, match="digital targets"):
        historical_pair_returns(before_source=source(), after_source=source(), before_pricer=DigitalPricer(),
            after_pricer=DigitalPricer(), before_curve=curve(100.),
            after_curve=curve(101.), current_spot=5000., episode_initial_spot=5000., book=book, claims=(claim,))


def test_joint_window_preserves_raw_returns_and_causal_pair_provenance(monkeypatch, tmp_path) -> None:
    dates = tuple(date(2023, 1, day) for day in (9, 10, 11, 12))
    visited = []

    class Archive:
        def dates(self):
            return dates

        def load(self, day):
            visited.append(day)
            return source(), {"source_sha256": day.isoformat()}

    directory = tmp_path / "licensed" / "SPX"
    directory.mkdir(parents=True)
    for day in dates:
        (directory / f"{day.isoformat()}.pdq").write_bytes(day.isoformat().encode())
    monkeypatch.setattr("quantdata.hedge2.historical.licensed_curve",
        lambda directory, day: curve(100. + dates.index(day)))
    monkeypatch.setattr("quantdata.hedge2.historical.DailyState.load",
        lambda directory, day: SimpleNamespace(decode=lambda: SimpleNamespace(pricer=DigitalPricer())))
    book = pd.DataFrame({"strike": [5000.], "bdte": [1.], "option_type": ["call"]})
    claim = Claim(label="1D", kind="digital", terminal_day=1, strike=5000., initial_spot=5000.)
    result = build_historical_returns(archive=Archive(), licensed_fit_directory=directory.parent, daily_directory=tmp_path,
        as_of_date=dates[2], current_spot=5000., book=book, claims=(claim,), window_transitions=2)
    np.testing.assert_allclose(result.weights @ result.instrument_gains, 0., atol=1e-14)
    np.testing.assert_allclose(result.raw_instrument_gains - result.instrument_gains,
        np.broadcast_to(result.removed_instrument_mean, result.raw_instrument_gains.shape))
    np.testing.assert_allclose(result.mapped_cash_strikes[:, 0], [100., 101.])
    np.testing.assert_array_equal(result.pair_end_dates, ["2023-01-10", "2023-01-11"])
    assert max(visited) <= dates[2]
    assert result.carry_inputs.shape == (2, 2, 2, 3)
    assert result.source_hashes.shape == result.curve_hashes.shape == (2, 2)
    assert result.weights @ result.claim_gains[:, 0] > .1
    premium = build_historical_returns(archive=Archive(), licensed_fit_directory=directory.parent, daily_directory=tmp_path,
        as_of_date=dates[2], current_spot=5000., book=book, claims=(claim,), window_transitions=2,
        return_basis="premium")
    np.testing.assert_allclose(premium.raw_instrument_gains,
        premium.raw_cash_gains / premium.starting_instrument_prices)
    np.testing.assert_allclose(premium.weights @ premium.instrument_gains, 0., atol=1e-14)
    np.testing.assert_allclose(premium.claim_gains, result.claim_gains)
    assert premium.return_basis == "premium"


def test_zero_strike_calls_use_source_forward_not_financing_forward() -> None:
    old = curve(100., forward=1.005, reference=1.01, discount=.99)
    book = pd.DataFrame({"strike": [0.], "bdte": [1.], "option_type": ["call"],
        "instrument_kind": ["zero_strike_call"]})
    claim = Claim(label="1D", kind="digital", terminal_day=1, strike=5000., initial_spot=5000.)
    gains, _ = historical_pair_returns(before_source=source(), after_source=source(),
        before_pricer=DigitalPricer(), after_pricer=DigitalPricer(),
        before_curve=old, after_curve=curve(102.), current_spot=5000., episode_initial_spot=5000.,
        book=book, claims=(claim,))
    assert gains[0] == pytest.approx(.99 * (102. - 101.) / 100.)
    assert gains[1] == pytest.approx((.99 * 102. - .99 * 100.5) / 100.)
