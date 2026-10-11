from datetime import date

import pytest
from cdxcore.subdir import CacheMustExistError

from quantdata.dysanos import download as dysanos_download
from quantdata.dysanos.flex import market as flex_market


@pytest.mark.parametrize(
    ("ticker", "cache_mode", "expected"),
    [
        ("SPX", "cacheonly", "ready"),
        ("AAPL", "on", "ready"),
    ],
)
def test_load_market_job_uses_expected_cache_mode(
    monkeypatch,
    ticker: str,
    cache_mode: str,
    expected: str,
) -> None:
    calls: dict[str, object] = {}

    def load_market_day(*args, **kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(dysanos_download, "load_market_day", load_market_day)

    result = dysanos_download._load_market_job(
        ticker,
        date(2024, 3, 5),
        geometry=dysanos_download.FlexGeometryConfig(),
        market_config=dysanos_download.FlexMarketConfig(),
        cache_only=ticker == "SPX",
    )

    assert result == expected
    assert calls["override_cache_mode"] == cache_mode


def test_spx_cache_validation_reports_missing_cache(monkeypatch) -> None:
    def load_market_day(*args, **kwargs):
        assert kwargs["override_cache_mode"] == "cacheonly"
        raise CacheMustExistError("missing market cache")

    monkeypatch.setattr(dysanos_download, "load_market_day", load_market_day)

    assert (
        dysanos_download._load_market_job(
            "SPX",
            date(2024, 3, 5),
            geometry=dysanos_download.FlexGeometryConfig(),
            market_config=dysanos_download.FlexMarketConfig(),
            cache_only=True,
        )
        == "cache-missing"
    )


def test_non_spx_cache_errors_are_not_hidden(monkeypatch) -> None:
    def load_market_day(*args, **kwargs):
        raise CacheMustExistError("missing source data")

    monkeypatch.setattr(dysanos_download, "load_market_day", load_market_day)

    with pytest.raises(CacheMustExistError, match="missing source data"):
        dysanos_download._load_market_job(
            "AAPL",
            date(2024, 3, 5),
            geometry=dysanos_download.FlexGeometryConfig(),
            market_config=dysanos_download.FlexMarketConfig(),
            cache_only=False,
        )


def test_spx_is_default_cache_only_even_when_not_first(monkeypatch) -> None:
    def load_market_day(*args, **kwargs):
        assert kwargs["override_cache_mode"] == "cacheonly"
        return object()

    monkeypatch.setattr(dysanos_download, "load_market_day", load_market_day)

    assert dysanos_download._load_market_job(
        "SPX",
        date(2024, 3, 5),
        geometry=dysanos_download.FlexGeometryConfig(),
        market_config=dysanos_download.FlexMarketConfig(),
        cache_only=True,
    ) == "ready"


def test_load_market_day_forwards_cache_mode(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def smooth_interpolated_market(*args, **kwargs):
        calls.update(kwargs)
        return {"cached": True}

    monkeypatch.setattr(
        flex_market,
        "smooth_interpolated_market",
        smooth_interpolated_market,
    )
    monkeypatch.setattr(
        flex_market,
        "market_day_from_payload",
        lambda payload, **kwargs: object(),
    )

    flex_market.load_market_day(
        "SPX",
        date(2024, 3, 5),
        override_cache_mode="cacheonly",
        verbose=None,
    )

    assert calls["override_cache_mode"] == "cacheonly"
