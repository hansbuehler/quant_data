from __future__ import annotations

from quantdata.vixdata.cboe import parse_cboe_vix_csv


def test_parse_cboe_vix_csv_sorts_and_validates() -> None:
    values = parse_cboe_vix_csv(
        b"DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2025,18,20,17,19\n01/02/2025,17,18,16,17.5\n"
    )
    assert values["date"].to_list()[0].isoformat() == "2025-01-02"
    assert values["close"].to_list() == [17.5, 19.0]


def test_parse_cboe_vix_csv_preserves_early_source_ohlc_anomaly() -> None:
    values = parse_cboe_vix_csv(
        b"DATE,OPEN,HIGH,LOW,CLOSE\n02/11/1992,19.24,18.57,17.61,17.70\n"
    )
    assert values["close"].item() == 17.7
