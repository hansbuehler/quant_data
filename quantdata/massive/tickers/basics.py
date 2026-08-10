"""
quantdata.massive.tickers.basics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Listing date and basic reference data for tickers via MassiveClient.
"""

from __future__ import annotations

from datetime import date

from cdxcore.err import verify
from cdxcore.version import version

from quantdata.massive import MassiveClient

MANUAL_LISTING_DATES: dict[str, date] = {
    "VXX": date(2019, 5, 1),
}
"""
Overrides for tickers whose Massive-reported list_date is too late to be used as the
data start date (e.g. a re-listed product).
"""


@version("0.1", dependencies=[MassiveClient.ticker_reference_data])
def get_listing_date(ticker: str, client: MassiveClient | None = None) -> date:
    """
    Return the listing date for ``ticker``.

    Applies :attr:`MANUAL_LISTING_DATES` overrides and raises if the ticker is not active.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL".
    client : MassiveClient | None
        Optional pre-constructed client; a new one is created if ``None``.

    Returns
    -------
    date
        The later of the Massive-reported list_date and any manual override.
    """
    client = MassiveClient() if client is None else client
    data = client.ticker_reference_data(ticker)

    verify(
        data.get("active", False),
        lambda: f"Ticker {ticker} is not currently active on Massive",
    )

    list_date = data.get("list_date")
    verify(
        list_date is not None,
        lambda: f"No list_date found for ticker {ticker} on Massive",
    )

    manual = MANUAL_LISTING_DATES.get(ticker, list_date)
    return min(list_date, manual)
