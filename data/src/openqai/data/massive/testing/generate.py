"""
Generate massive test data
^^^^^^^^^^^^^^^^^^^^^^^^^^

python -m generate

"""

import os
os.environ["QUANTDATA_TEST_MODE"] = "generate_tests" # do this before importing anything else

from quantdata import env# triggers construction of caches.
from quantdata.massive.testing.scope import TICKERS, DATES

def generate_day(ticker, d):
    from quantdata.massive.client import MassiveClient
    from quantdata.massive.processing import filter_valid_trades, filter_valid_quotes, align_quotes_to_trades, align_trades_to_quotes

    client = MassiveClient()
    ocv = client.fetch_day_summary(ticker, d, adjusted=True)
    print(ocv)
    ocv = client.fetch_day_summary(ticker, d, adjusted=False)
    print(ocv)
    trades = client.fetch_trades(ticker, d) # extracts trades, removed zero size and price, and excludes trades with NORMAL_TRADE_EXCLUDE_FLAGS
    print(trades)
    quotes = client.fetch_quotes(ticker, d)
    print(quotes)
    v = filter_valid_trades(ticker, d, True)
    print(v)
    v = filter_valid_quotes(ticker, d, False)
    print(v)
    v = align_quotes_to_trades(ticker, d, drop_unmatched = True)
    print(v)
    v = align_quotes_to_trades(ticker, d, drop_unmatched = False)
    print(v)
    v = align_trades_to_quotes(ticker, d, drop_unmatched = True)
    print(v)
    v = align_trades_to_quotes(ticker, d, drop_unmatched = False)
    print(v)
    v = align_quotes_to_trades(ticker, d, drop_unmatched = True, within_session = False)
    print(v)
    v = align_quotes_to_trades(ticker, d, drop_unmatched = False, within_session = False)
    print(v)
    v = align_trades_to_quotes(ticker, d, drop_unmatched = True, within_session = False)
    print(v)
    v = align_trades_to_quotes(ticker, d, drop_unmatched = False, within_session = False)
    print(v)
    
def main():
    for ticker in TICKERS:
        for d in DATES:
            print(f"Generating data for {ticker} {d}")
            generate_day(ticker, d)

if __name__ == "__main__":
    main()