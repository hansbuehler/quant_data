"""
Generate WIDB test data
"""

import os
os.environ["QUANTDATA_TEST_MODE"] = "generate_tests" # do this before importing anything else

from quantdata import env# triggers construction of caches.
from quantdata.biz_calendar import next_business_day
from cdxcore.verbose import Context

def generate_day(ticker, d):
    from quantdata.widb.day import collect_raw_day, enrich_day, ExpiryFilter, RawDayFilter
    from quantdata.widb.arbfree import mk_arb_free_day, MkArbFree

    raw_day_filter = RawDayFilter()
    expiry_filter = ExpiryFilter()
    mk_arb_free = MkArbFree()
    verbose = Context.all

    raw_day = collect_raw_day(ticker, d, raw_day_filter=raw_day_filter, verbose=verbose, raise_if_bad=False)
    print(raw_day)

    enr_day = enrich_day(ticker, d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=verbose, raise_if_bad=False)
    print(enr_day)

    af_day = mk_arb_free_day(ticker, d, mk_arb_free=mk_arb_free, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=verbose, raise_if_bad=False)
    print(af_day)


def main():
    tickers = ["SPX", "AAPL"]
    dates = ["2005-11-18", "2024-03-03", "2025-01-02", "2024-12-24", "2010-07-05", "2011-07-06"]
    dates = [next_business_day(d) for d in dates]

    for ticker in tickers:
        for d in dates:
            print(f"Generating data for {ticker} {d}")
            generate_day(ticker, d)


if __name__ == "__main__":
    main()