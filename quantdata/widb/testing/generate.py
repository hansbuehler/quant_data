"""
Generate WIDB test data
"""

import os
os.environ["QUANTDATA_TEST_MODE"] = "generate_tests" # do this before importing anything else

from quantdata import env# triggers construction of caches.
from cdxcore.verbose import Context
from quantdata.widb.testing.scope import TICKERS, DATES

def generate_day(ticker, d):
    from quantdata.widb.day import collect_raw_day, enrich_day, ExpiryFilter, RawDayFilter
    from quantdata.widb.arbfree import mk_arb_free_day, MkArbFree

    raw_day_filter = RawDayFilter()
    expiry_filter = ExpiryFilter()
    mk_arb_free = MkArbFree()
    verbose = Context.all

    raw_day = collect_raw_day(ticker, d, raw_day_filter=raw_day_filter, verbose=verbose, raise_if_bad_day=False)
    print(raw_day)

    enr_day = enrich_day(ticker, d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=verbose, raise_if_bad_day=False)
    print(enr_day)

    af_day = mk_arb_free_day(ticker, d, mk_arb_free=mk_arb_free, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=verbose, raise_if_bad_day=False)
    print(af_day)

def main():

    for ticker in TICKERS:
        for d in DATES:
            print(f"Generating data for {ticker} {d}")
            generate_day(ticker, d)


if __name__ == "__main__":
    main()