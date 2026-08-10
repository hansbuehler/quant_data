"""
WIDB Testing scope for locally stored sample testing data
"""

from quantdata.biz_calendar import next_business_day

TICKERS = ["SPX", "AAPL"]
DATES = [next_business_day(d) for d in ["2005-11-18", "2024-03-03", "2025-01-02", "2024-12-24", "2010-07-05", "2011-07-06"]]