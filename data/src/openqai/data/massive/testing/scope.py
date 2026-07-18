"""
WIDB Testing scope for locally stored sample testing data
"""

from openqai.core.biz_calendar import next_business_day

TICKERS = ["FICO", "TDG"]
DATES = [next_business_day(d) for d in ["2025-11-18", "2024-03-03", "2025-01-02", "2024-12-24", "2023-07-05", "2021-07-06"]]