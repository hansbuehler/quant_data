"""
Shared constants for intraday implied-vol Kalman filters.
"""

TRADING_SECONDS_PER_DAY = 6.5 * 3600.0
MIN_TTE_Y = 1e-10
MIN_SPOT = 1e-8
MIN_VOL = 1e-3
MAX_VOL = 5.0

PRIMARY_TRADE_FIELDS = ["trade_t_d", "trade_price", "trade_size"]

# Treat a quoted half-spread as an approximate 95% observation band.
SINGLE_OPTION_BID_ASK_95_Z = 1.959963984540054
