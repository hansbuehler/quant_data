"""
openqai.data.massive.trade_conds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OPRA (options) and spot (stocks) trade condition codes and bitmask flags.

Sources:
- https://massive.com/glossary/trade-conditions
- https://massive.com/glossary/conditions-indicators

Design:
- Keep the vendor condition IDs as IntEnum values (as returned in `conditions`).
- Provide IntFlag bitmasks for fast filtering/joins (int32).
- pack_conditions raises on unknown codes so vendor schema changes are explicit.

Use the function :func:`filter_valid_trades` in :mod:`openqai.data.massive.processing` to filter out bad trades based 
:data:`NORMAL_TRADE_EXCLUDE_FLAGS`.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Iterable

class ConditionCode(IntEnum):
    """
    Massive condition IDs (stocks 0..60-ish, options 201..248).

    See https://massive.com/glossary/trade-conditions and https://massive.com/glossary/conditions-indicators
    """

    # ---------------------------------------------------------------------
    # Stocks trade conditions (per Massive Conditions & Indicators)
    # ---------------------------------------------------------------------
    REGULAR_SALE = 0
    ACQUISITION = 1
    AVERAGE_PRICE_TRADE = 2
    STOCK_AUTOMATIC_EXECUTION = 3
    BUNCHED_TRADE = 4
    BUNCHED_SOLD_TRADE = 5
    CAP_ELECTION = 6
    CASH_SALE = 7
    CLOSING_PRINTS = 8
    CROSS_TRADE = 9
    DERIVATIVELY_PRICED = 10
    DISTRIBUTION = 11
    FORM_T_EXTENDED_HOURS = 12
    EXTENDED_HOURS_OOS = 13
    STOCK_INTERMARKET_SWEEP = 14
    MARKET_CENTER_OFFICIAL_CLOSE = 15
    MARKET_CENTER_OFFICIAL_OPEN = 16
    MARKET_CENTER_OPENING_TRADE = 17
    MARKET_CENTER_REOPENING_TRADE = 18
    MARKET_CENTER_CLOSING_TRADE = 19
    NEXT_DAY = 20
    PRICE_VARIATION_TRADE = 21
    PRIOR_REFERENCE_PRICE = 22
    RULE_155_AMEX = 23
    RULE_127_NYSE = 24
    OPENING_PRINTS = 25
    OPENED = 26
    STOPPED_STOCK_REGULAR = 27
    REOPENING_PRINTS = 28
    SELLER = 29
    SOLD_LAST = 30
    SOLD_OUT = 32
    SOLD_OUT_OF_SEQUENCE = 33
    SPLIT_TRADE = 34
    STOCK_OPTION_TRADE = 35
    YELLOW_FLAG_REGULAR = 36
    ODD_LOT_TRADE = 37
    CORRECTED_CONSOLIDATED_CLOSE = 38
    UNKNOWN = 39
    HELD_TRADES = 40
    TRADE_THRU_EXEMPT = 41
    NON_ELIGIBLE = 42
    NON_ELIGIBLE_EXTENDED = 43
    CANCELLED = 44
    RECOVERY = 45
    CORRECTION = 46
    AS_OF = 47
    AS_OF_CORRECTION = 48
    AS_OF_CANCEL = 49
    OOB = 50
    SUMMARY = 51
    CONTINGENT_TRADE = 52
    QUALIFIED_CONTINGENT_TRADE = 53
    ERRORED = 54
    OPENING_REOPENING_TRADE_DETAIL = 55
    PLACEHOLDER_E_TBD = 56
    PLACEHOLDER_611_EXEMPT = 59
    SSR_IN_EFFECT = 60

    # ---------------------------------------------------------------------
    # Options / OPRA-like trade conditions (Massive numeric IDs)
    # ---------------------------------------------------------------------
    CANCEL = 201
    LATE_AND_OOS = 202
    LAST_AND_CANCELED = 203
    LATE = 204
    OPENING_AND_CANCELED = 205
    OPENING_LATE_AND_OOS = 206
    ONLY_TRADE_AND_CANCELED = 207
    OPENING_LATE = 208

    AUTOMATIC_EXECUTION = 209
    REOPENING_TRADE = 210
    INTERMARKET_SWEEP = 219  # (options ISO)

    SINGLE_LEG_AUCTION_NON_ISO = 227
    SINGLE_LEG_AUCTION_ISO = 228
    SINGLE_LEG_CROSS_NON_ISO = 229
    SINGLE_LEG_CROSS_ISO = 230
    SINGLE_LEG_FLOOR_TRADE = 231

    MULTI_LEG_AUTO = 232
    MULTI_LEG_AUCTION = 233
    MULTI_LEG_CROSS = 234
    MULTI_LEG_FLOOR_TRADE = 235
    MULTI_LEG_AUTO_AGAINST_SINGLE_LEG = 236

    STOCK_OPTIONS_AUCTION = 237
    MULTI_LEG_AUCTION_AGAINST_SINGLE_LEG = 238
    MULTI_LEG_FLOOR_TRADE_AGAINST_SINGLE_LEG = 239
    STOCK_OPTIONS_AUTO = 240
    STOCK_OPTIONS_CROSS = 241
    STOCK_OPTIONS_FLOOR_TRADE = 242
    STOCK_OPTIONS_AUTO_AGAINST_SINGLE_LEG = 243
    STOCK_OPTIONS_AUCTION_AGAINST_SINGLE_LEG = 244
    STOCK_OPTIONS_FLOOR_TRADE_AGAINST_SINGLE_LEG = 245

    MULTI_LEG_FLOOR_PROPRIETARY_3PLUS = 246
    MULTILATERAL_COMPRESSION_PROPRIETARY = 247
    EXTENDED_HOURS_TRADE = 248


class ConditionFlag(IntFlag):
    """Semantic bitmask (int32)."""
    NONE = 0

    # reporting / data-quality
    LATE = 1 << 0
    OUT_OF_SEQUENCE = 1 << 1
    CANCELLED = 1 << 2
    CORRECTION = 1 << 3
    AS_OF = 1 << 4
    ERRORED = 1 << 5

    # execution style / structure
    ELECTRONIC = 1 << 6
    FLOOR = 1 << 7
    ISO = 1 << 8
    AUCTION = 1 << 9
    CROSS = 1 << 10
    COMPLEX = 1 << 11
    STOCK_OPTION = 1 << 12

    # session / special
    EXTENDED_HOURS = 1 << 13
    COMPRESSION = 1 << 14
    PROPRIETARY_PRODUCT = 1 << 15

    # stock-specific structure
    ODD_LOT = 1 << 16

    # stock “print types”
    OPENING_PRINT = 1 << 17
    CLOSING_PRINT = 1 << 18
    OFFICIAL_OPEN = 1 << 19
    OFFICIAL_CLOSE = 1 << 20

    # misc stock flags
    CONTINGENT = 1 << 21
    QCT = 1 << 22
    TRADE_THRU_EXEMPT = 1 << 23
    SSR = 1 << 24

CODE_TO_FLAG: dict[ConditionCode, ConditionFlag] = {
    # ---------------- options: cancel/late/sequence ----------------
    ConditionCode.CANCEL: ConditionFlag.CANCELLED,
    ConditionCode.LATE_AND_OOS: ConditionFlag.LATE | ConditionFlag.OUT_OF_SEQUENCE,
    ConditionCode.LAST_AND_CANCELED: ConditionFlag.CANCELLED,
    ConditionCode.LATE: ConditionFlag.LATE,
    ConditionCode.OPENING_AND_CANCELED: ConditionFlag.CANCELLED,
    ConditionCode.OPENING_LATE_AND_OOS: ConditionFlag.LATE | ConditionFlag.OUT_OF_SEQUENCE,
    ConditionCode.ONLY_TRADE_AND_CANCELED: ConditionFlag.CANCELLED,
    ConditionCode.OPENING_LATE: ConditionFlag.LATE,

    # options: execution/informational
    ConditionCode.AUTOMATIC_EXECUTION: ConditionFlag.ELECTRONIC,
    ConditionCode.REOPENING_TRADE: ConditionFlag.NONE,
    ConditionCode.INTERMARKET_SWEEP: ConditionFlag.ISO,

    # options: single-leg mechanisms
    ConditionCode.SINGLE_LEG_AUCTION_NON_ISO: ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC,
    ConditionCode.SINGLE_LEG_AUCTION_ISO: ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC | ConditionFlag.ISO,
    ConditionCode.SINGLE_LEG_CROSS_NON_ISO: ConditionFlag.CROSS | ConditionFlag.ELECTRONIC,
    ConditionCode.SINGLE_LEG_CROSS_ISO: ConditionFlag.CROSS | ConditionFlag.ELECTRONIC | ConditionFlag.ISO,
    ConditionCode.SINGLE_LEG_FLOOR_TRADE: ConditionFlag.FLOOR,

    # options: complex / multi-leg
    ConditionCode.MULTI_LEG_AUTO: ConditionFlag.COMPLEX | ConditionFlag.ELECTRONIC,
    ConditionCode.MULTI_LEG_AUCTION: ConditionFlag.COMPLEX | ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC,
    ConditionCode.MULTI_LEG_CROSS: ConditionFlag.COMPLEX | ConditionFlag.CROSS | ConditionFlag.ELECTRONIC,
    ConditionCode.MULTI_LEG_FLOOR_TRADE: ConditionFlag.COMPLEX | ConditionFlag.FLOOR,
    ConditionCode.MULTI_LEG_AUTO_AGAINST_SINGLE_LEG: ConditionFlag.COMPLEX | ConditionFlag.ELECTRONIC,

    # options: stock/options combo
    ConditionCode.STOCK_OPTIONS_AUCTION: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC,
    ConditionCode.MULTI_LEG_AUCTION_AGAINST_SINGLE_LEG: ConditionFlag.COMPLEX | ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC,
    ConditionCode.MULTI_LEG_FLOOR_TRADE_AGAINST_SINGLE_LEG: ConditionFlag.COMPLEX | ConditionFlag.FLOOR,
    ConditionCode.STOCK_OPTIONS_AUTO: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.ELECTRONIC,
    ConditionCode.STOCK_OPTIONS_CROSS: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.CROSS | ConditionFlag.ELECTRONIC,
    ConditionCode.STOCK_OPTIONS_FLOOR_TRADE: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.FLOOR,
    ConditionCode.STOCK_OPTIONS_AUTO_AGAINST_SINGLE_LEG: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.ELECTRONIC,
    ConditionCode.STOCK_OPTIONS_AUCTION_AGAINST_SINGLE_LEG: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.AUCTION | ConditionFlag.ELECTRONIC,
    ConditionCode.STOCK_OPTIONS_FLOOR_TRADE_AGAINST_SINGLE_LEG: ConditionFlag.STOCK_OPTION | ConditionFlag.COMPLEX | ConditionFlag.FLOOR,

    # options: proprietary / compression / extended hours
    ConditionCode.MULTI_LEG_FLOOR_PROPRIETARY_3PLUS: ConditionFlag.PROPRIETARY_PRODUCT | ConditionFlag.COMPLEX | ConditionFlag.FLOOR,
    ConditionCode.MULTILATERAL_COMPRESSION_PROPRIETARY: ConditionFlag.PROPRIETARY_PRODUCT | ConditionFlag.COMPRESSION,
    ConditionCode.EXTENDED_HOURS_TRADE: ConditionFlag.EXTENDED_HOURS,

    # ---------------- stocks: session/sequence ----------------
    ConditionCode.FORM_T_EXTENDED_HOURS: ConditionFlag.EXTENDED_HOURS,
    ConditionCode.EXTENDED_HOURS_OOS: ConditionFlag.EXTENDED_HOURS | ConditionFlag.OUT_OF_SEQUENCE,
    ConditionCode.SOLD_OUT_OF_SEQUENCE: ConditionFlag.OUT_OF_SEQUENCE,
    ConditionCode.SOLD_OUT: ConditionFlag.NONE,  # semantic “sold out” marker; keep as NONE unless you need it

    # stocks: mechanisms
    ConditionCode.CROSS_TRADE: ConditionFlag.CROSS,
    ConditionCode.STOCK_INTERMARKET_SWEEP: ConditionFlag.ISO,
    ConditionCode.ODD_LOT_TRADE: ConditionFlag.ODD_LOT,
    ConditionCode.STOCK_OPTION_TRADE: ConditionFlag.STOCK_OPTION,

    # stocks: opening/closing prints + official open/close
    ConditionCode.OPENING_PRINTS: ConditionFlag.OPENING_PRINT,
    ConditionCode.MARKET_CENTER_OPENING_TRADE: ConditionFlag.OPENING_PRINT,
    ConditionCode.MARKET_CENTER_REOPENING_TRADE: ConditionFlag.OPENING_PRINT,
    ConditionCode.CLOSING_PRINTS: ConditionFlag.CLOSING_PRINT,
    ConditionCode.MARKET_CENTER_CLOSING_TRADE: ConditionFlag.CLOSING_PRINT,
    ConditionCode.MARKET_CENTER_OFFICIAL_OPEN: ConditionFlag.OFFICIAL_OPEN,
    ConditionCode.MARKET_CENTER_OFFICIAL_CLOSE: ConditionFlag.OFFICIAL_CLOSE,
    ConditionCode.CORRECTED_CONSOLIDATED_CLOSE: ConditionFlag.OFFICIAL_CLOSE | ConditionFlag.CORRECTION,

    # stocks: corrections / as-of / errors
    ConditionCode.CORRECTION: ConditionFlag.CORRECTION,
    ConditionCode.AS_OF: ConditionFlag.AS_OF,
    ConditionCode.AS_OF_CORRECTION: ConditionFlag.AS_OF | ConditionFlag.CORRECTION,
    ConditionCode.AS_OF_CANCEL: ConditionFlag.AS_OF | ConditionFlag.CANCELLED,
    ConditionCode.CANCELLED: ConditionFlag.CANCELLED,
    ConditionCode.ERRORED: ConditionFlag.ERRORED,

    # stocks: contingent / QCT / exemptions / SSR
    ConditionCode.CONTINGENT_TRADE: ConditionFlag.CONTINGENT,
    ConditionCode.QUALIFIED_CONTINGENT_TRADE: ConditionFlag.CONTINGENT | ConditionFlag.QCT,
    ConditionCode.TRADE_THRU_EXEMPT: ConditionFlag.TRADE_THRU_EXEMPT,
    ConditionCode.SSR_IN_EFFECT: ConditionFlag.SSR,
}
""" Map from detailed condition codes to unified flags """

BAD_TRADE_CONDITIONS = (
    ConditionFlag.CANCELLED
    | ConditionFlag.LATE
    | ConditionFlag.OUT_OF_SEQUENCE
)
""" Bad trade conditions: cancelled, late reporting, or out-of-sequence. These are the most common data quality issues and are often filtered out in analyses. """

MICROSTRUCTURE_HARD_EXCLUDE = (
    BAD_TRADE_CONDITIONS
    | ConditionFlag.COMPRESSION
    | ConditionFlag.CORRECTION
    | ConditionFlag.AS_OF
    | ConditionFlag.ERRORED
)
""" 
Microstructure conditions to exclude for clean trade/quote analysis.
These include bad trade conditions compression, corrections, as-of trades, and errored trades.

We do *not* exclude  ``ConditionFlag.EXTENDED_HOURS`` here as extended hours trades can be exluded
easily later based on their time stamp, see :func:`filter_valid_trades` in :mod:`openqai.data.massive.processing`.
"""

NORMAL_TRADE_EXCLUDE_FLAGS = (
    MICROSTRUCTURE_HARD_EXCLUDE  # data quality issues (late, cancelled, corrections, etc.)
    | ConditionFlag.CROSS  # cross trades are reported separately, not normal market trades
    | ConditionFlag.OPENING_PRINT  # opening prints are special print conditions, not regular trades
    | ConditionFlag.CLOSING_PRINT  # closing prints are special print conditions, not regular trades
    | ConditionFlag.OFFICIAL_OPEN  # official open prices are reference/index, not actual trades
    | ConditionFlag.OFFICIAL_CLOSE  # official close prices are reference/index, not actual trades
    | ConditionFlag.CONTINGENT  # contingent trades are conditional, not unconditional normal trades
    | ConditionFlag.QCT  # qualified contingent trades are special conditions
    | ConditionFlag.TRADE_THRU_EXEMPT  # exemptions indicate special trade handling
)
"""
Flags for normal trade filtering: these conditions are commonly excluded when filtering for regular market trades. This includes all microstructure hard excludes plus cross trades, opening/closing prints, official open/closes, and contingent trades. 

We do not exclude ``ConditionFlag.ODD_LOT`` here.
"""

def pack_conditions(codes: Iterable[int]) -> int:
    """
    Pack Massive/Polygon `conditions` (iterable of ints) into an int32 semantic mask.
    Unknown codes raise ValueError.
    """
    flags = ConditionFlag.NONE
    for c in codes:
        try:
            cc = ConditionCode(int(c))
        except ValueError:
            raise ValueError(f"Unknown trade condition code: {c}") from None
        flags |= CODE_TO_FLAG.get(cc, ConditionFlag.NONE)
    return int(flags)
