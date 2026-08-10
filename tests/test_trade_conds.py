from __future__ import annotations

import pytest

from quantdata.massive.trade_conds import ConditionCode, ConditionFlag, pack_conditions


def test_pack_conditions_combines_known_flags() -> None:
    packed = pack_conditions([ConditionCode.LATE, ConditionCode.CROSS_TRADE, 248])

    assert packed == int(ConditionFlag.LATE | ConditionFlag.CROSS | ConditionFlag.EXTENDED_HOURS)


def test_pack_conditions_returns_zero_for_unmapped_known_codes() -> None:
    assert pack_conditions([ConditionCode.REGULAR_SALE, ConditionCode.REOPENING_TRADE]) == 0


def test_pack_conditions_raises_for_unknown_codes() -> None:
    with pytest.raises(ValueError, match="Unknown trade condition code: 999"):
        pack_conditions([999])