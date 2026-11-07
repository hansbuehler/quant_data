"""Decode floating log-DLV states without changing their reference geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.pricing import DlvPricer


@dataclass(frozen=True, kw_only=True)
class DecodedState:
    pricer: DlvPricer
    lower_clips: int
    upper_clips: int


@dataclass(frozen=True, kw_only=True)
class DailyState:
    day: date
    native: np.ndarray
    reference_support: np.ndarray
    config: DlvConfig

    @classmethod
    def load(cls, directory: Path, day: date) -> DailyState:
        configuration = json.loads((directory / "config.json").read_text())
        names = {field.name for field in fields(DlvConfig)}
        config = DlvConfig(**{name: value for name, value in configuration.items() if name in names})
        with np.load(directory / "days" / f"{day.isoformat()}.npz", allow_pickle=False) as saved:
            if not bool(saved["reconstruction_passed"]) or str(saved["date"].item()) != day.isoformat():
                raise ValueError(f"Daily state is not validated for {day}")
            return cls(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)

    def decode(self, native: np.ndarray | None = None) -> DecodedState:
        """Rebuild the daily floating grid; report every logarithmic guard."""
        values = self.native if native is None else np.asarray(native, dtype=float)
        shape = (len(self.config.business_dtes), self.config.num_inner_strikes + 1)
        if values.shape != (np.prod(shape),) or not np.isfinite(values).all():
            raise ValueError("Native log-DLV state is non-finite or has incompatible geometry")
        packed = values.reshape(shape)
        lower = np.r_[np.full(self.config.num_inner_strikes, 2. * np.log(self.config.min_local_vol)),
                      np.log(self.config.min_local_vol)]
        upper = np.r_[np.full(self.config.num_inner_strikes, 2. * np.log(self.config.max_local_vol)),
                      np.log(self.config.max_local_vol)]
        guarded = np.clip(packed, lower, upper)
        forward_vols = np.exp(guarded[:, -1])
        times = self.config.expiries_y
        largest_log_strike = (max(abs(value) for value in self.config.nms_range)
                              * np.sum(forward_vols * np.sqrt(np.diff(times, prepend=0.))))
        if largest_log_strike > 30.:
            raise ValueError("The shocked forward-vol geometry exceeds the numerical log-strike bound of 30")
        inner = shared_inner_strikes(forward_vols, config=self.config)
        lower_ratio = self.reference_support[0] / self.reference_support[1]
        upper_ratio = self.reference_support[-1] / self.reference_support[-2]
        support = np.r_[lower_ratio * inner[0], inner, upper_ratio * inner[-1]]
        pricer = DlvPricer(support=support, expiries_y=times, dlvsqs=np.exp(guarded[:, :-1]))
        return DecodedState(pricer=pricer, lower_clips=int(np.sum(packed < lower)),
                            upper_clips=int(np.sum(packed > upper)))
