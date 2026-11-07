"""Private daily spots and carry from the licensed qdata source cache."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedge2.source import SourceArchive


def daily_observation(licensed_fit_directory: Path, day: date) -> dict[str, object]:
    """Use the same PM-only parity/financing curves as hedge cash accounting."""
    source_path = licensed_fit_directory / "SPX" / f"{day.isoformat()}.pdq"
    curve = licensed_curve(licensed_fit_directory, day)
    frame = pd.read_parquet(source_path, columns=["spot_prev"])
    parity_1d, forward_1d, discount_1d = curve.at(1.)
    return {
        "date": day, "spot": curve.spot, "spot_prev_source": float(frame["spot_prev"].iloc[0]),
        "forward_parity_1d": float(parity_1d),
        "forward_reference_1d": float(forward_1d), "discount_1d": float(discount_1d),
        "first_source_business_dte": float(curve.business_dtes[0]),
        "source_cache_file": str(source_path),
    }


def prepare_observations(archive: SourceArchive, output: Path, *, licensed_fit_directory: Path,
                         start: date | None = None, end: date | None = None) -> pd.DataFrame:
    if start is not None and end is not None and end < start:
        raise ValueError("The observation date range must be ordered")
    days = tuple(day for day in archive.dates() if (start is None or day >= start) and (end is None or day <= end))
    if not days:
        raise ValueError("No source observations are available in the requested range")
    sessions = get_business_days(days[0], days[-1])
    session_index = {day: index for index, day in enumerate(sessions)}
    rows = []
    for index, day in enumerate(days):
        row = daily_observation(licensed_fit_directory, day)
        row["session_index"] = session_index[day]
        rows.append(row)
        if (index + 1) % 250 == 0:
            print(f"observations {index + 1}/{len(days)}", flush=True)
    frame = pd.DataFrame(rows)
    exact = np.diff(frame["session_index"].to_numpy(), prepend=-2) == 1
    returns = np.full(len(frame), np.nan)
    returns[1:] = frame["spot"].to_numpy()[1:] / frame["forward_reference_1d"].to_numpy()[:-1] - 1.
    returns[~exact] = np.nan
    frame["normalized_spot_return_into_date"] = returns
    frame["exact_previous_session"] = exact
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--licensed-fit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    frame = prepare_observations(SourceArchive(root=args.source_root), args.output,
                                 licensed_fit_directory=args.licensed_fit_directory, start=args.start, end=args.end)
    print(f"Saved {len(frame)} observations; exact transitions={frame['exact_previous_session'].sum()}", flush=True)


if __name__ == "__main__":
    main()
