from __future__ import annotations

import json
from pathlib import Path

import torch

from quantdata.dysanos.flex.plot_crossfit_q2_convergence import (
    load_direction,
    plot_histories,
)


def _checkpoint(path: Path, records: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "dysanos-convex-q-conditional-oce-v1",
            "history": records,
        },
        path,
    )


def _direction(root: Path, name: str, *, select_lbfgs: bool) -> Path:
    directory = root / name
    _checkpoint(
        directory / "adam" / "conditional-oce.pt",
        [
            {
                "epoch": 1,
                "full_population_oce": 0.1,
                "gradient_rms": 0.02,
            },
            {
                "epoch": 2,
                "full_population_oce": 0.2,
                "gradient_rms": 0.01,
                "validation_oce": 0.15,
            },
        ],
    )
    _checkpoint(
        directory / "lbfgs" / "conditional-oce.pt",
        [
            {
                "epoch": 1,
                "full_population_oce": 0.25,
                "gradient_rms": 0.005,
                "validation_oce": 0.17,
            }
        ],
    )
    selected_path = (
        directory / "lbfgs" / "conditional-oce-best-validation.pt"
        if select_lbfgs
        else directory / "adam" / "conditional-oce-best-validation.pt"
    )
    (directory / "selected").mkdir(parents=True, exist_ok=True)
    (directory / "selected" / "selection.json").write_text(
        json.dumps(
            {
                "selected": {
                    "path": str(selected_path),
                    "epoch": 1 if select_lbfgs else 2,
                }
            }
        )
    )
    return directory


def test_crossfit_history_concatenates_stages_and_marks_selection(
    tmp_path: Path,
) -> None:
    first = load_direction(
        _direction(tmp_path, "A-to-B", select_lbfgs=True),
        label="A to B",
    )
    second = load_direction(
        _direction(tmp_path, "B-to-A", select_lbfgs=False),
        label="B to A",
    )

    assert first.epochs.tolist() == [1, 2, 3]
    assert first.validation_epochs.tolist() == [2, 3]
    assert first.adam_epochs == 2
    assert first.selected_epoch == 3
    assert second.selected_epoch == 2

    output = tmp_path / "convergence.png"
    plot_histories((first, second), output)
    assert output.stat().st_size > 0
