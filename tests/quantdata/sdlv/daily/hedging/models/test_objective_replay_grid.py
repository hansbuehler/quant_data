from __future__ import annotations

import json
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.runners import (
    run_objective_replay_grid as grid,
)


def test_grid_records_a_failed_cell_without_skipping_later_cells(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference" / "run.json"
    records = reference.parent / "records"
    books = tmp_path / "books"
    records.mkdir(parents=True)
    books.mkdir()
    for index in range(44):
        label = f"2024-{index // 22 + 4:02d}-{index % 22 + 1:02d}"
        (records / f"{label}.json").touch()
        (books / f"{label}.npz").touch()
    reference.write_text("{}", encoding="utf-8")
    visited: list[str] = []

    def fake_replay(**arguments: object) -> None:
        destination = Path(arguments["output_directory"])
        visited.append(destination.name)
        if destination.name == "mean-variance-lambdainf":
            raise RuntimeError("position cap hit")
        output_records = destination / "records"
        output_records.mkdir(parents=True)
        for path in records.glob("2024-*.json"):
            (output_records / path.name).touch()

    monkeypatch.setattr(grid, "replay_return_books", fake_replay)
    grid.run_grid(
        model="diffusion-ft",
        book_directory=books,
        reference_run=reference,
        output_root=tmp_path / "output",
    )

    status = json.loads((tmp_path / "output" / "diffusion-ft" / "GRID_STATUS.json").read_text())
    assert len(visited) == 14
    assert status["cells"]["mean-variance-lambdainf"]["status"] == "failed"
    assert status["cells"]["mean-variance-lambdainf"]["error"] == "position cap hit"
    assert status["cells"]["cvar-lambdainf"]["status"] == "complete"


def test_grid_keeps_complete_cap_limited_cell_separate_from_certified(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference" / "run.json"
    records = reference.parent / "records"
    books = tmp_path / "books"
    records.mkdir(parents=True)
    books.mkdir()
    for index in range(44):
        label = f"2024-{index // 22 + 4:02d}-{index % 22 + 1:02d}"
        (records / f"{label}.json").touch()
        (books / f"{label}.npz").touch()
    reference.write_text("{}", encoding="utf-8")

    def fake_replay(**arguments: object) -> None:
        assert arguments["continue_cap_limited"] is True
        destination = Path(arguments["output_directory"])
        output_records = destination / "records"
        output_records.mkdir(parents=True)
        for path in records.glob("2024-*.json"):
            if destination.name == "mean-variance-lambdainf" and path.stem == "2024-04-01":
                failures = destination / "failures"
                failures.mkdir()
                (failures / path.name).touch()
            else:
                (output_records / path.name).touch()

    monkeypatch.setattr(grid, "replay_return_books", fake_replay)
    grid.run_grid(
        model="gromm-P",
        book_directory=books,
        reference_run=reference,
        output_root=tmp_path / "output",
    )

    status = json.loads((tmp_path / "output" / "gromm-P" / "GRID_STATUS.json").read_text())
    assert status["cells"]["mean-variance-lambdainf"]["status"] == "cap-limited"
    assert status["cells"]["mean-variance-lambdainf"]["cap_limited_dates"] == 1
