from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import weekly_production_matrix
from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    GeneratedProductionInputs,
    ProductionReadiness,
    ProductionResource,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_production_matrix import (
    WeeklyProductionMatrixConfig,
    build_weekly_production_matrix,
)
from quantdata.torch import torch


def _weekly_evidence(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "notionals.json"
    target.write_text("{}\n", encoding="utf-8")
    from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
        file_sha256,
    )

    preflight = tmp_path / "weekly-preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-input-preflight-v1",
                "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
                "target_notional_artifact": str(target),
                "target_notional_artifact_sha256": file_sha256(target),
                "target_count": 13,
                "episode_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return dataset, preflight


def test_weekly_matrix_has_complete_unique_case_partition(tmp_path: Path) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    cases = build_weekly_production_matrix(
        dataset=dataset,
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        weekly_input_preflight=preflight,
    )

    assert len(cases) == 94
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.model == "sanos-lv" for case in cases) == 2
    assert sum(case.model == "historical-sanos" for case in cases) == 26
    assert sum(case.model == "gromm" for case in cases) == 52
    assert sum(case.model == "gromm-q-dlv-vega" for case in cases) == 2
    assert sum(case.resource is ProductionResource.CPU for case in cases) == 28
    assert all(
        case.readiness is ProductionReadiness.READY
        for case in cases
        if case.resource is ProductionResource.CPU
    )
    assert all(
        case.readiness is ProductionReadiness.BLOCKED
        for case in cases
        if case.model == "gromm"
    )


def test_weekly_native_commands_pin_protocol_and_authenticated_sizing(
    tmp_path: Path,
) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    config = WeeklyProductionMatrixConfig()
    cases = build_weekly_production_matrix(
        dataset=dataset,
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        weekly_input_preflight=preflight,
    )

    for case in (value for value in cases if value.resource is ProductionResource.CPU):
        assert case.command is not None
        assert case.command[:3] == (
            "python",
            "-m",
            "quantdata.sdlv.daily.front_spot.hedging.runners.run_weekly",
        )
        assert case.command[case.command.index("--schedule") + 1] in {
            "frozen-weekly",
            "restarted-weekly",
        }
        assert case.command[case.command.index("--observation-count") + 1] == str(
            config.observation_count
        )
        target_sha256 = case.command[
            case.command.index("--target-notional-artifact-sha256") + 1
        ]
        assert len(target_sha256) == 64
        target_path = case.command[
            case.command.index("--target-notional-artifact") + 1
        ]
        assert Path(target_path).exists()


def test_weekly_generated_rolling_cases_require_selected_artifacts(
    tmp_path: Path,
) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    missing = {
        case.case_id: case
        for case in build_weekly_production_matrix(
            dataset=dataset,
            output_root=tmp_path / "outputs",
            checkpoint_root=tmp_path / "checkpoints",
            weekly_input_preflight=preflight,
        )
    }
    assert (
        missing["frozen-weekly-volgan-rolling-trained"].readiness
        is ProductionReadiness.BLOCKED
    )

    generated = _generated_inputs(tmp_path, dataset=dataset)
    supplied = {
        case.case_id: case
        for case in build_weekly_production_matrix(
            dataset=dataset,
            output_root=tmp_path / "outputs",
            checkpoint_root=tmp_path / "checkpoints",
            weekly_input_preflight=preflight,
            generated=generated,
        )
    }
    for schedule in ("frozen-weekly", "restarted-weekly"):
        for suffix, epochs in (
            ("volgan-rolling-trained", "7"),
            ("diffusion-ad-seq-vol-rolling-trained", "11"),
            ("diffusion-ad-seq-vol-ft-rolling-trained", "5"),
        ):
            case = supplied[f"{schedule}-{suffix}"]
            assert case.readiness is ProductionReadiness.WAITING_GATE
            assert case.command is not None
            assert case.command[case.command.index("--epochs") + 1] == epochs
            assert case.command[case.command.index("--objective") + 1] == "mean-variance"
            assert case.command[case.command.index("--risk-parameter") + 1] == "1"
            assert "--warm-start-checkpoint-sha256" in case.command

    generated_cases = tuple(
        case
        for case in supplied.values()
        if case.model
        in {"volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"}
        and case.command is not None
    )
    assert len(generated_cases) == 12
    for case in generated_cases:
        assert case.command[case.command.index("--objective") + 1] == "mean-variance"
        assert case.command[case.command.index("--risk-parameter") + 1] == "1"


@pytest.mark.parametrize(
    ("physical_accepted_count", "expected_p_readiness"),
    (
        (2, ProductionReadiness.BLOCKED),
        (3, ProductionReadiness.READY),
    ),
)
def test_weekly_gromm_q_and_conventional_do_not_depend_on_physical_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_accepted_count: int,
    expected_p_readiness: ProductionReadiness,
) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    architecture = tmp_path / "architecture.json"
    architecture.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "range.json"
    summary.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    config = WeeklyProductionMatrixConfig()
    monkeypatch.setattr(
        weekly_production_matrix,
        "load_gromm_architecture_selection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        weekly_production_matrix,
        "load_completed_rolling_gromm2_range",
        lambda *_args, **_kwargs: SimpleNamespace(
            evaluation_start=config.evaluation_start,
            evaluation_end=config.evaluation_end,
            eligible_origin_count=3,
            physical_accepted_count=physical_accepted_count,
        ),
    )
    cases = build_weekly_production_matrix(
        dataset=dataset,
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        weekly_input_preflight=preflight,
        gromm=weekly_production_matrix.GrommProductionInputs(
            market_artifact_root=artifact_root,
            architecture_selection=architecture,
            artifact_range_summary=summary,
        ),
    )
    conventional = tuple(
        case for case in cases if case.model == "gromm-q-dlv-vega"
    )
    deep_q = tuple(
        case for case in cases if case.model == "gromm" and case.measure == "Q"
    )
    deep_p = tuple(
        case for case in cases if case.model == "gromm" and case.measure == "P"
    )
    assert len(conventional) == 2
    assert all(case.readiness is ProductionReadiness.READY for case in conventional)
    assert all(case.readiness is ProductionReadiness.READY for case in deep_q)
    assert all(case.readiness is expected_p_readiness for case in deep_p)


def test_weekly_generated_matrix_rejects_initializer_tampering(
    tmp_path: Path,
) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    generated = _generated_inputs(tmp_path, dataset=dataset)
    (tmp_path / "diffusion.pt").write_bytes(b"tampered")

    cases = {
        case.case_id: case
        for case in build_weekly_production_matrix(
            dataset=dataset,
            output_root=tmp_path / "outputs",
            checkpoint_root=tmp_path / "checkpoints",
            weekly_input_preflight=preflight,
            generated=generated,
        )
    }

    assert (
        cases["frozen-weekly-diffusion-ad-seq-vol-rolling-trained"].readiness
        is ProductionReadiness.BLOCKED
    )
    assert "hash mismatch" in cases[
        "frozen-weekly-diffusion-ad-seq-vol-rolling-trained"
    ].blocked_reason


def test_weekly_matrix_blocks_every_case_without_preflight(tmp_path: Path) -> None:
    cases = build_weekly_production_matrix(
        dataset=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        checkpoint_root=tmp_path / "checkpoints",
        weekly_input_preflight=None,
    )

    assert all(case.readiness is ProductionReadiness.BLOCKED for case in cases)
    assert all("weekly input preflight" in case.blocked_reason for case in cases)


def test_weekly_production_matrix_cli_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, preflight = _weekly_evidence(tmp_path)
    output = tmp_path / "weekly-matrix.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_production_matrix",
            "--dataset",
            str(dataset),
            "--output-root",
            str(tmp_path / "outputs"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--weekly-input-preflight",
            str(preflight),
            "--output",
            str(output),
        ],
    )

    weekly_production_matrix.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "hedging-weekly-production-matrix-v1"
    assert payload["case_count"] == 94
    assert payload["ready_count"] == 28
    assert payload["config"]["evaluation_start"] == "2023-01-01"
    assert payload["config"]["evaluation_end"] == "2025-12-31"


def _generated_inputs(
    tmp_path: Path,
    *,
    dataset: Path,
) -> GeneratedProductionInputs:
    dataset_sha256 = file_sha256(dataset / "manifest.json")
    volgan_manifest, volgan_selection = _generated_pair(
        tmp_path,
        model="volgan",
        initializer_model="volgan",
        stem="volgan",
        epochs=7,
        dataset_sha256=dataset_sha256,
    )
    diffusion_manifest, diffusion_selection = _generated_pair(
        tmp_path,
        model="diffusion-ad-seq-vol",
        initializer_model="ad-seq-vol",
        stem="diffusion",
        epochs=11,
        dataset_sha256=dataset_sha256,
    )
    diffusion_ft_manifest, diffusion_ft_selection = _generated_pair(
        tmp_path,
        model="diffusion-ad-seq-vol-ft",
        initializer_model="ad-seq-vol-ft",
        stem="diffusion-ft",
        epochs=5,
        dataset_sha256=dataset_sha256,
        base_checkpoint=tmp_path / "diffusion.pt",
    )
    return GeneratedProductionInputs(
        volgan_initializer_manifest=volgan_manifest,
        volgan_epoch_selection=volgan_selection,
        diffusion_initializer_manifest=diffusion_manifest,
        diffusion_epoch_selection=diffusion_selection,
        diffusion_ft_initializer_manifest=diffusion_ft_manifest,
        diffusion_ft_epoch_selection=diffusion_ft_selection,
    )


def _generated_pair(
    tmp_path: Path,
    *,
    model: str,
    initializer_model: str,
    stem: str,
    epochs: int,
    dataset_sha256: str,
    base_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    checkpoint = tmp_path / f"{stem}.pt"
    variant = initializer_model if initializer_model != "volgan" else None
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_code_version": "test-code-version",
        "model_user_version": "test-user-version",
        "model_config_identity": f"{stem}-config",
        "precision": "float64",
        "training_start_date": "2021-09-01",
        "training_end_date": "2022-08-31",
        "dataset_manifest_sha256": dataset_sha256,
        "train_config": {"epochs": 3},
        "epoch": 2,
        "base_checkpoint_fingerprint": (
            None if base_checkpoint is None else file_sha256(base_checkpoint)
        ),
    }
    if variant is None:
        payload.update({"generator": {}, "discriminator": {}})
    else:
        payload.update({"variant": variant, "model": {}, "ema_model": {}})
    torch.save(payload, checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    base_sha256 = None if base_checkpoint is None else file_sha256(base_checkpoint)
    manifest = tmp_path / f"{stem}-initializer.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "hedging-generated-initializer-v1",
                "model": initializer_model,
                "dataset_manifest_sha256": dataset_sha256,
                "as_of_date": "2022-08-31",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "completed_epochs": 3,
                "base_checkpoint_path": (
                    None if base_checkpoint is None else str(base_checkpoint)
                ),
                "base_checkpoint_sha256": base_sha256,
            }
        ),
        encoding="utf-8",
    )
    losses = (1.0, 1.1)
    mean = float(np.mean(losses))
    standard_error = float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
    selection = tmp_path / f"{stem}-selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "hedging-generated-epoch-selection-v1",
                "model": model,
                "dataset_manifest_sha256": dataset_sha256,
                "warm_start_checkpoint": str(checkpoint),
                "warm_start_sha256": checkpoint_sha256,
                "warm_start_training_end_date": "2022-08-31",
                "base_checkpoint_sha256": base_sha256,
                "observation_count": 252,
                "one_standard_error_rule": True,
                "validation_origins": ["2022-09-01", "2022-11-01"],
                "selected_epochs": epochs,
                "scores": [
                    {
                        "epochs": epochs,
                        "losses": losses,
                        "mean": mean,
                        "standard_error": standard_error,
                        "fit_checkpoint_sha256": ["a" * 64, "b" * 64],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, selection
