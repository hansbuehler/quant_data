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
from quantdata.sdlv.daily.front_spot.hedging.runners import production_matrix
from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    DailyProductionMatrixConfig,
    GeneratedProductionInputs,
    GrommProductionInputs,
    ProductionReadiness,
    ProductionResource,
    build_daily_production_matrix,
)
from quantdata.torch import torch


def _matrix(
    *,
    dataset: Path = Path("/public/dataset"),
    gromm: GrommProductionInputs = GrommProductionInputs(),
    generated: GeneratedProductionInputs = GeneratedProductionInputs(),
):
    return build_daily_production_matrix(
        dataset=dataset,
        output_root=Path("/outputs"),
        checkpoint_root=Path("/checkpoints"),
        gromm=gromm,
        generated=generated,
    )


def test_daily_production_matrix_is_complete_unique_and_resource_partitioned() -> None:
    cases = _matrix()
    assert len(cases) == 54
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.model == "gromm" for case in cases) == 26
    assert sum(case.model == "gromm-q-dlv-vega" for case in cases) == 1
    assert all(
        case.resource is ProductionResource.GPU
        for case in cases
        if case.model in {"gromm", "volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"}
    )
    assert all(
        case.readiness is ProductionReadiness.BLOCKED
        for case in cases
        if case.model == "gromm"
    )


def test_native_commands_pin_common_daily_controls() -> None:
    config = DailyProductionMatrixConfig()
    cases = _matrix()
    native = tuple(
        case
        for case in cases
        if case.resource is ProductionResource.CPU
        and case.readiness is ProductionReadiness.READY
    )
    assert native
    for case in native:
        assert case.readiness is ProductionReadiness.READY
        assert case.command is not None
        assert case.command[0:3] == (
            "python",
            "-m",
            "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
        )
        assert case.command[case.command.index("--observation-count") + 1] == str(
            config.observation_count
        )
        assert case.command[case.command.index("--position-cap") + 1] == "1000"
        assert case.command[case.command.index("--device") + 1] == "cpu"
        assert case.command[case.command.index("--start") + 1] == "2023-01-01"
        assert case.command[case.command.index("--end") + 1] == "2025-12-31"


@pytest.mark.parametrize(
    ("physical_accepted_count", "expected_p_readiness"),
    (
        (2, ProductionReadiness.BLOCKED),
        (3, ProductionReadiness.READY),
    ),
)
def test_gromm_readiness_uses_authenticated_q_and_physical_range_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_accepted_count: int,
    expected_p_readiness: ProductionReadiness,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    architecture = tmp_path / "architecture.json"
    architecture.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "range.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    monkeypatch.setattr(
        production_matrix,
        "load_gromm_architecture_selection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        production_matrix,
        "load_completed_rolling_gromm2_range",
        lambda *_args, **_kwargs: SimpleNamespace(
            evaluation_start=DailyProductionMatrixConfig().evaluation_start,
            evaluation_end=DailyProductionMatrixConfig().evaluation_end,
            eligible_origin_count=3,
            physical_accepted_count=physical_accepted_count,
        ),
    )
    cases = _matrix(
        dataset=dataset,
        gromm=GrommProductionInputs(
            market_artifact_root=artifact_root,
            architecture_selection=architecture,
            artifact_range_summary=summary_path,
        ),
    )
    gromm_cases = tuple(case for case in cases if case.model == "gromm")
    conventional = next(case for case in cases if case.model == "gromm-q-dlv-vega")
    q_cases = tuple(case for case in gromm_cases if case.measure == "Q")
    p_cases = tuple(case for case in gromm_cases if case.measure == "P")
    assert all(case.readiness is ProductionReadiness.READY for case in q_cases)
    assert all(case.readiness is expected_p_readiness for case in p_cases)
    assert conventional.readiness is ProductionReadiness.READY
    assert conventional.command is not None
    assert "--market-artifact-range-summary" in conventional.command
    for case in q_cases:
        assert case.command is not None
        assert case.command[case.command.index("--measure") + 1] == "Q"
        assert case.command[case.command.index("--device") + 1] == "cuda"
        assert case.command[case.command.index("--market-artifact-root") + 1] == str(
            artifact_root
        )
        assert case.command[case.command.index("--architecture-selection") + 1] == str(
            architecture
        )
    if expected_p_readiness is ProductionReadiness.BLOCKED:
        assert all("2/3 origins" in case.blocked_reason for case in p_cases)


def test_generated_rolling_cases_require_selected_artifacts(tmp_path: Path) -> None:
    missing = {case.case_id: case for case in _matrix()}
    assert (
        missing["volgan-rolling-trained"].readiness
        is ProductionReadiness.BLOCKED
    )
    assert "initializer" in missing["volgan-rolling-trained"].blocked_reason

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    supplied = _generated_inputs(
        tmp_path,
        dataset=dataset,
    )
    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=supplied)
    }
    for case_id, epochs in (
        ("volgan-rolling-trained", "7"),
        ("diffusion-ad-seq-vol-rolling-trained", "11"),
        ("diffusion-ad-seq-vol-ft-rolling-trained", "5"),
    ):
        case = cases[case_id]
        assert case.readiness is ProductionReadiness.WAITING_GATE
        assert case.command is not None
        assert case.command[case.command.index("--epochs") + 1] == epochs
        assert case.command[case.command.index("--device") + 1] == "cuda"
        assert case.command[case.command.index("--objective") + 1] == "mean-variance"
        assert case.command[case.command.index("--risk-parameter") + 1] == "1"
        assert "--warm-start-checkpoint-sha256" in case.command


def test_production_matrix_cli_serializes_dates_and_generated_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "matrix.json"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_matrix",
            "--dataset",
            str(dataset),
            "--output-root",
            str(tmp_path / "outputs"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--output",
            str(output),
            "--volgan-initializer-manifest",
            str(generated.volgan_initializer_manifest),
            "--volgan-epoch-selection",
            str(generated.volgan_epoch_selection),
            "--diffusion-initializer-manifest",
            str(generated.diffusion_initializer_manifest),
            "--diffusion-epoch-selection",
            str(generated.diffusion_epoch_selection),
            "--diffusion-ft-initializer-manifest",
            str(generated.diffusion_ft_initializer_manifest),
            "--diffusion-ft-epoch-selection",
            str(generated.diffusion_ft_epoch_selection),
        ],
    )

    production_matrix.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["evaluation_start"] == "2023-01-01"
    assert payload["config"]["evaluation_end"] == "2025-12-31"
    case = next(
        value
        for value in payload["cases"]
        if value["case_id"] == "volgan-rolling-trained"
    )
    assert case["readiness"] == ProductionReadiness.WAITING_GATE.value
    assert case["command"][case["command"].index("--epochs") + 1] == "7"
    assert case["command"][
        case["command"].index("--warm-start-checkpoint") + 1
    ] == str(tmp_path / "volgan.pt")
    cases = {value["case_id"]: value for value in payload["cases"]}
    for case_id, initializer, epochs in (
        ("diffusion-ad-seq-vol-rolling-trained", tmp_path / "diffusion.pt", "11"),
        ("diffusion-ad-seq-vol-ft-rolling-trained", tmp_path / "diffusion-ft.pt", "5"),
    ):
        case = cases[case_id]
        assert case["readiness"] == ProductionReadiness.WAITING_GATE.value
        assert case["command"][case["command"].index("--epochs") + 1] == epochs
        assert case["command"][
            case["command"].index("--warm-start-checkpoint") + 1
        ] == str(initializer)
    fine_tuned = cases["diffusion-ad-seq-vol-ft-rolling-trained"]
    assert fine_tuned["command"][
        fine_tuned["command"].index("--base-checkpoint") + 1
    ] == str(tmp_path / "diffusion.pt")


def test_generated_matrix_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    (tmp_path / "volgan.pt").write_bytes(b"tampered")

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    case = cases["volgan-rolling-trained"]
    assert case.readiness is ProductionReadiness.BLOCKED
    assert "hash mismatch" in case.blocked_reason


def test_generated_matrix_rejects_forged_initializer_date(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    checkpoint = tmp_path / "volgan.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_end_date"] = "2023-01-03"
    torch.save(payload, checkpoint)
    manifest = json.loads(
        generated.volgan_initializer_manifest.read_text(encoding="utf-8")
    )
    manifest["checkpoint_sha256"] = file_sha256(checkpoint)
    generated.volgan_initializer_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    case = cases["volgan-rolling-trained"]
    assert case.readiness is ProductionReadiness.BLOCKED
    assert "checkpoint enters evaluation data" in case.blocked_reason


def test_generated_matrix_recomputes_one_se_choice(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    selection = json.loads(
        generated.volgan_epoch_selection.read_text(encoding="utf-8")
    )
    selection["selected_epochs"] = 3
    selection["scores"] = [
        {
            "epochs": 3,
            "losses": [2.0, 2.0],
            "mean": 2.0,
            "standard_error": 0.0,
            "fit_checkpoint_sha256": ["c" * 64, "d" * 64],
        },
        {
            "epochs": 7,
            "losses": [1.0, 1.0],
            "mean": 1.0,
            "standard_error": 0.0,
            "fit_checkpoint_sha256": ["e" * 64, "f" * 64],
        },
    ]
    generated.volgan_epoch_selection.write_text(
        json.dumps(selection),
        encoding="utf-8",
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    case = cases["volgan-rolling-trained"]
    assert case.readiness is ProductionReadiness.BLOCKED
    assert "not the recomputed one-SE choice" in case.blocked_reason


def test_generated_matrix_rejects_checkpoint_ft_lineage_forgery(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    checkpoint = tmp_path / "diffusion-ft.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["base_checkpoint_fingerprint"] = "f" * 64
    torch.save(payload, checkpoint)
    manifest = json.loads(
        generated.diffusion_ft_initializer_manifest.read_text(encoding="utf-8")
    )
    manifest["checkpoint_sha256"] = file_sha256(checkpoint)
    generated.diffusion_ft_initializer_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selection = json.loads(
        generated.diffusion_ft_epoch_selection.read_text(encoding="utf-8")
    )
    selection["warm_start_sha256"] = file_sha256(checkpoint)
    generated.diffusion_ft_epoch_selection.write_text(
        json.dumps(selection),
        encoding="utf-8",
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    case = cases["diffusion-ad-seq-vol-ft-rolling-trained"]
    assert case.readiness is ProductionReadiness.BLOCKED
    assert "another base checkpoint" in case.blocked_reason


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
    diffusion_checkpoint = tmp_path / "diffusion.pt"
    diffusion_ft_manifest, diffusion_ft_selection = _generated_pair(
        tmp_path,
        model="diffusion-ad-seq-vol-ft",
        initializer_model="ad-seq-vol-ft",
        stem="diffusion-ft",
        epochs=5,
        dataset_sha256=dataset_sha256,
        base_checkpoint=diffusion_checkpoint,
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
