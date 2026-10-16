import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import production_matrix
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    ECONOMIC_CAP_WARNING_STATUS,
    GPU_READINESS_SCHEMA,
    PRODUCTION_GATE_SCHEMA,
    PRODUCTION_SCOPES,
    QUANT_REVIEW_SCHEMA,
    generated_scope_binding,
    load_generated_initializer_evidence,
    load_generated_selection_evidence,
)
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
    assert len(cases) == 57
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.model == "gromm" for case in cases) == 26
    assert sum(case.model == "gromm-q-dlv-vega" for case in cases) == 1
    covariance_power = tuple(
        case
        for case in cases
        if case.model == "robust-covariance"
        and case.objective == "power-exponential"
    )
    assert tuple(case.risk_parameter for case in covariance_power) == (0.1, 1.0, 10.0)
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
        assert "--validate-dataset" in case.command
        assert "--allow-nonconvergence" not in case.command


def test_case_warning_metadata_requires_warning_gate_status() -> None:
    case = next(
        value
        for value in _matrix()
        if value.readiness is ProductionReadiness.READY
    )

    with pytest.raises(ValueError, match="requires the warning status"):
        replace(
            case,
            production_gate_status="pass",
            production_gate_warning_json=json.dumps(
                {"classification": "fitted-policy-position-cap"}
            ),
        )


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
    for case_id, epochs, model_label in (
        ("volgan-rolling-trained", "7", "volgan-model"),
        ("diffusion-ad-seq-vol-rolling-trained", "11", "diffusion-model"),
        (
            "diffusion-ad-seq-vol-ft-rolling-trained",
            "5",
            "diffusion-ft-model",
        ),
    ):
        case = cases[case_id]
        assert case.readiness is ProductionReadiness.WAITING_GATE
        assert case.command is not None
        assert case.command[case.command.index("--epochs") + 1] == epochs
        assert case.command[case.command.index("--device") + 1] == "cuda"
        assert case.command[case.command.index("--objective") + 1] == "mean-variance"
        assert case.command[case.command.index("--risk-parameter") + 1] == "1"
        assert case.command[case.command.index("--model-label") + 1] == model_label
        assert "--warm-start-checkpoint-sha256" in case.command
        assert case.command[case.command.index("--dataset-manifest-sha256") + 1] == (
            file_sha256(dataset / "manifest.json")
        )


def test_generated_production_gates_promote_only_authenticated_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    selected = _generated_inputs(tmp_path, dataset=dataset)
    monkeypatch.setattr(
        production_matrix,
        "reviewable_repository_commit",
        lambda: "a" * 40,
    )
    generated = replace(
        selected,
        **_generated_gates(tmp_path, dataset=dataset, generated=selected),
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    for case_id in (
        "volgan-public-settings",
        "volgan-rolling-trained",
        "diffusion-ad-seq-vol-public-settings",
        "diffusion-ad-seq-vol-rolling-trained",
        "diffusion-ad-seq-vol-ft-public-settings",
        "diffusion-ad-seq-vol-ft-rolling-trained",
    ):
        assert cases[case_id].readiness is ProductionReadiness.READY
        assert cases[case_id].reviewed_commit == "a" * 40
        assert cases[case_id].production_gate_sha256 is not None
        assert len(cases[case_id].production_gate_sha256) == 64
        assert cases[case_id].production_gate_path is not None
        assert cases[case_id].production_gate_status == "pass"
        assert cases[case_id].production_gate_warning_json is None


def test_generated_production_matrix_preserves_scope_cap_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    selected = _generated_inputs(tmp_path, dataset=dataset)
    monkeypatch.setattr(
        production_matrix,
        "reviewable_repository_commit",
        lambda: "a" * 40,
    )
    gates = _generated_gates(tmp_path, dataset=dataset, generated=selected)
    _mark_gate_warning(
        tmp_path,
        stem="volgan",
        scope="frozen-weekly-public",
    )
    generated = replace(selected, **gates)

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }
    public = cases["volgan-public-settings"]
    rolling = cases["volgan-rolling-trained"]

    assert public.readiness is ProductionReadiness.READY
    assert public.production_gate_status == ECONOMIC_CAP_WARNING_STATUS
    assert public.production_gate_warning_json is None
    assert public.to_dict()["production_gate_warning"] is None
    assert rolling.production_gate_status == ECONOMIC_CAP_WARNING_STATUS
    assert rolling.production_gate_warning_json is None


def test_generated_production_gate_rejects_tampered_readiness_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    selected = _generated_inputs(tmp_path, dataset=dataset)
    monkeypatch.setattr(
        production_matrix,
        "reviewable_repository_commit",
        lambda: "a" * 40,
    )
    gates = _generated_gates(
        tmp_path,
        dataset=dataset,
        generated=selected,
    )
    (tmp_path / "volgan-gpu-readiness.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )
    generated = replace(
        selected,
        **gates,
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    assert cases["volgan-public-settings"].readiness is ProductionReadiness.BLOCKED
    assert "gpu_readiness hash mismatch" in cases[
        "volgan-public-settings"
    ].blocked_reason


def test_generated_production_gate_binds_commit_and_selected_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    selected = _generated_inputs(tmp_path, dataset=dataset)
    gates = _generated_gates(tmp_path, dataset=dataset, generated=selected)
    generated = replace(selected, **gates)

    monkeypatch.setattr(
        production_matrix,
        "reviewable_repository_commit",
        lambda: "b" * 40,
    )
    stale = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }
    assert stale["volgan-public-settings"].readiness is ProductionReadiness.BLOCKED
    assert "another commit" in stale["volgan-public-settings"].blocked_reason

    pinned = {
        case.case_id: case
        for case in _matrix(
            dataset=dataset,
            generated=replace(generated, reviewed_commit="a" * 40),
        )
    }
    assert pinned["volgan-public-settings"].readiness is ProductionReadiness.READY
    assert pinned["volgan-public-settings"].reviewed_commit == "a" * 40

    monkeypatch.setattr(
        production_matrix,
        "reviewable_repository_commit",
        lambda: "a" * 40,
    )
    _rewrite_gate_scope(
        gates["volgan_production_gate"],
        scope="daily-rolling",
        selected_epochs=99,
    )
    mismatched = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }
    assert mismatched["volgan-public-settings"].readiness is ProductionReadiness.READY
    assert mismatched["volgan-rolling-trained"].readiness is ProductionReadiness.BLOCKED
    assert "evidence differs" in mismatched[
        "volgan-rolling-trained"
    ].blocked_reason


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
            "--volgan-selection-source-manifest",
            str(generated.volgan_selection_source_manifest),
            "--volgan-epoch-selection",
            str(generated.volgan_epoch_selection),
            "--diffusion-initializer-manifest",
            str(generated.diffusion_initializer_manifest),
            "--diffusion-selection-source-manifest",
            str(generated.diffusion_selection_source_manifest),
            "--diffusion-epoch-selection",
            str(generated.diffusion_epoch_selection),
            "--diffusion-ft-initializer-manifest",
            str(generated.diffusion_ft_initializer_manifest),
            "--diffusion-ft-selection-source-manifest",
            str(generated.diffusion_ft_selection_source_manifest),
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


def test_generated_matrix_accepts_distinct_budget_selection_source(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    source = tmp_path / "volgan-selection-source.pt"
    payload = torch.load(
        tmp_path / "volgan.pt",
        map_location="cpu",
        weights_only=False,
    )
    payload["training_end_date"] = "2022-07-01"
    torch.save(payload, source)
    source_manifest = tmp_path / "volgan-selection-source.json"
    manifest = json.loads(
        generated.volgan_initializer_manifest.read_text(encoding="utf-8")
    )
    manifest["as_of_date"] = "2022-07-01"
    manifest["checkpoint_path"] = str(source)
    manifest["checkpoint_sha256"] = file_sha256(source)
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    generated = replace(
        generated,
        volgan_selection_source_manifest=source_manifest,
    )
    selection = json.loads(
        generated.volgan_epoch_selection.read_text(encoding="utf-8")
    )
    selection["warm_start_checkpoint"] = str(source)
    selection["warm_start_sha256"] = file_sha256(source)
    selection["warm_start_training_end_date"] = "2022-07-01"
    generated.volgan_epoch_selection.write_text(
        json.dumps(selection),
        encoding="utf-8",
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    assert (
        cases["volgan-rolling-trained"].readiness
        is ProductionReadiness.WAITING_GATE
    )


def test_generated_matrix_rejects_selection_source_with_another_geometry(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    source = tmp_path / "volgan-selection-source.pt"
    payload = torch.load(
        tmp_path / "volgan.pt",
        map_location="cpu",
        weights_only=False,
    )
    payload["training_end_date"] = "2022-07-01"
    payload["model_config_identity"] = "another-geometry"
    torch.save(payload, source)
    source_manifest = tmp_path / "volgan-selection-source.json"
    manifest = json.loads(
        generated.volgan_initializer_manifest.read_text(encoding="utf-8")
    )
    manifest["as_of_date"] = "2022-07-01"
    manifest["checkpoint_path"] = str(source)
    manifest["checkpoint_sha256"] = file_sha256(source)
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    generated = replace(
        generated,
        volgan_selection_source_manifest=source_manifest,
    )
    selection = json.loads(
        generated.volgan_epoch_selection.read_text(encoding="utf-8")
    )
    selection["warm_start_checkpoint"] = str(source)
    selection["warm_start_sha256"] = file_sha256(source)
    selection["warm_start_training_end_date"] = "2022-07-01"
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
    assert "another model geometry" in case.blocked_reason


def test_generated_matrix_separates_ft_selection_and_production_bases(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)

    selection_base = tmp_path / "diffusion-selection-source.pt"
    base_payload = torch.load(
        tmp_path / "diffusion.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_payload["training_end_date"] = "2022-07-01"
    torch.save(base_payload, selection_base)
    selection_base_manifest = tmp_path / "diffusion-selection-source.json"
    base_manifest = json.loads(
        generated.diffusion_initializer_manifest.read_text(encoding="utf-8")
    )
    base_manifest["as_of_date"] = "2022-07-01"
    base_manifest["checkpoint_path"] = str(selection_base)
    base_manifest["checkpoint_sha256"] = file_sha256(selection_base)
    selection_base_manifest.write_text(json.dumps(base_manifest), encoding="utf-8")
    base_selection = json.loads(
        generated.diffusion_epoch_selection.read_text(encoding="utf-8")
    )
    base_selection["warm_start_checkpoint"] = str(selection_base)
    base_selection["warm_start_sha256"] = file_sha256(selection_base)
    base_selection["warm_start_training_end_date"] = "2022-07-01"
    generated.diffusion_epoch_selection.write_text(
        json.dumps(base_selection),
        encoding="utf-8",
    )

    selection_ft = tmp_path / "diffusion-ft-selection-source.pt"
    ft_payload = torch.load(
        tmp_path / "diffusion-ft.pt",
        map_location="cpu",
        weights_only=False,
    )
    ft_payload["training_end_date"] = "2022-07-01"
    ft_payload["base_checkpoint_fingerprint"] = file_sha256(selection_base)
    torch.save(ft_payload, selection_ft)
    selection_ft_manifest = tmp_path / "diffusion-ft-selection-source.json"
    ft_manifest = json.loads(
        generated.diffusion_ft_initializer_manifest.read_text(encoding="utf-8")
    )
    ft_manifest["as_of_date"] = "2022-07-01"
    ft_manifest["checkpoint_path"] = str(selection_ft)
    ft_manifest["checkpoint_sha256"] = file_sha256(selection_ft)
    ft_manifest["base_checkpoint_path"] = str(selection_base)
    ft_manifest["base_checkpoint_sha256"] = file_sha256(selection_base)
    selection_ft_manifest.write_text(json.dumps(ft_manifest), encoding="utf-8")
    ft_selection = json.loads(
        generated.diffusion_ft_epoch_selection.read_text(encoding="utf-8")
    )
    ft_selection["warm_start_checkpoint"] = str(selection_ft)
    ft_selection["warm_start_sha256"] = file_sha256(selection_ft)
    ft_selection["warm_start_training_end_date"] = "2022-07-01"
    ft_selection["base_checkpoint_sha256"] = file_sha256(selection_base)
    generated.diffusion_ft_epoch_selection.write_text(
        json.dumps(ft_selection),
        encoding="utf-8",
    )
    generated = replace(
        generated,
        diffusion_selection_source_manifest=selection_base_manifest,
        diffusion_ft_selection_source_manifest=selection_ft_manifest,
    )

    cases = {
        case.case_id: case
        for case in _matrix(dataset=dataset, generated=generated)
    }

    case = cases["diffusion-ad-seq-vol-ft-rolling-trained"]
    assert case.readiness is ProductionReadiness.WAITING_GATE
    assert case.command is not None
    assert case.command[case.command.index("--warm-start-checkpoint") + 1] == str(
        tmp_path / "diffusion-ft.pt"
    )
    assert case.command[case.command.index("--base-checkpoint") + 1] == str(
        tmp_path / "diffusion.pt"
    )


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


def test_generated_matrix_rejects_another_budget_selection_window(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    selection = json.loads(
        generated.volgan_epoch_selection.read_text(encoding="utf-8")
    )
    selection["observation_count"] = 252
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
    assert "126-observation window" in case.blocked_reason


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_checkpoint_path", "/tmp/another-base.pt"),
        ("base_checkpoint_sha256", "f" * 64),
    ),
)
def test_ft_initializer_sentinel_requires_exact_expected_base(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    dataset_sha256 = file_sha256(dataset / "manifest.json")
    generated = _generated_inputs(tmp_path, dataset=dataset)
    manifest = json.loads(
        generated.diffusion_ft_initializer_manifest.read_text(encoding="utf-8")
    )
    manifest[field] = value
    generated.diffusion_ft_initializer_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another base"):
        load_generated_initializer_evidence(
            initializer_manifest=generated.diffusion_ft_initializer_manifest,
            expected_model="diffusion-ad-seq-vol-ft",
            expected_dataset_manifest_sha256=dataset_sha256,
            expected_base_initializer_manifest=(
                generated.diffusion_initializer_manifest
            ),
        )


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
        volgan_selection_source_manifest=volgan_manifest,
        volgan_epoch_selection=volgan_selection,
        diffusion_initializer_manifest=diffusion_manifest,
        diffusion_selection_source_manifest=diffusion_manifest,
        diffusion_epoch_selection=diffusion_selection,
        diffusion_ft_initializer_manifest=diffusion_ft_manifest,
        diffusion_ft_selection_source_manifest=diffusion_ft_manifest,
        diffusion_ft_epoch_selection=diffusion_ft_selection,
    )


def _generated_gates(
    tmp_path: Path,
    *,
    dataset: Path,
    generated: GeneratedProductionInputs,
) -> dict[str, Path]:
    dataset_sha256 = file_sha256(dataset / "manifest.json")
    base = load_generated_initializer_evidence(
        initializer_manifest=generated.diffusion_initializer_manifest,
        expected_model="diffusion-ad-seq-vol",
        expected_dataset_manifest_sha256=dataset_sha256,
    )
    evidence = {
        "volgan": load_generated_selection_evidence(
            initializer_manifest=generated.volgan_initializer_manifest,
            selection_source_initializer_manifest=(
                generated.volgan_selection_source_manifest
            ),
            epoch_selection=generated.volgan_epoch_selection,
            expected_model="volgan",
            expected_dataset_manifest_sha256=dataset_sha256,
        ),
        "diffusion-ad-seq-vol": load_generated_selection_evidence(
            initializer_manifest=generated.diffusion_initializer_manifest,
            selection_source_initializer_manifest=(
                generated.diffusion_selection_source_manifest
            ),
            epoch_selection=generated.diffusion_epoch_selection,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=dataset_sha256,
        ),
        "diffusion-ad-seq-vol-ft": load_generated_selection_evidence(
            initializer_manifest=generated.diffusion_ft_initializer_manifest,
            selection_source_initializer_manifest=(
                generated.diffusion_ft_selection_source_manifest
            ),
            epoch_selection=generated.diffusion_ft_epoch_selection,
            expected_model="diffusion-ad-seq-vol-ft",
            expected_dataset_manifest_sha256=dataset_sha256,
            base_initializer_manifest=generated.diffusion_initializer_manifest,
            selection_base_initializer_manifest=(
                generated.diffusion_selection_source_manifest
            ),
        ),
    }
    values: dict[str, Path] = {}
    for field, model, stem in (
        ("volgan_production_gate", "volgan", "volgan"),
        (
            "diffusion_production_gate",
            "diffusion-ad-seq-vol",
            "diffusion",
        ),
        (
            "diffusion_ft_production_gate",
            "diffusion-ad-seq-vol-ft",
            "diffusion-ft",
        ),
    ):
        public_base = base if model.endswith("-ft") else None
        bindings = {
            scope: generated_scope_binding(
                selection=(evidence[model] if scope.endswith("-rolling") else None),
                base_initializer=public_base,
            )
            for scope in PRODUCTION_SCOPES
        }
        common = {
            "model": model,
            "dataset_manifest_sha256": dataset_sha256,
            "status": "pass",
            "reviewed_commit": "a" * 40,
            "scope_bindings": bindings,
        }
        review = tmp_path / f"{stem}-quant-review.json"
        readiness = tmp_path / f"{stem}-gpu-readiness.json"
        review.write_text(
            json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
            encoding="utf-8",
        )
        readiness.write_text(
            json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
            encoding="utf-8",
        )
        gate = tmp_path / f"{stem}-production-gate.json"
        gate.write_text(
            json.dumps(
                {
                    "schema": PRODUCTION_GATE_SCHEMA,
                    **common,
                    "quant_review": {
                        "path": str(review),
                        "sha256": file_sha256(review),
                    },
                    "gpu_readiness": {
                        "path": str(readiness),
                        "sha256": file_sha256(readiness),
                    },
                }
            ),
            encoding="utf-8",
        )
        values[field] = gate
    return values


def _mark_gate_warning(
    tmp_path: Path,
    *,
    stem: str,
    scope: str,
) -> dict[str, object]:
    warning: dict[str, object] = {
        "classification": "fitted-policy-position-cap",
        "fit_cap_hits": 12_090,
        "executed_cap_hits": 0,
        "message": "pricing policies reach the cap; the executed hedge does not",
    }
    warnings = {scope: warning}
    readiness = tmp_path / f"{stem}-gpu-readiness.json"
    readiness_payload = json.loads(readiness.read_text(encoding="utf-8"))
    readiness_payload.update(
        {
            "status": ECONOMIC_CAP_WARNING_STATUS,
            "warnings": warnings,
            "failures": {},
        }
    )
    readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
    gate = tmp_path / f"{stem}-production-gate.json"
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    gate_payload.update(
        {
            "status": ECONOMIC_CAP_WARNING_STATUS,
            "warnings": warnings,
        }
    )
    gate_payload["gpu_readiness"]["sha256"] = file_sha256(readiness)
    gate.write_text(json.dumps(gate_payload), encoding="utf-8")
    return warning


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
                "model_config": {"label": f"{stem}-model"},
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
                "observation_count": 126,
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


def _rewrite_gate_scope(
    gate_path: Path,
    *,
    scope: str,
    selected_epochs: int,
) -> None:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["scope_bindings"][scope]["selected_epochs"] = selected_epochs
    for name in ("quant_review", "gpu_readiness"):
        artifact_path = Path(gate[name]["path"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["scope_bindings"][scope]["selected_epochs"] = selected_epochs
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        gate[name]["sha256"] = file_sha256(artifact_path)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
