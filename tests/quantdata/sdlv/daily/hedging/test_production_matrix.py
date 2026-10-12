from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.runners.production_matrix import (
    DailyProductionMatrixConfig,
    GeneratedProductionInputs,
    ProductionReadiness,
    ProductionResource,
    build_daily_production_matrix,
)


def _matrix(*, generated: GeneratedProductionInputs = GeneratedProductionInputs()):
    return build_daily_production_matrix(
        dataset=Path("/public/dataset"),
        output_root=Path("/outputs"),
        checkpoint_root=Path("/checkpoints"),
        generated=generated,
    )


def test_daily_production_matrix_is_complete_unique_and_resource_partitioned() -> None:
    cases = _matrix()
    assert len(cases) == 53
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.model == "gromm" for case in cases) == 26
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


def test_generated_rolling_cases_require_selected_artifacts() -> None:
    missing = {case.case_id: case for case in _matrix()}
    assert (
        missing["volgan-rolling-trained"].readiness
        is ProductionReadiness.BLOCKED
    )
    assert "initializer" in missing["volgan-rolling-trained"].blocked_reason

    supplied = GeneratedProductionInputs(
        volgan_initializer=Path("/training/volgan-2022.pt"),
        volgan_update_epochs=7,
        diffusion_initializer=Path("/training/diffusion-2022.pt"),
        diffusion_update_epochs=11,
        diffusion_ft_base=Path("/training/diffusion-base.pt"),
        diffusion_ft_initializer=Path("/training/diffusion-ft-2022.pt"),
        diffusion_ft_update_epochs=5,
    )
    cases = {case.case_id: case for case in _matrix(generated=supplied)}
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
