"""Stable presentation labels for authenticated hedging policy identities."""

_MEASURE_LABELS = {
    "P": "P",
    "Q": "Q",
    "historical-P": "historical P",
    "Q-price/historical-transition": "Q price / historical path",
}

_MODEL_VARIANTS = {
    "sanos-lv": "reference-delta-eight-factor-vega",
    "robust-covariance": "independent-shrinkage-tyler",
    "historical-sanos": "rolling-fixed-contract-scenarios",
    "gromm-q-dlv-vega": "q-expectation-eight-factor-dlv-vega",
    "gromm": "rolling-16-factor-eight-substep-deep-hedge",
}

_GENERATED_MODELS = {
    "volgan",
    "diffusion-ad-seq-vol",
    "diffusion-ad-seq-vol-ft",
}

_VARIANT_LABELS = {
    "reference-delta-eight-factor-vega": "delta + 8 DLV vegas",
    "independent-shrinkage-tyler": "shrinkage Tyler",
    "rolling-fixed-contract-scenarios": "SANOS scenarios",
    "q-expectation-eight-factor-dlv-vega": "delta + 8 DLV vegas",
    "rolling-16-factor-eight-substep-deep-hedge": "deep hedge",
    "published-settings-full-rolling-fit": "public settings",
    "fixed-2022-source-selected-update": "rolling trained",
}

_WEEKLY_SCHEDULES = ("frozen-weekly", "restarted-weekly")


def canonical_policy_variant(
    *,
    case_id: str,
    model: str,
    schedule: str | None = None,
) -> str:
    """Derive the precommitted policy variant independently of matrix text."""
    if model in _MODEL_VARIANTS:
        base_variant = _MODEL_VARIANTS[model]
    elif model in _GENERATED_MODELS:
        if case_id.endswith("public-settings"):
            base_variant = "published-settings-full-rolling-fit"
        elif case_id.endswith("rolling-trained"):
            base_variant = "fixed-2022-source-selected-update"
        else:
            raise ValueError(f"unknown generated policy case: {case_id}")
    else:
        raise ValueError(f"unknown policy model: {model}")
    if schedule is None:
        return base_variant
    if schedule not in _WEEKLY_SCHEDULES:
        raise ValueError(f"unknown hedge schedule: {schedule}")
    return f"{schedule}-{base_variant}"


def policy_specification_label(*, measure: str, variant: str) -> str:
    """Return a concise label without discarding measure or variant identity."""
    measure_label = _MEASURE_LABELS.get(measure, measure.replace("-", " "))
    base_variant = variant
    for schedule in _WEEKLY_SCHEDULES:
        prefix = f"{schedule}-"
        if variant.startswith(prefix):
            base_variant = variant.removeprefix(prefix)
            break
    try:
        label = _VARIANT_LABELS[base_variant]
    except KeyError as error:
        raise ValueError(f"unknown policy variant: {variant}") from error
    return f"{measure_label}; {label}"
