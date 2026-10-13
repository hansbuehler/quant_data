"""Early process-environment guards for staged JAX command-line runners."""

from __future__ import annotations

import os
from collections.abc import Sequence


def force_cpu_for_prepare_cli(arguments: Sequence[str]) -> bool:
    """Hide GPUs before JAX imports when the requested CLI stage is prepare."""
    stage = None
    for index, argument in enumerate(arguments):
        if argument == "--stage" and index + 1 < len(arguments):
            stage = arguments[index + 1]
            break
        if argument.startswith("--stage="):
            stage = argument.partition("=")[2]
            break
    if stage != "prepare":
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    return True
