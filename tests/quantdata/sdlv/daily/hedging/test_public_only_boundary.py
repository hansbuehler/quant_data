from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

_RUNTIME_ROOT = Path("quantdata/sdlv/daily/front_spot/hedging")
_RUNTIME_DIRECTORIES = ("benchmark", "claims", "models", "runners")
_FORBIDDEN_IMPORT_PREFIXES = (
    "quantdata.dysanos",
    "quantdata.widb",
    "quantdata.sdlv.daily.front_spot.hedging.data.export_public_candidate",
)
_FORBIDDEN_EXACT_IMPORTS = ("quantdata.sdlv.daily",)


def test_runtime_models_do_not_import_private_source_pipeline() -> None:
    violations: list[str] = []
    for directory_name in _RUNTIME_DIRECTORIES:
        directory = _RUNTIME_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: list[tuple[int, str]] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend((node.lineno, alias.name) for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported.append((node.lineno, node.module))
            for line, module in imported:
                if module in _FORBIDDEN_EXACT_IMPORTS or module.startswith(
                    _FORBIDDEN_IMPORT_PREFIXES
                ):
                    violations.append(f"{path}:{line}: {module}")
    assert violations == []


def test_public_panel_guard_rejects_other_objects() -> None:
    with pytest.raises(TypeError, match="PublicHedgingPanel"):
        require_public_hedging_panel(object())  # type: ignore[arg-type]
