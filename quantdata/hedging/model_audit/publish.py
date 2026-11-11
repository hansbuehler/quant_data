"""Publish only the completed, checksummed audit on the existing hedging branch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from quantdata.hedging.model_audit.run import _hash, _json

SOURCE_PATHS = (
    "quantdata/dysanos/flex/stochastic_volatility_diffusion.py",
    "quantdata/hedging/USE.md",
    "quantdata/hedging/model_audit",
    "tests/quantdata/hedging/test_model_audit.py",
    "tests/quantdata/hedging/test_model_audit_finalize.py",
    "tests/quantdata/hedging/test_model_inference.py",
    "tests/quantdata/hedging/test_rank_comparison.py",
)
TEST_PATHS = (
    "tests/quantdata/hedging/test_model_audit.py",
    "tests/quantdata/hedging/test_model_audit_finalize.py",
    "tests/quantdata/hedging/test_model_inference.py",
    "tests/quantdata/hedging/test_rank_comparison.py",
    "tests/quantdata/dysanos/flex/test_stochastic_volatility_diffusion.py",
)


def _command(repository: Path, *command: str) -> str:
    result = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    result.check_returncode()
    return result.stdout.strip()


def publish(repository: Path, output: Path, *, branch: str, expected_head: str) -> dict:
    """Reject unexpected branch/history/index changes and never force-push or merge main."""
    ready = json.loads((output / "READY_TO_PUBLISH.json").read_text())
    retained = Path(ready["retained_directory"])
    if not ready["complete"] or not retained.is_relative_to(repository):
        raise ValueError("The completed deck does not belong to this source worktree")
    if _command(repository, "git", "branch", "--show-current") != branch or branch == "main":
        raise ValueError("Publishing requires the explicitly authorized feature branch, never main")
    if _command(repository, "git", "rev-parse", "HEAD") != expected_head:
        raise ValueError("The source checkout changed while diagnostics ran; review before publishing")
    if _command(repository, "git", "diff", "--cached", "--name-only"):
        raise ValueError("The index already contains staged changes; review before publishing")
    manifest = json.loads((retained / "RETAINED_ARTIFACTS.json").read_text())
    if manifest["raw_licensed_quotes_included"] or not manifest["large_files_external"]:
        raise ValueError("The retained deck violates the artifact scope")
    artifacts = []
    for name, identity in manifest["files"].items():
        path = retained / name
        if _hash(path) != identity["sha256"]:
            raise ValueError(f"Retained artifact changed after validation: {path}")
        artifacts.append(str(path.relative_to(repository)))
    artifacts.append(str((retained / "RETAINED_ARTIFACTS.json").relative_to(repository)))
    code_manifest = json.loads((retained / "MANIFEST.json").read_text())
    for name, identity in code_manifest["code_files"].items():
        path = repository / "quantdata/hedging/model_audit" / name
        if _hash(path) != identity["sha256"]:
            raise ValueError(f"Audit source changed after the final deck was built: {path}")
    _command(repository, sys.executable, "-m", "pytest", "-q", *TEST_PATHS)
    _command(repository, sys.executable, "-m", "ruff", "check", "--select", "F,I",
        "quantdata/hedging/model_audit", *TEST_PATHS[:-1])
    _command(repository, sys.executable, "-m", "compileall", "-q", "quantdata/hedging/model_audit",
        "quantdata/dysanos/flex/stochastic_volatility_diffusion.py")
    _command(repository, "git", "fetch", "origin")
    if _command(repository, "git", "rev-parse", f"origin/{branch}") != expected_head:
        raise ValueError("The remote feature branch advanced; review before publishing")
    _command(repository, "git", "add", "--", *SOURCE_PATHS)
    _command(repository, "git", "add", "-f", "--", *artifacts)
    _command(repository, "git", "diff", "--cached", "--check")
    _command(repository, "git", "diff", "--cached", "--stat")
    _command(repository, "git", "commit", "-m", "Complete model-dynamics audit with sixteen-factor DYSANOS")
    commit = _command(repository, "git", "rev-parse", "HEAD")
    _command(repository, "git", "push", "origin", f"HEAD:refs/heads/{branch}")
    remote = _command(repository, "git", "ls-remote", "origin", f"refs/heads/{branch}").split()[0]
    if remote != commit:
        raise RuntimeError("The remote branch does not match the published commit")
    return {"commit": commit, "branch": branch, "remote": "origin", "pushed": True,
        "main_modified": False, "deck": str(retained / "MODEL_DYNAMICS_REPORT.pdf")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    with FileLock(arguments.output / ".publish.lock"):
        if (arguments.output / "PUBLICATION.json").exists():
            print((arguments.output / "PUBLICATION.json").read_text(), flush=True)
            return
        while not (arguments.output / "READY_TO_PUBLISH.json").exists():
            print("Publication waits for all 44 origins, every model and the completed sixteen-factor deck", flush=True)
            time.sleep(60.)
        result = publish(arguments.repository, arguments.output, branch=arguments.branch,
            expected_head=arguments.expected_head)
        _json(arguments.output / "PUBLICATION.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
