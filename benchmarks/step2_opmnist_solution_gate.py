#!/usr/bin/env python3
"""Audit Step 2 OPMNIST artifacts against the solution gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "examples"
    / "The Alberta Plan"
    / "Step2"
    / "step2_upgd_memory_opmnist.py"
)
DEFAULT_RESULT_PATH = (
    REPO_ROOT
    / "outputs"
    / "step2_canonical"
    / "upgd_memory_opmnist_latest_best_800block_1seed_results.json"
)


def load_runner_module() -> Any:
    """Load the OPMNIST runner module from its path with spaces."""
    spec = importlib.util.spec_from_file_location(
        "step2_upgd_memory_opmnist_solution_gate",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OPMNIST runner at {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_payload(path: Path) -> dict[str, Any]:
    """Load one result JSON artifact."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(dict[str, Any], payload)


def audit_result(path: Path, *, min_seeds: int = 3) -> dict[str, Any]:
    """Return solution status plus artifact path metadata."""
    module = load_runner_module()
    payload = load_payload(path)
    status = module.opmnist_solution_status(payload, min_seeds=min_seeds)
    return {
        "schema": "alberta.step2.opmnist_solution_gate.audit.v1",
        "artifact_path": str(path),
        "status": status,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_path",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Step 2 OPMNIST result JSON to audit.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=3,
        help="Minimum completed full-scale seeds required for solution status.",
    )
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path to write the audit JSON.",
    )
    parser.add_argument(
        "--allow-unsolved",
        action="store_true",
        help="Return exit code 0 even when solved_opmnist_step2 is false.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact audit and return a process exit code."""
    args = parse_args(argv)
    if args.min_seeds <= 0:
        raise ValueError("--min-seeds must be positive")
    audit = audit_result(args.result_path, min_seeds=args.min_seeds)
    rendered = json.dumps(audit, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    solved = bool(audit["status"]["solved_opmnist_step2"])
    if solved or args.allow_unsolved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
