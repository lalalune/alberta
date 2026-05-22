#!/usr/bin/env python3
"""Export local security-gym rollout records for oracle-review pipelines.

The artifact is deliberately scoped to the local ``security-gym`` rollout. It
creates the `(state, action, reward, outcome)` JSONL stream needed by downstream
LLM/oracle review code without claiming the unavailable ``rlsecd`` daemon loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alberta_framework.security import (
    SecurityFeatureSchema,
    SecurityRolloutStep,
    security_rollout_step_to_oracle_experience,
    validate_security_oracle_experience,
)

try:
    from benchmarks.security_gym_counterfactual_rollout import (
        DEFAULT_SECURITY_GYM,
        run_benchmark,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from security_gym_counterfactual_rollout import (  # type: ignore[no-redef]
        DEFAULT_SECURITY_GYM,
        run_benchmark,
    )

DEFAULT_OUTPUT = Path("outputs/security_gym_oracle_experience/records.jsonl")
DEFAULT_MANIFEST = Path("outputs/security_gym_oracle_experience/manifest.json")


def export_oracle_experience(
    *,
    security_gym_root: Path = DEFAULT_SECURITY_GYM,
    max_steps: int = 48,
    policy: str = "oracle_block_malicious",
    output: Path = DEFAULT_OUTPUT,
    manifest_output: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Run the local rollout and export oracle experience records as JSONL."""
    benchmark = run_benchmark(
        security_gym_root,
        max_steps,
        include_rollout_records=True,
    )
    rollout_records = benchmark.get("rollout_records", {})
    if not isinstance(rollout_records, dict) or policy not in rollout_records:
        raise ValueError(f"policy {policy!r} was not present in rollout records")
    schema = SecurityFeatureSchema.from_dict(
        benchmark["feature_schema"]  # type: ignore[arg-type]
    )
    steps = [
        SecurityRolloutStep.from_dict(record)
        for record in rollout_records[policy]  # type: ignore[index]
    ]
    records = [security_rollout_step_to_oracle_experience(step) for step in steps]
    validate_security_oracle_experience(records, schema)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.outcome["label"])
        counts[label] = counts.get(label, 0) + 1
    manifest = {
        "schema": "alberta.security_gym.oracle_experience_manifest.v1",
        "claim_scope": "local_security_gym_oracle_experience_export",
        "records_path": str(output),
        "n_records": len(records),
        "policy": policy,
        "feature_schema": schema.to_dict(),
        "outcome_counts": counts,
        "source_rollout_schema": benchmark.get("schema"),
        "source_rollout_passed": benchmark.get("passed"),
        "passed": bool(len(records) >= 20 and benchmark.get("passed") is True),
        "boundary": (
            "exports local security-gym oracle experience records; does not "
            "prove unavailable rlsecd daemon or production-log integration"
        ),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security-gym-root", type=Path, default=DEFAULT_SECURITY_GYM)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--policy", default="oracle_block_malicious")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export and print the manifest."""
    args = parse_args(argv)
    manifest = export_oracle_experience(
        security_gym_root=args.security_gym_root,
        max_steps=args.max_steps,
        policy=args.policy,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
