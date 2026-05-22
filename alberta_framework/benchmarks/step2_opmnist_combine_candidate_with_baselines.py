#!/usr/bin/env python3
"""Combine completed OPMNIST baselines with candidate-only seed results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
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
PROTOCOL_KEYS = (
    "is_true_mnist",
    "is_full_mnist_split",
    "n_train",
    "n_test",
    "n_permutations",
    "task_block_size",
    "steps",
    "matches_dohare_opmnist_core_protocol",
    "matches_dohare_opmnist_published_task_count",
    "prediction_before_update_every_step",
    "task_id_provided_to_learner",
    "test_views_cover_all_permutations",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner_module() -> Any:
    """Load the OPMNIST runner module from its path with spaces."""
    spec = importlib.util.spec_from_file_location(
        "step2_upgd_memory_opmnist_mixed_combine",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OPMNIST runner at {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_payload(path: Path) -> dict[str, Any]:
    """Load one result JSON object."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(dict[str, Any], payload)


def records_by_seed(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return result records keyed by seed."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("payload must contain non-empty records")
    keyed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("seed"), int):
            raise ValueError("all records must be JSON objects with integer seeds")
        seed = int(record["seed"])
        if seed in keyed:
            raise ValueError(f"duplicate seed {seed}")
        keyed[seed] = record
    return keyed


def validate_protocol_compatible(
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
) -> None:
    """Validate merge-critical protocol compatibility for two seed records."""
    if baseline_record.get("seed") != candidate_record.get("seed"):
        raise ValueError("baseline and candidate seed mismatch")
    baseline_dataset = baseline_record.get("dataset")
    candidate_dataset = candidate_record.get("dataset")
    if not isinstance(baseline_dataset, dict) or not isinstance(candidate_dataset, dict):
        raise ValueError("records must contain dataset metadata")
    for key in PROTOCOL_KEYS:
        if baseline_dataset.get(key) != candidate_dataset.get(key):
            raise ValueError(
                f"protocol mismatch for seed {baseline_record.get('seed')} "
                f"key {key}: {baseline_dataset.get(key)!r} "
                f"!= {candidate_dataset.get(key)!r}"
            )


def single_record_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return the only record from a candidate split payload."""
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"{path} must contain exactly one seed record")
    record = records[0]
    if not isinstance(record, dict):
        raise ValueError(f"{path} record must be a JSON object")
    return record


def manifest_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Return provenance for one source artifact."""
    manifest = payload.get("manifest")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest": manifest if isinstance(manifest, dict) else None,
    }


def combine_payload(
    baseline_payload: dict[str, Any],
    baseline_path: Path,
    candidate_payloads: Sequence[dict[str, Any]],
    candidate_paths: Sequence[Path],
    *,
    baseline_methods: Sequence[str],
    candidate_method: str,
) -> dict[str, Any]:
    """Build a strict mixed-method OPMNIST artifact."""
    runner = load_runner_module()
    baseline_records = records_by_seed(baseline_payload)
    candidate_records = [
        single_record_payload(payload, path)
        for payload, path in zip(candidate_payloads, candidate_paths, strict=True)
    ]
    seeds = sorted(int(record["seed"]) for record in candidate_records)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate candidate seeds: {seeds}")
    missing = [seed for seed in seeds if seed not in baseline_records]
    if missing:
        raise ValueError(f"baseline artifact lacks candidate seed(s): {missing}")

    records: list[dict[str, Any]] = []
    method_order = [*baseline_methods, candidate_method]
    for candidate_record in sorted(candidate_records, key=lambda row: int(row["seed"])):
        seed = int(candidate_record["seed"])
        baseline_record = baseline_records[seed]
        validate_protocol_compatible(baseline_record, candidate_record)
        baseline_methods_dict = baseline_record.get("methods")
        candidate_methods_dict = candidate_record.get("methods")
        if not isinstance(baseline_methods_dict, dict):
            raise ValueError(f"baseline seed {seed} lacks methods")
        if not isinstance(candidate_methods_dict, dict):
            raise ValueError(f"candidate seed {seed} lacks methods")
        missing_baselines = [
            method for method in baseline_methods if method not in baseline_methods_dict
        ]
        if missing_baselines:
            raise ValueError(
                f"baseline seed {seed} lacks method(s): {missing_baselines}"
            )
        if candidate_method not in candidate_methods_dict:
            raise ValueError(f"candidate seed {seed} lacks {candidate_method}")
        record = {
            "dataset_name": candidate_record.get(
                "dataset_name",
                baseline_record.get("dataset_name", "permuted_mnist_like"),
            ),
            "seed": seed,
            "dataset": dict(candidate_record["dataset"]),
            "methods": {
                **{method: baseline_methods_dict[method] for method in baseline_methods},
                candidate_method: candidate_methods_dict[candidate_method],
            },
        }
        records.append(record)

    aggregate = runner.aggregate_records(records)
    candidate_config = candidate_payloads[0].get("config")
    config = dict(candidate_config) if isinstance(candidate_config, dict) else {}
    config.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "combined_from_completed_baselines": True,
            "baseline_result_path": str(baseline_path),
            "candidate_result_paths": [str(path) for path in candidate_paths],
            "n_seeds": len(records),
            "seeds": seeds,
            "methods": method_order,
            "runner": "step2_upgd_memory_opmnist_mixed_method_combiner",
        }
    )
    datasets = {"permuted_mnist_like": dict(records[-1]["dataset"])}
    datasets["permuted_mnist_like"]["combined_from_completed_baselines"] = True
    datasets["permuted_mnist_like"]["baseline_result_path"] = str(baseline_path)
    datasets["permuted_mnist_like"]["candidate_result_paths"] = [
        str(path) for path in candidate_paths
    ]

    merged = {
        "config": config,
        "datasets": datasets,
        "records": records,
        "primary_method": getattr(runner, "PRIMARY_METHOD"),
        "mlp_methods": runner.mlp_method_names(method_order),
        "candidate_methods": runner.candidate_method_names(method_order),
        "aggregate": {"permuted_mnist_like": aggregate},
        "baseline_result_path": str(baseline_path),
        "candidate_result_paths": [str(path) for path in candidate_paths],
        "manifest": {
            "schema": "alberta.step2.upgd_memory_opmnist.mixed_method_manifest.v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "combine_script": str(Path(__file__).resolve()),
            "combine_script_sha256": sha256_file(Path(__file__).resolve()),
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "methods": method_order,
            "seeds": seeds,
            "baseline_methods": list(baseline_methods),
            "candidate_method": candidate_method,
            "baseline_artifact": manifest_row(baseline_path, baseline_payload),
            "candidate_split_results": [
                manifest_row(path, payload)
                for path, payload in zip(candidate_paths, candidate_payloads, strict=True)
            ],
        },
        "evidence_level": "mixed_completed_baselines_and_candidate_opmnist_seed_splits",
    }
    merged["solution_status"] = runner.opmnist_solution_status(merged)
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--candidate-method", type=str, required=True)
    parser.add_argument(
        "--baseline-methods",
        type=str,
        default="mlp_h128,mlp_h128_sharp",
        help="Comma-separated baseline method names to copy from baseline artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-summary", type=Path, default=None)
    parser.add_argument("candidate_results", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Combine artifacts and write the mixed-method result JSON."""
    args = parse_args(argv)
    baseline_methods = [
        method.strip()
        for method in args.baseline_methods.split(",")
        if method.strip()
    ]
    if not baseline_methods:
        raise ValueError("--baseline-methods must include at least one method")
    baseline_payload = load_payload(args.baseline_result)
    candidate_payloads = [load_payload(path) for path in args.candidate_results]
    combined = combine_payload(
        baseline_payload,
        args.baseline_result,
        candidate_payloads,
        args.candidate_results,
        baseline_methods=baseline_methods,
        candidate_method=args.candidate_method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_summary is not None:
        runner = load_runner_module()
        runner.write_note(combined, args.write_summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(combined["records"]),
                "methods": combined["manifest"]["methods"],
                "solution_status": combined["solution_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
