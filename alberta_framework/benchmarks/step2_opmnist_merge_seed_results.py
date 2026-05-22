#!/usr/bin/env python3
"""Merge split Step 2 OPMNIST seed result JSON files."""

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
        "step2_upgd_memory_opmnist_merge",
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


def extract_records(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten records from split result payloads."""
    records: list[dict[str, Any]] = []
    for payload in payloads:
        payload_records = payload.get("records")
        if not isinstance(payload_records, list) or not payload_records:
            raise ValueError("each input result must contain at least one record")
        for record in payload_records:
            if not isinstance(record, dict):
                raise ValueError("all input records must be JSON objects")
            records.append(record)
    return records


def validate_records(records: Sequence[dict[str, Any]], runner: Any) -> None:
    """Validate merge-critical seed, method, and protocol compatibility."""
    seeds = [record.get("seed") for record in records]
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("all records must have integer seed fields")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in split results: {seeds}")
    runner.method_names_from_records(list(records))
    reference_dataset = records[0].get("dataset")
    if not isinstance(reference_dataset, dict):
        raise ValueError("records must contain dataset metadata")
    for record in records[1:]:
        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("records must contain dataset metadata")
        for key in PROTOCOL_KEYS:
            if dataset.get(key) != reference_dataset.get(key):
                raise ValueError(
                    f"protocol mismatch for {key}: "
                    f"{dataset.get(key)!r} != {reference_dataset.get(key)!r}"
                )


def merged_config(
    payloads: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    input_paths: Sequence[Path],
) -> dict[str, Any]:
    """Create merged config metadata."""
    first_config = payloads[0].get("config")
    config = dict(first_config) if isinstance(first_config, dict) else {}
    seeds = sorted(int(record["seed"]) for record in records)
    reference_dataset = records[0]["dataset"]
    config.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "merged_from_seed_splits": True,
            "split_result_paths": [str(path) for path in input_paths],
            "n_seeds": len(seeds),
            "seeds": seeds,
            "mnist_source": config.get("mnist_source", "openml"),
            "steps": config.get("steps", reference_dataset.get("steps")),
            "n_permutations": config.get(
                "n_permutations",
                reference_dataset.get("n_permutations"),
            ),
            "task_block_size": config.get(
                "task_block_size",
                reference_dataset.get("task_block_size"),
            ),
        }
    )
    return config


def split_result_manifest(
    payloads: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    input_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    """Return per-seed provenance for split results consumed by the merge."""
    path_by_payload_index = [Path(path) for path in input_paths]
    manifest_rows: list[dict[str, Any]] = []
    for payload, path in zip(payloads, path_by_payload_index, strict=True):
        payload_records = payload.get("records")
        if not isinstance(payload_records, list) or not payload_records:
            raise ValueError("each input result must contain at least one record")
        manifest = payload.get("manifest")
        seed_values = [
            record.get("seed")
            for record in payload_records
            if isinstance(record, dict)
        ]
        manifest_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "seeds": seed_values,
                "manifest": manifest if isinstance(manifest, dict) else None,
            }
        )
    record_seeds = sorted(int(record["seed"]) for record in records)
    return sorted(
        manifest_rows,
        key=lambda row: min(int(seed) for seed in row["seeds"])
        if row["seeds"]
        else max(record_seeds, default=0) + 1,
    )


def merged_manifest(
    payloads: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    input_paths: Sequence[Path],
    method_names: Sequence[str],
) -> dict[str, Any]:
    """Build provenance metadata for the merged artifact."""
    split_results = split_result_manifest(payloads, records, input_paths)
    return {
        "schema": "alberta.step2.upgd_memory_opmnist.merge_manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "merge_script": str(Path(__file__).resolve()),
        "merge_script_sha256": sha256_file(Path(__file__).resolve()),
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "methods": list(method_names),
        "seeds": sorted(int(record["seed"]) for record in records),
        "split_results": split_results,
    }


def merge_payloads(
    payloads: Sequence[dict[str, Any]],
    input_paths: Sequence[Path],
) -> dict[str, Any]:
    """Merge compatible split OPMNIST results into one auditable payload."""
    if not payloads:
        raise ValueError("at least one input result is required")
    runner = load_runner_module()
    records = extract_records(payloads)
    validate_records(records, runner)
    records = sorted(records, key=lambda record: int(record["seed"]))
    method_names = runner.method_names_from_records(records)
    mlp_names = runner.mlp_method_names(method_names)
    aggregate = runner.aggregate_records(records)
    datasets = {"permuted_mnist_like": dict(records[-1]["dataset"])}
    datasets["permuted_mnist_like"]["merged_from_seed_splits"] = True
    datasets["permuted_mnist_like"]["split_result_paths"] = [
        str(path) for path in input_paths
    ]
    merged = {
        "config": merged_config(payloads, records, input_paths),
        "datasets": datasets,
        "records": records,
        "primary_method": getattr(runner, "PRIMARY_METHOD"),
        "mlp_methods": mlp_names,
        "candidate_methods": runner.candidate_method_names(method_names),
        "aggregate": {"permuted_mnist_like": aggregate},
        "split_results": [str(path) for path in input_paths],
        "manifest": merged_manifest(payloads, records, input_paths, method_names),
        "evidence_level": "merged_upgd_memory_opmnist_seed_splits",
    }
    merged["solution_status"] = runner.opmnist_solution_status(merged)
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        help="Optional Markdown summary path written with the runner's note writer.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Merge input result files and write the combined JSON."""
    args = parse_args(argv)
    payloads = [load_payload(path) for path in args.result_paths]
    merged = merge_payloads(payloads, args.result_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_summary is not None:
        runner = load_runner_module()
        runner.write_note(merged, args.write_summary)
    print(
        json.dumps(
            {"output": str(args.output), "solution_status": merged["solution_status"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
