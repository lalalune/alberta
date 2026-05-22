#!/usr/bin/env python3
"""Audit the remaining unchecked Alberta Plan TODO items.

This gate is intentionally stricter than the numbered step solution gate.  The
numbered Alberta Plan gates can all pass while external daemon or hardware
demonstrations remain unproven in this checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".")
TODO_PATTERN = re.compile(r"^- \[(?P<mark>[ xX])\] (?P<text>.+)$")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _todo_items(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    todos: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        match = TODO_PATTERN.match(line)
        if match:
            checked = match.group("mark").lower() == "x"
            todos.append(
                {"line": idx, "text": match.group("text"), "checked": checked}
            )
    return todos


def _unchecked_todos(path: Path) -> list[dict[str, Any]]:
    return [
        {"line": item["line"], "text": item["text"]}
        for item in _todo_items(path)
        if not item["checked"]
    ]


def _robot_or_sim_to_real_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if "robot" in item["text"].lower() or "sim-to-real" in item["text"].lower()
    ]


def _requires_real_robot_artifact(text: str) -> bool:
    lowered = text.lower()
    return "real robot" in lowered or "physical robot" in lowered


def _requires_sim_to_real_surrogate_artifact(text: str) -> bool:
    lowered = text.lower()
    return "sim-to-real" in lowered and "surrogate" in lowered


def audit_remaining_todos(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Return whether all TODO-listed Alberta Plan work is proven."""
    todo_path = root / "TODO.md"
    todo_items = _todo_items(todo_path)
    unchecked = [
        {"line": item["line"], "text": item["text"]}
        for item in todo_items
        if not item["checked"]
    ]
    master_gate_mod = _load_module(root / "benchmarks/alberta_plan_solution_gate.py")
    master_gate = master_gate_mod.run_alberta_plan_gate(root=root)
    external_audit = _load_json(root / "outputs/rlsecd_external_audit/status.json")
    external_acceptance_spec = _load_json(
        root / "outputs/rlsecd_external_acceptance/spec.json"
    )
    external_acceptance_status = _load_json(
        root / "outputs/rlsecd_external_acceptance/status.json"
    )
    security_rollout = _load_json(
        root / "outputs/security_gym_counterfactual_rollout/results.json"
    )
    oracle_export = _load_json(
        root / "outputs/security_gym_oracle_experience/manifest.json"
    )
    sim_to_real_transfer = _load_json(
        root / "outputs/prototype_sim_to_real_transfer/results.json"
    )
    real_robot_acceptance = _load_json(root / "outputs/real_robot_acceptance/status.json")
    sim_to_real_accepted = bool(
        sim_to_real_transfer
        and sim_to_real_transfer.get("accepted_sim_to_real_transfer")
    )
    real_robot_accepted = bool(
        real_robot_acceptance
        and real_robot_acceptance.get("schema")
        == "alberta.real_robot_acceptance_status.v1"
        and real_robot_acceptance.get("accepted_real_robot_demonstration") is True
    )
    prototype_evidence = {
        "roadmap_mentions_robot_boundary": "running in real time on a robot"
        in (root / "ROADMAP.md").read_text(encoding="utf-8"),
        "prototype_cartpole_evidence": (
            root / "outputs/prototype_end_to_end/results.json"
        ).exists(),
        "prototype_sim_to_real_transfer": {
            "path": "outputs/prototype_sim_to_real_transfer/results.json",
            "exists": sim_to_real_transfer is not None,
            "accepted": sim_to_real_accepted,
            "claim_scope": (
                sim_to_real_transfer.get("claim_scope")
                if sim_to_real_transfer
                else None
            ),
            "target_final_mean_reward": (
                sim_to_real_transfer.get("evidence", {}).get(
                    "target_final_mean_reward"
                )
                if sim_to_real_transfer
                else None
            ),
            "boundary": (
                sim_to_real_transfer.get("boundary")
                if sim_to_real_transfer
                else None
            ),
        },
    }
    external_items = [
        item for item in unchecked if item["text"].startswith("External:")
    ]
    unchecked_robot_items = _robot_or_sim_to_real_items(unchecked)
    checked_robot_items = _robot_or_sim_to_real_items(
        [item for item in todo_items if item["checked"]]
    )
    robot_items = list(unchecked_robot_items)
    for item in checked_robot_items:
        text = str(item["text"])
        if _requires_real_robot_artifact(text) and not real_robot_accepted:
            robot_items.append(
                {
                    "line": item["line"],
                    "text": item["text"],
                    "reason": "checked but missing accepted real robot acceptance artifact",
                }
            )
        elif _requires_sim_to_real_surrogate_artifact(text) and not sim_to_real_accepted:
            robot_items.append(
                {
                    "line": item["line"],
                    "text": item["text"],
                    "reason": "checked but missing accepted sim-to-real surrogate artifact",
                }
            )
    all_numbered_steps_accepted = bool(master_gate.get("all_steps_accepted"))
    all_todos_proven = not unchecked and not robot_items
    remaining_boundaries = []
    if external_items:
        remaining_boundaries.append("unchecked external rlsecd daemon items")
    if robot_items:
        remaining_boundaries.append("unproven robot/sim-to-real demonstration items")
    boundary = (
        "All numbered Steps 1-12 are accepted locally, but the full objective "
        f"still has {', '.join(remaining_boundaries)}."
        if remaining_boundaries
        else (
            "All numbered Steps 1-12 and TODO-listed local evidence items are "
            "accepted in this checkout."
        )
    )
    return {
        "schema": "alberta.plan.remaining_todo_gate.v1",
        "accepted_numbered_steps_1_to_12": all_numbered_steps_accepted,
        "all_todos_proven": all_todos_proven,
        "accepted_full_alberta_plan_objective": bool(
            all_numbered_steps_accepted and all_todos_proven
        ),
        "unchecked_todos": unchecked,
        "unproven_external_items": external_items,
        "unproven_robot_or_sim_to_real_items": robot_items,
        "evidence": {
            "master_gate_summary": {
                "all_steps_accepted": master_gate.get("all_steps_accepted"),
                "per_step_accepted": master_gate.get("per_step_accepted"),
            },
            "rlsecd_external_audit": {
                "path": "outputs/rlsecd_external_audit/status.json",
                "exists": external_audit is not None,
                "passed": external_audit.get("passed") if external_audit else None,
                "rlsecd_available": (
                    external_audit.get("rlsecd_available") if external_audit else None
                ),
                "missing_required_repos": (
                    external_audit.get("missing_required_repos")
                    if external_audit
                    else None
                ),
                "github_owner_candidates": (
                    external_audit.get("github_owner_candidates")
                    if external_audit
                    else None
                ),
            },
            "rlsecd_external_acceptance_spec": {
                "path": "outputs/rlsecd_external_acceptance/spec.json",
                "exists": external_acceptance_spec is not None,
                "schema": (
                    external_acceptance_spec.get("schema")
                    if external_acceptance_spec
                    else None
                ),
                "status": (
                    external_acceptance_spec.get("status")
                    if external_acceptance_spec
                    else None
                ),
                "n_items": (
                    len(external_acceptance_spec.get("items", []))
                    if external_acceptance_spec
                    else None
                ),
            },
            "rlsecd_external_acceptance_status": {
                "path": "outputs/rlsecd_external_acceptance/status.json",
                "exists": external_acceptance_status is not None,
                "schema": (
                    external_acceptance_status.get("schema")
                    if external_acceptance_status
                    else None
                ),
                "accepted": (
                    external_acceptance_status.get("accepted")
                    if external_acceptance_status
                    else None
                ),
                "n_passed": (
                    external_acceptance_status.get("n_passed")
                    if external_acceptance_status
                    else None
                ),
                "n_items": (
                    external_acceptance_status.get("n_items")
                    if external_acceptance_status
                    else None
                ),
                "boundary": (
                    external_acceptance_status.get("boundary")
                    if external_acceptance_status
                    else None
                ),
            },
            "security_gym_counterfactual_rollout": {
                "path": "outputs/security_gym_counterfactual_rollout/results.json",
                "exists": security_rollout is not None,
                "passed": (
                    security_rollout.get("passed") if security_rollout else None
                ),
                "boundary": (
                    security_rollout.get("boundary") if security_rollout else None
                ),
            },
            "security_gym_oracle_experience_export": {
                "path": "outputs/security_gym_oracle_experience/manifest.json",
                "exists": oracle_export is not None,
                "passed": oracle_export.get("passed") if oracle_export else None,
                "n_records": (
                    oracle_export.get("n_records") if oracle_export else None
                ),
                "records_path": (
                    oracle_export.get("records_path") if oracle_export else None
                ),
                "boundary": (
                    oracle_export.get("boundary") if oracle_export else None
                ),
            },
            "prototype_agent_evidence": prototype_evidence,
            "real_robot_acceptance": {
                "path": "outputs/real_robot_acceptance/status.json",
                "exists": real_robot_acceptance is not None,
                "accepted": real_robot_accepted,
                "schema": (
                    real_robot_acceptance.get("schema")
                    if real_robot_acceptance
                    else None
                ),
                "boundary": (
                    real_robot_acceptance.get("boundary")
                    if real_robot_acceptance
                    else None
                ),
            },
        },
        "boundary": boundary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write-status", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the remaining TODO audit."""
    args = parse_args(argv)
    status = audit_remaining_todos(args.root)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    return 0 if status["accepted_full_alberta_plan_objective"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
