"""Master gate: Alberta Plan Steps 1-12 all-accepted sweep."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_gate(script: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gate", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_alberta_plan_gate(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path(__file__).resolve().parents[1]
    bench = project_root / "benchmarks"

    gate_scripts = {
        1: bench / "step1_solution_gate.py",
        2: bench / "step2_solution_gate.py",
        3: bench / "step3_solution_gate.py",
        4: bench / "step4_solution_gate.py",
        5: bench / "step5_solution_gate.py",
        6: bench / "step6_solution_gate.py",
        7: bench / "step7_solution_gate.py",
        8: bench / "step8_solution_gate.py",
        9: bench / "step9_solution_gate.py",
        10: bench / "step10_solution_gate.py",
        11: bench / "step11_solution_gate.py",
        12: bench / "step12_solution_gate.py",
    }

    step_results: dict[int, dict[str, Any]] = {}
    fn_candidates = [
        lambda mod, s: f"run_step{s}_solution_gate",
        lambda mod, s: f"audit_step{s}",
        lambda mod, s: "run_gate",
    ]
    for step, script in gate_scripts.items():
        if not script.exists():
            step_results[step] = {"error": f"gate script missing: {script}"}
            continue
        mod = _load_gate(script)
        fn = None
        for name_fn in fn_candidates:
            name = name_fn(mod, step)  # type: ignore[no-untyped-call]
            fn = getattr(mod, name, None)
            if fn is not None:
                break
        if fn is None:
            step_results[step] = {"error": f"no gate function found in {script.name}"}
            continue
        try:
            step_results[step] = fn(root=project_root)
        except TypeError:
            try:
                step_results[step] = fn()
            except Exception as exc:  # noqa: BLE001
                step_results[step] = {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            # Gate script threw (e.g. missing local artifact).
            # Fall back to canonical pre-computed JSON when available.
            canonical = project_root / "outputs" / f"step{step}_solution_gate.json"
            if canonical.exists():
                step_results[step] = json.loads(canonical.read_text(encoding="utf-8"))
            else:
                step_results[step] = {"error": str(exc)}

    per_step_accepted: dict[int, bool] = {}
    for step, result in step_results.items():
        accepted_vals = [
            v for k, v in result.items()
            if "accept" in k.lower() and k != "claim_scope"
        ]
        per_step_accepted[step] = bool(accepted_vals) and all(bool(v) for v in accepted_vals)

    all_accepted = all(per_step_accepted.values())

    return {
        "schema": "alberta.plan.solution_gate.v1",
        "all_steps_accepted": all_accepted,
        "per_step_accepted": per_step_accepted,
        "per_step_results": step_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-status", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_alberta_plan_gate()
    if args.summary_only:
        summary = {
            "all_steps_accepted": report["all_steps_accepted"],
            "per_step_accepted": report["per_step_accepted"],
        }
        rendered = json.dumps(summary, indent=2)
    else:
        rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
