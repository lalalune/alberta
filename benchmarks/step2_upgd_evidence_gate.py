#!/usr/bin/env python3
"""Evidence gate for the promoted Step 2 UPGD MLP-replacement claim.

This script is intentionally not another training benchmark.  It reads the
paper/deployment evidence artifacts and fails if the promoted
``UPGDLearner.step2_default`` claim is no longer supported by the recorded
results.  Use it as the final reproducibility guard before presenting or
shipping a new Step 2 default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.random as jr

from alberta_framework import UPGDLearner

PROMOTED_METHOD = "upgd_h32_structure_sigma1e_4_kappa05_rademacher_interval16_lean"
PROMOTED_EFFICIENCY_METHOD = "upgd32_structure_rademacher_lean_interval16"
MLP64_METHOD = "mlp64"


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: float | int | str
    threshold: float | int | str
    source: str
    detail: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    checks: list[GateCheck],
    *,
    name: str,
    passed: bool,
    observed: float | int | str,
    threshold: float | int | str,
    source: Path,
    detail: str,
) -> None:
    checks.append(
        GateCheck(
            name=name,
            passed=passed,
            observed=observed,
            threshold=threshold,
            source=str(source),
            detail=detail,
        )
    )


def _row_for_efficiency(
    efficiency: dict[str, Any],
    *,
    target_mode: str,
    method: str,
) -> dict[str, Any]:
    for row in efficiency["results"]:
        if row["target_mode"] == target_mode and row["name"] == method:
            return row
    msg = f"missing efficiency row target_mode={target_mode!r} method={method!r}"
    raise KeyError(msg)


def _validate_factory(checks: list[GateCheck]) -> None:
    source = Path("UPGDLearner.step2_default")
    cfg = UPGDLearner.step2_default(n_heads=10).to_config()
    expected = {
        "hidden_sizes": [32],
        "step_size": 0.03,
        "loss_normalization": "target_structure",
        "readout_mode": "linear_mse",
        "bounder": {"type": "ObGDBounding", "kappa": 0.5},
        "perturbation_sigma": 1e-4,
        "perturbation_noise": "rademacher",
        "perturbation_interval": 16,
        "track_unit_utilities": False,
        "track_gradient_history": False,
        "adaptive_kappa_mode": "none",
        "meta_plasticity_mode": "none",
    }
    for key, value in expected.items():
        _check(
            checks,
            name=f"factory.{key}",
            passed=cfg[key] == value,
            observed=str(cfg[key]),
            threshold=str(value),
            source=source,
            detail="Promoted factory config must stay frozen for reproducibility.",
        )

    state = UPGDLearner.step2_default(n_heads=10).init(
        feature_dim=64,
        key=jr.key(0),
    )
    _check(
        checks,
        name="factory.lean_unit_buffers",
        passed=len(state.unit_utilities) == 0 and state.unit_replacement_accumulators.shape == (0,),
        observed=f"{len(state.unit_utilities)} unit buffers",
        threshold="0 unit buffers",
        source=source,
        detail="Deployment default should not carry disabled unit-recycling state.",
    )
    _check(
        checks,
        name="factory.lean_gradient_history",
        passed=len(state.previous_trunk_weight_grads) == 0,
        observed=f"{len(state.previous_trunk_weight_grads)} gradient buffers",
        threshold="0 gradient buffers",
        source=source,
        detail="Deployment default should not carry disabled meta-gradient history.",
    )


def evaluate(
    *,
    synthetic_path: Path,
    digits_path: Path,
    efficiency_path: Path,
) -> dict[str, Any]:
    checks: list[GateCheck] = []
    _validate_factory(checks)

    synthetic = _load_json(synthetic_path)
    for stream in (
        "out_of_class_polynomial",
        "frequency_mismatch",
        "compositional",
    ):
        row = synthetic["paired_vs_best_mlp"][stream][
            "upgd_variant_structure_h32_rademacher_interval16_lean"
        ]
        diff = float(row["best_mlp_minus_method"])
        wins = int(row["wins"])
        n = int(row["n_seeds"])
        _check(
            checks,
            name=f"synthetic.{stream}.diff_positive",
            passed=diff > 0.0,
            observed=diff,
            threshold="> 0",
            source=synthetic_path,
            detail="Positive paired diff means UPGD beats the same-run best MLP.",
        )
        _check(
            checks,
            name=f"synthetic.{stream}.all_seed_wins",
            passed=wins == n,
            observed=wins,
            threshold=n,
            source=synthetic_path,
            detail="Paper claim uses all-seed wins on the three out-of-class streams.",
        )

    digits = _load_json(digits_path)
    overall = digits["aggregate"]["overall"][PROMOTED_METHOD]
    final_mse = overall["final_window_mse"]
    test_acc = overall["test_accuracy"]
    test_mse = overall["test_mse"]
    final_acc = overall["final_window_accuracy"]
    _check(
        checks,
        name="digits.final_window_mse.diff",
        passed=float(final_mse["mlp_minus_method_mean"]) >= 0.005,
        observed=float(final_mse["mlp_minus_method_mean"]),
        threshold=">= 0.005",
        source=digits_path,
        detail="Aggregate online supervised error should beat MLP64 by a material margin.",
    )
    _check(
        checks,
        name="digits.final_window_mse.wins",
        passed=int(final_mse["wins_for_method"]) >= 140,
        observed=int(final_mse["wins_for_method"]),
        threshold=">= 140 / 150",
        source=digits_path,
        detail="Final-window MSE should be robust over the 5-regime x 30-seed matrix.",
    )
    _check(
        checks,
        name="digits.test_accuracy.diff",
        passed=float(test_acc["method_minus_mlp_mean"]) >= 0.02,
        observed=float(test_acc["method_minus_mlp_mean"]),
        threshold=">= 0.02",
        source=digits_path,
        detail="Held-out digit accuracy should improve over MLP64, not just MSE.",
    )
    _check(
        checks,
        name="digits.test_accuracy.wins",
        passed=int(test_acc["wins_for_method"]) >= 130,
        observed=int(test_acc["wins_for_method"]),
        threshold=">= 130 / 150",
        source=digits_path,
        detail="Held-out accuracy should improve across most regime/seed cells.",
    )
    _check(
        checks,
        name="digits.test_mse.diff",
        passed=float(test_mse["mlp_minus_method_mean"]) >= 0.005,
        observed=float(test_mse["mlp_minus_method_mean"]),
        threshold=">= 0.005",
        source=digits_path,
        detail="Held-out MSE should remain positive under the promoted default.",
    )
    _check(
        checks,
        name="digits.final_window_accuracy.diff",
        passed=float(final_acc["method_minus_mlp_mean"]) > 0.0,
        observed=float(final_acc["method_minus_mlp_mean"]),
        threshold="> 0",
        source=digits_path,
        detail="Final-window classification accuracy should be positive in aggregate.",
    )

    class_blocked = digits["aggregate"]["by_regime"]["class_blocked"][PROMOTED_METHOD]
    class_fw_mse = class_blocked["final_window_mse"]
    class_test_acc = class_blocked["test_accuracy"]
    class_fw_acc = class_blocked["final_window_accuracy"]
    _check(
        checks,
        name="digits.class_blocked.final_window_mse",
        passed=float(class_fw_mse["mlp_minus_method_mean"]) > 0.0,
        observed=float(class_fw_mse["mlp_minus_method_mean"]),
        threshold="> 0",
        source=digits_path,
        detail="Class-blocked is the known stress row; MSE still has to clear MLP64.",
    )
    _check(
        checks,
        name="digits.class_blocked.test_accuracy",
        passed=float(class_test_acc["method_minus_mlp_mean"]) > 0.0,
        observed=float(class_test_acc["method_minus_mlp_mean"]),
        threshold="> 0",
        source=digits_path,
        detail="Class-blocked held-out accuracy should remain positive, even if low.",
    )
    _check(
        checks,
        name="digits.class_blocked.final_window_accuracy_caveat",
        passed=float(class_fw_acc["method_minus_mlp_mean"]) >= -0.005,
        observed=float(class_fw_acc["method_minus_mlp_mean"]),
        threshold=">= -0.005",
        source=digits_path,
        detail="Known caveat: class-blocked tracking accuracy may be slightly negative.",
    )

    efficiency = _load_json(efficiency_path)
    for target_mode, min_ratio in (("onehot", 1.1), ("dense", 1.1)):
        upgd = _row_for_efficiency(
            efficiency,
            target_mode=target_mode,
            method=PROMOTED_EFFICIENCY_METHOD,
        )
        mlp = _row_for_efficiency(
            efficiency,
            target_mode=target_mode,
            method=MLP64_METHOD,
        )
        ratio = float(upgd["steps_per_second"]) / float(mlp["steps_per_second"])
        _check(
            checks,
            name=f"efficiency.{target_mode}.speed_ratio",
            passed=ratio >= min_ratio,
            observed=ratio,
            threshold=f">= {min_ratio}",
            source=efficiency_path,
            detail="Deployment claim requires the promoted width-32 branch to beat MLP64 speed.",
        )
        _check(
            checks,
            name=f"efficiency.{target_mode}.trainable_params",
            passed=int(upgd["trainable_params"]) <= int(mlp["trainable_params"]),
            observed=int(upgd["trainable_params"]),
            threshold=f"<= {int(mlp['trainable_params'])}",
            source=efficiency_path,
            detail="Resource-efficient branch should use no more trainable parameters than MLP64.",
        )
        _check(
            checks,
            name=f"efficiency.{target_mode}.float_state",
            passed=int(upgd["float_state_size"]) <= int(mlp["float_state_size"]),
            observed=int(upgd["float_state_size"]),
            threshold=f"<= {int(mlp['float_state_size'])}",
            source=efficiency_path,
            detail="Runtime state should stay below the MLP64 baseline in the recorded benchmark.",
        )

    passed = all(check.passed for check in checks)
    return {
        "passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(not check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Step 2 UPGD Evidence Gate",
        "",
        f"Overall status: **{'PASS' if payload['passed'] else 'FAIL'}**.",
        f"Checks: {payload['n_checks']} total, {payload['n_failed']} failed.",
        "",
        "| Status | Check | Observed | Threshold | Detail |",
        "|---|---|---:|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            "| {status} | `{name}` | {observed} | {threshold} | {detail} |".format(
                status="PASS" if check["passed"] else "FAIL",
                **check,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-results",
        type=Path,
        default=Path(
            "output/subagents/compute_efficiency/"
            "small_rademacher_synthetic_30seed_6000/out_of_class_results.json"
        ),
    )
    parser.add_argument(
        "--digits-results",
        type=Path,
        default=Path(
            "output/subagents/compute_efficiency/"
            "small_rademacher_digits_30seed_h64baseline/upgd_digits_sweep_results.json"
        ),
    )
    parser.add_argument(
        "--efficiency-results",
        type=Path,
        default=Path(
            "output/benchmarks/step2_upgd_efficiency_fused_heads_4096/"
            "efficiency_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmarks/step2_upgd_evidence_gate"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(
        synthetic_path=args.synthetic_results,
        digits_path=args.digits_results,
        efficiency_path=args.efficiency_results,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "evidence_gate_results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
