#!/usr/bin/env python3
"""D09: fixed-operator feature search for the Step 2 compositional blocker.

The main blocker left by D07 is ``synthetic_compositional``: the target is a
non-stationary two-layer tanh oracle, and arccosine/NNGP KRLS did not beat the
fair MLP.  This runner tests a narrower hypothesis:

* keep the nonlinear operator fixed at birth,
* update only a linear readout at every time step,
* use normalized LMS rather than a trained MLP trunk or a router.

The strongest candidate is a random tanh feature operator.  It is an online
extreme-learning-machine style learner: random nonlinear observables plus a
multi-head linear readout.  The ablations also include deeper random tanh
observables, random Fourier features, Chebyshev tensor products, and Volterra
products.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    evaluate_mlp_classifier,
    expand_dataset_names,
    is_higher_better,
    make_dataset,
    make_mlp,
    paired_diff,
    run_mlp_stream,
    stderr,
    summarize_prequential,
)
from step2_expert_mixture import DIGITS_REGIMES, N_DIGIT_CLASSES  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d09_compositional_operator_search")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d09_compositional_operator_search.md"
)


class OperatorState(NamedTuple):
    """State for fixed nonlinear observables plus an online linear readout."""

    output_weights: jax.Array
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    coord_idx: jax.Array
    degree_idx: jax.Array
    signs: jax.Array


class OperatorUpdateResult(NamedTuple):
    """Result for one fixed-operator online update."""

    state: OperatorState


@dataclass(frozen=True)
class OperatorSpec:
    """Configuration for one fixed nonlinear feature operator."""

    name: str
    kind: str
    width: int
    step_size: float
    weight_scale: float = 1.0
    inner_width: int = 0
    order: int = 3
    degree: int = 3
    input_scale: float = 2.0
    feature_clip: float = 3.0


class FixedOperatorLearner:
    """Normalized LMS over one fixed nonlinear feature map."""

    def __init__(self, n_heads: int, spec: OperatorSpec) -> None:
        self._n_heads = int(n_heads)
        self._spec = spec

    @property
    def spec(self) -> OperatorSpec:
        """Return the operator specification."""
        return self._spec

    def _feature_dim(self, raw_feature_dim: int) -> int:
        if self._spec.kind == "deep_tanh":
            return 1 + self._spec.inner_width + self._spec.width
        if self._spec.kind == "fourier":
            return 1 + 2 * self._spec.width
        return 1 + self._spec.width

    def init(self, feature_dim: int, key: jax.Array) -> OperatorState:
        """Initialize fixed random observables and zero output weights."""
        key_w1, key_b1, key_w2, key_b2, key_coord, key_degree, key_sign = jr.split(
            key,
            7,
        )
        width = int(self._spec.width)
        inner_width = int(max(self._spec.inner_width, 1))
        order = int(max(self._spec.order, 1))
        scale = self._spec.weight_scale / math.sqrt(max(feature_dim, 1))
        inner_scale = self._spec.weight_scale / math.sqrt(max(inner_width, 1))
        w1_rows = inner_width if self._spec.kind == "deep_tanh" else width
        w1 = scale * jr.normal(key_w1, (w1_rows, feature_dim), dtype=jnp.float32)
        b1 = jr.uniform(
            key_b1,
            (w1_rows,),
            dtype=jnp.float32,
            minval=-self._spec.weight_scale,
            maxval=self._spec.weight_scale,
        )
        w2 = inner_scale * jr.normal(
            key_w2,
            (width, inner_width),
            dtype=jnp.float32,
        )
        b2 = jr.uniform(
            key_b2,
            (width,),
            dtype=jnp.float32,
            minval=-self._spec.weight_scale,
            maxval=self._spec.weight_scale,
        )
        coord_idx = jr.randint(
            key_coord,
            (width, order),
            minval=0,
            maxval=feature_dim,
            dtype=jnp.int32,
        )
        degree_idx = jr.randint(
            key_degree,
            (width, order),
            minval=1,
            maxval=max(self._spec.degree, 1) + 1,
            dtype=jnp.int32,
        )
        signs = jr.choice(
            key_sign,
            jnp.asarray([-1.0, 1.0], dtype=jnp.float32),
            shape=(width,),
        )
        output_weights = jnp.zeros(
            (self._n_heads, self._feature_dim(feature_dim)),
            dtype=jnp.float32,
        )
        return OperatorState(
            output_weights=output_weights,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            coord_idx=coord_idx,
            degree_idx=degree_idx,
            signs=signs,
        )

    def _chebyshev_values(self, z: jax.Array, degree_idx: jax.Array) -> jax.Array:
        """Return Chebyshev ``T_n(z)`` for degrees one through five."""
        t1 = z
        t2 = 2.0 * z * z - 1.0
        t3 = 4.0 * z * z * z - 3.0 * z
        t4 = 8.0 * z**4 - 8.0 * z * z + 1.0
        t5 = 16.0 * z**5 - 20.0 * z**3 + 5.0 * z
        values = jnp.stack([t1, t2, t3, t4, t5], axis=0)
        clipped_degree = jnp.clip(degree_idx, 1, 5) - 1
        return jnp.take_along_axis(values, clipped_degree[None, ...], axis=0)[0]

    def _features(self, state: OperatorState, observation: jax.Array) -> jax.Array:
        """Compute the fixed nonlinear feature vector for one observation."""
        spec = self._spec
        if spec.kind == "tanh":
            hidden = jnp.tanh(state.w1 @ observation + state.b1)
            return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), hidden])
        if spec.kind == "deep_tanh":
            inner = jnp.tanh(state.w1 @ observation + state.b1)
            outer = jnp.tanh(state.w2 @ inner + state.b2)
            return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), inner, outer])
        if spec.kind == "fourier":
            projection = state.w1 @ observation + state.b1
            return jnp.concatenate(
                [
                    jnp.ones(1, dtype=jnp.float32),
                    jnp.sin(projection),
                    jnp.cos(projection),
                ]
            )
        if spec.kind == "cheb_tensor":
            z = jnp.tanh(observation / jnp.asarray(spec.input_scale, dtype=jnp.float32))
            selected = z[state.coord_idx]
            terms = self._chebyshev_values(selected, state.degree_idx)
            products = state.signs * jnp.prod(terms, axis=1)
            return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), products])
        if spec.kind == "volterra":
            z = jnp.clip(
                observation,
                -jnp.asarray(spec.feature_clip, dtype=jnp.float32),
                jnp.asarray(spec.feature_clip, dtype=jnp.float32),
            ) / jnp.asarray(spec.feature_clip, dtype=jnp.float32)
            selected = z[state.coord_idx]
            products = state.signs * jnp.prod(selected ** state.degree_idx, axis=1)
            return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), products])
        raise ValueError(f"unknown operator kind: {spec.kind}")

    def predict(self, state: OperatorState, observation: jax.Array) -> jax.Array:
        """Predict all heads from the current readout."""
        return state.output_weights @ self._features(state, observation)

    def update(
        self,
        state: OperatorState,
        observation: jax.Array,
        target: jax.Array,
    ) -> OperatorUpdateResult:
        """Apply one normalized LMS update to active target heads."""
        features = self._features(state, observation)
        prediction = state.output_weights @ features
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        errors = jnp.where(active, safe_target - prediction, 0.0)
        normalizer = 1.0 + jnp.sum(features * features)
        delta = (
            jnp.asarray(self._spec.step_size, dtype=jnp.float32)
            * errors[:, None]
            * features[None, :]
            / normalizer
        )
        return OperatorUpdateResult(
            state=OperatorState(
                output_weights=state.output_weights + delta,
                w1=state.w1,
                b1=state.b1,
                w2=state.w2,
                b2=state.b2,
                coord_idx=state.coord_idx,
                degree_idx=state.degree_idx,
                signs=state.signs,
            )
        )


def masked_mse_jax(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error over active heads."""
    active = ~jnp.isnan(target)
    safe_target = jnp.where(active, target, 0.0)
    active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(active, (prediction - safe_target) ** 2, 0.0)) / active_count


def run_operator_stream(
    learner: FixedOperatorLearner,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> tuple[OperatorState, np.ndarray]:
    """Run one fixed-operator learner and record prequential metrics."""
    state = learner.init(feature_dim=int(observations.shape[1]), key=key)

    def step_fn(
        carry: OperatorState,
        sample: tuple[jax.Array, jax.Array],
    ) -> tuple[OperatorState, jax.Array]:
        obs, tgt = sample
        pred = learner.predict(carry, obs)
        loss = masked_mse_jax(pred, tgt)
        result = learner.update(carry, obs, tgt)
        metric = jnp.asarray([loss, jnp.argmax(pred).astype(jnp.float32)])
        return result.state, metric

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics, dtype=np.float64)


def evaluate_operator_classifier(
    learner: FixedOperatorLearner,
    state: OperatorState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final fixed-operator classifier on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def method_name(spec: OperatorSpec) -> str:
    """Return a stable compact method name for one operator spec."""
    scale = f"{spec.weight_scale:g}".replace(".", "p")
    eta = f"{spec.step_size:g}".replace(".", "p")
    if spec.kind == "deep_tanh":
        return (
            f"deep_tanh_lms_i{spec.inner_width}_w{spec.width}_"
            f"eta{eta}_s{scale}"
        )
    if spec.kind in {"cheb_tensor", "volterra"}:
        return (
            f"{spec.kind}_lms_w{spec.width}_d{spec.degree}_o{spec.order}_"
            f"eta{eta}_s{scale}"
        )
    return f"{spec.kind}_lms_w{spec.width}_eta{eta}_s{scale}"


def stable_method_offset(name: str) -> int:
    """Return a deterministic small integer offset for PRNG splitting."""
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % 10_000


def all_candidate_specs() -> dict[str, OperatorSpec]:
    """Return the named operator ablation grid."""
    specs = [
        OperatorSpec("tanh128", "tanh", width=128, step_size=0.4),
        OperatorSpec("tanh256", "tanh", width=256, step_size=0.4),
        OperatorSpec("tanh512", "tanh", width=512, step_size=0.4),
        OperatorSpec("tanh256_eta02", "tanh", width=256, step_size=0.2),
        OperatorSpec("tanh256_eta06", "tanh", width=256, step_size=0.6),
        OperatorSpec("tanh256_scale05", "tanh", width=256, step_size=0.4, weight_scale=0.5),
        OperatorSpec("tanh256_scale2", "tanh", width=256, step_size=0.4, weight_scale=2.0),
        OperatorSpec("deep_tanh", "deep_tanh", width=256, inner_width=128, step_size=0.3),
        OperatorSpec("fourier", "fourier", width=256, step_size=0.3),
        OperatorSpec("cheb_tensor", "cheb_tensor", width=256, step_size=0.2, degree=5),
        OperatorSpec("volterra", "volterra", width=256, step_size=0.2, degree=3),
    ]
    return {method_name(spec): spec for spec in specs}


def default_method_names(preset: str) -> tuple[str, ...]:
    """Return method names for a preset."""
    specs = all_candidate_specs()
    if preset == "winner":
        return ("tanh_lms_w256_eta0p4_s1",)
    if preset == "focused":
        return (
            "tanh_lms_w128_eta0p4_s1",
            "tanh_lms_w256_eta0p4_s1",
            "tanh_lms_w512_eta0p4_s1",
            "tanh_lms_w256_eta0p2_s1",
            "tanh_lms_w256_eta0p6_s1",
            "tanh_lms_w256_eta0p4_s0p5",
            "tanh_lms_w256_eta0p4_s2",
            "deep_tanh_lms_i128_w256_eta0p3_s1",
            "fourier_lms_w256_eta0p3_s1",
            "cheb_tensor_lms_w256_d5_o3_eta0p2_s1",
            "volterra_lms_w256_d3_o3_eta0p2_s1",
        )
    if preset == "smoke":
        return ("tanh_lms_w32_eta0p4_s1",)
    if preset != "all":
        raise ValueError(f"unknown preset: {preset}")
    return tuple(specs)


def parse_method_specs(methods_spec: str, preset: str) -> list[OperatorSpec]:
    """Parse a comma-separated method list or preset into operator specs."""
    all_specs = all_candidate_specs()
    if preset == "smoke":
        smoke = OperatorSpec("tanh32", "tanh", width=32, step_size=0.4)
        all_specs[method_name(smoke)] = smoke
    method_names = (
        tuple(item.strip() for item in methods_spec.split(",") if item.strip())
        if methods_spec
        else default_method_names(preset)
    )
    unknown = sorted(set(method_names).difference(all_specs))
    if unknown:
        valid = ", ".join(sorted(all_specs))
        raise ValueError(f"unknown methods {unknown}; valid methods: {valid}")
    return [all_specs[name] for name in method_names]


def method_metric_keys(methods: dict[str, dict[str, float]]) -> list[str]:
    """Return all scalar metric keys present on any method."""
    keys: set[str] = set()
    for metrics in methods.values():
        keys.update(metrics)
    return sorted(keys)


def compare_to_group(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
    group: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one method against the per-seed best member of a baseline group."""
    diffs: list[float] = []
    best_methods: list[str] = []
    for record in records:
        methods = record["methods"]
        if metric not in methods[method]:
            continue
        group_values = {
            name: float(methods[name][metric])
            for name in group
            if name in methods and metric in methods[name]
        }
        if not group_values:
            continue
        if is_higher_better(metric):
            best_name = max(group_values, key=group_values.__getitem__)
        else:
            best_name = min(group_values, key=group_values.__getitem__)
        best_methods.append(best_name)
        diffs.append(
            paired_diff(float(methods[method][metric]), group_values[best_name], metric)
        )
    diff_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "method": method,
        "metric": metric,
        "paired_diff_mean_positive_favors_method": float(np.mean(diff_arr))
        if diff_arr.size
        else 0.0,
        "paired_diff_stderr": stderr(diff_arr) if diff_arr.size else 0.0,
        "wins_for_method": int(np.sum(diff_arr > 0.0)),
        "wins_for_baseline": int(np.sum(diff_arr < 0.0)),
        "ties": int(np.sum(diff_arr == 0.0)),
        "n": int(diff_arr.shape[0]),
        "diffs": diff_arr.tolist(),
        "best_baseline_counts": dict(
            sorted((name, best_methods.count(name)) for name in set(best_methods))
        ),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    candidate_methods: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate paired-seed records and add candidate-vs-MLP comparisons."""
    aggregate: dict[str, Any] = {}
    for dataset in sorted({record["dataset_name"] for record in records}):
        dataset_records = [r for r in records if r["dataset_name"] == dataset]
        method_names = list(dataset_records[0]["methods"])
        dataset_agg: dict[str, Any] = {}
        for method in method_names:
            metric_rows: dict[str, Any] = {}
            for metric in method_metric_keys(
                {method: dataset_records[0]["methods"][method]}
            ):
                values = np.asarray(
                    [
                        r["methods"][method][metric]
                        for r in dataset_records
                        if metric in r["methods"][method]
                    ],
                    dtype=np.float64,
                )
                if values.size:
                    metric_rows[metric] = {
                        "mean": float(np.mean(values)),
                        "stderr": stderr(values),
                        "values": values.tolist(),
                    }
            dataset_agg[method] = metric_rows
        primary_metrics = [
            metric
            for metric in (
                "final_window_mse",
                "online_mean_mse",
                "test_mse",
                "final_window_accuracy",
                "online_mean_accuracy",
                "test_accuracy",
            )
            if metric in dataset_records[0]["methods"][method_names[0]]
        ]
        comparisons: dict[str, Any] = {}
        for metric in primary_metrics:
            comparisons[metric] = {
                method: compare_to_group(dataset_records, method, metric, MLP_METHODS)
                for method in candidate_methods
            }
            best_candidate_by_seed: list[str] = []
            diffs: list[float] = []
            for record in dataset_records:
                methods = record["methods"]
                candidate_values = {
                    method: methods[method][metric]
                    for method in candidate_methods
                    if metric in methods[method]
                }
                if not candidate_values:
                    continue
                if is_higher_better(metric):
                    best_candidate = max(candidate_values, key=candidate_values.__getitem__)
                    best_mlp = max(MLP_METHODS, key=lambda name: methods[name][metric])
                else:
                    best_candidate = min(candidate_values, key=candidate_values.__getitem__)
                    best_mlp = min(MLP_METHODS, key=lambda name: methods[name][metric])
                best_candidate_by_seed.append(best_candidate)
                diffs.append(
                    paired_diff(
                        float(methods[best_candidate][metric]),
                        float(methods[best_mlp][metric]),
                        metric,
                    )
                )
            diff_arr = np.asarray(diffs, dtype=np.float64)
            comparisons[metric]["best_candidate_vs_best_mlp"] = {
                "paired_diff_mean_positive_favors_candidate": float(np.mean(diff_arr))
                if diff_arr.size
                else 0.0,
                "paired_diff_stderr": stderr(diff_arr) if diff_arr.size else 0.0,
                "wins_for_candidate": int(np.sum(diff_arr > 0.0)),
                "wins_for_mlp": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n": int(diff_arr.shape[0]),
                "diffs": diff_arr.tolist(),
                "best_candidate_counts": dict(
                    sorted(
                        (name, best_candidate_by_seed.count(name))
                        for name in set(best_candidate_by_seed)
                    )
                ),
            }
        dataset_agg["comparisons"] = comparisons
        aggregate[dataset] = dataset_agg
    return aggregate


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell for Markdown."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a detailed Markdown assessment."""
    cfg = results["config"]
    lines = [
        "# D09 Compositional Operator Search",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate preset="
            f"{cfg['candidate_preset']}; methods={', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "All D09 candidates are single online predictors. They do not consume MLP "
        "predictions, do not route over experts, and do not train a neural trunk. "
        "Each candidate uses a fixed nonlinear operator initialized at birth and "
        "a normalized-LMS readout updated every time step.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                (
                    "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                    "Runtime s | Readout dim |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in dataset_agg.items():
            if method == "comparisons":
                continue
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} | "
                f"{metric_cell(row, 'runtime_s')} | "
                f"{metric_cell(row, 'readout_dim')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_candidate_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_candidate']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_candidate']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-candidate counts {best['best_candidate_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_candidate_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_candidate']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_candidate']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Assessment",
            "",
            "The canonical signal to look for is not per-dataset selection. It is a "
            "fixed operator configuration, especially `tanh_lms_w256_eta0p4_s1`, "
            "beating the best fair MLP on the compositional blocker under paired "
            "seeds. Spillover benchmarks are diagnostic only: a stateful digits "
            "loss here means this operator should become one bank inside the "
            "larger resource-managed learner, not the whole Step 2 solution.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="synthetic_compositional")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--candidate-preset",
        choices=("focused", "winner", "all", "smoke"),
        default="focused",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated candidate method names. Empty uses --candidate-preset.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")
    if not 0.0 <= args.mlp_sparsity < 1.0:
        raise ValueError("--mlp-sparsity must be in [0, 1)")


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    candidate_specs: list[OperatorSpec],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all methods for one paired dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    for method in MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        learner = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            learner,
            observations,
            targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_mlp_classifier(learner, state, x_test, y_test))

    for spec in candidate_specs:
        name = method_name(spec)
        print(f"{dataset_name} seed={seed}: running {name}")
        operator = FixedOperatorLearner(n_heads=int(targets.shape[1]), spec=spec)
        t0 = time.time()
        state, metrics = run_operator_stream(
            operator,
            observations,
            targets,
            jr.key(seed + 70_000 + stable_method_offset(name)),
        )
        methods[name] = summarize_prequential(metrics, args.final_window, labels)
        methods[name].update(
            {
                "runtime_s": float(time.time() - t0),
                "readout_dim": float(state.output_weights.shape[1]),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[name].update(
                evaluate_operator_classifier(operator, state, x_test, y_test)
            )

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "methods": methods,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def main() -> None:
    """Run the D09 experiment and write JSON/Markdown outputs."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "synthetic_compositional"
        args.candidate_preset = "smoke"
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    candidate_specs = parse_method_specs(args.methods, args.candidate_preset)
    candidate_methods = tuple(method_name(spec) for spec in candidate_specs)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name,
                seed,
                candidate_specs,
                args,
            )
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta
    results = {
        "config": {
            "datasets": datasets,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "feature_dim": args.feature_dim,
            "noise_std": args.noise_std,
            "rare_period": args.rare_period,
            "train_fraction": args.train_fraction,
            "phase_length": args.phase_length,
            "mask_keep_fraction": args.mask_keep_fraction,
            "mask_noise_std": args.mask_noise_std,
            "mlp_step_size": args.mlp_step_size,
            "mlp_sparsity": args.mlp_sparsity,
            "candidate_preset": args.candidate_preset,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "standalone_fixed_operator_compositional_blocker_probe",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    if args.note_path:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()
