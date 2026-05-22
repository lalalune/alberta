#!/usr/bin/env python3
"""Shared-state temperature sweep for no-gate UPGD on OPMNIST.

The runner updates one ``UPGDLearner.step2_default(..., readout_mode="softmax_ce")``
state and scores fixed probability-temperature readouts in parallel. This keeps
the experiment a single learner with fixed deployment transforms, while avoiding
duplicated UPGD updates for each temperature.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step2_published_stressors as pub  # noqa: E402

from alberta_framework.core.upgd import UPGDLearner  # noqa: E402


def empty_accumulator(final_window: int) -> dict[str, Any]:
    """Create a JSON/pickle-friendly streaming accumulator."""
    return {
        "n_steps": 0,
        "loss_sum": 0.0,
        "correct_sum": 0.0,
        "final_losses": np.zeros(final_window, dtype=np.float64),
        "final_correct": np.zeros(final_window, dtype=np.float64),
    }


def update_accumulator(
    accumulator: dict[str, Any],
    losses: np.ndarray,
    correct: np.ndarray,
    final_window: int,
) -> dict[str, Any]:
    """Add one metrics chunk to a streaming accumulator."""
    losses = np.asarray(losses, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    previous_count = min(int(accumulator["n_steps"]), final_window)
    loss_tail = np.concatenate(
        [accumulator["final_losses"][:previous_count], losses]
    )[-final_window:]
    correct_tail = np.concatenate(
        [accumulator["final_correct"][:previous_count], correct]
    )[-final_window:]
    final_losses = np.zeros(final_window, dtype=np.float64)
    final_correct = np.zeros(final_window, dtype=np.float64)
    final_losses[: loss_tail.shape[0]] = loss_tail
    final_correct[: correct_tail.shape[0]] = correct_tail
    return {
        "n_steps": int(accumulator["n_steps"] + losses.shape[0]),
        "loss_sum": float(accumulator["loss_sum"] + np.sum(losses)),
        "correct_sum": float(accumulator["correct_sum"] + np.sum(correct)),
        "final_losses": final_losses,
        "final_correct": final_correct,
    }


def summarize_accumulator(accumulator: dict[str, Any], final_window: int) -> dict[str, float]:
    """Return online and final-window metrics."""
    n_steps = int(accumulator["n_steps"])
    if n_steps <= 0:
        raise RuntimeError("cannot summarize an empty accumulator")
    final_count = min(n_steps, final_window)
    return {
        "online_mean_mse": float(accumulator["loss_sum"] / n_steps),
        "online_mean_accuracy": float(accumulator["correct_sum"] / n_steps),
        "final_window_mse": float(np.mean(accumulator["final_losses"][:final_count])),
        "final_window_accuracy": float(
            np.mean(accumulator["final_correct"][:final_count])
        ),
    }


def temperature_probs(predictions: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a fixed probability-temperature transform."""
    if temperature == 1.0:
        return predictions
    powered = np.power(np.maximum(predictions, 1e-8), 1.0 / temperature)
    return powered / np.sum(powered, axis=1, keepdims=True)


class TemperaturePredictionWrapper:
    """Prediction-only wrapper used for held-out evaluation."""

    def __init__(self, base: UPGDLearner, temperature: float):
        self.base = base
        self.temperature = float(temperature)

    def predict(self, state: Any, observation: Any) -> Any:
        """Predict with fixed probability temperature."""
        prediction = self.base.predict(state, observation)
        if self.temperature == 1.0:
            return prediction
        powered = jnp.power(jnp.maximum(prediction, 1e-8), 1.0 / self.temperature)
        return powered / jnp.sum(powered)


@functools.partial(jax.jit, static_argnums=(0,))
def scan_raw_predictions(
    learner: UPGDLearner,
    state: Any,
    observations: Any,
    targets: Any,
) -> tuple[Any, Any]:
    """Update one base UPGD learner and return raw softmax predictions."""

    def step(carry: Any, batch: tuple[Any, Any]) -> tuple[Any, Any]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, result.predictions

    return jax.lax.scan(step, state, (observations, targets))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--final-window", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=60_000)
    parser.add_argument(
        "--opmnist-fraction",
        type=float,
        default=None,
        help=(
            "Run a complete-task fraction of the 800-task protocol. For example, "
            "0.01 runs eight 60,000-example permutation blocks."
        ),
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_upgd_memory_opmnist_single_upgd_full"),
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        default="softmax_h128_temperature_sweep_800task",
    )
    parser.add_argument("--force-restart", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run or resume the temperature sweep."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"{args.result_prefix}_resume.pkl"
    progress_path = args.output_dir / f"{args.result_prefix}_progress.jsonl"
    result_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    if args.force_restart:
        for path in (ckpt_path, progress_path, result_path, summary_path):
            if path.exists():
                path.unlink()

    dataset_args = argparse.Namespace(
        mnist_source="openml",
        mnist_split="canonical",
        allow_openml_download=args.allow_openml_download,
        allow_torchvision_download=False,
        openml_data_home=None,
        torchvision_data_home=None,
        openml_n_retries=2,
        openml_retry_delay=1.0,
        train_fraction=0.7,
        max_train_examples=None,
        max_test_examples=None,
    )
    seed = int(args.seed)
    n_permutations = pub.DOHARE_OPMNIST_TASKS
    task_block_size = pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
    steps = pub.DOHARE_OPMNIST_TOTAL_STEPS
    if args.opmnist_fraction is not None:
        if not 0.0 < args.opmnist_fraction <= 1.0:
            raise ValueError("--opmnist-fraction must be in (0, 1]")
        requested = int(round(pub.DOHARE_OPMNIST_TOTAL_STEPS * args.opmnist_fraction))
        steps = max(task_block_size, requested - requested % task_block_size)
    dataset = pub.load_mnist_like_source(dataset_args, seed)
    feature_dim = int(dataset.x_train.shape[1])
    stream_seed = seed + 10_000
    feature_orders = pub.make_feature_orders(
        seed=stream_seed,
        feature_dim=feature_dim,
        n_permutations=n_permutations,
        include_identity_permutation=False,
    )
    observed_task_ids = pub.opmnist_task_ids_for_steps(
        steps=steps,
        n_permutations=n_permutations,
        task_block_size=task_block_size,
    )
    test_task_ids = pub.opmnist_test_task_ids(
        observed_task_ids=observed_task_ids,
        n_permutations=n_permutations,
        max_test_permutation_views=800,
        evaluate_all_permutation_views=True,
    )
    learner = UPGDLearner.step2_default(
        n_heads=10,
        hidden_sizes=(args.hidden_size,),
        readout_mode="softmax_ce",
    )
    state = learner.init(feature_dim, jr.key(seed + 31_337))
    accumulators = {
        str(float(temp)): empty_accumulator(args.final_window)
        for temp in args.temperatures
    }
    completed_steps = 0
    elapsed_s = 0.0
    if ckpt_path.exists():
        payload = pickle.loads(ckpt_path.read_bytes())
        checkpoint_hidden_size = payload.get("hidden_size", 128)
        if int(checkpoint_hidden_size) != int(args.hidden_size):
            raise ValueError(
                "checkpoint hidden size does not match --hidden-size: "
                f"{checkpoint_hidden_size} != {args.hidden_size}"
            )
        state = payload["state"]
        accumulators = payload.get("accumulators", payload.get("accs"))
        if accumulators is None:
            raise KeyError("checkpoint is missing accumulators")
        feature_orders = payload["feature_orders"]
        completed_steps = int(payload["completed_steps"])
        elapsed_s = float(payload.get("elapsed_s", 0.0))
        print(f"temperature sweep resumed {completed_steps}/{steps} from {ckpt_path}")

    while completed_steps < steps:
        chunk_steps = min(args.chunk_size, steps - completed_steps)
        t0 = time.time()
        observations, targets, labels = pub.make_permuted_classification_chunk(
            dataset=dataset,
            start_step=completed_steps,
            chunk_steps=chunk_steps,
            seed=stream_seed,
            n_permutations=n_permutations,
            task_block_size=task_block_size,
            sample_with_replacement=False,
            task_sampling="sequential_epoch",
            feature_orders=feature_orders,
        )
        state, raw_predictions = scan_raw_predictions(
            learner,
            state,
            observations,
            targets,
        )
        raw_np = np.asarray(raw_predictions)
        targets_np = np.asarray(targets)
        labels_np = np.asarray(labels)
        for temp in args.temperatures:
            probs = temperature_probs(raw_np, float(temp))
            losses = np.mean((probs - targets_np) ** 2, axis=1)
            correct = (np.argmax(probs, axis=1) == labels_np).astype(np.float32)
            key = str(float(temp))
            accumulators[key] = update_accumulator(
                accumulators[key],
                losses,
                correct,
                args.final_window,
            )
        completed_steps += chunk_steps
        chunk_elapsed = time.time() - t0
        elapsed_s += chunk_elapsed
        progress_row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "completed_steps": int(completed_steps),
            "completed_full_task_blocks": int(completed_steps // task_block_size),
            "chunk_elapsed_s": float(chunk_elapsed),
            "steps_per_second": float(chunk_steps / max(chunk_elapsed, 1e-12)),
            "elapsed_s": float(elapsed_s),
        }
        ckpt_path.write_bytes(
            pickle.dumps(
                {
                    "state": state,
                    "accumulators": accumulators,
                    "feature_orders": feature_orders,
                    "completed_steps": completed_steps,
                    "elapsed_s": elapsed_s,
                    "temperatures": list(map(float, args.temperatures)),
                    "hidden_size": int(args.hidden_size),
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        with progress_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(progress_row, sort_keys=True) + "\n")
        print(
            f"temperature sweep streamed {completed_steps}/{steps} "
            f"({completed_steps // task_block_size} blocks, "
            f"{progress_row['steps_per_second']:.1f} steps/s)"
        )

    methods: dict[str, dict[str, float]] = {}
    for temp in args.temperatures:
        temp_f = float(temp)
        name = (
            f"upgd_structure_softmax_h{args.hidden_size}_"
            f"temp{str(temp_f).replace('.', 'p')}"
        )
        row = summarize_accumulator(accumulators[str(temp_f)], args.final_window)
        row.update(
            pub.evaluate_classifier_feature_orders(
                learner=TemperaturePredictionWrapper(learner, temp_f),
                state=state,
                dataset=dataset,
                feature_orders=feature_orders,
                test_task_ids=test_task_ids,
            )
        )
        methods[name] = row

    results = {
        "schema": "alberta.step2.opmnist.temperature_sweep.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_steps": int(completed_steps),
        "completed_full_task_blocks": int(completed_steps // task_block_size),
        "test_permutation_views": len(test_task_ids),
        "test_views_cover_all_permutations": set(test_task_ids)
        == set(range(n_permutations)),
        "matches_dohare_opmnist_core_protocol": True,
        "matches_dohare_opmnist_published_task_count": completed_steps == steps,
        "hidden_size": int(args.hidden_size),
        "temperatures": list(map(float, args.temperatures)),
        "methods": methods,
        "baseline_reference": (
            "outputs/step2_upgd_memory_opmnist_single_upgd_full/"
            "single_upgd_h128_800task_eval800views_results.json"
        ),
        "elapsed_s": float(elapsed_s),
    }
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# Softmax H{args.hidden_size} Temperature Sweep",
        "",
        f"- Completed blocks: `{completed_steps // task_block_size}`",
        f"- Test views: `{len(test_task_ids)}`",
        "",
        "| Method | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in methods.items():
        lines.append(
            f"| `{name}` | {row['online_mean_mse']:.6f} | "
            f"{row['online_mean_accuracy']:.6f} | "
            f"{row['final_window_mse']:.6f} | "
            f"{row['final_window_accuracy']:.6f} | "
            f"{row['test_mse']:.6f} | {row['test_accuracy']:.6f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result_path": str(result_path), "summary_path": str(summary_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
