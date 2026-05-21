#!/usr/bin/env python3
"""Exp01 Step 2 moonshot: replicator-mutator feature population.

This is a deliberately small online supervised prototype.  A population of
feature recipes feeds a linear multi-head readout.  Each feature receives a
multiplicative fitness update from its recent ablation loss reduction.  The
three revisions are:

* Rev A: selection only over a fixed feature population.
* Rev B: selection plus replacement of low-fitness features by mutated or
  composed variants of high-fitness features.
* Rev C: Rev B plus a small elite recipe archive that can be reintroduced
  after drift/loss spikes or context-boundary recurrence.

The benchmark reuses the Step 2 paired-seed convention: each method consumes
the same stream realization for a seed, and methods are compared by online
mean MSE plus final-window MSE against the better fair MLP baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
)

DEFAULT_OUTPUT_DIR = Path("output/moonshots/exp01_replicator_mutator")
STEP_SIZE = 0.03
SPARSITY = 0.5
OBGD_KAPPA = 2.0


@dataclass(frozen=True)
class MethodConfig:
    """Metadata for baseline methods."""

    name: str
    method_type: str
    hidden_sizes: tuple[int, ...]
    step_size: float = STEP_SIZE
    sparsity: float = SPARSITY
    use_layer_norm: bool = True
    perturbation_sigma: float | None = None
    utility_decay: float | None = None
    perturbation_beta: float | None = None
    perturbation_interval: int | None = None


BASELINES: tuple[MethodConfig, ...] = (
    MethodConfig("mlp64", "mlp", (64,)),
    MethodConfig("mlp64_64", "mlp", (64, 64)),
    MethodConfig(
        "upgd64_sigma3e_4",
        "upgd",
        (64,),
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
)


@dataclass(frozen=True)
class Recipe:
    """A hashable feature recipe tree."""

    op: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class ReplicatorConfig:
    """Configuration for one revision of the feature population."""

    name: str
    population_size: int = 96
    readout_lr: float = 0.01
    bias_lr: float = 0.003
    weight_decay: float = 1e-5
    utility_decay: float = 0.97
    norm_decay: float = 0.995
    selection_eta: float = 0.08
    max_gate: float = 3.0
    min_gate: float = 0.15
    mutation_enabled: bool = False
    elite_memory_enabled: bool = False
    mutation_interval: int = 80
    replacement_fraction: float = 0.04
    mutation_burn_in: int = 300
    archive_size: int = 16
    context_length_hint: int = 300
    seed_triples: bool = True


REVISIONS: tuple[ReplicatorConfig, ...] = (
    ReplicatorConfig(name="rev_a_selection"),
    ReplicatorConfig(
        name="rev_b_mutation",
        mutation_enabled=True,
    ),
    ReplicatorConfig(
        name="rev_c_elite_memory",
        mutation_enabled=True,
        elite_memory_enabled=True,
    ),
)


def method_key(seed: int, method_name: str) -> jax.Array:
    """Return a deterministic JAX PRNG key for one seed/method."""
    checksum = zlib.crc32(method_name.encode("utf-8")) & 0x7FFFFFFF
    return jr.fold_in(jr.key(seed), checksum)


def recipe_to_json(recipe: Recipe) -> Any:
    """Serialize a recursive recipe into JSON-safe data."""
    values: list[Any] = []
    for arg in recipe.args:
        values.append(recipe_to_json(arg) if isinstance(arg, Recipe) else arg)
    return {"op": recipe.op, "args": values}


def recipe_key(recipe: Recipe) -> str:
    """Stable key for recipe de-duplication."""
    return json.dumps(recipe_to_json(recipe), sort_keys=True)


def eval_recipe(recipe: Recipe, x: np.ndarray, cache: dict[Recipe, float]) -> float:
    """Evaluate a recipe on one observation."""
    if recipe in cache:
        return cache[recipe]
    op = recipe.op
    args = recipe.args
    if op == "raw":
        value = float(x[int(args[0])])
    elif op == "square":
        value = float(x[int(args[0])] ** 2)
    elif op == "pair":
        value = float(x[int(args[0])] * x[int(args[1])])
    elif op == "triple":
        value = float(x[int(args[0])] * x[int(args[1])] * x[int(args[2])])
    elif op == "prod":
        value = eval_recipe(args[0], x, cache) * eval_recipe(args[1], x, cache)
    elif op == "tanh":
        value = float(np.tanh(eval_recipe(args[0], x, cache)))
    elif op == "sin":
        idx, omega, phase = args
        value = float(np.sin(float(omega) * x[int(idx)] + float(phase)))
    else:
        raise ValueError(f"unknown recipe op: {op}")
    if not np.isfinite(value):
        value = 0.0
    cache[recipe] = float(np.clip(value, -20.0, 20.0))
    return cache[recipe]


def all_triples(feature_dim: int) -> list[Recipe]:
    """Enumerate strict triple-product recipes."""
    return [
        Recipe("triple", (i, j, k))
        for i in range(feature_dim)
        for j in range(i + 1, feature_dim)
        for k in range(j + 1, feature_dim)
    ]


def initial_recipes(
    feature_dim: int,
    population_size: int,
    rng: np.random.Generator,
    seed_triples: bool,
) -> list[Recipe]:
    """Create the initial mixed recipe population."""
    recipes: list[Recipe] = [Recipe("raw", (i,)) for i in range(feature_dim)]
    recipes.extend(Recipe("square", (i,)) for i in range(feature_dim))
    recipes.extend(
        Recipe("pair", (i, j)) for i in range(feature_dim) for j in range(i + 1, feature_dim)
    )
    if seed_triples:
        triples = all_triples(feature_dim)
        rng.shuffle(triples)
        recipes.extend(triples)
    while len(recipes) < population_size:
        idx = int(rng.integers(feature_dim))
        omega = float(rng.choice([0.5, 1.0, 2.0, 3.0, 5.0]))
        phase = float(rng.uniform(-math.pi, math.pi))
        recipes.append(Recipe("sin", (idx, omega, phase)))
    return unique_prefix(recipes, population_size)


def unique_prefix(recipes: Sequence[Recipe], n: int) -> list[Recipe]:
    """Keep the first n unique recipes."""
    out: list[Recipe] = []
    seen: set[str] = set()
    for recipe in recipes:
        key = recipe_key(recipe)
        if key not in seen:
            seen.add(key)
            out.append(recipe)
        if len(out) == n:
            return out
    return out


def random_recipe(feature_dim: int, rng: np.random.Generator) -> Recipe:
    """Sample one grammar recipe."""
    op = str(rng.choice(["raw", "square", "pair", "triple", "sin"]))
    if op == "raw":
        return Recipe("raw", (int(rng.integers(feature_dim)),))
    if op == "square":
        return Recipe("square", (int(rng.integers(feature_dim)),))
    if op == "pair":
        i, j = sorted(rng.choice(feature_dim, size=2, replace=False).tolist())
        return Recipe("pair", (int(i), int(j)))
    if op == "triple":
        i, j, k = sorted(rng.choice(feature_dim, size=3, replace=False).tolist())
        return Recipe("triple", (int(i), int(j), int(k)))
    idx = int(rng.integers(feature_dim))
    return Recipe(
        "sin",
        (
            idx,
            float(rng.choice([0.5, 1.0, 2.0, 3.0, 5.0])),
            float(rng.uniform(-math.pi, math.pi)),
        ),
    )


def mutate_recipe(
    parent: Recipe,
    feature_dim: int,
    rng: np.random.Generator,
) -> Recipe:
    """Create a mutated/composed child of a high-fitness parent."""
    op = str(rng.choice(["prod_raw", "tanh", "sin_near", "random"], p=[0.45, 0.2, 0.2, 0.15]))
    if op == "prod_raw":
        raw = Recipe("raw", (int(rng.integers(feature_dim)),))
        if parent.op == "raw":
            i = int(parent.args[0])
            j = int(raw.args[0])
            if i != j:
                return Recipe("pair", tuple(sorted((i, j))))
        if parent.op == "pair":
            vals = {*parent.args, int(raw.args[0])}
            if len(vals) == 3:
                return Recipe("triple", tuple(sorted(int(v) for v in vals)))
        return Recipe("prod", (parent, raw))
    if op == "tanh":
        return Recipe("tanh", (parent,))
    if op == "sin_near":
        if parent.op == "sin":
            idx, omega, phase = parent.args
            return Recipe(
                "sin",
                (
                    int(idx),
                    float(max(0.1, float(omega) + rng.normal(0.0, 0.25))),
                    float(float(phase) + rng.normal(0.0, 0.4)),
                ),
            )
        return Recipe("tanh", (parent,))
    return random_recipe(feature_dim, rng)


def feature_values(recipes: Sequence[Recipe], x: np.ndarray) -> np.ndarray:
    """Evaluate all features for one observation."""
    cache: dict[Recipe, float] = {}
    return np.asarray([eval_recipe(recipe, x, cache) for recipe in recipes], dtype=np.float64)


def active_error(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return active mask and prediction error."""
    active = np.isfinite(target)
    safe_target = np.where(active, target, 0.0)
    err = np.where(active, safe_target - pred, 0.0)
    return active, err


def run_replicator(
    config: ReplicatorConfig,
    observations: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Run one replicator-mutator revision on a materialized stream."""
    rng = np.random.default_rng(seed)
    num_steps, feature_dim = observations.shape
    n_tasks = targets.shape[1]
    recipes = initial_recipes(
        feature_dim,
        config.population_size,
        rng,
        seed_triples=config.seed_triples,
    )
    population = len(recipes)
    weights = 0.01 * rng.normal(size=(population, n_tasks))
    bias = np.zeros(n_tasks, dtype=np.float64)
    log_fitness = np.zeros(population, dtype=np.float64)
    utility = np.zeros(population, dtype=np.float64)
    ages = np.zeros(population, dtype=np.int64)
    feat_mean = np.zeros(population, dtype=np.float64)
    feat_var = np.ones(population, dtype=np.float64)
    archive: dict[str, dict[str, Any]] = {}
    loss_curve = np.zeros(num_steps, dtype=np.float64)
    replacement_count = 0
    archive_reintroductions = 0
    loss_fast = 0.0
    loss_slow = 0.0

    for step in range(num_steps):
        x = observations[step].astype(np.float64)
        target = targets[step].astype(np.float64)
        raw_vals = feature_values(recipes, x)

        delta = raw_vals - feat_mean
        feat_mean += (1.0 - config.norm_decay) * delta
        feat_var = config.norm_decay * feat_var + (1.0 - config.norm_decay) * np.square(
            raw_vals - feat_mean
        )
        z = (raw_vals - feat_mean) / np.sqrt(feat_var + 1e-6)
        z = np.clip(z, -5.0, 5.0)
        gates = np.exp(np.clip(log_fitness - np.mean(log_fitness), -4.0, 4.0))
        gates = gates / max(np.mean(gates), 1e-8)
        gates = np.clip(gates, config.min_gate, config.max_gate)
        phi = z * gates

        pred = bias + phi @ weights
        active, err = active_error(pred, target)
        active_count = max(float(np.sum(active)), 1.0)
        loss = float(np.sum(np.square(err)) / active_count)
        loss_curve[step] = loss

        contrib = phi[:, None] * weights
        active_contrib = contrib[:, active]
        active_err = err[active]
        loss_reduction = np.mean(
            2.0 * active_contrib * active_err[None, :] + np.square(active_contrib),
            axis=1,
        )
        utility = config.utility_decay * utility + (1.0 - config.utility_decay) * loss_reduction
        centered = utility - np.median(utility)
        log_fitness += config.selection_eta * np.clip(centered, -1.0, 1.0)
        log_fitness = np.clip(log_fitness, -5.0, 5.0)

        scale = 1.0 / active_count
        weights[:, active] += config.readout_lr * np.outer(phi, err[active]) * scale
        weights *= 1.0 - config.weight_decay
        bias[active] += config.bias_lr * err[active] * scale
        ages += 1

        loss_fast = 0.97 * loss_fast + 0.03 * loss
        loss_slow = 0.995 * loss_slow + 0.005 * loss
        drift_signal = step > 100 and loss_fast > 1.20 * max(loss_slow, 1e-8)
        recurrence_signal = (
            config.context_length_hint > 0 and step > 0 and step % config.context_length_hint == 0
        )

        if config.elite_memory_enabled and (
            recurrence_signal or step % (2 * config.mutation_interval) == 0
        ):
            elite_count = min(config.archive_size, max(1, population // 8))
            for idx in np.argsort(-utility)[:elite_count]:
                key = recipe_key(recipes[int(idx)])
                old_score = float(archive.get(key, {}).get("score", -np.inf))
                score = float(utility[int(idx)])
                if score > old_score:
                    archive[key] = {
                        "recipe": recipes[int(idx)],
                        "score": score,
                        "step": int(step),
                    }
            if len(archive) > config.archive_size:
                keep = sorted(
                    archive.items(),
                    key=lambda item: item[1]["score"],
                    reverse=True,
                )[: config.archive_size]
                archive = dict(keep)

        should_mutate = (
            config.mutation_enabled
            and step >= config.mutation_burn_in
            and step % config.mutation_interval == 0
        )
        if should_mutate:
            n_replace = max(1, int(round(config.replacement_fraction * population)))
            replace_indices = np.argsort(log_fitness + 0.01 * np.minimum(ages, 1000))[:n_replace]
            parent_pool = np.argsort(log_fitness)[-max(4, population // 4) :]
            existing = {recipe_key(recipe) for recipe in recipes}
            archive_items = sorted(
                archive.values(),
                key=lambda item: item["score"],
                reverse=True,
            )
            for rank, idx_raw in enumerate(replace_indices):
                idx = int(idx_raw)
                use_archive = (
                    config.elite_memory_enabled
                    and archive_items
                    and (drift_signal or recurrence_signal or rank % 3 == 0)
                )
                if use_archive:
                    child = archive_items[rank % len(archive_items)]["recipe"]
                    archive_reintroductions += 1
                else:
                    parent = recipes[int(rng.choice(parent_pool))]
                    child = mutate_recipe(parent, feature_dim, rng)
                tries = 0
                while recipe_key(child) in existing and tries < 12:
                    parent = recipes[int(rng.choice(parent_pool))]
                    child = mutate_recipe(parent, feature_dim, rng)
                    tries += 1
                existing.discard(recipe_key(recipes[idx]))
                existing.add(recipe_key(child))
                recipes[idx] = child
                weights[idx, :] = 0.01 * rng.normal(size=n_tasks)
                log_fitness[idx] = float(np.median(log_fitness))
                utility[idx] = 0.0
                ages[idx] = 0
                feat_mean[idx] = 0.0
                feat_var[idx] = 1.0
                replacement_count += 1

    top_indices = np.argsort(-utility)[:10]
    top_features = [
        {
            "rank": int(rank + 1),
            "utility": float(utility[int(idx)]),
            "log_fitness": float(log_fitness[int(idx)]),
            "recipe": recipe_to_json(recipes[int(idx)]),
        }
        for rank, idx in enumerate(top_indices)
    ]
    return {
        "metrics": loss_curve,
        "diagnostics": {
            "replacement_count": int(replacement_count),
            "archive_size": int(len(archive)),
            "archive_reintroductions": int(archive_reintroductions),
            "top_features": top_features,
        },
    }


def make_mlp(config: MethodConfig, n_heads: int) -> MultiHeadMLPLearner:
    """Create a Step 2 fair MLP comparator."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=config.hidden_sizes,
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=config.sparsity,
        use_layer_norm=config.use_layer_norm,
    )


def make_upgd(config: MethodConfig, n_heads: int) -> UPGDLearner:
    """Create the single UPGD comparator."""
    if config.perturbation_sigma is None:
        raise ValueError("UPGD config missing perturbation_sigma")
    return UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=config.hidden_sizes,
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=config.sparsity,
        use_layer_norm=config.use_layer_norm,
        perturbation_sigma=config.perturbation_sigma,
        utility_decay=float(config.utility_decay),
        perturbation_beta=float(config.perturbation_beta),
        perturbation_interval=int(config.perturbation_interval),
    )


def active_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error over active target heads."""
    active = ~jnp.isnan(targets)
    safe_targets = jnp.where(active, targets, 0.0)
    squared_error = jnp.where(active, (predictions - safe_targets) ** 2, 0.0)
    denom = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return jnp.sum(squared_error) / denom


def run_online_regression(
    learner: Any,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
) -> np.ndarray:
    """Run one JAX learner and return a per-step MSE curve."""
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        obs, tgt = inputs
        result = learner.update(carry, obs, tgt)
        return result.state, active_mse(result.predictions, tgt)

    _, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    return np.asarray(metrics, dtype=np.float64)


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a Step 2 stream into arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    observations.block_until_ready()
    targets.block_until_ready()
    return observations, targets


def stream_factories(context_length: int) -> dict[str, tuple[Any, int, int]]:
    """Return supported out-of-class Step 2 streams."""
    return {
        "polynomial": (
            lambda: OutOfClassPolynomialStream(
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=context_length,
                active_triples_per_context=2,
                noise_std=0.05,
            ),
            8,
            3,
        ),
        "frequency": (
            lambda: FrequencyMismatchStream(
                feature_dim=4,
                n_tasks=2,
                n_components_per_task=3,
                n_contexts=4,
                context_length=context_length,
                noise_std=0.05,
            ),
            4,
            2,
        ),
    }


def summarize_curve(curve: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize one loss curve."""
    window = min(final_window, curve.shape[0])
    return {
        "online_mean_mse": float(np.mean(curve)),
        "final_window_mse": float(np.mean(curve[-window:])),
        "first_half_mse": float(np.mean(curve[: curve.shape[0] // 2])),
        "second_half_mse": float(np.mean(curve[curve.shape[0] // 2 :])),
    }


def stderr(values: Sequence[float]) -> float:
    """Sample standard error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by scenario and method."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], record["method"])].append(record)

    aggregate: dict[str, dict[str, Any]] = defaultdict(dict)
    for (scenario, method), group in sorted(grouped.items()):
        metrics = sorted(group[0]["metrics"])
        row: dict[str, Any] = {"n_seeds": len(group)}
        for metric in metrics:
            vals = [float(record["metrics"][metric]) for record in group]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_stderr"] = stderr(vals)
            row[f"{metric}_per_seed"] = vals
        aggregate[scenario][method] = row
    return dict(aggregate)


def paired_vs_best_mlp(
    records: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pair every method against the better fair MLP by final-window MSE."""
    scenario_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_records[record["scenario"]].append(record)

    out: dict[str, dict[str, Any]] = {}
    for scenario, group in sorted(scenario_records.items()):
        best_mlp = min(
            ("mlp64", "mlp64_64"),
            key=lambda name: aggregate[scenario][name]["final_window_mse_mean"],
        )
        by_key = {(record["method"], int(record["seed"])): record for record in group}
        seeds = sorted(int(record["seed"]) for record in group if record["method"] == best_mlp)
        scenario_out: dict[str, Any] = {"_best_mlp": best_mlp}
        for method in sorted({record["method"] for record in group}):
            if method == best_mlp:
                continue
            diffs = []
            for seed in seeds:
                base = by_key[(best_mlp, seed)]["metrics"]["final_window_mse"]
                val = by_key[(method, seed)]["metrics"]["final_window_mse"]
                diffs.append(float(base - val))
            arr = np.asarray(diffs, dtype=np.float64)
            sd = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0
            scenario_out[method] = {
                "metric": "final_window_mse",
                "positive_means_method_beats_best_mlp": True,
                "mean_diff": float(np.mean(arr)),
                "stderr": stderr(diffs),
                "wins": int(np.sum(arr > 0.0)),
                "losses": int(np.sum(arr < 0.0)),
                "ties": int(np.sum(arr == 0.0)),
                "n_seeds": int(arr.shape[0]),
                "cohens_d": float(np.mean(arr) / sd) if sd > 0.0 else 0.0,
                "diffs_per_seed": diffs,
            }
        out[scenario] = scenario_out
    return out


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated CLI argument."""
    return [part.strip() for part in value.split(",") if part.strip()]


def method_config_json(config: MethodConfig) -> dict[str, Any]:
    """Serialize baseline config."""
    data = asdict(config)
    data["hidden_sizes"] = list(config.hidden_sizes)
    return data


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run all streams, baselines, and three feature-population revisions."""
    requested = parse_csv(args.streams)
    factories = stream_factories(args.context_length)
    unknown = sorted(set(requested) - set(factories))
    if unknown:
        raise ValueError(f"unknown streams: {unknown}")

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    t0 = time.time()

    for stream_name in requested:
        factory, feature_dim, n_tasks = factories[stream_name]
        scenario = f"synthetic_{stream_name}"
        print(f"\n=== {scenario}: seeds={args.n_seeds}, steps={args.steps} ===")
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            observations, targets = collect_stream_arrays(
                factory(),
                args.steps,
                method_key(seed, f"{stream_name}_stream"),
            )
            obs_np = np.asarray(observations, dtype=np.float64)
            tgt_np = np.asarray(targets, dtype=np.float64)

            for config in BASELINES:
                learner = (
                    make_mlp(config, n_tasks)
                    if config.method_type == "mlp"
                    else make_upgd(config, n_tasks)
                )
                print(f"  seed={seed} method={config.name}")
                curve = run_online_regression(
                    learner,
                    method_key(seed, f"{stream_name}_{config.name}"),
                    observations,
                    targets,
                )
                records.append(
                    {
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": config.name,
                        "method_family": config.method_type,
                        "method_config": method_config_json(config),
                        "metrics": summarize_curve(curve, args.final_window),
                        "loss_curve": curve.tolist() if args.save_curves else None,
                    }
                )

            for revision in REVISIONS:
                revision_kwargs = {
                    **asdict(revision),
                    "population_size": args.population_size,
                    "context_length_hint": args.context_length,
                    "seed_triples": not args.no_seed_triples,
                }
                if args.replacement_fraction is not None:
                    revision_kwargs["replacement_fraction"] = args.replacement_fraction
                if args.selection_eta is not None:
                    revision_kwargs["selection_eta"] = args.selection_eta
                if args.mutation_interval is not None:
                    revision_kwargs["mutation_interval"] = args.mutation_interval
                revision = ReplicatorConfig(**revision_kwargs)
                print(f"  seed={seed} method={revision.name}")
                result = run_replicator(
                    revision,
                    obs_np,
                    tgt_np,
                    seed=seed + zlib.crc32(f"{stream_name}_{revision.name}".encode()),
                )
                curve = result["metrics"]
                records.append(
                    {
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": revision.name,
                        "method_family": "replicator_mutator",
                        "method_config": asdict(revision),
                        "metrics": summarize_curve(curve, args.final_window),
                        "loss_curve": curve.tolist() if args.save_curves else None,
                    }
                )
                diagnostics.append(
                    {
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": revision.name,
                        **result["diagnostics"],
                    }
                )

    aggregate = aggregate_records(records)
    paired = paired_vs_best_mlp(records, aggregate)
    return {
        "experiment": "exp01_replicator_mutator",
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "streams": requested,
            "context_length": args.context_length,
            "population_size": args.population_size,
            "save_curves": args.save_curves,
            "revisions": [asdict(rev) for rev in REVISIONS],
            "baselines": [method_config_json(cfg) for cfg in BASELINES],
        },
        "records": records,
        "aggregate": aggregate,
        "paired_vs_best_mlp": {"final_window_mse": paired},
        "diagnostics": diagnostics,
        "wall_clock_s": float(time.time() - t0),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write flat per-run metric records."""
    fieldnames = [
        "scenario",
        "stream",
        "seed",
        "method",
        "method_family",
        "online_mean_mse",
        "final_window_mse",
        "first_half_mse",
        "second_half_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "scenario": record["scenario"],
                "stream": record["stream"],
                "seed": record["seed"],
                "method": record["method"],
                "method_family": record["method_family"],
            }
            row.update(record["metrics"])
            writer.writerow(row)


def fmt(row: dict[str, Any], metric: str) -> str:
    """Format aggregate mean +/- stderr."""
    return f"{row[f'{metric}_mean']:.6f} +/- {row[f'{metric}_stderr']:.6f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown summary in the output directory."""
    aggregate = results["aggregate"]
    paired = results["paired_vs_best_mlp"]["final_window_mse"]
    lines = [
        "# Exp01 Replicator-Mutator",
        "",
        f"Wall clock: {results['wall_clock_s']:.1f}s.",
        (
            f"Seeds: {results['config']['n_seeds']}; "
            f"steps: {results['config']['steps']}; "
            f"final window: {results['config']['final_window']}."
        ),
        "",
        "Positive paired differences mean lower final-window MSE than the best fair MLP.",
        "",
    ]
    for scenario, by_method in sorted(aggregate.items()):
        best_mlp = paired[scenario]["_best_mlp"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"Best fair MLP: `{best_mlp}`.",
                "",
                "| Method | Final-window MSE | Online mean MSE | Paired diff | Wins |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        ordered = sorted(by_method, key=lambda method: by_method[method]["final_window_mse_mean"])
        for method in ordered:
            row = by_method[method]
            pair = paired[scenario].get(method)
            diff = f"{pair['mean_diff']:+.6f}" if pair is not None else ""
            wins = f"{pair['wins']}/{pair['n_seeds']}" if pair is not None else ""
            lines.append(
                f"| `{method}` | {fmt(row, 'final_window_mse')} | "
                f"{fmt(row, 'online_mean_mse')} | {diff} | {wins} |"
            )
        winners = [
            method
            for method in ("rev_a_selection", "rev_b_mutation", "rev_c_elite_memory")
            if paired[scenario].get(method, {}).get("mean_diff", -np.inf) > 0.0
            and paired[scenario].get(method, {}).get("wins", 0)
            > paired[scenario].get(method, {}).get("losses", 0)
        ]
        lines.extend(["", f"Replicator revisions beating best MLP: {winners or 'none'}.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=450)
    parser.add_argument("--context-length", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=96)
    parser.add_argument("--streams", default="polynomial,frequency")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-curves", action="store_true")
    parser.add_argument("--no-seed-triples", action="store_true")
    parser.add_argument("--replacement-fraction", type=float, default=None)
    parser.add_argument("--selection-eta", type=float, default=None)
    parser.add_argument("--mutation-interval", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run experiment and write artifacts."""
    args = parse_args()
    if args.steps <= 0 or args.n_seeds <= 0 or args.final_window <= 0:
        raise ValueError("steps, n-seeds, and final-window must be positive")
    if args.population_size < 16:
        raise ValueError("population-size must be at least 16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(args)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "records.csv"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(csv_path, results["records"])
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
