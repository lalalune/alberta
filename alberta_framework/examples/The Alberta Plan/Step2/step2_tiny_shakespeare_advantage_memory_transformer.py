#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare advantage-gated memory transformer.

This runner tests a resource manager whose gate is updated from measured
fast-vs-slow loss advantage instead of backpropagating the current token loss
through a diagnostic gate. If memory reduces loss, the gate opens; if memory
hurts, it closes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_tiny_shakespeare_proto_basis_transformer import (
    init_hybrid_transformer,
    make_proto_block,
    reset_value_row_if_novel,
    select_center_slot,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    TINY_SHAKESPEARE_URL,
    causal_attention_sequence,
    clip_grads,
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    encode_text,
    ensure_tiny_shakespeare,
    eval_transformer,
    ffn_transform,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    stderr,
    summarize_online,
    transformer_logits,
)

from alberta_framework.core.prototype_basis import (
    PrototypeBasisBlock,
    PrototypeBasisParams,
    PrototypeBasisState,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured in result artifacts."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    proto_count: int
    eval_steps: int
    eval_batch_size: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    fast_lr: float
    slow_lr: float
    grad_clip: float
    proto_update_rate: float
    proto_novelty_threshold: float
    proto_bandwidth: float
    proto_adaptive_bandwidth: bool
    proto_bandwidth_update_rate: float
    gate_init_logit: float
    gate_lr: float
    gate_decay: float
    gate_max: float
    advantage_margin: float
    gate_l2: float
    gate_mode: str
    gate_objective: str
    replay_size: int
    train_loss_mode: str
    memory_loss_weight: float
    reset_mode: str
    data_path: str
    output_dir: str
    seed: int


@chex.dataclass(frozen=True)
class AdvantageMemoryState:
    """Slow memory state plus resource-manager traces."""

    proto_state: PrototypeBasisState
    gate_logit: jax.Array
    advantage_ema: jax.Array
    init_value: jax.Array
    step_count: jax.Array


def sha256_text(value: str) -> str:
    """Return a sha256 digest for manifest strings."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_manifest(path: Path) -> dict[str, Any]:
    """Return path, byte count, and sha256 with failure details if unavailable."""
    resolved = path.expanduser().resolve()
    manifest: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": False,
        "bytes": None,
        "sha256": None,
    }
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                byte_count += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        return manifest
    manifest.update(
        {
            "exists": True,
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
        }
    )
    return manifest


def command_manifest(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run a provenance command without raising on missing tools or failures."""
    manifest: dict[str, Any] = {
        "command": command,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        manifest["stderr"] = f"{type(exc).__name__}: {exc}"
        return manifest
    manifest.update(
        {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    )
    return manifest


def command_stdout(result: dict[str, Any]) -> str | None:
    """Extract successful command stdout."""
    if result.get("ok") and result.get("stdout"):
        return str(result["stdout"])
    return None


def git_manifest(cwd: Path) -> dict[str, Any]:
    """Capture git commit, branch, and dirty status with safe fallbacks."""
    commands = {
        "root": command_manifest(["git", "rev-parse", "--show-toplevel"], cwd=cwd),
        "commit": command_manifest(["git", "rev-parse", "HEAD"], cwd=cwd),
        "branch": command_manifest(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
        ),
        "describe": command_manifest(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=cwd,
        ),
        "status_short": command_manifest(["git", "status", "--short"], cwd=cwd),
    }
    status_result = commands["status_short"]
    status = str(status_result["stdout"]) if status_result.get("ok") else None
    return {
        "root": command_stdout(commands["root"]),
        "commit": command_stdout(commands["commit"]),
        "branch": command_stdout(commands["branch"]),
        "describe": command_stdout(commands["describe"]),
        "dirty": None if status is None else bool(status),
        "status_short": None if status is None else status.splitlines(),
        "status_short_sha256": None if status is None else sha256_text(status),
        "commands": commands,
    }


def package_version(package: str) -> str | None:
    """Return an installed package version without failing manifest creation."""
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive manifest fallback.
        return f"unavailable: {type(exc).__name__}: {exc}"


def device_manifest() -> dict[str, Any]:
    """Capture JAX backend and devices without making runs fail."""
    manifest: dict[str, Any] = {
        "default_backend": None,
        "process_index": None,
        "process_count": None,
        "local_device_count": None,
        "device_count": None,
        "devices": [],
    }
    try:
        devices = jax.devices()
        manifest.update(
            {
                "default_backend": jax.default_backend(),
                "process_index": jax.process_index(),
                "process_count": jax.process_count(),
                "local_device_count": jax.local_device_count(),
                "device_count": jax.device_count(),
                "devices": [
                    {
                        "id": int(getattr(device, "id", -1)),
                        "platform": getattr(device, "platform", None),
                        "device_kind": getattr(device, "device_kind", str(device)),
                        "process_index": int(getattr(device, "process_index", -1)),
                    }
                    for device in devices
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - backend-specific fallback.
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    return manifest


def environment_manifest() -> dict[str, Any]:
    """Capture platform, Python, package, and selected runtime settings."""
    uname = platform.uname()
    selected_env = {
        name: os_value
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "JAX_ENABLE_X64",
            "JAX_PLATFORM_NAME",
            "PYTHONPATH",
            "TPU_VISIBLE_DEVICES",
            "XLA_FLAGS",
        )
        if (os_value := os_environ_get(name)) is not None
    }
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": {
                "system": uname.system,
                "node": uname.node,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
                "processor": uname.processor,
            },
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "jax": getattr(jax, "__version__", None),
            "jaxlib": package_version("jaxlib"),
            "numpy": getattr(np, "__version__", None),
            "chex": package_version("chex"),
        },
        "jax": device_manifest(),
        "environment_variables": selected_env,
    }


def os_environ_get(name: str) -> str | None:
    """Small wrapper to keep environment capture easy to test and type-check."""
    return os.environ.get(name)


def json_safe(value: Any) -> Any:
    """Convert argparse/config values into JSON-safe manifest data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def key_manifest(key: jax.Array) -> list[int]:
    """Return explicit JAX PRNG key data for reproducible seed derivation."""
    return [int(item) for item in np.asarray(jr.key_data(key)).reshape(-1)]


def source_manifest() -> dict[str, Any]:
    """Capture hashes for this runner and the local helper modules it imports."""
    runner = Path(__file__).resolve()
    helper_dir = runner.parent
    return {
        "source_file": file_manifest(runner),
        "local_dependencies": {
            "step2_tiny_shakespeare_proto_basis_transformer": file_manifest(
                helper_dir / "step2_tiny_shakespeare_proto_basis_transformer.py",
            ),
            "step2_tiny_shakespeare_upgd_ffn_transformer": file_manifest(
                helper_dir / "step2_tiny_shakespeare_upgd_ffn_transformer.py",
            ),
        },
    }


def init_advantage_params_state(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    gate_init_logit: float,
    gate_mode: str,
) -> tuple[dict[str, Any], AdvantageMemoryState]:
    """Initialize transformer params and advantage-gated memory state."""
    params, proto_state = init_hybrid_transformer(
        key,
        block=block,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
    )
    if gate_mode == "prototype":
        gate_logit = jnp.full(
            (block.config.n_prototypes,),
            gate_init_logit,
            dtype=jnp.float32,
        )
    else:
        gate_logit = jnp.asarray(gate_init_logit, dtype=jnp.float32)
    state = AdvantageMemoryState(
        proto_state=proto_state,
        gate_logit=gate_logit,
        advantage_ema=jnp.asarray(0.0, dtype=jnp.float32),
        init_value=jnp.zeros((d_model,), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    return params, state


def model_parts(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    context: jax.Array,
    *,
    placement: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return base logits, memory logits, basis input, activations, and gate."""
    attn_hidden = causal_attention_sequence(params["attn"], context)[-1]
    if placement == "pre_ffn_kv":
        basis_input = attn_hidden
        base_hidden = ffn_transform(params["ffn"], basis_input)
    else:
        basis_input = ffn_transform(params["ffn"], attn_hidden)
        base_hidden = basis_input
    activations = block.activations(state.proto_state, basis_input)
    gate = jax.nn.sigmoid(state.gate_logit)
    if state.gate_logit.ndim == 0:
        residual = gate * block.transform(params["proto"], activations)
    else:
        proto_params = params["proto"]
        gated_activations = activations * gate
        bias_scale = jnp.sum(gated_activations)
        residual = gated_activations @ proto_params.values + bias_scale * proto_params.bias
    if placement == "pre_ffn_kv":
        memory_hidden = ffn_transform(params["ffn"], basis_input + residual)
    else:
        memory_hidden = basis_input + residual
    readout = params["readout"]
    base_logits = base_hidden @ readout["w"] + readout["b"]
    memory_logits = memory_hidden @ readout["w"] + readout["b"]
    return base_logits, memory_logits, basis_input, activations, gate


def advantage_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    context: jax.Array,
    *,
    placement: str,
) -> jax.Array:
    """Return memory-gated logits for evaluation."""
    return model_parts(block, params, state, context, placement=placement)[1]


def prediction_advantage(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    context: jax.Array,
    label: jax.Array,
    *,
    placement: str,
) -> jax.Array:
    """Return fast-minus-memory loss advantage for one example."""
    base_logits, memory_logits, _, _, _ = model_parts(
        block,
        params,
        state,
        context,
        placement=placement,
    )
    return cross_entropy_from_logits(base_logits, label) - cross_entropy_from_logits(
        memory_logits,
        label,
    )


def sgd_step_decoupled(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    fast_lr: float,
    slow_lr: float,
) -> dict[str, Any]:
    """Apply separate rates to fast transformer and slow memory values."""

    def step_tree(tree: Any, grad_tree: Any, lr: float) -> Any:
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, tree, grad_tree)

    return {
        "attn": step_tree(params["attn"], grads["attn"], fast_lr),
        "ffn": step_tree(params["ffn"], grads["ffn"], fast_lr),
        "readout": step_tree(params["readout"], grads["readout"], fast_lr),
        "proto": step_tree(params["proto"], grads["proto"], slow_lr),
    }


def reset_proto_row(
    params: dict[str, Any],
    state: AdvantageMemoryState,
    slot: jax.Array,
    novel: jax.Array,
    *,
    reset_mode: str,
) -> dict[str, Any]:
    """Reset replaced prototype value rows."""
    if reset_mode == "none":
        return params
    if reset_mode == "zero":
        return reset_value_row_if_novel(params, slot, novel)
    proto_params = params["proto"]
    row = proto_params.values[slot]
    new_row = jnp.where(novel, state.init_value, row)
    new_proto = PrototypeBasisParams(
        values=proto_params.values.at[slot].set(new_row),
        bias=proto_params.bias,
    )
    return {**params, "proto": new_proto}


def run_advantage_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    gate_max: float,
    advantage_margin: float,
    gate_l2: float,
    gate_init_logit: float,
    gate_objective: str,
    replay_size: int,
    train_loss_mode: str,
    memory_loss_weight: float,
    reset_mode: str,
) -> tuple[dict[str, Any], AdvantageMemoryState, np.ndarray]:
    """Train one advantage-gated memory transformer variant."""
    replay_capacity = max(1, replay_size)

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: AdvantageMemoryState,
    ) -> tuple[tuple[Any, ...], jax.Array]:
        def step(
            carry: tuple[Any, ...],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[Any, ...], jax.Array]:
            params, state, replay_contexts, replay_labels, replay_count, replay_index = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[
                jax.Array,
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                base_logits, memory_logits, basis_input, activations, gate = model_parts(
                    block,
                    candidate,
                    state,
                    context,
                    placement=placement,
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                memory_loss = cross_entropy_from_logits(memory_logits, label)
                if train_loss_mode == "memory":
                    train_loss = memory_loss
                else:
                    weight = jnp.asarray(memory_loss_weight, dtype=jnp.float32)
                    train_loss = (1.0 - weight) * base_loss + weight * memory_loss
                return train_loss, (base_loss, memory_logits, basis_input, activations, gate)

            (_train_loss, (base_loss, logits, basis_input, activations, gate)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(params)
            )
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled(
                params,
                grads,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )
            memory_loss = cross_entropy_from_logits(logits, label)
            advantage = base_loss - memory_loss
            slot, novel = select_center_slot(block, state.proto_state, basis_input)
            new_proto_state, center_metrics = block.update_centers(
                state.proto_state,
                basis_input,
            )
            new_params = reset_proto_row(
                new_params,
                state,
                slot,
                novel,
                reset_mode=reset_mode,
            )
            if gate_objective == "replay":
                replay_slot = jnp.mod(replay_index, replay_capacity)
                replay_state = AdvantageMemoryState(
                    proto_state=new_proto_state,
                    gate_logit=state.gate_logit,
                    advantage_ema=state.advantage_ema,
                    init_value=state.init_value,
                    step_count=state.step_count,
                )
                replay_advantage = prediction_advantage(
                    block,
                    new_params,
                    replay_state,
                    replay_contexts[replay_slot],
                    replay_labels[replay_slot],
                    placement=placement,
                )
                gate_update_advantage = jnp.where(
                    replay_count > 0,
                    replay_advantage,
                    advantage,
                )
            else:
                gate_update_advantage = advantage
            gate_signal = gate_update_advantage - jnp.asarray(
                advantage_margin,
                dtype=jnp.float32,
            )
            if state.gate_logit.ndim == 0:
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                new_gate_logit = (
                    jnp.asarray(gate_decay, dtype=jnp.float32) * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
            else:
                credit = activations / jnp.maximum(jnp.max(activations), 1e-6)
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                active_decay = 1.0 - credit * (
                    1.0 - jnp.asarray(gate_decay, dtype=jnp.float32)
                )
                new_gate_logit = (
                    active_decay * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * credit
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
                slots = jnp.arange(state.gate_logit.shape[0], dtype=slot.dtype)
                new_gate_logit = jnp.where(
                    (slots == slot) & novel,
                    jnp.asarray(gate_init_logit, dtype=jnp.float32),
                    new_gate_logit,
                )
            max_logit = jnp.log(
                jnp.asarray(gate_max, dtype=jnp.float32)
                / jnp.maximum(1.0 - jnp.asarray(gate_max, dtype=jnp.float32), 1e-6)
            )
            new_gate_logit = jnp.clip(new_gate_logit, -8.0, jnp.minimum(max_logit, 8.0))
            residual = block.transform(new_params["proto"], activations)
            improved = advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * residual,
                state.init_value,
            )
            new_state = AdvantageMemoryState(
                proto_state=new_proto_state,
                gate_logit=new_gate_logit,
                advantage_ema=0.99 * state.advantage_ema + 0.01 * advantage,
                init_value=init_value,
                step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
            )
            write_slot = jnp.mod(replay_index, replay_capacity)
            new_replay_contexts = replay_contexts.at[write_slot].set(context)
            new_replay_labels = replay_labels.at[write_slot].set(label)
            new_replay_count = jnp.minimum(
                replay_count + jnp.array(1, dtype=jnp.int32),
                jnp.array(replay_capacity, dtype=jnp.int32),
            )
            new_replay_index = jnp.mod(
                replay_index + jnp.array(1, dtype=jnp.int32),
                jnp.array(replay_capacity, dtype=jnp.int32),
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    memory_loss,
                    acc,
                    base_loss,
                    advantage,
                    jnp.mean(gate),
                    jnp.mean(new_gate_logit),
                    center_metrics[0],
                    center_metrics[1],
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    gate_update_advantage,
                ]
            )
            return (
                new_params,
                new_state,
                new_replay_contexts,
                new_replay_labels,
                new_replay_count,
                new_replay_index,
            ), metrics

        replay_contexts = jnp.zeros(
            (replay_capacity, contexts.shape[1]),
            dtype=contexts.dtype,
        )
        replay_labels = jnp.zeros((replay_capacity,), dtype=labels.dtype)
        initial = (
            params,
            state,
            replay_contexts,
            replay_labels,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(step, initial, (contexts, labels))

    (final_params, final_state, *_), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_advantage_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
    eval_batch_size: int = 0,
) -> dict[str, float]:
    """Evaluate advantage-gated memory transformer."""
    if should_stream_eval(labels, eval_batch_size):
        return eval_logits_batched(
            contexts,
            labels,
            eval_batch_size=eval_batch_size,
            logits_for_context=lambda ctx: advantage_logits(
                block,
                params,
                state,
                ctx,
                placement=placement,
            ),
            nll_key="eval_nll",
            accuracy_key="eval_accuracy",
            perplexity_key="eval_perplexity",
        )

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(
            lambda ctx: advantage_logits(block, params, state, ctx, placement=placement)
        )(contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, labels)
        acc = jnp.argmax(logits, axis=1) == labels
        return losses, acc.astype(jnp.float32)

    losses, acc = run()
    losses.block_until_ready()
    mean_loss = float(jnp.mean(losses))
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": float(jnp.mean(acc)),
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def eval_advantage_fast_only(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
    eval_batch_size: int = 0,
) -> dict[str, float]:
    """Evaluate the fast branch after memory-regularized training."""
    if should_stream_eval(labels, eval_batch_size):
        return eval_logits_batched(
            contexts,
            labels,
            eval_batch_size=eval_batch_size,
            logits_for_context=lambda ctx: model_parts(
                block,
                params,
                state,
                ctx,
                placement=placement,
            )[0],
            nll_key="eval_fast_nll",
            accuracy_key="eval_fast_accuracy",
            perplexity_key="eval_fast_perplexity",
        )

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(
            lambda ctx: model_parts(block, params, state, ctx, placement=placement)[0]
        )(contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, labels)
        acc = jnp.argmax(logits, axis=1) == labels
        return losses, acc.astype(jnp.float32)

    losses, acc = run()
    losses.block_until_ready()
    mean_loss = float(jnp.mean(losses))
    return {
        "eval_fast_nll": mean_loss,
        "eval_fast_accuracy": float(jnp.mean(acc)),
        "eval_fast_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def effective_eval_batch_size(eval_steps: int, eval_batch_size: int) -> int:
    """Return the held-out batch size used by evaluation helpers."""
    if eval_batch_size <= 0:
        return eval_steps
    return min(eval_steps, eval_batch_size)


def eval_batch_count(eval_steps: int, eval_batch_size: int) -> int:
    """Return the number of held-out batches for one seed."""
    effective = effective_eval_batch_size(eval_steps, eval_batch_size)
    return (eval_steps + effective - 1) // effective


def should_stream_eval(labels: jax.Array, eval_batch_size: int) -> bool:
    """Return whether evaluation should use smaller JAX batches."""
    return 0 < eval_batch_size < int(labels.shape[0])


def eval_logits_batched(
    contexts: jax.Array,
    labels: jax.Array,
    *,
    eval_batch_size: int,
    logits_for_context: Callable[[jax.Array], jax.Array],
    nll_key: str,
    accuracy_key: str,
    perplexity_key: str,
) -> dict[str, float]:
    """Evaluate logits in fixed-size chunks and aggregate over all examples."""
    total_examples = int(labels.shape[0])
    if total_examples <= 0:
        raise ValueError("held-out evaluation requires at least one example")

    @jax.jit
    def run_batch(
        batch_contexts: jax.Array,
        batch_labels: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(logits_for_context)(batch_contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, batch_labels)
        acc = jnp.argmax(logits, axis=1) == batch_labels
        return jnp.sum(losses), jnp.sum(acc.astype(jnp.float32))

    loss_sum = 0.0
    acc_sum = 0.0
    for start in range(0, total_examples, eval_batch_size):
        stop = min(start + eval_batch_size, total_examples)
        batch_loss, batch_acc = run_batch(contexts[start:stop], labels[start:stop])
        batch_loss.block_until_ready()
        loss_sum += float(batch_loss)
        acc_sum += float(batch_acc)

    mean_loss = loss_sum / total_examples
    return {
        nll_key: mean_loss,
        accuracy_key: acc_sum / total_examples,
        perplexity_key: float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def eval_baseline_transformer(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    eval_batch_size: int,
) -> dict[str, float]:
    """Evaluate the baseline transformer, optionally streaming held-out batches."""
    if not should_stream_eval(labels, eval_batch_size):
        return eval_transformer(params, contexts, labels)
    return eval_logits_batched(
        contexts,
        labels,
        eval_batch_size=eval_batch_size,
        logits_for_context=lambda ctx: transformer_logits(params, ctx),
        nll_key="eval_nll",
        accuracy_key="eval_accuracy",
        perplexity_key="eval_perplexity",
    )


def summarize_advantage(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize online and gate diagnostics."""
    online = summarize_online(metrics[:, :2], final_window)
    window = metrics[-min(final_window, metrics.shape[0]) :]
    online.update(
        {
            "final_window_base_nll": float(np.mean(window[:, 2])),
            "final_window_advantage": float(np.mean(window[:, 3])),
            "final_window_gate": float(np.mean(window[:, 4])),
            "final_window_gate_logit": float(np.mean(window[:, 5])),
            "final_window_active_prototypes": float(np.mean(window[:, 6])),
            "final_window_allocation_rate": float(np.mean(window[:, 7])),
            "final_window_active_features": float(np.mean(window[:, 8])),
        }
    )
    if metrics.shape[1] > 9:
        online["final_window_gate_update_advantage"] = float(np.mean(window[:, 9]))
    return online


def aggregate_metric(records: list[dict[str, Any]], method: str, metric: str) -> np.ndarray:
    """Collect metric values across seeds."""
    fallback = {
        "final_window_base_nll": "final_window_nll",
        "eval_fast_nll": "eval_nll",
        "eval_fast_perplexity": "eval_perplexity",
    }
    values = []
    for row in records:
        if row["method"] != method:
            continue
        summary = row["summary"]
        if metric in summary:
            values.append(summary[metric])
        else:
            values.append(summary[fallback[metric]])
    return np.asarray(
        values,
        dtype=np.float64,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "advantage_post_ffn_memory",
        "advantage_pre_ffn_kv_memory",
    ]
    labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "advantage_post_ffn_memory": "Advantage Post-FFN",
        "advantage_pre_ffn_kv_memory": "Advantage Pre-FFN KV",
    }
    metrics = [
        "final_window_nll",
        "final_window_base_nll",
        "final_window_accuracy",
        "eval_nll",
        "eval_fast_nll",
        "eval_accuracy",
        "eval_perplexity",
        "eval_fast_perplexity",
        "train_s",
        "train_steps_per_s",
    ]
    lower = {
        "final_window_nll",
        "final_window_base_nll",
        "eval_nll",
        "eval_fast_nll",
        "eval_perplexity",
        "eval_fast_perplexity",
        "train_s",
    }
    lines = [
        "# Tiny Shakespeare Advantage-Gated Memory Transformer",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        "",
        "## Architecture and State",
        "",
        "| Method | Trainable params | Trainable bytes | "
        "Extra state elements | Extra state bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        profile = payload["profiles"][method]
        lines.append(
            f"| `{labels[method]}` | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    lines.extend(["", "## Metrics", ""])
    lines.append("| Metric | " + " | ".join(labels[m] for m in methods) + " |")
    lines.append("|---|" + "---:|" * len(methods))
    for metric in metrics:
        cells = []
        for method in methods:
            values = aggregate_metric(records, method, metric)
            cells.append(f"{np.mean(values):.4f} +/- {stderr(values):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(["", "## Diffs vs Baseline", ""])
    lines.append("| Metric | Advantage Post-FFN | Advantage Pre-FFN KV |")
    lines.append("|---|---:|---:|")
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        cells = []
        for method in methods[1:]:
            values = aggregate_metric(records, method, metric)
            diff = baseline - values if metric in lower else values - baseline
            cells.append(f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    for method in methods[1:]:
        lines.extend(["", f"## {labels[method]} Diagnostics", ""])
        lines.append("| Metric | Mean +/- stderr |")
        lines.append("|---|---:|")
        rows = [row for row in records if row["method"] == method]
        for metric in [
            "final_window_base_nll",
            "final_window_advantage",
            "final_window_gate_update_advantage",
            "final_window_gate",
            "final_window_gate_logit",
            "final_window_active_prototypes",
            "final_window_allocation_rate",
            "final_window_active_features",
        ]:
            values = np.asarray([row["summary"][metric] for row in rows], dtype=np.float64)
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Held-out eval batch size. Use 0 for legacy full-context evaluation.",
    )
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--fast-lr", type=float, default=0.15)
    parser.add_argument("--slow-lr", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.0002)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--proto-adaptive-bandwidth", action="store_true")
    parser.add_argument("--proto-bandwidth-update-rate", type=float, default=0.1)
    parser.add_argument("--gate-init-logit", type=float, default=-2.0)
    parser.add_argument("--gate-lr", type=float, default=1.0)
    parser.add_argument("--gate-decay", type=float, default=0.995)
    parser.add_argument("--gate-max", type=float, default=0.9997)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--gate-l2", type=float, default=0.0)
    parser.add_argument(
        "--gate-mode",
        choices=("scalar", "prototype"),
        default="scalar",
        help="Use one global slow-path gate or one utility gate per prototype.",
    )
    parser.add_argument(
        "--gate-objective",
        choices=("current", "replay"),
        default="current",
        help="Update gates from current-token advantage or delayed replay advantage.",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=128,
        help="Ring-buffer size for --gate-objective replay.",
    )
    parser.add_argument(
        "--train-loss-mode",
        choices=("memory", "blend"),
        default="memory",
        help="Use memory-only loss or blend fast/base and memory losses for gradients.",
    )
    parser.add_argument(
        "--memory-loss-weight",
        type=float,
        default=1.0,
        help="Memory loss weight for --train-loss-mode blend.",
    )
    parser.add_argument("--reset-mode", choices=("none", "zero", "meta_ema"), default="meta_ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/advantage_memory_transformer"),
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments."""
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.d_model < 1 or args.mlp_hidden < 1 or args.proto_count < 1:
        raise ValueError("--d-model, --mlp-hidden, and --proto-count must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if min(args.baseline_lr, args.fast_lr, args.slow_lr, args.gate_lr) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 < args.proto_update_rate <= 1.0:
        raise ValueError("--proto-update-rate must be in (0, 1]")
    if args.proto_novelty_threshold < 0.0:
        raise ValueError("--proto-novelty-threshold must be non-negative")
    if args.proto_bandwidth <= 0.0:
        raise ValueError("--proto-bandwidth must be positive")
    if not 0.0 <= args.proto_bandwidth_update_rate <= 1.0:
        raise ValueError("--proto-bandwidth-update-rate must be in [0, 1]")
    if not 0.0 <= args.gate_decay <= 1.0:
        raise ValueError("--gate-decay must be in [0, 1]")
    if not 0.0 < args.gate_max < 1.0:
        raise ValueError("--gate-max must be in (0, 1)")
    if not 0.0 <= args.memory_loss_weight <= 1.0:
        raise ValueError("--memory-loss-weight must be in [0, 1]")
    if args.replay_size < 1:
        raise ValueError("--replay-size must be positive")


def main() -> None:
    """Run the experiment."""
    run_started_at = dt.datetime.now(dt.UTC).isoformat()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = metadata["vocab_size"]
    final_window = args.final_window if args.final_window > 0 else args.eval_steps
    eval_batch_size = effective_eval_batch_size(args.eval_steps, args.eval_batch_size)
    block = make_proto_block(args)
    data_info = file_manifest(args.data_path)
    data_info.update(
        {
            "source_url": TINY_SHAKESPEARE_URL,
            "encoding": "utf-8",
            "text_char_count": len(text),
            "token_count": int(tokens.shape[0]),
            "split_token_index": split,
            "train_token_count": int(train_tokens.shape[0]),
            "eval_token_count": int(eval_tokens.shape[0]),
            "vocab_size": vocab_size,
            "vocab_sha256": sha256_text("".join(metadata["itos"])),
        }
    )

    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        proto_count=args.proto_count,
        eval_steps=args.eval_steps,
        eval_batch_size=eval_batch_size,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=args.proto_adaptive_bandwidth,
        proto_bandwidth_update_rate=args.proto_bandwidth_update_rate,
        gate_init_logit=args.gate_init_logit,
        gate_lr=args.gate_lr,
        gate_decay=args.gate_decay,
        gate_max=args.gate_max,
        advantage_margin=args.advantage_margin,
        gate_l2=args.gate_l2,
        gate_mode=args.gate_mode,
        gate_objective=args.gate_objective,
        replay_size=args.replay_size,
        train_loss_mode=args.train_loss_mode,
        memory_loss_weight=args.memory_loss_weight,
        reset_mode=args.reset_mode,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    profile_key = jr.fold_in(root, 999)
    raw_args = json_safe(vars(args))
    raw_args["effective_final_window"] = final_window
    raw_args["effective_eval_batch_size"] = eval_batch_size
    baseline_profile = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    advantage_profile, advantage_state = init_advantage_params_state(
        profile_key,
        block=block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_logit=args.gate_init_logit,
        gate_mode=args.gate_mode,
    )
    replay_state_elements = (
        args.replay_size * args.block_size + args.replay_size + 2
        if args.gate_objective == "replay"
        else 0
    )
    replay_state_bytes = 4 * replay_state_elements
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile),
            "trainable_bytes": count_array_bytes(baseline_profile),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "advantage_post_ffn_memory": {
            "trainable_params": count_array_elements(advantage_profile),
            "trainable_bytes": count_array_bytes(advantage_profile),
            "state_elements": count_array_elements(advantage_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(advantage_state, include_int=True)
            + replay_state_bytes,
        },
        "advantage_pre_ffn_kv_memory": {
            "trainable_params": count_array_elements(advantage_profile),
            "trainable_bytes": count_array_bytes(advantage_profile),
            "state_elements": count_array_elements(advantage_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(advantage_state, include_int=True)
            + replay_state_bytes,
        },
    }

    records: list[dict[str, Any]] = []
    seed_runs: list[dict[str, Any]] = []
    methods = (
        "baseline_ffn_transformer",
        "advantage_post_ffn_memory",
        "advantage_pre_ffn_kv_memory",
    )
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        run_key = jr.fold_in(root, seed_idx)
        param_key, offset_key = jr.split(run_key, 2)
        train_max_start = int(train_tokens.shape[0]) - args.block_size - 1
        eval_max_start = int(eval_tokens.shape[0]) - args.block_size - 1
        max_offset = max(1, int(train_tokens.shape[0]) - args.block_size - args.steps - 1)
        train_offset = int(jr.randint(offset_key, (), 0, max_offset))
        eval_offset = seed_idx * args.eval_steps
        train_effective_mod = max(1, train_max_start)
        eval_effective_mod = max(1, eval_max_start)
        data_offsets = {
            "train_offset": train_offset,
            "train_effective_offset": train_offset % train_effective_mod,
            "train_max_start": train_max_start,
            "train_sample_max_offset": max_offset,
            "train_steps": args.steps,
            "eval_offset": eval_offset,
            "eval_effective_offset": eval_offset % eval_effective_mod,
            "eval_max_start": eval_max_start,
            "eval_steps": args.eval_steps,
            "eval_batch_size": eval_batch_size,
            "eval_batches": eval_batch_count(args.eval_steps, eval_batch_size),
            "block_size": args.block_size,
        }
        seed_keys = {
            "run_key": key_manifest(run_key),
            "param_key": key_manifest(param_key),
            "offset_key": key_manifest(offset_key),
        }
        seed_runs.append(
            {
                "seed_index": seed_idx,
                "base_seed": args.seed,
                "keys": seed_keys,
                "methods": {
                    method: {
                        **data_offsets,
                        "param_key": seed_keys["param_key"],
                        "run_key": seed_keys["run_key"],
                    }
                    for method in methods
                },
            }
        )
        contexts, labels = make_examples(
            train_tokens,
            steps=args.steps,
            block_size=args.block_size,
            offset=train_offset,
        )
        eval_contexts, eval_labels = make_examples(
            eval_tokens,
            steps=args.eval_steps,
            block_size=args.block_size,
            offset=eval_offset,
        )
        baseline_params = init_transformer_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
        )
        method_start = time.perf_counter()
        final_baseline, baseline_metrics = run_baseline_transformer(
            baseline_params,
            contexts,
            labels,
            step_size=args.baseline_lr,
            grad_clip=args.grad_clip,
        )
        train_s = time.perf_counter() - method_start
        summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_baseline_transformer(
                final_baseline,
                eval_contexts,
                eval_labels,
                eval_batch_size=eval_batch_size,
            ),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "baseline_ffn_transformer",
                "data_offsets": data_offsets,
                "summary": summary,
            }
        )
        print(
            f"seed={seed_idx} baseline: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, train_s={train_s:.2f}"
        )

        for method, placement in (
            ("advantage_post_ffn_memory", "post_ffn"),
            ("advantage_pre_ffn_kv_memory", "pre_ffn_kv"),
        ):
            params, state = init_advantage_params_state(
                param_key,
                block=block,
                vocab_size=vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                ffn_hidden=args.mlp_hidden,
                gate_init_logit=args.gate_init_logit,
                gate_mode=args.gate_mode,
            )
            method_start = time.perf_counter()
            final_params, final_state, metrics = run_advantage_memory_transformer(
                block,
                params,
                state,
                contexts,
                labels,
                placement=placement,
                fast_lr=args.fast_lr,
                slow_lr=args.slow_lr,
                grad_clip=args.grad_clip,
                gate_lr=args.gate_lr,
                gate_decay=args.gate_decay,
                gate_max=args.gate_max,
                advantage_margin=args.advantage_margin,
                gate_l2=args.gate_l2,
                gate_init_logit=args.gate_init_logit,
                gate_objective=args.gate_objective,
                replay_size=args.replay_size,
                train_loss_mode=args.train_loss_mode,
                memory_loss_weight=args.memory_loss_weight,
                reset_mode=args.reset_mode,
            )
            final_state.step_count.block_until_ready()
            train_s = time.perf_counter() - method_start
            summary = {
                **summarize_advantage(metrics, final_window),
                **eval_advantage_memory_transformer(
                    block,
                    final_params,
                    final_state,
                    eval_contexts,
                    eval_labels,
                    placement=placement,
                    eval_batch_size=eval_batch_size,
                ),
                **eval_advantage_fast_only(
                    block,
                    final_params,
                    final_state,
                    eval_contexts,
                    eval_labels,
                    placement=placement,
                    eval_batch_size=eval_batch_size,
                ),
                "train_s": train_s,
                "train_steps_per_s": args.steps / train_s,
            }
            records.append(
                {
                    "seed": seed_idx,
                    "method": method,
                    "data_offsets": data_offsets,
                    "summary": summary,
                }
            )
            print(
                f"seed={seed_idx} {method}: fw_nll={summary['final_window_nll']:.3f}, "
                f"eval_ppl={summary['eval_perplexity']:.2f}, "
                f"adv={summary['final_window_advantage']:.4f}, "
                f"gate={summary['final_window_gate']:.3f}, train_s={train_s:.2f}"
            )

    elapsed_s = time.perf_counter() - start
    manifest = {
        "schema_version": "advantage_memory_transformer_manifest_v1",
        "run_started_at_utc": run_started_at,
        "run_completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "command": shlex.join([sys.executable, *sys.argv]),
        "executable": sys.executable,
        "config": {
            "raw_args": raw_args,
            "effective_config": asdict(config),
            "prototype_block": block.to_config(),
        },
        "data": data_info,
        "source": source_manifest(),
        "git": git_manifest(Path.cwd()),
        "environment": environment_manifest(),
        "evaluation": {
            "eval_steps_per_seed": args.eval_steps,
            "eval_batch_size": eval_batch_size,
            "eval_batches_per_seed": eval_batch_count(args.eval_steps, eval_batch_size),
            "aggregation": "weighted mean over all held-out examples; no subsampling",
        },
        "prng": {
            "library": "jax.random",
            "root_seed": args.seed,
            "root_key": key_manifest(root),
            "profile_key": key_manifest(profile_key),
            "derivation": [
                "root = jr.key(seed)",
                "profile_key = jr.fold_in(root, 999)",
                "run_key = jr.fold_in(root, seed_idx)",
                "param_key, offset_key = jr.split(run_key, 2)",
                "train_offset = jr.randint(offset_key, (), 0, max_offset)",
                "eval_offset = seed_idx * eval_steps",
            ],
        },
        "seed_runs": seed_runs,
        "elapsed_s": elapsed_s,
    }
    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "profiles": profiles,
        "prototype_block": block.to_config(),
        "elapsed_s": elapsed_s,
        "manifest": manifest,
        "records": records,
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
