#!/usr/bin/env python3
"""Generate the Step 2 transformer baseline fairness ledger.

The ledger is intentionally declarative. It freezes which extra baselines and
mechanism ablations are required before a replay-capped transformer memory
claim can be treated as paperworthy, records which entries are runnable with
the current scripts, and names the runner hooks still needed for the rest.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "alberta.step2.transformer_baseline_fairness_ledger.v1"
DATE_FROZEN = "2026-05-07"
PRIMARY_METHOD = "advantage_post_ffn_memory"
BASELINE_METHOD = "baseline_ffn_transformer"
PRIMARY_METRICS = ("final_window_nll", "eval_nll")
HORIZONS = (3000, 5000, 10000)
SMOKE_HORIZONS = (300,)
DEFAULT_SEEDS = 30
SMOKE_SEEDS = 2
DEFAULT_EVAL_STEPS = 4096
SMOKE_EVAL_STEPS = 128
DEFAULT_EVAL_BATCH_SIZE = 512
SMOKE_EVAL_BATCH_SIZE = 128
DEFAULT_FINAL_WINDOW = 512
SMOKE_FINAL_WINDOW = 128
TRAIN_END_BYTE = 800_000
VALIDATION_END_BYTE = 950_000
TINY_SHAKESPEARE_BYTES = 1_115_394
CANDIDATE_REPLAY_STATE_BYTES = 26_008
CANDIDATE_TRAINABLE_PARAMS = 15_617
BASELINE_H64_TRAINABLE_PARAMS = 13_537
PARAMETER_MATCHED_HIDDEN = 96
WIDER_HIDDEN = 128

CONFIRMATORY_WRAPPER = "benchmarks/step2_transformer_confirmatory_paperworthy_runner.py"
ADVANTAGE_RUNNER = (
    "examples/The Alberta Plan/Step2/"
    "step2_tiny_shakespeare_advantage_memory_transformer.py"
)
THROUGHPUT_RUNNER = "benchmarks/step2_transformer_memory_throughput.py"
VALIDATION_DECISION_SUMMARY = (
    "outputs/step2_new_directions/"
    "advantage_memory_transformer_confirmatory_validation_30seed/"
    "confirmatory_decision_summary.json"
)

BASE_RUNNER_FLAGS = {
    "block-size": "32",
    "d-model": "32",
    "mlp-hidden": "64",
    "proto-count": "64",
    "baseline-lr": "0.15",
    "fast-lr": "0.15",
    "slow-lr": "0.1",
    "grad-clip": "1.0",
    "proto-update-rate": "0.3",
    "proto-novelty-threshold": "0.0002",
    "proto-bandwidth": "0.01",
    "gate-init-logit": "-3.0",
    "gate-lr": "0.5",
    "gate-decay": "0.995",
    "gate-max": "0.15",
    "advantage-margin": "0.0",
    "gate-l2": "0.1",
    "gate-mode": "scalar",
    "gate-objective": "replay",
    "replay-size": "128",
    "train-loss-mode": "memory",
    "memory-loss-weight": "1.0",
    "reset-mode": "meta_ema",
    "seed": "0",
}

FFN_LR_GRID = ("0.05", "0.10", "0.15", "0.20", "0.30")


@dataclass(frozen=True)
class SplitSpec:
    """One frozen direct-run split."""

    preset: str
    train_fraction: str
    data_path: str
    train_range: tuple[int, int]
    eval_range: tuple[int, int | str]
    output_root: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "train_fraction": self.train_fraction,
            "data_path": self.data_path,
            "train_range": list(self.train_range),
            "eval_range": list(self.eval_range),
            "output_root": self.output_root,
        }


@dataclass(frozen=True)
class CommandRecord:
    """A concrete command or a desired command blocked on runner hooks."""

    entry_id: str
    preset: str
    command_kind: str
    status: str
    command: tuple[str, ...]
    horizon: int | None = None
    output_dir: str | None = None
    sweep_values: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "preset": self.preset,
            "command_kind": self.command_kind,
            "status": self.status,
            "horizon": self.horizon,
            "sweep_values": self.sweep_values,
            "command": list(self.command),
            "output_dir": self.output_dir,
            "notes": list(self.notes),
            "shell": shell_join(self.command),
        }


@dataclass(frozen=True)
class LedgerEntry:
    """One frozen baseline, ablation, or attribution entry."""

    entry_id: str
    title: str
    tier: str
    family: str
    status: str
    command_mode: str
    paper_question: str
    paperworthiness: str
    tuning_policy: str
    selection_rule: str
    result_methods: tuple[str, ...]
    primary_or_secondary: str
    metrics: tuple[str, ...]
    config_overrides: dict[str, str] = field(default_factory=dict)
    sweep_grid: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_entries: tuple[str, ...] = ()
    required_hooks: tuple[str, ...] = ()
    hook_flags: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.entry_id,
            "title": self.title,
            "tier": self.tier,
            "family": self.family,
            "status": self.status,
            "command_mode": self.command_mode,
            "paper_question": self.paper_question,
            "paperworthiness": self.paperworthiness,
            "tuning_policy": self.tuning_policy,
            "selection_rule": self.selection_rule,
            "result_methods": list(self.result_methods),
            "primary_or_secondary": self.primary_or_secondary,
            "metrics": list(self.metrics),
            "config_overrides": self.config_overrides,
            "sweep_grid": {key: list(values) for key, values in self.sweep_grid.items()},
            "source_entries": list(self.source_entries),
            "required_hooks": list(self.required_hooks),
            "hook_flags": self.hook_flags,
            "notes": list(self.notes),
        }


def split_specs() -> dict[str, SplitSpec]:
    """Return frozen split specs used by direct runner commands."""
    root = "outputs/step2_new_directions/transformer_baseline_fairness_splits"
    return {
        "smoke": SplitSpec(
            preset="smoke",
            train_fraction="0.84210526315789469",
            data_path=f"{root}/smoke/data/tinyshakespeare_confirmatory_smoke.txt",
            train_range=(0, TRAIN_END_BYTE),
            eval_range=(TRAIN_END_BYTE, VALIDATION_END_BYTE),
            output_root=f"{root}/smoke",
        ),
        "validation": SplitSpec(
            preset="validation",
            train_fraction="0.84210526315789469",
            data_path=f"{root}/validation/data/tinyshakespeare_confirmatory_validation.txt",
            train_range=(0, TRAIN_END_BYTE),
            eval_range=(TRAIN_END_BYTE, VALIDATION_END_BYTE),
            output_root=f"{root}/validation",
        ),
        "lockbox": SplitSpec(
            preset="lockbox",
            train_fraction="0.82867720329730654",
            data_path=f"{root}/lockbox/data/tinyshakespeare_confirmatory_lockbox.txt",
            train_range=(0, TRAIN_END_BYTE),
            eval_range=(VALIDATION_END_BYTE, "EOF"),
            output_root=f"{root}/lockbox",
        ),
    }


def ledger_entries() -> tuple[LedgerEntry, ...]:
    """Return frozen ledger entries."""
    required = "required_for_paperworthiness"
    exploratory = "exploratory_or_mechanism_only"
    primary = "primary_family"
    secondary = "secondary_report"
    return (
        LedgerEntry(
            entry_id="optimizer_matched_ffn_sgd",
            title="Optimizer-matched clipped-SGD FFN",
            tier=required,
            family="baseline",
            status="available_in_confirmatory_wrapper",
            command_mode="confirmatory_wrapper",
            paper_question=(
                "Does replay-capped memory beat the same fast transformer trained "
                "with the same clipped online SGD optimizer?"
            ),
            paperworthiness=(
                "Required primary comparator. Without it there is no paired claim "
                "against the tuned FFN transformer."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; LR 0.15 and H=64 are inherited from the frozen protocol.",
            result_methods=(BASELINE_METHOD,),
            primary_or_secondary=primary,
            metrics=PRIMARY_METRICS,
            notes=(
                "The wrapper also runs the primary memory candidate and the report generator.",
            ),
        ),
        LedgerEntry(
            entry_id="parameter_matched_ffn_h96",
            title="Parameter-matched FFN, H=96",
            tier=required,
            family="baseline",
            status="available_with_direct_runner",
            command_mode="advantage_runner_baseline_rows",
            paper_question=(
                "Does the memory result survive a no-memory FFN with the same trainable "
                "parameter count as H=64/P=64 memory?"
            ),
            paperworthiness=(
                "Required to rule out the 2080 extra prototype value parameters as the cause."
            ),
            tuning_policy="validation_only_grid",
            selection_rule=(
                "Run the frozen LR grid on validation. Select the lowest mean validation "
                "eval_nll at 10000 steps; ties choose the smaller LR. Run only the selected "
                "config on lockbox."
            ),
            result_methods=(BASELINE_METHOD,),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS,
            config_overrides={"mlp-hidden": str(PARAMETER_MATCHED_HIDDEN)},
            sweep_grid={"baseline-lr": FFN_LR_GRID},
            notes=(
                f"H=96 gives {CANDIDATE_TRAINABLE_PARAMS} trainable parameters, matching "
                "the H=64/P=64 memory candidate.",
                "The current runner still emits memory rows for H=96; ignore them "
                "for this baseline.",
            ),
        ),
        LedgerEntry(
            entry_id="wider_ffn_h128",
            title="Wider FFN, H=128",
            tier=required,
            family="baseline",
            status="available_with_direct_runner",
            command_mode="advantage_runner_baseline_rows",
            paper_question=(
                "Does a larger no-memory FFN close the margin under the same online data "
                "and optimizer family?"
            ),
            paperworthiness=(
                "Required because the observed memory margins are small and could "
                "be a width effect."
            ),
            tuning_policy="validation_only_grid",
            selection_rule=(
                "Run the frozen LR grid on validation. Select the lowest mean validation "
                "eval_nll at 10000 steps; ties choose the smaller LR. Run only the selected "
                "config on lockbox."
            ),
            result_methods=(BASELINE_METHOD,),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS,
            config_overrides={"mlp-hidden": str(WIDER_HIDDEN)},
            sweep_grid={"baseline-lr": FFN_LR_GRID},
            notes=(
                "This is intentionally not parameter matched; it tests ordinary scale-up pressure.",
            ),
        ),
        LedgerEntry(
            entry_id="state_matched_ffn_control",
            title="State-matched FFN control",
            tier=required,
            family="baseline",
            status="hook_needed",
            command_mode="desired_advantage_runner",
            paper_question=(
                "Does equal learner state and temporal state-update traffic explain the memory win?"
            ),
            paperworthiness=(
                "Required if the paper claims an algorithmic memory effect rather than a state "
                "or update-budget effect."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; match the replay candidate state bytes exactly.",
            result_methods=(BASELINE_METHOD,),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS,
            config_overrides={"mlp-hidden": str(PARAMETER_MATCHED_HIDDEN)},
            hook_flags={
                "baseline-state-control-bytes": str(CANDIDATE_REPLAY_STATE_BYTES),
                "baseline-state-control-update": "ring_ema",
                "baseline-state-control-read": "false",
            },
            required_hooks=(
                "Add a baseline-only no-op state buffer that is updated every step, "
                "recorded in profiles, and never read by prediction or gradients.",
            ),
            notes=(
                f"Target replay candidate state: {CANDIDATE_REPLAY_STATE_BYTES} bytes.",
            ),
        ),
        LedgerEntry(
            entry_id="wall_clock_matched_ffn",
            title="Wall-clock-matched FFN",
            tier=required,
            family="baseline",
            status="hook_needed",
            command_mode="desired_wall_clock_wrapper",
            paper_question=(
                "Could the FFN match or beat memory if trained for the same wall-clock budget?"
            ),
            paperworthiness=(
                "Required because the replay path is materially slower than the FFN path."
            ),
            tuning_policy="validation_only_grid",
            selection_rule=(
                "Use validation to choose the FFN width/LR from the frozen grid under equal "
                "candidate train_s. Apply the selected recipe once on lockbox."
            ),
            result_methods=(BASELINE_METHOD,),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("train_s", "train_steps_per_s"),
            sweep_grid={"mlp-hidden": ("64", "96", "128"), "baseline-lr": FFN_LR_GRID},
            required_hooks=(
                "Add a wrapper that reads candidate train_s per seed/horizon, runs FFN until "
                "the same wall-clock budget is consumed, and records actual steps consumed.",
                "Separate JIT compile time from hot-loop time before matching budgets.",
            ),
            notes=(
                f"Use {THROUGHPUT_RUNNER} as the calibration probe, but do not use throughput "
                "numbers to retune candidate flags.",
            ),
        ),
        LedgerEntry(
            entry_id="no_replay_current_gate",
            title="No-replay current-token gate",
            tier=required,
            family="mechanism_ablation",
            status="available_with_direct_runner",
            command_mode="advantage_runner_candidate_rows",
            paper_question=(
                "Is delayed replay utility necessary, or is current-token advantage enough?"
            ),
            paperworthiness=(
                "Required to attribute the result to delayed validation-like utility rather than "
                "any advantage gate."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; this is a single preregistered mechanism ablation.",
            result_methods=(PRIMARY_METHOD, "advantage_pre_ffn_kv_memory"),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("final_window_gate", "final_window_advantage"),
            config_overrides={"gate-objective": "current", "replay-size": "1"},
        ),
        LedgerEntry(
            entry_id="no_cap_replay_gate",
            title="No-cap replay gate",
            tier=required,
            family="mechanism_ablation",
            status="available_with_direct_runner_practical_no_cap",
            command_mode="advantage_runner_candidate_rows",
            paper_question="Is the tight resource cap necessary for the memory result?",
            paperworthiness=(
                "Required to show the positive result is not hiding an uncapped memory "
                "failure mode."
            ),
            tuning_policy="frozen_single_config",
            selection_rule=(
                "No selection; gate_max is raised to the runner's practical no-cap value."
            ),
            result_methods=(PRIMARY_METHOD, "advantage_pre_ffn_kv_memory"),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("final_window_gate", "final_window_active_prototypes"),
            config_overrides={"gate-max": "0.9997"},
            notes=(
                "The current runner requires 0 < gate_max < 1, so 0.9997 is the exact "
                "preexisting practical no-cap setting.",
            ),
        ),
        LedgerEntry(
            entry_id="fixed_gate_cap_only",
            title="Fixed-gate cap-only memory",
            tier=required,
            family="mechanism_ablation",
            status="available_with_direct_runner",
            command_mode="advantage_runner_candidate_rows",
            paper_question=(
                "Does a fixed small residual path explain the gain without learned utility gating?"
            ),
            paperworthiness=(
                "Required to isolate learned gate adaptation from a static residual ensemble."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; gate stays at sigmoid(-3) under the same cap.",
            result_methods=(PRIMARY_METHOD, "advantage_pre_ffn_kv_memory"),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("final_window_gate", "final_window_advantage"),
            config_overrides={"gate-lr": "0.0", "gate-decay": "1.0"},
        ),
        LedgerEntry(
            entry_id="placement_pre_ffn_kv",
            title="Pre-FFN KV placement check",
            tier=required,
            family="mechanism_ablation",
            status="available_from_primary_results",
            command_mode="derived_from_primary_results",
            paper_question=(
                "Is the reported post-FFN placement robust against placement cherry-picking?"
            ),
            paperworthiness=(
                "Required as a secondary check. It must not replace the frozen post-FFN primary "
                "method after validation or lockbox outcomes are known."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; report the pre-FFN KV method emitted by the same runs.",
            result_methods=("advantage_pre_ffn_kv_memory",),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("eval_fast_nll",),
            source_entries=("optimizer_matched_ffn_sgd",),
        ),
        LedgerEntry(
            entry_id="fast_only_memory_training_attribution",
            title="Fast-only deployment attribution",
            tier=required,
            family="attribution",
            status="available_from_primary_results",
            command_mode="derived_from_primary_results",
            paper_question=(
                "Is the gain present in deployed memory logits, or only in the fast branch after "
                "memory-regularized training?"
            ),
            paperworthiness=(
                "Required to state whether the mechanism is inference-time memory or training-time "
                "auxiliary regularization."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; compare eval_nll and eval_fast_nll from the same rows.",
            result_methods=(PRIMARY_METHOD,),
            primary_or_secondary=secondary,
            metrics=("eval_nll", "eval_fast_nll", "final_window_nll", "final_window_base_nll"),
            source_entries=("optimizer_matched_ffn_sgd",),
        ),
        LedgerEntry(
            entry_id="fast_loss_only_training_ablation",
            title="Fast-loss-only training ablation",
            tier=required,
            family="attribution",
            status="available_with_direct_runner",
            command_mode="advantage_runner_candidate_rows",
            paper_question=(
                "Does the candidate need gradients from the memory loss, or only the side-channel "
                "memory/gate updates?"
            ),
            paperworthiness=(
                "Required for training-attribution claims; secondary for the primary "
                "performance claim."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; set memory_loss_weight to 0 in blend mode.",
            result_methods=(PRIMARY_METHOD, "advantage_pre_ffn_kv_memory"),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("eval_fast_nll", "final_window_base_nll"),
            config_overrides={"train-loss-mode": "blend", "memory-loss-weight": "0.0"},
        ),
        LedgerEntry(
            entry_id="residual_stop_gradient",
            title="Residual stop-gradient ablation",
            tier=exploratory,
            family="mechanism_ablation",
            status="hook_needed",
            command_mode="desired_advantage_runner",
            paper_question=(
                "Are gains due to gradient flow through the memory residual, or the nonparametric "
                "state/update path?"
            ),
            paperworthiness=(
                "Exploratory unless the paper makes a gradient-path mechanism claim."
            ),
            tuning_policy="frozen_single_config",
            selection_rule="No selection; single stop-gradient variant.",
            result_methods=(PRIMARY_METHOD, "advantage_pre_ffn_kv_memory"),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("eval_fast_nll",),
            hook_flags={"memory-residual-gradient": "stop"},
            required_hooks=(
                "Add a flag that wraps the memory residual or prototype transform in "
                "lax.stop_gradient without changing center/gate temporal updates.",
            ),
        ),
        LedgerEntry(
            entry_id="frozen_fast_pretrained_memory",
            title="Frozen-fast pretrained-memory ablation",
            tier=exploratory,
            family="mechanism_ablation",
            status="hook_needed",
            command_mode="desired_two_phase_runner",
            paper_question=(
                "Can memory improve a pretrained frozen fast transformer, or does it need joint "
                "fast-path adaptation?"
            ),
            paperworthiness=(
                "Exploratory unless the paper claims memory can be added to a fixed fast model."
            ),
            tuning_policy="frozen_single_config",
            selection_rule=(
                "No selection; train FFN under the primary baseline, freeze it, then "
                "train memory."
            ),
            result_methods=(PRIMARY_METHOD,),
            primary_or_secondary=secondary,
            metrics=PRIMARY_METRICS + ("eval_fast_nll",),
            hook_flags={
                "initialize-fast-from-baseline-checkpoint": "true",
                "freeze-fast-after-init": "true",
            },
            required_hooks=(
                "Add checkpoint handoff from the trained FFN baseline into the memory runner.",
                "Freeze attention, FFN, and readout after handoff while still updating "
                "centers, gate, and prototype values.",
            ),
            notes=(
                "Setting --fast-lr 0 from random initialization is not this ablation.",
            ),
        ),
    )


def preset_horizons(preset: str) -> tuple[int, ...]:
    """Return horizons for a preset."""
    return SMOKE_HORIZONS if preset == "smoke" else HORIZONS


def preset_seeds(preset: str) -> int:
    """Return seed count for a preset."""
    return SMOKE_SEEDS if preset == "smoke" else DEFAULT_SEEDS


def preset_eval_steps(preset: str) -> int:
    """Return held-out eval contexts for a preset."""
    return SMOKE_EVAL_STEPS if preset == "smoke" else DEFAULT_EVAL_STEPS


def preset_eval_batch_size(preset: str) -> int:
    """Return eval batch size for a preset."""
    return SMOKE_EVAL_BATCH_SIZE if preset == "smoke" else DEFAULT_EVAL_BATCH_SIZE


def preset_final_window(preset: str) -> int:
    """Return final-window width for a preset."""
    return SMOKE_FINAL_WINDOW if preset == "smoke" else DEFAULT_FINAL_WINDOW


def output_token(raw: str) -> str:
    """Return a path-safe token for a flag value."""
    return raw.replace(".", "p").replace("-", "m")


def shell_join(command: tuple[str, ...] | list[str]) -> str:
    """Return a copy-pastable shell command."""
    return " ".join(shlex.quote(part) for part in command)


def flag_items(flags: dict[str, str]) -> list[str]:
    """Flatten CLI flags in stable insertion order."""
    items: list[str] = []
    for key, value in flags.items():
        items.extend([f"--{key}", value])
    return items


def sweep_combinations(grid: dict[str, tuple[str, ...]]) -> tuple[dict[str, str], ...]:
    """Return frozen grid combinations."""
    if not grid:
        return ({},)
    keys = tuple(grid)
    combos: list[dict[str, str]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, values, strict=True)))
    return tuple(combos)


def sweep_suffix(values: dict[str, str]) -> str:
    """Return output-dir suffix for a sweep combination."""
    if not values:
        return ""
    parts = [
        f"{key.replace('-', '')}{output_token(value)}"
        for key, value in values.items()
    ]
    return "_" + "_".join(parts)


def split_materialization_command(preset: str) -> CommandRecord:
    """Build the dry-run split-materialization command for direct runners."""
    split = split_specs()[preset]
    command = [
        "python",
        CONFIRMATORY_WRAPPER,
        "--preset",
        preset,
        "--dry-run",
        "--steps",
        "300",
        "--seeds",
        "1",
        "--eval-steps",
        "16",
        "--eval-batch-size",
        "16",
        "--final-window",
        "16",
        "--output-root",
        split.output_root,
    ]
    status = "available_now"
    notes = ["Creates the derived corpus consumed by direct runner ledger entries."]
    if preset == "lockbox":
        command.append("--allow-lockbox-without-validation")
        status = "dry_run_planning_only"
        notes.append(
            "Lockbox split planning is allowed only because this command is a dry run; "
            "it must not evaluate or train."
        )
    return CommandRecord(
        entry_id="split_materialization",
        preset=preset,
        command_kind="precondition",
        status=status,
        command=tuple(command),
        output_dir=split.output_root,
        notes=tuple(notes),
    )


def wrapper_command(preset: str) -> CommandRecord:
    """Build the primary confirmatory wrapper command."""
    output_root = (
        "outputs/step2_new_directions/"
        f"advantage_memory_transformer_confirmatory_{preset}_30seed"
    )
    if preset == "smoke":
        output_root = (
            "outputs/step2_new_directions/"
            "advantage_memory_transformer_confirmatory_smoke"
        )
    command = [
        "python",
        CONFIRMATORY_WRAPPER,
        "--preset",
        preset,
        "--seeds",
        str(preset_seeds(preset)),
        "--steps",
        *(str(step) for step in preset_horizons(preset)),
        "--eval-steps",
        str(preset_eval_steps(preset)),
        "--eval-batch-size",
        str(preset_eval_batch_size(preset)),
        "--final-window",
        str(preset_final_window(preset)),
        "--output-root",
        output_root,
    ]
    status = "available_now"
    notes = ["Runs the primary memory candidate and optimizer-matched FFN baseline."]
    if preset == "lockbox":
        command.extend(
            [
                "--validation-decision-summary",
                VALIDATION_DECISION_SUMMARY,
            ]
        )
        status = "locked_until_validation_clears"
        notes.append(
            "The wrapper refuses lockbox unless the supplied validation summary has "
            "clears_confirmatory_bar exactly true."
        )
    return CommandRecord(
        entry_id="optimizer_matched_ffn_sgd",
        preset=preset,
        command_kind="training_and_report",
        status=status,
        command=tuple(command),
        output_dir=output_root,
        notes=tuple(notes),
    )


def direct_runner_commands(entry: LedgerEntry, preset: str) -> tuple[CommandRecord, ...]:
    """Build direct advantage-runner commands for an entry."""
    split = split_specs()[preset]
    commands: list[CommandRecord] = []
    for horizon in preset_horizons(preset):
        for combo in sweep_combinations(entry.sweep_grid):
            flags = {
                **BASE_RUNNER_FLAGS,
                **entry.config_overrides,
                **combo,
                **entry.hook_flags,
                "steps": str(horizon),
                "seeds": str(preset_seeds(preset)),
                "eval-steps": str(preset_eval_steps(preset)),
                "eval-batch-size": str(preset_eval_batch_size(preset)),
                "final-window": str(preset_final_window(preset)),
                "train-fraction": split.train_fraction,
                "data-path": split.data_path,
                "output-dir": (
                    "outputs/step2_new_directions/transformer_baseline_fairness/"
                    f"{preset}/{entry.entry_id}/{horizon}_{preset_seeds(preset)}seed"
                    f"{sweep_suffix(combo)}"
                ),
            }
            command = ("python", ADVANTAGE_RUNNER, *flag_items(flags))
            if entry.status == "hook_needed":
                status = "hook_needed"
                notes = list(entry.required_hooks)
            else:
                status = "available_now"
                notes = []
            if preset == "lockbox":
                status = (
                    "hook_needed_and_locked_until_validation_clears"
                    if entry.status == "hook_needed"
                    else "plan_only_locked_until_validation_clears"
                )
                notes.append(
                    "Do not run this direct lockbox command manually. It is a frozen "
                    "plan entry only; execution requires a validation-cleared summary "
                    "and a gated orchestration path."
                )
            commands.append(
                CommandRecord(
                    entry_id=entry.entry_id,
                    preset=preset,
                    command_kind="training",
                    status=status,
                    command=tuple(command),
                    horizon=horizon,
                    output_dir=flags["output-dir"],
                    sweep_values=combo,
                    notes=tuple(notes),
                )
            )
    return tuple(commands)


def wall_clock_commands(entry: LedgerEntry, preset: str) -> tuple[CommandRecord, ...]:
    """Build desired wall-clock wrapper commands."""
    source_root = (
        "outputs/step2_new_directions/advantage_memory_transformer_confirmatory_smoke"
        if preset == "smoke"
        else (
            "outputs/step2_new_directions/"
            f"advantage_memory_transformer_confirmatory_{preset}_30seed"
        )
    )
    command = (
        "python",
        "benchmarks/step2_transformer_wall_clock_matched_baselines.py",
        "--preset",
        preset,
        "--source-results-root",
        source_root,
        "--baseline-hidden-grid",
        ",".join(entry.sweep_grid["mlp-hidden"]),
        "--baseline-lr-grid",
        ",".join(entry.sweep_grid["baseline-lr"]),
        "--match-method",
        PRIMARY_METHOD,
        "--primary-metrics",
        ",".join(PRIMARY_METRICS),
        "--output-root",
        f"outputs/step2_new_directions/transformer_baseline_fairness/{preset}/{entry.entry_id}",
    )
    return (
        CommandRecord(
            entry_id=entry.entry_id,
            preset=preset,
            command_kind="desired_training_wrapper",
            status="hook_needed",
            command=command,
            notes=entry.required_hooks,
        ),
    )


def two_phase_commands(entry: LedgerEntry, preset: str) -> tuple[CommandRecord, ...]:
    """Build desired two-phase frozen-fast commands."""
    split = split_specs()[preset]
    command = (
        "python",
        "benchmarks/step2_transformer_frozen_fast_memory_ablation.py",
        "--preset",
        preset,
        "--steps",
        *(str(step) for step in preset_horizons(preset)),
        "--seeds",
        str(preset_seeds(preset)),
        "--eval-steps",
        str(preset_eval_steps(preset)),
        "--eval-batch-size",
        str(preset_eval_batch_size(preset)),
        "--final-window",
        str(preset_final_window(preset)),
        "--train-fraction",
        split.train_fraction,
        "--data-path",
        split.data_path,
        "--baseline-hidden",
        "64",
        "--baseline-lr",
        "0.15",
        "--memory-config",
        "replay_cap_post_ffn_v1",
        "--freeze-fast-after-baseline",
        "--output-root",
        f"outputs/step2_new_directions/transformer_baseline_fairness/{preset}/{entry.entry_id}",
    )
    return (
        CommandRecord(
            entry_id=entry.entry_id,
            preset=preset,
            command_kind="desired_two_phase_training",
            status="hook_needed",
            command=command,
            notes=entry.required_hooks,
        ),
    )


def throughput_command(preset: str = "smoke") -> CommandRecord:
    """Build a current throughput calibration command."""
    command = (
        "python",
        THROUGHPUT_RUNNER,
        "--steps",
        "256" if preset == "smoke" else "1024",
        "--repeats",
        "3",
        "--eval-steps",
        "128" if preset == "smoke" else "512",
        "--final-window",
        "128" if preset == "smoke" else "512",
        "--output-dir",
        f"outputs/benchmarks/step2_transformer_memory_throughput_{preset}",
    )
    return CommandRecord(
        entry_id="wall_clock_matched_ffn",
        preset=preset,
        command_kind="calibration",
        status="available_now",
        command=command,
        output_dir=f"outputs/benchmarks/step2_transformer_memory_throughput_{preset}",
        notes=("Calibration only; not a matched training baseline by itself.",),
    )


def commands_for_entry(entry: LedgerEntry, preset: str) -> tuple[CommandRecord, ...]:
    """Return command records for an entry and preset."""
    if entry.command_mode == "confirmatory_wrapper":
        return (wrapper_command(preset),)
    if entry.command_mode in {
        "advantage_runner_baseline_rows",
        "advantage_runner_candidate_rows",
        "desired_advantage_runner",
    }:
        return direct_runner_commands(entry, preset)
    if entry.command_mode == "desired_wall_clock_wrapper":
        return (throughput_command(preset), *wall_clock_commands(entry, preset))
    if entry.command_mode == "desired_two_phase_runner":
        return two_phase_commands(entry, preset)
    return ()


def check_entries(entries: tuple[LedgerEntry, ...]) -> None:
    """Validate internal ledger consistency."""
    ids = [entry.entry_id for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate ledger ids: {duplicates}")
    for entry in entries:
        if entry.tier == "required_for_paperworthiness" and not entry.paperworthiness:
            raise ValueError(f"{entry.entry_id} lacks paperworthiness rationale")
        if entry.status == "hook_needed" and not entry.required_hooks:
            raise ValueError(f"{entry.entry_id} is hook_needed without required_hooks")
        if entry.command_mode == "derived_from_primary_results" and not entry.source_entries:
            raise ValueError(f"{entry.entry_id} is derived but has no source entry")
        if entry.sweep_grid and entry.tuning_policy != "validation_only_grid":
            raise ValueError(f"{entry.entry_id} has a grid without validation-only tuning")


def ledger_payload() -> dict[str, object]:
    """Build the machine-readable ledger payload."""
    entries = ledger_entries()
    check_entries(entries)
    commands_by_entry: dict[str, dict[str, list[dict[str, object]]]] = {}
    for entry in entries:
        commands_by_entry[entry.entry_id] = {
            preset: [command.to_dict() for command in commands_for_entry(entry, preset)]
            for preset in ("smoke", "validation", "lockbox")
        }
    return {
        "schema": SCHEMA,
        "date_frozen": DATE_FROZEN,
        "environment_setup": "source .venv/bin/activate",
        "primary_method": PRIMARY_METHOD,
        "baseline_method": BASELINE_METHOD,
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_horizons": list(HORIZONS),
        "primary_seed_count": DEFAULT_SEEDS,
        "candidate_config": BASE_RUNNER_FLAGS,
        "parameter_accounting": {
            "baseline_h64_trainable_params": BASELINE_H64_TRAINABLE_PARAMS,
            "candidate_h64_p64_trainable_params": CANDIDATE_TRAINABLE_PARAMS,
            "parameter_matched_ffn_hidden": PARAMETER_MATCHED_HIDDEN,
            "candidate_replay_state_bytes": CANDIDATE_REPLAY_STATE_BYTES,
        },
        "splits": {key: split.to_dict() for key, split in split_specs().items()},
        "split_materialization_commands": {
            preset: split_materialization_command(preset).to_dict()
            for preset in ("smoke", "validation", "lockbox")
        },
        "lockbox_policy": {
            "rule": (
                "Run lockbox only after validation selection is frozen and the confirmatory "
                "decision summary has clears_confirmatory_bar exactly true. Do not alter "
                "candidate flags, baseline grids, primary metrics, horizons, seeds, or "
                "selection rules after reading validation outcomes."
            ),
            "validation_only_grid_rule": (
                "For grid baselines, select exactly one config on validation using the entry's "
                "selection_rule, then run that selected config once on lockbox."
            ),
            "gating_artifact": VALIDATION_DECISION_SUMMARY,
            "direct_runner_lockbox_commands": (
                "Plan-only. Direct runner commands do not enforce lockbox access themselves; "
                "they must be launched only through a gated orchestration path."
            ),
        },
        "entries": [entry.to_dict() for entry in entries],
        "commands_by_entry": commands_by_entry,
    }


def status_label(entry: LedgerEntry) -> str:
    """Return compact status text."""
    if entry.status == "hook_needed":
        return "hook needed"
    if entry.command_mode == "derived_from_primary_results":
        return "from primary results"
    return "runnable now"


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    """Write the human-readable Markdown ledger."""
    entries = ledger_entries()
    required_entries = [
        entry for entry in entries if entry.tier == "required_for_paperworthiness"
    ]
    exploratory_entries = [
        entry for entry in entries if entry.tier != "required_for_paperworthiness"
    ]
    lines = [
        "# Transformer Baseline Fairness Ledger",
        "",
        f"Date frozen: `{DATE_FROZEN}`.",
        "",
        "This ledger freezes validation-only baselines and ablations for the Step 2 "
        "replay-capped advantage-memory transformer. The adjacent JSON file is the "
        "machine-readable source of truth generated by "
        "`benchmarks/step2_transformer_baseline_fairness_ledger.py`.",
        "",
        "The primary candidate remains `advantage_post_ffn_memory` with primary metrics "
        "`final_window_nll` and `eval_nll` at horizons `3000`, `5000`, and `10000`. "
        "Perplexity, placement, fast-only metrics, gate diagnostics, and compute are "
        "secondary. No entry below may be used to retune candidate flags after validation.",
        "",
        "## Lockbox Rule",
        "",
        "Run validation first. For single-config entries, lockbox uses the same frozen "
        "config. For validation-grid baselines, validation selects exactly one config "
        "using the predeclared selection rule; only that selected config is run on "
        "lockbox. The lockbox wrapper command must include "
        "`--validation-decision-summary` pointing to a validation summary whose "
        "`clears_confirmatory_bar` is exactly `true`; otherwise it refuses before "
        "split materialization. Direct-runner lockbox commands in the JSON are "
        "plan-only entries, not permission to bypass the wrapper. Do not inspect "
        "lockbox to choose widths, learning rates, placement, gate caps, replay "
        "settings, or reporting metrics.",
        "",
        "## Required For Paperworthiness",
        "",
        "| Entry | Family | Status | Why it is required |",
        "|---|---|---|---|",
    ]
    for entry in required_entries:
        lines.append(
            f"| `{entry.entry_id}` | {entry.family} | {status_label(entry)} | "
            f"{entry.paperworthiness} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory Or Mechanism-Only",
            "",
            "| Entry | Status | Scope |",
            "|---|---|---|",
        ]
    )
    for entry in exploratory_entries:
        lines.append(f"| `{entry.entry_id}` | {status_label(entry)} | {entry.paperworthiness} |")

    lines.extend(
        [
            "",
            "## Split Precommands",
            "",
            "Direct runner entries require a materialized validation or lockbox corpus. "
            "These commands only create the derived corpus and manifest; they do not train. "
            "The lockbox precommand is dry-run planning only.",
            "",
        ]
    )
    for preset in ("smoke", "validation", "lockbox"):
        command = split_materialization_command(preset)
        lines.extend(
            [
                "```bash",
                "source .venv/bin/activate",
                shell_join(command.command),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Primary Confirmatory Commands",
            "",
            "These run the primary candidate plus the optimizer-matched FFN baseline and "
            "generate the paired paperworthy report.",
            "",
        ]
    )
    for preset in ("smoke", "validation", "lockbox"):
        command = wrapper_command(preset)
        lines.extend(
            [
                "```bash",
                "source .venv/bin/activate",
                shell_join(command.command),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Entry Ledger",
            "",
        ]
    )
    commands_by_entry = payload["commands_by_entry"]
    if not isinstance(commands_by_entry, dict):
        raise TypeError("commands_by_entry must be a dict")
    for entry in entries:
        lines.extend(
            [
                f"### `{entry.entry_id}`",
                "",
                f"- Title: {entry.title}",
                f"- Tier: `{entry.tier}`",
                f"- Status: `{entry.status}`",
                f"- Question: {entry.paper_question}",
                f"- Tuning policy: `{entry.tuning_policy}`",
                f"- Selection rule: {entry.selection_rule}",
                f"- Result methods: {', '.join(f'`{method}`' for method in entry.result_methods)}",
                f"- Metrics: {', '.join(f'`{metric}`' for metric in entry.metrics)}",
            ]
        )
        if entry.config_overrides:
            overrides = ", ".join(
                f"`--{key} {value}`"
                for key, value in entry.config_overrides.items()
            )
            lines.append(f"- Frozen overrides: {overrides}")
        if entry.sweep_grid:
            grid = ", ".join(
                f"`--{key}` in [{', '.join(values)}]" for key, values in entry.sweep_grid.items()
            )
            lines.append(f"- Frozen validation grid: {grid}")
        if entry.required_hooks:
            lines.append("- Remaining hooks: " + " ".join(entry.required_hooks))
        if entry.notes:
            lines.append("- Notes: " + " ".join(entry.notes))
        entry_commands = commands_by_entry.get(entry.entry_id, {})
        smoke_commands = []
        if isinstance(entry_commands, dict):
            raw_smoke = entry_commands.get("smoke", [])
            if isinstance(raw_smoke, list):
                smoke_commands = raw_smoke
        if smoke_commands:
            first = smoke_commands[0]
            if isinstance(first, dict) and isinstance(first.get("shell"), str):
                lines.extend(
                    [
                        "",
                        "Smoke command example:",
                        "",
                        "```bash",
                        "source .venv/bin/activate",
                    ]
                )
                lines.append(str(first["shell"]))
                if len(smoke_commands) > 1:
                    lines.append(
                        f"# ... {len(smoke_commands) - 1} more smoke command(s) in JSON"
                    )
                lines.extend(["```", ""])
        else:
            lines.append(
                "- Command source: derived from source entry results; "
                "no extra training command."
            )
            lines.append("")

    hook_entries = [entry for entry in entries if entry.required_hooks]
    lines.extend(
        [
            "## Remaining Code Hooks Needed",
            "",
            "| Entry | Required hook |",
            "|---|---|",
        ]
    )
    for entry in hook_entries:
        lines.append(f"| `{entry.entry_id}` | {' '.join(entry.required_hooks)} |")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "Smoke command-generation check:",
            "",
            "```bash",
            "source .venv/bin/activate",
            "python benchmarks/step2_transformer_baseline_fairness_ledger.py "
            "--check --print-commands --preset smoke --entry no_replay_current_gate",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write stable pretty JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("docs/research/step2_new_directions/transformer_baseline_fairness_ledger.json"),
    )
    parser.add_argument(
        "--md-path",
        type=Path,
        default=Path("docs/research/step2_new_directions/transformer_baseline_fairness_ledger.md"),
    )
    parser.add_argument("--check", action="store_true", help="Validate ledger consistency.")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print generated shell commands for one preset/entry without running them.",
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "validation", "lockbox"),
        default="smoke",
        help="Preset used by --print-commands.",
    )
    parser.add_argument(
        "--entry",
        default=None,
        help="Ledger entry id for --print-commands. Defaults to split precommand plus all entries.",
    )
    return parser.parse_args()


def print_commands(entry_id: str | None, preset: str) -> None:
    """Print generated shell commands for smoke validation."""
    print("# environment")
    print("source .venv/bin/activate")
    print("# split precondition")
    print(shell_join(split_materialization_command(preset).command))
    for entry in ledger_entries():
        if entry_id is not None and entry.entry_id != entry_id:
            continue
        commands = commands_for_entry(entry, preset)
        if not commands:
            print(f"# {entry.entry_id}: derived from {', '.join(entry.source_entries)}")
            continue
        for command in commands:
            print(f"# {entry.entry_id}: {command.status}")
            print(shell_join(command.command))


def main() -> int:
    """Generate or check the ledger."""
    args = parse_args()
    payload = ledger_payload()
    if args.check:
        check_entries(ledger_entries())
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.md_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_path, payload)
    write_markdown(args.md_path, payload)
    if args.print_commands:
        print_commands(args.entry, args.preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
