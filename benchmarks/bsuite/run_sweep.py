#!/usr/bin/env python3
"""Run bsuite sweeps across agents, seeds, and experiment families."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alberta_framework import Timer  # noqa: E402
from benchmarks.bsuite._bsuite_path import (  # noqa: E402
    add_bsuite_to_path,
    bsuite_missing_message,
)
from benchmarks.bsuite.configs import (  # noqa: E402
    ALL_EXPERIMENTS,
    BOTTLENECK_AGENTS,
    PRIMARY_EXPERIMENTS,
    SECONDARY_EXPERIMENTS,
    STANDARD_AGENTS,
)

logger = logging.getLogger(__name__)
DEFAULT_COMPARISON_STEPS = 200


def require_bsuite() -> tuple[Any, Any, Any]:
    """Import bsuite modules after local helper functions have been imported."""
    add_bsuite_to_path()
    try:
        return (
            importlib.import_module("bsuite"),
            importlib.import_module("bsuite.sweep"),
            importlib.import_module("bsuite.baselines.experiment"),
        )
    except ModuleNotFoundError as exc:
        if exc.name == "bsuite":
            raise ModuleNotFoundError(bsuite_missing_message()) from exc
        raise


class SweepJob(NamedTuple):
    """A concrete bsuite run with a stable output directory name."""

    agent_name: str
    output_agent_name: str
    bsuite_id: str
    seed: int


def seeded_agent_name(agent_name: str, seed: int, include_seed: bool) -> str:
    """Return the output directory name for an agent/seed run."""
    if include_seed:
        return f"{agent_name}_seed{seed}"
    return agent_name


def build_sweep_jobs(
    agent_names: list[str],
    bsuite_ids: list[str],
    seeds: list[int],
    include_seed_in_name: bool,
) -> list[SweepJob]:
    """Build the concrete run list for a sweep."""
    return [
        SweepJob(
            agent_name=agent_name,
            output_agent_name=seeded_agent_name(
                agent_name,
                seed,
                include_seed=include_seed_in_name,
            ),
            bsuite_id=bsuite_id,
            seed=seed,
        )
        for agent_name in agent_names
        for seed in seeds
        for bsuite_id in bsuite_ids
    ]


def get_bsuite_ids_for_experiment(
    experiment_name: str,
    sweep_ids: list[str] | None = None,
) -> list[str]:
    """Get all bsuite_ids for a given experiment name."""
    if sweep_ids is None:
        _, sweep, _ = require_bsuite()
        ids = list(sweep.SWEEP)
    else:
        ids = sweep_ids
    return [
        bsuite_id for bsuite_id in ids if bsuite_id.split("/")[0] == experiment_name
    ]


def experiment_names_from_bsuite_ids(bsuite_ids: list[str]) -> list[str]:
    """Return experiment names from explicit bsuite ids, preserving first use."""
    return list(dict.fromkeys(bsuite_id.split("/")[0] for bsuite_id in bsuite_ids))


def resolve_bsuite_ids(
    experiment_names: list[str],
    explicit_bsuite_ids: list[str] | None = None,
    max_ids_per_experiment: int | None = None,
    sweep_ids: list[str] | None = None,
) -> list[str]:
    """Resolve CLI experiment/id selection into concrete bsuite ids."""
    if explicit_bsuite_ids:
        return list(dict.fromkeys(explicit_bsuite_ids))

    all_bsuite_ids: list[str] = []
    for exp_name in experiment_names:
        ids = get_bsuite_ids_for_experiment(exp_name, sweep_ids=sweep_ids)
        if max_ids_per_experiment is not None:
            ids = ids[:max_ids_per_experiment]
        if not ids:
            logger.warning("No bsuite_ids found for experiment: %s", exp_name)
        all_bsuite_ids.extend(ids)
    return all_bsuite_ids


def run_agent_on_id(
    agent_name: str,
    bsuite_id: str,
    save_path: str,
    mode: str = "continuing",
    num_steps: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
    verbose: bool = False,
    output_agent_name: str | None = None,
) -> None:
    """Run a single agent on a single bsuite_id."""
    bsuite, sweep, experiment = require_bsuite()
    from benchmarks.bsuite.run_single import make_agent, run_continuing
    from benchmarks.bsuite.wrappers import ContinuingWrapper

    agent_save_path = str(Path(save_path) / (output_agent_name or agent_name))

    try:
        if mode == "standard":
            env = bsuite.load_and_record(
                bsuite_id=bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=overwrite,
            )
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )
            num_episodes = sweep.EPISODES[bsuite_id]
            experiment.run(agent, env, num_episodes, verbose=verbose)
        else:
            raw_env = bsuite.load_and_record(
                bsuite_id=bsuite_id,
                save_path=agent_save_path,
                logging_mode="csv",
                overwrite=overwrite,
            )
            env = ContinuingWrapper(raw_env, mode="continuing")
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )
            steps = num_steps
            if steps is None:
                num_episodes = sweep.EPISODES[bsuite_id]
                steps = num_episodes * 1000
            run_continuing(agent, env, steps)
    except Exception:
        logger.exception("Failed: %s on %s", agent_name, bsuite_id)
        raise


def run_continual_sequence(
    agent_name: str,
    bsuite_ids: list[str],
    save_path: str,
    steps_per_task: int = 10000,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Run a single persistent agent across a sequence of environments."""
    bsuite, _, _ = require_bsuite()
    from benchmarks.bsuite.run_single import make_agent, run_continuing
    from benchmarks.bsuite.wrappers import ContinuingWrapper

    agent_save_path = str(Path(save_path) / f"{agent_name}_continual")
    agent = None

    for task_idx, bsuite_id in enumerate(bsuite_ids):
        print(f"  Task {task_idx + 1}/{len(bsuite_ids)}: {bsuite_id}")

        raw_env = bsuite.load_and_record(
            bsuite_id=bsuite_id,
            save_path=agent_save_path,
            logging_mode="csv",
            overwrite=overwrite,
        )
        env = ContinuingWrapper(raw_env, mode="continuing")

        if agent is None:
            agent = make_agent(
                agent_type=agent_name,
                obs_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                config_name=agent_name,
                seed=seed,
            )

        run_continuing(agent, env, steps_per_task)
        print(f"    Completed {steps_per_task} steps on {bsuite_id}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run bsuite sweep across agents")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/bsuite",
        help="Base directory for results (default: output/bsuite)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=None,
        help="Experiment names to run (e.g., catch catch_scale)",
    )
    parser.add_argument(
        "--bsuite-ids",
        nargs="+",
        type=str,
        default=None,
        help="Explicit bsuite ids to run (e.g., catch/0 cartpole/0)",
    )
    parser.add_argument(
        "--max-ids-per-experiment",
        type=int,
        default=None,
        help="Limit each experiment family to its first N bsuite ids",
    )
    parser.add_argument("--all-primary", action="store_true")
    parser.add_argument("--all-secondary", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        default=None,
        help="Agent config names to run",
    )
    parser.add_argument(
        "--sarsa-vs-q",
        action="store_true",
        help="Run SARSA-vs-Q comparison (default agents: autostep, sarsa)",
    )
    parser.add_argument(
        "--step4-comparison",
        action="store_true",
        help="Run Q/SARSA/actor-critic control comparison",
    )
    parser.add_argument(
        "--horde-ac",
        action="store_true",
        help=(
            "Run a Horde-AC vs Q/SARSA/AC comparison "
            "(default agents: autostep_bottleneck, sarsa_bottleneck, "
            "actor_critic, horde_ac)"
        ),
    )
    parser.add_argument("--q-agent", type=str, default="autostep")
    parser.add_argument("--sarsa-agent", type=str, default="sarsa")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to run. Comparison modes default to 0 1 2.",
    )
    parser.add_argument(
        "--comparison-report",
        type=str,
        default=None,
        help="Write a Markdown comparison report after the sweep",
    )
    parser.add_argument("--bottleneck", action="store_true")
    parser.add_argument("--continual-sequence", nargs="+", type=str, default=None)
    parser.add_argument("--steps-per-task", type=int, default=10000)
    parser.add_argument(
        "--mode",
        type=str,
        default="continuing",
        choices=["continuing", "standard"],
    )
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-scythe", action="store_true")

    args = parser.parse_args()

    if args.use_scythe:
        warnings.warn(
            "Scythe integration is a placeholder and currently a no-op.",
            stacklevel=2,
        )

    if args.continual_sequence:
        agent_names = args.agents or STANDARD_AGENTS
        with Timer("Continual sequence sweep"):
            for agent_name in agent_names:
                print(f"\n=== Agent: {agent_name} ===")
                run_continual_sequence(
                    agent_name=agent_name,
                    bsuite_ids=args.continual_sequence,
                    save_path=args.save_path,
                    steps_per_task=args.steps_per_task,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
        return

    if args.bsuite_ids:
        experiment_names = experiment_names_from_bsuite_ids(args.bsuite_ids)
    elif (
        args.sarsa_vs_q or args.step4_comparison or args.horde_ac
    ) and not any(
        [args.all, args.all_primary, args.all_secondary, args.experiments]
    ):
        experiment_names = ["catch", "cartpole"]
    elif args.all:
        experiment_names = ALL_EXPERIMENTS
    elif args.all_primary:
        experiment_names = PRIMARY_EXPERIMENTS
    elif args.all_secondary:
        experiment_names = SECONDARY_EXPERIMENTS
    elif args.experiments:
        experiment_names = args.experiments
    else:
        experiment_names = ["catch"]

    if args.horde_ac:
        agent_names = args.agents or [
            "autostep_bottleneck",
            "sarsa_bottleneck",
            "actor_critic",
            "horde_ac",
        ]
    elif args.step4_comparison:
        agent_names = args.agents or [args.q_agent, args.sarsa_agent, "actor_critic"]
    elif args.sarsa_vs_q:
        agent_names = args.agents or [args.q_agent, args.sarsa_agent]
    else:
        agent_names = args.agents or STANDARD_AGENTS
    if args.bottleneck:
        agent_names = agent_names + BOTTLENECK_AGENTS

    seeds = args.seeds or (
        [0, 1, 2]
        if args.sarsa_vs_q or args.step4_comparison or args.horde_ac
        else [args.seed]
    )
    include_seed_in_name = (
        args.seeds is not None
        or args.sarsa_vs_q
        or args.step4_comparison
        or args.horde_ac
    )
    all_bsuite_ids = resolve_bsuite_ids(
        experiment_names=experiment_names,
        explicit_bsuite_ids=args.bsuite_ids,
        max_ids_per_experiment=args.max_ids_per_experiment,
    )
    jobs = build_sweep_jobs(
        agent_names=agent_names,
        bsuite_ids=all_bsuite_ids,
        seeds=seeds,
        include_seed_in_name=include_seed_in_name,
    )

    num_steps = args.num_steps
    if (
        num_steps is None
        and (args.sarsa_vs_q or args.step4_comparison or args.horde_ac)
        and args.mode == "continuing"
    ):
        num_steps = DEFAULT_COMPARISON_STEPS
        print(
            "Using bounded comparison continuing horizon: "
            f"{num_steps} steps per job. Pass --num_steps to override."
        )

    total_runs = len(jobs)
    print(
        f"Running {total_runs} total experiments "
        f"({len(agent_names)} agents x {len(seeds)} seeds x "
        f"{len(all_bsuite_ids)} bsuite_ids)"
    )

    with Timer("bsuite sweep"):
        for run_count, job in enumerate(jobs, start=1):
            print(
                f"\n[{run_count}/{total_runs}] {job.output_agent_name} "
                f"on {job.bsuite_id}"
            )
            try:
                run_agent_on_id(
                    agent_name=job.agent_name,
                    bsuite_id=job.bsuite_id,
                    save_path=args.save_path,
                    mode=args.mode,
                    num_steps=num_steps,
                    seed=job.seed,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                    output_agent_name=job.output_agent_name,
                )
            except Exception:
                logger.exception(
                    "Failed: %s on %s, continuing...",
                    job.output_agent_name,
                    job.bsuite_id,
                )

    print(f"\nResults saved to {args.save_path}")

    if args.step4_comparison or args.horde_ac:
        from benchmarks.bsuite.analysis import (
            format_step4_control_report,
            load_results,
            write_markdown_report,
        )

        if args.horde_ac:
            baseline_agent = (
                "autostep_bottleneck"
                if "autostep_bottleneck" in agent_names
                else agent_names[0]
            )
        else:
            baseline_agent = args.q_agent
        report = format_step4_control_report(
            load_results(args.save_path),
            control_agents=agent_names,
            baseline_agent=baseline_agent,
            experiments=experiment_names,
            metric="auto",
        )
        print("\n" + report)
        if args.comparison_report:
            write_markdown_report(args.comparison_report, report)
    elif args.sarsa_vs_q or args.comparison_report:
        from benchmarks.bsuite.analysis import (
            format_sarsa_q_report,
            load_results,
            write_markdown_report,
        )

        report = format_sarsa_q_report(
            load_results(args.save_path),
            q_agent=args.q_agent,
            sarsa_agent=args.sarsa_agent,
            experiments=experiment_names,
            metric="auto",
        )
        print("\n" + report)
        if args.comparison_report:
            write_markdown_report(args.comparison_report, report)


if __name__ == "__main__":
    main()
