#!/usr/bin/env python3
"""Bounded Horde-AC canonicality sweep against Q/SARSA controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alberta_framework.benchmarks.bsuite.analysis import (  # noqa: E402
    LOWER_IS_BETTER,
    final_preferred_metric_rows,
    load_results,
)

DEFAULT_AGENTS = [
    "autostep_bottleneck",
    "sarsa_bottleneck",
    "actor_critic",
    "horde_ac",
    "horde_ac_tuned",
    "horde_ac_pairwise",
]
DEFAULT_BSUITE_IDS = ["catch/0", "cartpole/0"]


def _signed_improvement(
    candidate: pd.Series,
    baseline: pd.Series,
    metric: pd.Series,
) -> pd.Series:
    """Return positive values when candidate beats baseline."""
    lower = metric.map(lambda name: LOWER_IS_BETTER.get(str(name), False))
    return pd.Series(
        np.where(lower, baseline - candidate, candidate - baseline),
        index=candidate.index,
    )


def _stats(values: pd.Series) -> dict[str, Any]:
    """Summarize paired improvements."""
    clean = values.dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return {"n": 0}
    std = float(clean.std(ddof=1)) if n > 1 else 0.0
    se = std / float(np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(clean.mean()),
        "std": std,
        "ci95": 1.96 * se,
        "wins": int((clean > 0.0).sum()),
        "ties": int((clean == 0.0).sum()),
        "losses": int((clean < 0.0).sum()),
        "cohen_dz": float(clean.mean() / std) if std > 0.0 else 0.0,
    }


def summarize_results(
    save_path: Path,
    agents: list[str],
    q_agent: str,
    sarsa_agent: str,
) -> dict[str, Any]:
    """Build candidate-vs-Q and candidate-vs-SARSA paired summaries."""
    rows = final_preferred_metric_rows(load_results(str(save_path)))
    rows.to_csv(save_path / "preferred_metric_rows.csv", index=False)
    values = rows.pivot_table(
        index=["seed", "bsuite_id", "experiment", "metric"],
        columns="agent",
        values="value",
        aggfunc="last",
    ).reset_index()
    values.to_csv(save_path / "paired_metric_table.csv", index=False)

    summary: dict[str, Any] = {
        "agents": agents,
        "q_agent": q_agent,
        "sarsa_agent": sarsa_agent,
        "n_rows": int(len(values)),
        "comparisons": {},
    }
    for candidate in agents:
        if candidate in {q_agent, sarsa_agent} or candidate not in values.columns:
            continue
        candidate_summary: dict[str, Any] = {}
        for baseline in (q_agent, sarsa_agent):
            if baseline not in values.columns:
                continue
            improvement = _signed_improvement(
                values[candidate],
                values[baseline],
                values["metric"],
            )
            values[f"{candidate}_vs_{baseline}"] = improvement
            candidate_summary[f"vs_{baseline}"] = {
                "overall": _stats(improvement),
                "by_experiment": {
                    str(experiment): _stats(group[f"{candidate}_vs_{baseline}"])
                    for experiment, group in values.groupby("experiment", sort=True)
                },
            }
        summary["comparisons"][candidate] = candidate_summary

    values.to_csv(save_path / "paired_improvements.csv", index=False)
    with open(save_path / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a compact promotion-oriented summary."""
    lines = [
        "# Horde-AC Canonical Sweep Summary",
        "",
        f"Agents: `{', '.join(summary['agents'])}`",
        f"Q baseline: `{summary['q_agent']}`",
        f"SARSA baseline: `{summary['sarsa_agent']}`",
        "",
        "| candidate | baseline | n | mean improvement | 95% CI | wins | ties | losses | dz |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, comparisons in summary["comparisons"].items():
        for baseline_key, payload in comparisons.items():
            stats = payload["overall"]
            lines.append(
                "| "
                f"{candidate} | {baseline_key.removeprefix('vs_')} | "
                f"{stats.get('n', 0)} | "
                f"{stats.get('mean', 0.0):.4f} | "
                f"+/- {stats.get('ci95', 0.0):.4f} | "
                f"{stats.get('wins', 0)} | "
                f"{stats.get('ties', 0)} | "
                f"{stats.get('losses', 0)} | "
                f"{stats.get('cohen_dz', 0.0):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("output/subagents/horde_ac_canonical_search/sweep"),
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bsuite-ids", nargs="+", default=DEFAULT_BSUITE_IDS)
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--q-agent", default="autostep_bottleneck")
    parser.add_argument("--sarsa-agent", default="sarsa_bottleneck")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only summarize existing CSV outputs.",
    )
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)
    if not args.no_run:
        command = [
            sys.executable,
            "benchmarks/bsuite/run_sweep.py",
            "--horde-ac",
            "--agents",
            *args.agents,
            "--bsuite-ids",
            *args.bsuite_ids,
            "--num_steps",
            str(args.num_steps),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--save_path",
            str(args.save_path),
            "--comparison-report",
            str(args.save_path / "horde_ac_control_report.md"),
            "--overwrite",
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    summary = summarize_results(
        save_path=args.save_path,
        agents=args.agents,
        q_agent=args.q_agent,
        sarsa_agent=args.sarsa_agent,
    )
    write_summary_markdown(summary, args.save_path / "summary.md")
    print((args.save_path / "summary.md").read_text())


if __name__ == "__main__":
    main()
