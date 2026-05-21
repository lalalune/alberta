"""Analysis tools for bsuite benchmark results.

Load CSV results, compare agents, generate plots, and export summaries.
Supports both bsuite's built-in scoring (standard mode) and online
performance metrics (continuing mode).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from bsuite.logging import csv_load
except ModuleNotFoundError:
    from benchmarks.bsuite._bsuite_path import bsuite_missing_message

    class _MissingCSVLoad:
        """Lazy bsuite CSV loader used when helper code runs without bsuite."""

        @staticmethod
        def load_bsuite(path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
            raise ModuleNotFoundError(bsuite_missing_message())

    csv_load = _MissingCSVLoad()

PREFERRED_CONTROL_METRICS = ("total_regret", "episode_return", "reward")
LOWER_IS_BETTER = {"total_regret": True, "episode_return": False, "reward": False}
_SEEDED_AGENT_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def load_results(
    save_path: str,
    agent_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load CSV results for all agents.

    Parameters
    ----------
    save_path : str
        Base directory containing agent subdirectories.
    agent_names : list of str, optional
        Specific agent names to load. If None, loads all subdirectories.

    Returns
    -------
    dict mapping agent_name -> DataFrame
        Results for each agent.
    """
    base_path = Path(save_path)
    results: dict[str, pd.DataFrame] = {}

    if agent_names is None:
        agent_dirs = [
            d
            for d in base_path.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        ]
        agent_names = [d.name for d in agent_dirs]

    for name in agent_names:
        agent_dir = base_path / name
        if not agent_dir.exists():
            continue
        try:
            df, _ = csv_load.load_bsuite(str(agent_dir))
            results[name] = df
        except Exception as e:
            print(f"Warning: Could not load results for {name}: {e}")

    return results


def load_representation_logs(
    save_path: str,
    agent_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load representation utility logs for an agent.

    Parameters
    ----------
    save_path : str
        Base directory containing agent subdirectories.
    agent_name : str
        Agent name.

    Returns
    -------
    dict mapping bsuite_id -> list of snapshots
    """
    agent_dir = Path(save_path) / agent_name
    logs: dict[str, list[dict[str, Any]]] = {}

    for json_file in agent_dir.glob("representation_*.json"):
        # Extract bsuite_id from filename: representation_catch_0.json -> catch/0
        name = json_file.stem.replace("representation_", "")
        bsuite_id = name.replace("_", "/", 1)
        with open(json_file) as f:
            logs[bsuite_id] = json.load(f)

    return logs


def compute_online_metrics(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Compute online performance metrics from bsuite results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw results DataFrame.
    window : int
        Smoothing window for running statistics. Default: 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional online metrics.
    """
    result = df.copy()

    if "total_regret" in result.columns:
        result["regret_rate"] = result.groupby("bsuite_id")["total_regret"].diff().fillna(0)
        result["running_regret_rate"] = (
            result.groupby("bsuite_id")["regret_rate"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    if "reward" in result.columns:
        result["running_reward"] = (
            result.groupby("bsuite_id")["reward"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    return result


def _split_agent_seed(agent_name: str) -> tuple[str, int]:
    """Split ``agent_seed3`` names emitted by run_sweep comparison mode."""
    match = _SEEDED_AGENT_RE.match(agent_name)
    if match is None:
        return agent_name, 0
    return match.group("base"), int(match.group("seed"))


def _metric_direction(metric: str) -> int:
    """Return +1 when higher is better and -1 when lower is better."""
    return -1 if LOWER_IS_BETTER.get(metric, False) else 1


def _improvement(candidate: pd.Series, baseline: pd.Series, metric: pd.Series) -> pd.Series:
    """Compute signed candidate improvement over baseline per metric."""
    lower_is_better = metric.map(lambda name: LOWER_IS_BETTER.get(str(name), False))
    return pd.Series(
        np.where(lower_is_better, baseline - candidate, candidate - baseline),
        index=candidate.index,
    )


def final_metric_rows(
    results: dict[str, pd.DataFrame],
    metric: str,
    experiments: list[str] | None = None,
) -> pd.DataFrame:
    """Return the final non-null metric value per agent, seed, and bsuite id."""
    rows: list[dict[str, Any]] = []
    experiment_filter = set(experiments or [])

    for output_agent, df in sorted(results.items()):
        if metric not in df.columns or "bsuite_id" not in df.columns:
            continue
        base_agent, seed = _split_agent_seed(output_agent)
        working = df.copy()
        working["experiment"] = working["bsuite_id"].str.split("/").str[0]
        if experiment_filter:
            working = working[working["experiment"].isin(experiment_filter)]
        if working.empty:
            continue

        for bsuite_id, group in working.groupby("bsuite_id", sort=True):
            values = group[metric].dropna()
            if values.empty:
                continue
            experiment_name = str(group["experiment"].iloc[0])
            rows.append(
                {
                    "agent": base_agent,
                    "output_agent": output_agent,
                    "seed": seed,
                    "bsuite_id": bsuite_id,
                    "experiment": experiment_name,
                    "metric": metric,
                    "value": float(values.iloc[-1]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "agent",
                "output_agent",
                "seed",
                "bsuite_id",
                "experiment",
                "metric",
                "value",
            ]
        )
    return pd.DataFrame(rows).sort_values(["bsuite_id", "seed", "agent"]).reset_index(
        drop=True
    )


def final_preferred_metric_rows(
    results: dict[str, pd.DataFrame],
    experiments: list[str] | None = None,
) -> pd.DataFrame:
    """Return final rows using each task's first available control metric."""
    rows: list[pd.DataFrame] = []

    for metric in PREFERRED_CONTROL_METRICS:
        metric_rows = final_metric_rows(results, metric=metric, experiments=experiments)
        if metric_rows.empty:
            continue
        metric_rows = metric_rows.copy()
        metric_rows["_metric_rank"] = PREFERRED_CONTROL_METRICS.index(metric)
        rows.append(metric_rows)

    if not rows:
        return final_metric_rows(results, metric=PREFERRED_CONTROL_METRICS[0])

    all_rows = pd.concat(rows, ignore_index=True)
    task_metric = (
        all_rows.groupby(["bsuite_id", "experiment", "_metric_rank", "metric"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values(["bsuite_id", "_metric_rank"])
        .drop_duplicates(["bsuite_id"], keep="first")
    )
    selected = all_rows.merge(
        task_metric[["bsuite_id", "metric"]],
        on=["bsuite_id", "metric"],
        how="inner",
    )
    return (
        selected.drop(columns=["_metric_rank"])
        .sort_values(["bsuite_id", "seed", "agent"])
        .reset_index(drop=True)
    )


def compare_sarsa_vs_q(
    results: dict[str, pd.DataFrame],
    q_agent: str = "autostep",
    sarsa_agent: str = "sarsa",
    metric: str = "total_regret",
    experiments: list[str] | None = None,
) -> pd.DataFrame:
    """Pair final SARSA and Q-learning metrics by seed and bsuite id."""
    if metric == "auto":
        return compare_sarsa_vs_q_preferred_metric(
            results,
            q_agent=q_agent,
            sarsa_agent=sarsa_agent,
            experiments=experiments,
        )

    rows = final_metric_rows(results, metric=metric, experiments=experiments)
    if rows.empty:
        return pd.DataFrame()
    q_rows = rows[rows["agent"] == q_agent].rename(columns={"value": "q_value"})
    sarsa_rows = rows[rows["agent"] == sarsa_agent].rename(
        columns={"value": "sarsa_value"}
    )
    pairs = q_rows.merge(
        sarsa_rows,
        on=["seed", "bsuite_id", "experiment", "metric"],
        suffixes=("_q", "_sarsa"),
    )
    if pairs.empty:
        return pairs
    pairs["improvement_vs_q"] = _improvement(
        pairs["sarsa_value"],
        pairs["q_value"],
        pairs["metric"],
    )
    keep = [
        "seed",
        "bsuite_id",
        "experiment",
        "metric",
        "q_value",
        "sarsa_value",
        "improvement_vs_q",
    ]
    return pairs[keep].sort_values(["bsuite_id", "seed"]).reset_index(drop=True)


def compare_sarsa_vs_q_preferred_metric(
    results: dict[str, pd.DataFrame],
    q_agent: str = "autostep",
    sarsa_agent: str = "sarsa",
    experiments: list[str] | None = None,
) -> pd.DataFrame:
    """Compare SARSA and Q-learning using each task's preferred metric."""
    rows = final_preferred_metric_rows(results, experiments=experiments)
    if rows.empty:
        return pd.DataFrame()
    q_rows = rows[rows["agent"] == q_agent].rename(columns={"value": "q_value"})
    sarsa_rows = rows[rows["agent"] == sarsa_agent].rename(
        columns={"value": "sarsa_value"}
    )
    pairs = q_rows.merge(
        sarsa_rows,
        on=["seed", "bsuite_id", "experiment", "metric"],
        suffixes=("_q", "_sarsa"),
    )
    if pairs.empty:
        return pairs
    pairs["improvement_vs_q"] = _improvement(
        pairs["sarsa_value"],
        pairs["q_value"],
        pairs["metric"],
    )
    keep = [
        "seed",
        "bsuite_id",
        "experiment",
        "metric",
        "q_value",
        "sarsa_value",
        "improvement_vs_q",
    ]
    return pairs[keep].sort_values(["bsuite_id", "seed"]).reset_index(drop=True)


def compare_step4_control(
    results: dict[str, pd.DataFrame],
    control_agents: list[str] | None = None,
    baseline_agent: str = "autostep",
    metric: str = "auto",
    experiments: list[str] | None = None,
) -> pd.DataFrame:
    """Compare Step 4 control candidates against a baseline Q learner."""
    agents = control_agents or [baseline_agent, "sarsa", "actor_critic"]
    rows = (
        final_preferred_metric_rows(results, experiments=experiments)
        if metric == "auto"
        else final_metric_rows(results, metric=metric, experiments=experiments)
    )
    if rows.empty:
        return pd.DataFrame()

    rows = rows[rows["agent"].isin(agents)]
    values = rows.pivot_table(
        index=["seed", "bsuite_id", "experiment", "metric"],
        columns="agent",
        values="value",
        aggfunc="last",
    ).reset_index()
    if baseline_agent not in values.columns:
        return pd.DataFrame()

    for agent_name in agents:
        if agent_name == baseline_agent or agent_name not in values.columns:
            continue
        values[f"{agent_name}_improvement_vs_{baseline_agent}"] = _improvement(
            values[agent_name],
            values[baseline_agent],
            values["metric"],
        )

    return values.sort_values(["bsuite_id", "seed"]).reset_index(drop=True)


def _summarize_pairs(
    pairs: pd.DataFrame,
    improvement_columns: list[str],
) -> pd.DataFrame:
    """Summarize paired comparison rows by experiment and overall."""
    if pairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for experiment_name, group in pairs.groupby("experiment", sort=True):
        row: dict[str, Any] = {
            "scope": experiment_name,
            "n": int(len(group)),
            "metrics": ", ".join(sorted(group["metric"].unique())),
        }
        for col in improvement_columns:
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_wins"] = int((group[col] > 0).sum())
        rows.append(row)

    overall: dict[str, Any] = {
        "scope": "overall",
        "n": int(len(pairs)),
        "metrics": ", ".join(sorted(pairs["metric"].unique())),
    }
    for col in improvement_columns:
        if col in pairs:
            overall[f"{col}_mean"] = float(pairs[col].mean())
            overall[f"{col}_wins"] = int((pairs[col] > 0).sum())
    rows.append(overall)
    return pd.DataFrame(rows)


def format_sarsa_q_report(
    results: dict[str, pd.DataFrame],
    q_agent: str = "autostep",
    sarsa_agent: str = "sarsa",
    metric: str = "total_regret",
    experiments: list[str] | None = None,
) -> str:
    """Format a Markdown SARSA-vs-Q comparison report."""
    pairs = compare_sarsa_vs_q(
        results,
        q_agent=q_agent,
        sarsa_agent=sarsa_agent,
        metric=metric,
        experiments=experiments,
    )
    lines = [
        "# SARSA vs Q-learning",
        "",
        f"Metric: `{metric}`",
        f"Q agent: `{q_agent}`",
        f"SARSA agent: `{sarsa_agent}`",
        "",
    ]
    if pairs.empty:
        lines.append("No paired results found.")
        return "\n".join(lines)

    summary = _summarize_pairs(pairs, ["improvement_vs_q"])
    lines.extend(
        [
            "## Summary",
            "",
            str(summary.to_markdown(index=False)),
            "",
            "## Paired Final Metrics",
            "",
            str(pairs.to_markdown(index=False)),
        ]
    )
    return "\n".join(lines)


def format_step4_control_report(
    results: dict[str, pd.DataFrame],
    control_agents: list[str] | None = None,
    baseline_agent: str = "autostep",
    metric: str = "auto",
    experiments: list[str] | None = None,
) -> str:
    """Format a Markdown Step 4 control comparison report."""
    agents = control_agents or [baseline_agent, "sarsa", "actor_critic"]
    pairs = compare_step4_control(
        results,
        control_agents=agents,
        baseline_agent=baseline_agent,
        metric=metric,
        experiments=experiments,
    )
    improvement_columns = [
        f"{agent}_improvement_vs_{baseline_agent}"
        for agent in agents
        if agent != baseline_agent
    ]
    lines = [
        "# Step 4 Control Comparison",
        "",
        f"Metric: `{metric}`",
        f"Baseline agent: `{baseline_agent}`",
        f"Control agents: `{', '.join(agents)}`",
        "",
    ]
    if pairs.empty:
        lines.append("No paired results found.")
        return "\n".join(lines)

    summary = _summarize_pairs(pairs, improvement_columns)
    lines.extend(
        [
            "## Summary",
            "",
            str(summary.to_markdown(index=False)),
            "",
            "## Paired Final Metrics",
            "",
            str(pairs.to_markdown(index=False)),
        ]
    )
    return "\n".join(lines)


def write_markdown_report(output_path: str | Path, report: str) -> None:
    """Write a Markdown report, creating parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def compare_agents_bar(
    results: dict[str, pd.DataFrame],
    metric: str = "total_regret",
    experiment_name: str | None = None,
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Create bar chart comparing agents on a metric.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    metric : str
        Column name to compare. Default: 'total_regret'.
    experiment_name : str, optional
        Filter to this experiment. If None, uses all data.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison figure.
    """
    agent_names = []
    means = []
    stds = []

    for name, df in sorted(results.items()):
        if experiment_name:
            df = df[df["bsuite_id"].str.startswith(experiment_name)]
        if metric not in df.columns:
            continue

        # Get final value per bsuite_id
        final_vals = df.groupby("bsuite_id")[metric].last()
        agent_names.append(name)
        means.append(float(final_vals.mean()))
        stds.append(float(final_vals.std()))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agent_names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Agent Comparison: {metric}")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_learning_curves(
    results: dict[str, pd.DataFrame],
    bsuite_id: str,
    metric: str = "total_regret",
    window: int = 100,
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot learning curves for all agents on a single bsuite_id.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    bsuite_id : str
        The bsuite experiment id to plot.
    metric : str
        Column name to plot. Default: 'total_regret'.
    window : int
        Smoothing window. Default: 100.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The learning curves figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in sorted(results.items()):
        mask = df["bsuite_id"] == bsuite_id
        agent_df = df[mask].copy()
        if agent_df.empty or metric not in agent_df.columns:
            continue

        values = agent_df[metric].values
        if window > 1 and len(values) > window:
            smoothed = pd.Series(values).rolling(window, min_periods=1).mean().values
        else:
            smoothed = values

        ax.plot(smoothed, label=name, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{bsuite_id}: {metric}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_representation_evolution(
    logs: list[dict[str, Any]],
    metric: str = "trunk_step_sizes",
    title: str | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot representation utility evolution over time.

    Parameters
    ----------
    logs : list of dict
        Representation snapshots from AlbertaAgent.
    metric : str
        Key in snapshot dicts to plot. Default: 'trunk_step_sizes'.
    title : str, optional
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The representation evolution figure.
    """
    steps = [snap["step"] for snap in logs]
    values = np.array([snap[metric] for snap in logs])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(values.shape[1]):
        ax.plot(steps, values[:, i], alpha=0.5, label=f"Layer {i}")

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Representation: {metric}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def generate_summary_table(
    results: dict[str, pd.DataFrame],
    metric: str = "total_regret",
    experiments: list[str] | None = None,
    fmt: str = "markdown",
) -> str:
    """Generate a summary table comparing agents across experiments.

    Parameters
    ----------
    results : dict mapping agent_name -> DataFrame
        Results for each agent.
    metric : str
        Column to summarize. Default: 'total_regret'.
    experiments : list of str, optional
        Experiment names to include. If None, includes all.
    fmt : str
        Output format: 'markdown' or 'latex'. Default: 'markdown'.

    Returns
    -------
    str
        Formatted table.
    """
    # Collect data
    rows: list[dict[str, Any]] = []
    agent_names = sorted(results.keys())

    # Get all experiment names from data
    all_experiments: set[str] = set()
    for df in results.values():
        all_experiments.update(df["bsuite_id"].str.split("/").str[0].unique())

    if experiments:
        all_experiments = {e for e in all_experiments if e in experiments}

    for exp_name in sorted(all_experiments):
        row: dict[str, Any] = {"experiment": exp_name}
        for agent_name in agent_names:
            df = results[agent_name]
            mask = df["bsuite_id"].str.startswith(exp_name)
            exp_df = df[mask]
            if exp_df.empty or metric not in exp_df.columns:
                row[agent_name] = "N/A"
                continue
            final_vals = exp_df.groupby("bsuite_id")[metric].last()
            mean = final_vals.mean()
            std = final_vals.std()
            row[agent_name] = f"{mean:.1f} +/- {std:.1f}"
        rows.append(row)

    # Format
    if not rows:
        return "No results to display."

    summary_df = pd.DataFrame(rows)

    if fmt == "latex":
        return str(summary_df.to_latex(index=False))
    return str(summary_df.to_markdown(index=False))


def main() -> None:
    """CLI for analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bsuite results")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/bsuite",
        help="Base directory for results",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        default=None,
        help="Agent names to compare",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="total_regret",
        help="Metric to compare (default: total_regret)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table",
    )

    args = parser.parse_args()

    results = load_results(args.save_path, args.agents)
    if not results:
        print(f"No results found in {args.save_path}")
        return

    print(f"Loaded results for agents: {list(results.keys())}")

    if args.summary:
        table = generate_summary_table(results, metric=args.metric)
        print("\n" + table)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        compare_agents_bar(
            results,
            metric=args.metric,
            output_path=str(output_dir / "agent_comparison.png"),
        )


if __name__ == "__main__":
    main()
