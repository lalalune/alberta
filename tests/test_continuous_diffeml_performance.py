"""Tests for the continuous DiffEML performance probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_performance_module() -> Any:
    """Import the performance probe from its path with spaces."""
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "alberta_framework" / "examples"
        / "The Alberta Plan"
        / "Step2"
        / "step2_continuous_diffeml_performance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "step2_continuous_diffeml_performance",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load continuous DiffEML performance module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PERF = load_performance_module()


def tiny_config() -> Any:
    """Return a tiny benchmark config for non-timing helper tests."""
    return PERF.BenchmarkConfig(
        batch_size=4,
        input_dim=5,
        output_dim=2,
        width=6,
        depth=2,
        repeats=1,
        warmups=0,
        seed=0,
        output_dir="unused",
    )


def test_parameter_summary_shows_sparse_compilation_compression() -> None:
    """Compiled sparse inference should store far less than soft selectors."""
    state = PERF.init_benchmark_state(tiny_config())
    counts = PERF.parameter_summary(state)

    assert counts["sparse_compiled_with_indices"] < counts["sparse_soft_trainable"]
    assert counts["sparse_compiled_scalars_only"] < counts["sparse_compiled_with_indices"]
    assert counts["sparse_soft_to_compiled_ratio"] > 1.0


def test_operation_estimates_separate_training_and_compiled_costs() -> None:
    """The harness should expose soft-selector work apart from compiled gathers."""
    estimates = PERF.operation_estimates(tiny_config())

    assert estimates["eml_nodes_per_example"] == 12
    assert estimates["sparse_compiled_gathers_per_example"] == 24
    assert estimates["approx_lut_entries"] == 514
    assert estimates["approx_poly_sqrt_per_example"] == 12
    assert (
        estimates["sparse_soft_selector_muls_per_example"]
        > estimates["sparse_compiled_gathers_per_example"]
    )
