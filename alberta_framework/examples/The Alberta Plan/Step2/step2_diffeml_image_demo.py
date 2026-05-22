#!/usr/bin/env python3
"""Run the package DiffEML image demonstration from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alberta_framework.core.diffeml_image import (
    build_config,
    json_default,
    parse_hidden_sizes,
    parse_stage_depths,
    run_demo,
    validate_args,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("digits", "mnist", "cifar"),
        default=["digits", "mnist", "cifar"],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/diffeml_image_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--max-test", type=int, default=300)
    parser.add_argument(
        "--feature-mode",
        choices=("variance_pixels", "threshold_pixels", "detector_thresholds"),
        default="variance_pixels",
        help=(
            "variance_pixels keeps the old median-thresholded high-variance pixels; "
            "threshold_pixels creates DiffLogic-style per-pixel threshold bits; "
            "detector_thresholds thresholds fixed edge/contrast/color features."
        ),
    )
    parser.add_argument("--input-bits", type=int, default=512)
    parser.add_argument(
        "--pixel-thresholds",
        type=int,
        default=1,
        help="Number of fixed threshold bits per raw pixel in threshold_pixels mode.",
    )
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument(
        "--wiring-mode",
        choices=(
            "random",
            "residual_random",
            "butterfly",
            "benes",
            "permuted_butterfly",
            "permuted_benes",
            "class_bank_random",
            "affine_expander",
            "butterfly_class_bank",
            "local",
            "local_hierarchy",
            "local_tree_hierarchy",
        ),
        default="random",
        help=(
            "Fixed circuit topology: random input-skip pairs, residual random "
            "previous-feature/raw-input pairs, butterfly stages, a Benes-style "
            "forward/reverse butterfly schedule, permuted variants, or "
            "class-bank random final layers, deterministic affine-expander wiring, "
            "butterfly class-bank wiring, or image-local EML gate-tree wiring."
        ),
    )
    parser.add_argument(
        "--local-patch-size",
        type=int,
        default=3,
        help="Odd local receptive-field size for local EML wiring.",
    )
    parser.add_argument(
        "--tree-stage-depths",
        type=parse_stage_depths,
        default=(2, 2, 2, 2),
        help=("Comma-separated EML tree depths for local_tree_hierarchy, for example 2,2,3,3."),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--head-l2", type=float, default=1e-4)
    parser.add_argument("--gate-init-scale", type=float, default=0.2)
    parser.add_argument("--threshold-init-scale", type=float, default=0.1)
    parser.add_argument("--direction-init-scale", type=float, default=0.2)
    parser.add_argument("--head-init-scale", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eml-template-depth", type=int, default=2)
    parser.add_argument("--eml-eps", type=float, default=0.05)
    parser.add_argument(
        "--gate-mode",
        choices=("eml_template", "eml_threshold", "truth_table"),
        default="eml_template",
        help=(
            "eml_template evaluates executable nested EML threshold templates; "
            "eml_threshold uses one learned EML threshold per node; "
            "truth_table uses the earlier truth-table interpolation baseline."
        ),
    )
    parser.add_argument(
        "--eml-threshold-temperature",
        type=float,
        default=0.75,
        help="Sigmoid temperature inside relaxed executable EML thresholds.",
    )
    parser.add_argument(
        "--hard-loss-weight",
        type=float,
        default=0.5,
        help="Straight-through loss weight on the hardened EML circuit features.",
    )
    parser.add_argument(
        "--input-drop-rate",
        type=float,
        default=0.0,
        help="Training-only binary input bit drop rate for circuit regularization.",
    )
    parser.add_argument(
        "--feature-drop-rate",
        type=float,
        default=0.0,
        help="Training-only final EML feature drop rate before the classifier head.",
    )
    parser.add_argument(
        "--residual-gate",
        choices=("none", "left", "right", "or"),
        default="none",
        help="Gate selector to bias at initialization for residual/local circuits.",
    )
    parser.add_argument(
        "--residual-gate-bias",
        type=float,
        default=0.0,
        help="Positive logit bias applied to the selected residual gate.",
    )
    parser.add_argument(
        "--head-mode",
        choices=("linear", "group_sum", "class_vote", "signed_class_vote"),
        default="linear",
        help=(
            "Trainable linear head, fixed grouped summation, learned discrete "
            "class-vote readout, or signed class-vote readout."
        ),
    )
    parser.add_argument(
        "--group-sum-tau",
        type=float,
        default=30.0,
        help="Divisor for group_sum logits.",
    )
    parser.add_argument(
        "--readout-entropy-weight",
        type=float,
        default=0.0,
        help="Class-vote-only penalty that hardens soft class assignments.",
    )
    parser.add_argument(
        "--readout-balance-weight",
        type=float,
        default=0.0,
        help="Class-vote-only penalty that discourages collapsed class assignments.",
    )
    parser.add_argument(
        "--packed-eval",
        action="store_true",
        help="Also evaluate the hardened selector as a bit-packed Boolean circuit.",
    )
    parser.add_argument(
        "--compare-mlp",
        action="store_true",
        help="Train a ReLU MLP baseline on the exact same binary features.",
    )
    parser.add_argument(
        "--mlp-hidden-sizes",
        type=parse_hidden_sizes,
        default=(512,),
        help="Comma-separated MLP hidden sizes; use '' for a linear softmax baseline.",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=0,
        help="MLP epochs; 0 reuses --epochs.",
    )
    parser.add_argument("--mlp-step-size", type=float, default=0.001)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--mlp-init-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    """Run the demonstration."""
    args = parse_args()
    validate_args(args)
    payload = run_demo(build_config(args), args.data_dir)
    text = json.dumps(payload, indent=2, default=json_default)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
