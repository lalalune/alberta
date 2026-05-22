# Step 2 Worker E Route Telemetry Analysis

Data inspected:

- `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json`
- `outputs/step2_conclusive_pruned_hedge_blend_stacker006_all_10seed/results.json`
- all `outputs/step2_conclusive_synthpoly_focus_*/results.json` directories present locally

Definitions:

- Current best: `step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed`.
- Previous base: `step2_conclusive_pruned_hedge_blend_stacker006_all_10seed`.
- Gap vs MLP = best same-run fair MLP final-window MSE minus conclusive final-window MSE. Positive is a conclusive win.
- Best MLP oracle is the minimum over `mlp_32x32_s01_no_ln`, `mlp_64x64_s01_no_ln`, `mlp_32x32`, `mlp_h64`, `mlp_h128`, and `mlp_h64_64`.
- Best expert oracle is the minimum over standalone fixed-feature experts: `recursive_features`, `polynomial_features`, `fourier_features`, and `tanh_random_features`.
- Deployment routes are the largest `meta_route_fraction_*` values stored under the conclusive method.

## Executive Readout

The current best fixed the broad `synthetic_polynomial` problem but paid for it with three small-to-moderate regressions on non-polynomial regimes. `synthetic_polynomial` improved from 5/5 wins/losses to 9/1, with mean conclusive final MSE improving from 0.9177 to 0.8854. The residual losses are now localized to `controlled_rare` seeds 8 and 9, `synthetic_compositional` seed 5, and `synthetic_polynomial` seed 6.

The mechanism is clear from route fractions. The safe-polynomial deployment family moved probability mass away from the previous base's all-convex, all-selector, safe-recursive, and stacker-heavy routes. That solved the hard polynomial seeds 3, 7, 8, and 9, but it also displaced good base routes on `controlled_rare` and slightly diluted the tanh/random-feature solution on `synthetic_compositional`.

## Current vs Previous Base

| Benchmark | Current con MSE | Current diff vs MLP | Current W/L | Base con MSE | Base diff vs MLP | Base W/L | Current loss seeds |
|---|---:|---:|---:|---:|---:|---:|---|
| controlled_rare | 0.0499 | +0.0144 | 8/2 | 0.0489 | +0.0154 | 10/0 | 8, 9 |
| synthetic_compositional | 0.2242 | +0.0434 | 9/1 | 0.2194 | +0.0483 | 10/0 | 5 |
| synthetic_polynomial | 0.8854 | +0.0613 | 9/1 | 0.9177 | +0.0290 | 5/5 | 6 |

Interpretation:

- `synthetic_polynomial`: fixed at the aggregate and seed-count level. The current best cuts mean conclusive MSE by 0.0323 and removes four of the previous five MLP losses.
- `controlled_rare`: aggregate damage is small, but seed 8 is a real routing regression. Base matched the polynomial expert at 0.0265; current rose to 0.0703.
- `synthetic_compositional`: the only current loss is tiny, 0.0013 worse than best MLP, but base was also slightly better than the current route on the same seed.

## Current Losing Seeds

| Benchmark | Seed | Conclusive | Best MLP | Best expert | Gap vs MLP | Base conclusive | Dominant deployment routes | Selector internals |
|---|---:|---:|---|---|---:|---:|---|---|
| controlled_rare | 8 | 0.0703 | `mlp_64x64_s01_no_ln` 0.0632 | `polynomial_features` 0.0265 | -0.0071 | 0.0265 | expert_mlp_32x32 25.3%, all_selector 20.5%, safe_polynomial_mlp_h64_64 18.5%, mlp_convex 6.6% | all: polynomial_features 38.3%, tanh_random_features 22.0%, mlp_64x64_s01_no_ln 20.8%; mlp: mlp_64x64_s01_no_ln 64.4%, mlp_32x32 22.9% |
| controlled_rare | 9 | 0.0451 | `mlp_64x64_s01_no_ln` 0.0394 | `polynomial_features` 0.0423 | -0.0058 | 0.0392 | expert_mlp_32x32 20.8%, safe_polynomial_mlp_h128 12.8%, safe_recursive_mlp_32x32 12.5%, safe_polynomial_mlp_64x64_s01_no_ln 11.4% | all: mlp_64x64_s01_no_ln 65.2%, tanh_random_features 9.0%, polynomial_features 8.0%; mlp: mlp_64x64_s01_no_ln 80.4%, mlp_32x32_s01_no_ln 7.6% |
| synthetic_compositional | 5 | 0.1027 | `mlp_h128` 0.1014 | `tanh_random_features` 0.1021 | -0.0013 | 0.1007 | all_convex 59.4%, expert_mlp_32x32 20.8%, safe_polynomial_mlp_h128 4.8%, mlp_convex 3.4% | all: tanh_random_features 96.5%, polynomial_features 2.0%, mlp_h64 1.1%; mlp: mlp_h128 70.6%, mlp_h64 29.1% |
| synthetic_polynomial | 6 | 0.7801 | `mlp_32x32_s01_no_ln` 0.7688 | `polynomial_features` 0.8344 | -0.0112 | 0.8114 | expert_mlp_32x32 20.8%, safe_polynomial_mlp_32x32 14.4%, safe_polynomial_mlp_32x32_s01_no_ln 10.7%, safe_recursive_mlp_32x32 10.4% | all: polynomial_features 40.0%, recursive_features 22.7%, mlp_32x32 16.2%; mlp: mlp_32x32 52.2%, mlp_64x64_s01_no_ln 30.6% |

Loss diagnosis:

- `controlled_rare` seed 8 is the highest-priority failure. The best expert oracle is much better than both the best MLP and current conclusive route. The previous base landed exactly on the polynomial expert final MSE, while the current route spent 25.3% on `expert_mlp_32x32` and 18.5% on `safe_polynomial_mlp_h64_64`.
- `controlled_rare` seed 9 is a near-MLP miss. The internal selectors correctly prefer `mlp_64x64_s01_no_ln`, but the deployment route spreads mass across expert-MLP and safe-polynomial/safe-recursive hybrids.
- `synthetic_compositional` seed 5 is not a structural blocker. The all-selector internals overwhelmingly identify `tanh_random_features` at 96.5%, and the loss is only 0.0013. The regression comes from dilution of the base's all-convex route by safe-polynomial and MLP-convex routes.
- `synthetic_polynomial` seed 6 is the remaining polynomial near miss. The current route improved over base by 0.0313 MSE, but the best MLP is `mlp_32x32_s01_no_ln`, while the MLP selector mostly favors `mlp_32x32` and `mlp_64x64_s01_no_ln`.

## What Fixed Synthetic Polynomial

The previous base lost on synthetic-polynomial seeds 3, 6, 7, 8, and 9. The current best fixes 3, 7, 8, and 9, while seed 6 remains a smaller loss.

| Seed | Base gap | Current gap | Base con | Current con | Best MLP | Best expert | Current top routes | Previous base top routes |
|---:|---:|---:|---:|---:|---|---|---|---|
| 3 | -0.1057 | +0.0033 | 1.7130 | 1.6039 | `mlp_32x32` 1.6073 | `recursive_features` 1.5275 | expert_mlp_32x32 27.4%, safe_polynomial_mlp_32x32 24.8%, safe_polynomial_mlp_32x32_s01_no_ln 9.3% | all_convex 34.1%, expert_mlp_32x32 20.8%, all_selector 16.5% |
| 6 | -0.0425 | -0.0112 | 0.8114 | 0.7801 | `mlp_32x32_s01_no_ln` 0.7688 | `polynomial_features` 0.8344 | expert_mlp_32x32 20.8%, safe_polynomial_mlp_32x32 14.4%, safe_polynomial_mlp_32x32_s01_no_ln 10.7% | safe_recursive_mlp_32x32_s01_no_ln 26.6%, expert_mlp_32x32 20.8%, all_convex 12.2% |
| 7 | -0.0062 | +0.0043 | 0.1811 | 0.1706 | `mlp_64x64_s01_no_ln` 0.1750 | `recursive_features` 0.1939 | expert_mlp_32x32 22.5%, safe_recursive_mlp_64x64_s01_no_ln 18.8%, safe_polynomial_mlp_64x64_s01_no_ln 10.2% | safe_recursive_mlp_64x64_s01_no_ln 38.5%, expert_mlp_32x32 20.8%, stacked_predictions 13.6% |
| 8 | -0.0027 | +0.0816 | 1.2842 | 1.2000 | `mlp_32x32_s01_no_ln` 1.2816 | `polynomial_features` 1.2699 | all_selector 21.3%, expert_mlp_32x32 20.8%, safe_polynomial_mlp_32x32 16.4% | all_selector 37.6%, stacked_predictions 22.2%, expert_mlp_32x32 20.8% |
| 9 | -0.0209 | +0.0591 | 1.3086 | 1.2286 | `mlp_64x64_s01_no_ln` 1.2877 | `recursive_features` 1.3040 | expert_mlp_32x32 20.8%, safe_polynomial_mlp_32x32 13.9%, stacked_predictions 12.1% | stacked_predictions 32.8%, all_selector 23.0%, expert_mlp_32x32 20.8% |

The decisive fix is not just "more selector." The best focused run is `safe_poly_plus_selector100`, which adds safe-polynomial routes while keeping a 100-step selector cadence. Runs without safe-polynomial routes generally remain around 2 to 6 wins and 4 to 8 losses.

## Focused Synthetic-Polynomial Ablation Map

| Focus run | Diff vs MLP | W/L | Loss seeds | Con MSE | Route signature |
|---|---:|---:|---|---:|---|
| safe_poly_plus_selector100 | +0.0613 | 9/1 | 6 | 0.8854 | safe_poly 0.36, safe_rec 0.16, stack 0.09, all_conv 0.07, all_sel 0.07, expert_mlp 0.25 |
| safe_poly_plus_recursive | +0.0570 | 5/5 | 0,1,3,6,7 | 0.8897 | safe_poly 0.60, safe_rec 0.10, stack 0.03, all_conv 0.04, all_sel 0.03, expert_mlp 0.21 |
| safe_poly_plus_selector_score_window | +0.0569 | 5/5 | 0,1,3,6,7 | 0.8898 | safe_poly 0.60, safe_rec 0.09, stack 0.03, all_conv 0.04, all_sel 0.03, expert_mlp 0.21 |
| safe_poly_plus_stacker012 | +0.0566 | 5/5 | 0,1,3,6,7 | 0.8901 | safe_poly 0.57, safe_rec 0.10, stack 0.05, all_conv 0.04, all_sel 0.03, expert_mlp 0.21 |
| safe_poly_plus_poly03_selector100 | +0.0514 | 8/2 | 6,7 | 0.8953 | safe_poly 0.31, safe_rec 0.17, stack 0.08, all_conv 0.06, all_sel 0.13, expert_mlp 0.24 |
| safe_poly_only | +0.0497 | 5/5 | 0,1,3,6,7 | 0.8970 | safe_poly 0.62, safe_rec 0.00, stack 0.05, all_conv 0.05, all_sel 0.07, expert_mlp 0.21 |
| safe_poly_plus_poly03 | +0.0448 | 6/4 | 0,5,6,7 | 0.9018 | safe_poly 0.51, safe_rec 0.14, stack 0.02, all_conv 0.05, all_sel 0.08, expert_mlp 0.21 |
| selector_score_window | +0.0392 | 5/5 | 0,1,3,6,7 | 0.9074 | safe_poly 0.00, safe_rec 0.22, stack 0.21, all_conv 0.21, all_sel 0.15, expert_mlp 0.21 |
| poly_step03 | +0.0380 | 6/4 | 0,5,6,7 | 0.9087 | safe_poly 0.00, safe_rec 0.23, stack 0.11, all_conv 0.27, all_sel 0.18, expert_mlp 0.21 |
| stacker012 | +0.0351 | 4/6 | 0,1,3,6,7,9 | 0.9116 | safe_poly 0.00, safe_rec 0.19, stack 0.27, all_conv 0.20, all_sel 0.14, expert_mlp 0.21 |
| selector100 | +0.0337 | 5/5 | 0,3,5,6,9 | 0.9130 | safe_poly 0.00, safe_rec 0.21, stack 0.21, all_conv 0.16, all_sel 0.14, expert_mlp 0.26 |
| selector150 | +0.0277 | 3/7 | 0,1,3,5,6,7,8 | 0.9190 | safe_poly 0.00, safe_rec 0.21, stack 0.24, all_conv 0.15, all_sel 0.14, expert_mlp 0.24 |
| no_safe_recursive | +0.0262 | 5/5 | 0,3,6,7,8 | 0.9205 | safe_poly 0.00, safe_rec 0.00, stack 0.31, all_conv 0.28, all_sel 0.19, expert_mlp 0.21 |
| no_stacker | +0.0213 | 2/8 | 0,1,3,4,6,7,8,9 | 0.9253 | safe_poly 0.00, safe_rec 0.31, stack 0.00, all_conv 0.22, all_sel 0.19, expert_mlp 0.26 |
| poly_step07 | +0.0208 | 5/5 | 1,3,6,7,9 | 0.9259 | safe_poly 0.00, safe_rec 0.18, stack 0.26, all_conv 0.11, all_sel 0.24, expert_mlp 0.21 |
| stacker003 | +0.0200 | 3/7 | 0,3,5,6,7,8,9 | 0.9266 | safe_poly 0.00, safe_rec 0.26, stack 0.15, all_conv 0.22, all_sel 0.16, expert_mlp 0.21 |
| selector200 | +0.0170 | 3/7 | 0,1,3,5,6,7,8 | 0.9297 | safe_poly 0.00, safe_rec 0.23, stack 0.23, all_conv 0.15, all_sel 0.17, expert_mlp 0.22 |
| no_stacker_no_safe_recursive | +0.0140 | 4/6 | 0,1,3,6,7,8 | 0.9327 | safe_poly 0.00, safe_rec 0.00, stack 0.00, all_conv 0.32, all_sel 0.25, expert_mlp 0.37 |

The focused map says `safe_poly_plus_selector100` is the only 9/1 synthetic-polynomial route in this set. High safe-polynomial mass alone is not enough; the 0.57 to 0.62 safe-poly variants still go 5/5. The winning setting keeps safe-polynomial mass moderate, preserves expert-MLP exposure, and uses selector100 to avoid overcommitting to the wrong route family.

## Recommended Next Experiments

1. Add a safe-polynomial deployment guard for non-polynomial regimes. Minimal target: preserve `safe_poly_plus_selector100` behavior on `synthetic_polynomial`, but cap or block safe-polynomial MLP routes when the all-selector is dominated by `tanh_random_features` or when `polynomial_features` is not a clear fixed-expert winner. This should directly test the `synthetic_compositional` seed 5 and `controlled_rare` seed 9 regressions.

2. Add an expert-oracle fallback for controlled rare events. Seed 8 shows the strongest signal: `polynomial_features` is 0.0265, the previous base conclusive is 0.0265, and current conclusive is 0.0703. A low-risk test is a final-window route-risk guard that prefers the fixed expert when all-selector polynomial mass is high and the polynomial expert's recent loss is below the deployed hybrid route.

3. Rescue synthetic-polynomial seed 6 by biasing toward the no-LN small MLP route inside polynomial-safe routing. The residual gap is small, but the best MLP is `mlp_32x32_s01_no_ln` while the MLP selector favors `mlp_32x32` and `mlp_64x64_s01_no_ln`. A focused run should test stronger no-LN promotion or an identity-preserving safe-polynomial route that follows the current best MLP selector more tightly.

## Top Hypotheses

1. Safe-polynomial routes are necessary for synthetic-polynomial closure, but their deployment must be context-gated. Without gating, they steal probability mass from good rare/compositional routes.

2. `controlled_rare` seed 8 is not an MLP-capacity problem. It is an expert-oracle routing miss where the polynomial fixed expert is already the right answer.

3. The remaining `synthetic_polynomial` loss is a route identity problem, not a missing route family. The correct family is active, but the route mix underweights the specific no-LN small MLP that wins seed 6.
