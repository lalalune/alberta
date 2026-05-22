# Step 2 Worker B Rare-Regime Failure Analysis

Worker scope: controlled rare failure analysis for the current
`step2_conclusive_learner.py` best run. No source files were edited.

## Current Best Diagnosis

Current best artifact:
`outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json`.

Current controlled_rare aggregate:

- Mean final-window MSE delta vs same-run best MLP: `+0.014419 +/- 0.005125`.
- Seed profile: `8/2/0`.
- Losing seeds: seed 8 `-0.007094`, seed 9 `-0.005773`.

Seed 8:

- Conclusive final-window MSE: `0.070299`.
- Best MLP: `mlp_64x64_s01_no_ln`, `0.063205`.
- Best expert: `polynomial_features`, `0.026506`.
- Final-window route mix: `all_selector` 236/300, `mlp_convex` 54/300,
  `stacked_predictions` 5/300, `safe_polynomial_mlp_32x32` 5/300.
- `all_selector` selected `polynomial_features` for all 300 final-window
  steps and was good (`0.016319` on those routed steps).
- `mlp_convex` was the failure route: 54 final-window steps with routed MSE
  `0.317054`; on rare-active final-window steps its routed MSE was `2.419970`.

Seed 9:

- Conclusive final-window MSE: `0.045129`.
- Best MLP and best expert: `mlp_64x64_s01_no_ln`, `0.039356`.
- Final-window route mix churned across safe routes:
  `safe_polynomial_mlp_64x64_s01_no_ln` 58/300,
  `safe_polynomial_mlp_32x32` 53/300,
  `safe_recursive_mlp_32x32_s01_no_ln` 44/300,
  `safe_polynomial_mlp_h64_64` 33/300,
  `safe_polynomial_mlp_h128` 32/300, plus smaller routes.
- The worst rare-active spikes came from safe/selector routes that looked good
  on frequent steps: `safe_polynomial_mlp_h128` rare-step MSE `1.123436`,
  `mlp_selector` rare-step MSE `0.650317`,
  `safe_polynomial_mlp_h64` rare-step MSE `0.499639`.

Interpretation: this is primarily rare-target underweighting in route scoring.
The frequent head dominates the route window; routes with excellent non-rare
loss can carry large rare-step spikes. Seed 8 is late-window route churn into
`mlp_convex`; seed 9 is safe-route churn among routes whose frequent-step loss
masks rare-step risk. It is not a polynomial-specialist overfit in seed 8,
because `polynomial_features` is the oracle best expert there.

## Focused Controlled-Rare Runs

All focused runs used the current-best command base:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks controlled_rare \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

| Variant | Mean diff | Seed profile | Seed 8 | Seed 9 |
|---|---:|---:|---:|---:|
| Current best | `+0.014419` | `8/2/0` | `-0.007094` | `-0.005773` |
| selector-window 150 | `+0.021181` | `9/1/0` | `+0.036743` | `-0.008390` |
| selector-window 200 | `+0.019462` | `9/1/0` | `+0.037115` | `-0.009299` |
| recursive-only safe routes | `+0.012628` | `8/2/0` | `-0.007302` | `-0.001811` |
| disable safe_polynomial | `+0.012628` | `8/2/0` | `-0.007302` | `-0.001811` |
| route-switch-margin 0.005 | `+0.017836` | `9/1/0` | `+0.020240` | `-0.009339` |
| route-switch-margin 0.010 | `+0.018256` | `9/1/0` | `+0.020240` | `-0.002594` |
| hedge-discount 0.990 | `+0.013881` | `8/2/0` | `-0.006826` | `-0.005773` |
| hedge-discount 0.999 | `+0.016041` | `9/1/0` | `+0.007887` | `-0.005775` |
| expert-selector-score-source window | `+0.014246` | `8/2/0` | `-0.007093` | `-0.005927` |
| selector-window 150 + disable safe_polynomial | `+0.019058` | `9/1/0` | `+0.036699` | `-0.001500` |
| selector-window 200 + disable safe_polynomial | `+0.018949` | `9/1/0` | `+0.036699` | `-0.002422` |
| selector-window 150 + route-switch-margin 0.010 | `+0.018954` | `10/0/0` | `+0.035429` | `+0.001315` |
| selector-window 150 + disable safe_polynomial + margin 0.010 | `+0.016825` | `9/1/0` | `+0.036699` | `+0.000522` |
| selector-window 200 + disable safe_polynomial + margin 0.010 | `+0.016962` | `9/0/1` | `+0.036699` | `+0.000000` |
| selector-window 150 + route-switch-margin 0.0075 | `+0.020537` | `10/0/0` | `+0.035675` | `+0.001365` |
| selector-window 150 + route-switch-margin 0.009 | `+0.018954` | `10/0/0` | `+0.035429` | `+0.001365` |

Best controlled_rare-only candidate:
`--selector-window 150 --route-switch-margin 0.0075`, but transfer was worse
than the `0.010` candidate.

## Transfer And Full Validation

The promoted full-suite candidate was:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 150 \
  --route-switch-margin 0.010 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_worker_b_w150_switch010_all_10seed \
  --note-path docs/research/step2_worker_b_w150_switch010_all_10seed.md
```

Full all-suite final-window MSE seed totals:

- Current best: wins/losses/ties `126/4/10`.
- Candidate: wins/losses/ties `126/4/10`.

Datasets whose seed-loss profile changed:

| Dataset | Current | Candidate | Change |
|---|---:|---:|---:|
| controlled_rare | `8/2/0`, `+0.014419` | `10/0/0`, `+0.018954` | removes 2 losses |
| synthetic_compositional | `9/1/0`, `+0.043445` | `10/0/0`, `+0.047270` | removes 1 loss |
| synthetic_polynomial | `9/1/0`, `+0.061279` | `7/3/0`, `+0.063029` | adds 2 losses |
| synthetic_frequency | `10/0/0`, `+0.804476` | `9/1/0`, `+0.811877` | adds 1 loss |

The candidate remains aggregate-positive on all 14 final-window MSE comparisons
and preserves the class-blocked held-out accuracy result (`+0.349165`,
`10/0/0`). It does not beat the current best on total seed losses; it ties.

## Conclusion

The controlled_rare losses can be eliminated without source changes using
selector-window 150 plus route-switch-margin 0.010. However, the full all-suite
validation shows a seed-loss relocation rather than a net seed-loss reduction:
controlled_rare and synthetic_compositional are fixed, but synthetic_polynomial
and synthetic_frequency absorb the losses. This is not a new best by total
seed losses.

Recommended next source patch, if allowed: add rare-aware route scoring for
masked multi-head targets, e.g. maintain per-head route windows or upweight
rare-active target heads in route loss. Existing flags can stabilize route
churn, but they cannot distinguish frequent-head wins from rare-head risk.
