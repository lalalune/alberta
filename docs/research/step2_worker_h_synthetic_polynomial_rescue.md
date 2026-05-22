# Step 2 Worker H: Synthetic Polynomial Rescue

Date: 2026-05-05

Scope: no source edits. This worker used only existing flags in
`examples/The Alberta Plan/Step2/step2_conclusive_learner.py` and wrote
`outputs/step2_worker_h_*` plus this note.

## Baseline

Current promoted command family:

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
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

Baseline synthetic-polynomial result:

| Result | Value |
|---|---:|
| Output | `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json` |
| Mean final-window MSE delta vs best fair MLP | `+0.06128` |
| Seed count | `9/1/0` |
| Losing seed | `6` |
| Seed-6 delta | `-0.01122` |

Positive deltas favor the conclusive learner.

## Candidate Results

Focused synthetic-polynomial runs, 10 seeds, 1200 steps, final-window 300,
warmup 250:

| Candidate | Output | Delta vs best MLP | Seeds W/L/T | Seed-6 delta | Losing seeds |
|---|---|---:|---:|---:|---|
| selector window 120 | `outputs/step2_worker_h_poly_selector_window_120/results.json` | `+0.06517` | `8/2/0` | `-0.01127` | `1, 6` |
| route switch margin 0.001 | `outputs/step2_worker_h_poly_route_switch_0p001/results.json` | `+0.06241` | `7/3/0` | `-0.01076` | `0, 1, 6` |
| selector window 110 | `outputs/step2_worker_h_poly_selector_window_110/results.json` | `+0.06234` | `8/2/0` | `-0.00835` | `1, 6` |
| route switch margin 0.003 | `outputs/step2_worker_h_poly_route_switch_0p003/results.json` | `+0.06175` | `8/2/0` | `-0.00980` | `1, 6` |
| baseline/current | `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json` | `+0.06128` | `9/1/0` | `-0.01122` | `6` |
| route switch margin 0.002 | `outputs/step2_worker_h_poly_route_switch_0p002/results.json` | `+0.06088` | `7/3/0` | `-0.01201` | `0, 1, 6` |
| disable top mixes, window 120 | `outputs/step2_worker_h_poly_disable_top_mixes_window120/results.json` | `+0.06082` | `7/3/0` | `-0.01092` | `1, 5, 6` |
| disable top mixes, window 110 | `outputs/step2_worker_h_poly_disable_top_mixes_window110/results.json` | `+0.06016` | `7/3/0` | `-0.00663` | `1, 5, 6` |
| disable `safe_polynomial_mlp_32x32` | `outputs/step2_worker_h_poly_disable_safe_poly_mlp32/results.json` | `+0.05944` | `7/3/0` | `-0.01593` | `0, 3, 6` |
| disable top mixes | `outputs/step2_worker_h_poly_disable_top_mixes/results.json` | `+0.05855` | `7/3/0` | `-0.00634` | `0, 5, 6` |
| polynomial-safe only | `outputs/step2_worker_h_poly_safe_poly_only/results.json` | `+0.05548` | `6/4/0` | `-0.04080` | `0, 1, 3, 6` |
| selector window 90 | `outputs/step2_worker_h_poly_selector_window_90/results.json` | `+0.05286` | `6/4/0` | `-0.00865` | `1, 3, 6, 7` |
| disable safe-polynomial family | `outputs/step2_worker_h_poly_disable_safe_polynomial/results.json` | `+0.03371` | `5/5/0` | `-0.02119` | `0, 3, 5, 6, 9` |
| MLP-only routes | `outputs/step2_worker_h_poly_disable_all_non_mlp/results.json` | `-0.00891` | `1/9/0` | `+0.00262` | `0, 1, 2, 3, 4, 5, 7, 8, 9` |

## Weak3 Checks

The top two viable candidates were selector-window 120 and selector-window 110.
Selector-window 120 had the largest aggregate polynomial delta. Selector-window
110 had the smaller seed-6 miss among high-delta, 8/2 candidates.

Selector-window 120 weak3 execution was interrupted before `results.json` was
written, after completing all controlled-rare and synthetic-compositional curve
files and polynomial seeds 0-7. The table below uses exact final-window metrics
computed from completed curve files for controlled-rare and compositional, and
the completed focused polynomial `results.json` for synthetic-polynomial.

| Candidate | Controlled rare | Synthetic compositional | Synthetic polynomial | Weak3 total |
|---|---:|---:|---:|---:|
| current baseline | `+0.01442`, `8/2/0` | `+0.04344`, `9/1/0` | `+0.06128`, `9/1/0` | `26/4/0` |
| selector window 110 | `+0.01568`, `8/2/0` | `+0.04492`, `9/1/0` | `+0.06234`, `8/2/0` | `25/5/0` |
| selector window 120 | `+0.01788`, `9/1/0` | `+0.04492`, `9/1/0` | `+0.06517`, `8/2/0` | `26/4/0` |

No candidate got weak3 losses below the current four losses, so no all-suite
confirmation was triggered.

## Interpretation

The focused evidence says the current baseline remains the best strict
synthetic-polynomial candidate because it has the best seed count: `9/1/0`.
Several variants improve aggregate MSE but add small seed losses. That is not a
good tradeoff for the current Step 2 closure criterion, which is trying to
remove all known losses rather than maximize aggregate margin.

The MLP-only route test is useful mechanistic evidence. It flips seed 6
positive (`+0.00262`) but collapses the aggregate result to `1/9/0`. This shows
that seed 6 can be rescued by blocking specialist displacement, but doing so
globally destroys the specialist wins that make the portfolio strong.

The surgical route tests did not work. Disabling only
`safe_polynomial_mlp_32x32` pushed seed 6 to another safe-polynomial anchor and
worsened the miss. Disabling the entire safe-polynomial family also worsened
seed 6 and added multiple losses. Therefore the missing mechanism is not a
static route blacklist. It is a context-sensitive guard that can recognize when
specialist interpolation is locally worse than the best small MLP.

## Recommendation

Do not promote any Worker H flag-only candidate.

The best next source edit, outside Worker H scope, is a causal MLP-floor or
route-risk guard that activates only when a safe specialist route is close to
the MLP selector and the MLP selector has lower recent loss. Static disabling
is too blunt: it either leaves seed 6 unresolved or removes the very specialist
routes that produce the large polynomial wins.
