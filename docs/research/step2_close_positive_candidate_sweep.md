# Step 2 Close-Positive Candidate Sweep

This note records the follow-up sweep over candidates that were already close
or positive. The goal was not to add another unrelated mechanism. The goal was
to grind the strongest signals and identify the best current Step 2 learner
against the same-run best fair MLP comparator.

## Candidates Tested

1. Recursive retention feature construction.
2. Recursive retention plus signed-tanh scaffolds.
3. A causal Hedge selector over the two recursive-retention variants.
4. The pruned conclusive learner: recursive, polynomial, Fourier, random-tanh,
   and fair MLP experts under discounted Hedge, with UPGD and dynamic sparse
   excluded from deployment routing.
5. Focused route/stacker/selector variants for the remaining
   synthetic-polynomial weakness.
6. Polynomial-safe interpolation routes plus shorter causal selector windows,
   which were the best transfer-improving refinement.

All promoted comparisons use same-run best fair MLP baselines. Positive MSE
deltas mean `best_mlp_mse - conclusive_mse`, so positive favors the candidate.

## Recursive-Only Result

Recursive-only feature construction is not the best Step 2 learner.

The retention-tanh24 sweep showed real signal but too much instability:

- `outputs/step2_recursive_retention_tanh24_suite_5seed_2500/SUMMARY.md`
- `single_mechanism_retention`: beats best fair MLP on 5/6 controlled tasks.
- `single_mechanism_retention_tanh24`: beats best fair MLP on 4/6 tasks and
  ties within 0.02 MSE on 1/6 tasks.
- On nonlinear, `single_mechanism_retention_tanh24` wins 3/5 seeds but has mean
  final-window MSE `0.0898`, while best fair MLP is `0.0597`.

A causal recursive-only Hedge selector improves the nonlinear near miss but
does not close it:

- `outputs/step2_recursive_retention_hedge_keymatched_nonlinear_5seed_2500/SUMMARY.md`
- Best variant: `recursive_retention_hedge_recent`.
- Nonlinear final-window MSE: `0.0617 +/- 0.0152`.
- Best fair MLP nonlinear final-window MSE: `0.0597 +/- 0.0023`.
- Paired delta: `-0.0020`; recursive hedge wins 3/5 seeds and ties within
  0.02 MSE, but does not beat MLP on mean.

Longer nonlinear training does not rescue recursive-only construction:

- `outputs/step2_recursive_retention_hedge_nonlinear_5seed_5000/SUMMARY.md`
- At 5000 steps, best fair MLP improves to `0.0320` final-window MSE, while
  recursive retention/hedge variants remain far behind.

Conclusion: recursive retention is important evidence for compositional feature
construction, but the best broad Step 2 learner must be a portfolio/resource
allocation learner, not recursive construction alone.

## Conclusive Learner Result

The strongest current all-suite learner is:

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
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed \
  --note-path docs/research/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed.md
```

Primary output:

- `docs/research/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed.md`
- `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json`

Aggregate result:

| Dataset | Mean final-MSE delta vs best MLP | Wins/losses/ties |
|---|---:|---:|
| controlled_frequency | +0.155974 | 10/0/0 |
| controlled_interaction | +0.147745 | 10/0/0 |
| controlled_nonlinear | +0.020404 | 10/0/0 |
| controlled_polynomial | +0.532656 | 10/0/0 |
| controlled_rare | +0.014419 | 8/2/0 |
| controlled_triple | +0.579143 | 10/0/0 |
| digits_class_blocked | +0.000000 | 0/0/10 |
| digits_iid | +0.006427 | 10/0/0 |
| digits_label_drift | +0.006958 | 10/0/0 |
| digits_mask_noise | +0.006157 | 10/0/0 |
| digits_permuted_pixels | +0.007925 | 10/0/0 |
| synthetic_compositional | +0.043445 | 9/1/0 |
| synthetic_frequency | +0.804476 | 10/0/0 |
| synthetic_polynomial | +0.061279 | 9/1/0 |

This is nonnegative on 14/14 aggregate final-window MSE comparisons and
positive on 13/14. The only non-positive aggregate is class-blocked digits
online MSE, which is an exact tie against the best MLP route; the same run has
a large held-out accuracy advantage on class-blocked digits:

- class-blocked held-out accuracy delta: `+0.3492 +/- 0.0080`, wins 10/10.

Relative to the previous 10-seed base command with selector window 300 and
recursive-only safe routes, this variant trades small mean losses on rare,
compositional, and frequency regimes for a better broad seed-level profile:

- controlled nonlinear improves from `+0.006764`, `8/2/0` to `+0.020404`,
  `10/0/0`.
- synthetic polynomial improves from `+0.028963`, `5/5/0` to `+0.061279`,
  `9/1/0`.
- total non-class-blocked final-window MSE seed losses fall from 7 to 4.

## Focused Synthetic-Polynomial Sweep

The previous pruned learner was already positive on mean synthetic-polynomial
MSE but was unstable under 10 seeds. Focused sweeps showed:

| Variant | Focused synthetic-polynomial result |
|---|---|
| base 10-seed stacker run, selector window 300 | mean +0.028963, 5/5 wins/losses |
| disable `stacked_predictions` | mean +0.021333, 2/8 wins/losses |
| disable recursive-safe routes | mean +0.026188, 5/5 wins/losses |
| selector window 100 only | mean +0.033707, 5/5 wins/losses |
| polynomial-safe routes only | mean +0.056958, 5/5 wins/losses |
| polynomial step size 0.3 only | mean +0.037972, 6/4 wins/losses |
| polynomial-safe routes plus step size 0.3 | mean +0.044823, 6/4 wins/losses |
| polynomial-safe routes plus selector window 100 | mean +0.061279, 9/1 wins/losses |
| polynomial-safe routes plus step size 0.3 and selector window 100 | mean +0.051393, 8/2 wins/losses |

The winning refinement is polynomial-safe interpolation plus a shorter causal
selector window. The interpretation is simple: synthetic polynomial needs the
polynomial expert to be allowed to interpolate with MLP anchors, and the final
window is too late-changing for the longer route window.

## Verdict

Best learner found in this pass:

`step2_conclusive_learner.py` with pruned discounted-Hedge routing,
`--selector-window 100`, `--safe-route-sources recursive_features,polynomial_features`,
`--stacker-step-size 0.006`, and UPGD/dynamic-sparse disabled for deployment
routing.

What this supports:

- A strong current-matrix Step 2 claim: nonnegative versus the same-run best
  fair MLP on all 14 aggregate final-window MSE comparisons.
- Strict positive aggregate improvement on 13/14 comparisons.
- Strong held-out digits accuracy improvements, including class-blocked
  retention accuracy.
- A clear best current learner among the close-positive candidates tested.
- Better seed-level robustness than the previous 10-seed base run.

What it still does not support:

- An unqualified "beats MLP on every seed" claim. Controlled rare still has two
  seed-level losses, synthetic compositional has one, and synthetic polynomial
  has one.
- A single universal feature-construction mechanism. The winning mechanism is
  allocation among multiple live predictors.
- Full published-scale universal Step 2 closure. OPMNIST 800-task completion
  remains separate from this all-suite learner result.
