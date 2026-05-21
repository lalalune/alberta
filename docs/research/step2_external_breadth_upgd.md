# Step 2 External Breadth: Promoted UPGD vs MLP64

Date: 2026-05-05

## Scope

This workstream checks the current promoted Step 2 UPGD learner on bundled
sklearn-style non-synthetic streams, with no portfolio, retained MLP head,
replay buffer, or MLP fallback.

The external-suite `upgd` method was patched to instantiate:

```python
UPGDLearner.step2_default(
    n_heads=n_classes,
    hidden_sizes=(64,),
    step_size=0.03,
    readout_mode="softmax_ce" if task_kind == "multiclass" else "linear_mse",
)
```

That means the reported UPGD branch uses target-structure loss normalization,
the promoted conservative ObGD bounder, low-noise Rademacher perturbations, and
the native softmax/CE readout for multiclass classification. The suite config
still contains legacy CLI fields such as `perturbation_sigma`, but those fields
are not passed into the promoted UPGD factory after this patch. The MLP baseline
is the suite's existing `MultiHeadMLPLearner` with one hidden layer of 64 units.

## Commands

Smoke check:

```bash
.venv/bin/python 'examples/The Alberta Plan/Step2/step2_external_suite.py' \
  --smoke \
  --methods mlp,upgd \
  --output-dir output/subagents/external_breadth_sklearn/smoke_promoted
```

Main 5-seed sklearn breadth run:

```bash
.venv/bin/python 'examples/The Alberta Plan/Step2/step2_external_suite.py' \
  --steps 3000 \
  --n-seeds 5 \
  --final-window 500 \
  --hidden-size 64 \
  --methods mlp,upgd \
  --benchmarks digits_shuffled,digits_class_blocked,digits_permuted,digits_mask_noise,wine_shuffled,breast_cancer_shuffled,diabetes_regression \
  --output-dir output/subagents/external_breadth_sklearn/promoted_step2_default_5seed
```

Post-hoc CE pass:

```bash
.venv/bin/python - <<'PY'
# Reused the external-suite loader and same 5 seeds/config to compute
# classification CE. MLP logits were softmaxed for CE evaluation; UPGD used
# native probabilities from step2_default(readout_mode="softmax_ce").
PY
```

Harness test:

```bash
.venv/bin/python -m pytest tests/test_step2_external_suite.py -q
```

## Main Results

Source:
`output/subagents/external_breadth_sklearn/promoted_step2_default_5seed/external_suite_results.json`

Total suite wall clock: 170.4 s. The runner does not record per-method compute,
so compute win/loss by method is not available from this run.

| Dataset/regime | Task | FW MSE MLP | FW MSE UPGD | FW acc MLP | FW acc UPGD | Test MSE MLP | Test MSE UPGD | Test acc MLP | Test acc UPGD | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| digits_shuffled | multiclass | 0.0203 | 0.0036 | 0.9652 | 0.9772 | 0.0224 | 0.0066 | 0.9484 | 0.9566 | UPGD beats MLP |
| digits_class_blocked | multiclass | 0.0040 | 0.0049 | 0.9836 | 0.9656 | 0.1315 | 0.0294 | 0.1132 | 0.7948 | tracking loss, test win |
| digits_permuted | multiclass | 0.0470 | 0.0278 | 0.8116 | 0.8024 | 0.0972 | 0.0522 | 0.3505 | 0.6305 | MSE/test win, FW acc loss |
| digits_mask_noise | multiclass | 0.0419 | 0.0181 | 0.8148 | 0.8736 | 0.0391 | 0.0099 | 0.8913 | 0.9332 | UPGD beats MLP |
| wine_shuffled | multiclass | 0.0026 | 0.0000 | 1.0000 | 1.0000 | 0.0152 | 0.0054 | 0.9849 | 0.9887 | MSE win, acc saturated |
| breast_cancer_shuffled | multiclass | 0.0219 | 0.0066 | 0.9820 | 0.9916 | 0.0301 | 0.0178 | 0.9696 | 0.9789 | UPGD beats MLP |
| diabetes_regression | regression | 0.5051 | 0.5901 | n/a | n/a | 0.8409 | 0.8935 | n/a | n/a | UPGD loses MLP |

Paired final-window MSE reading:

- UPGD beats MLP on 5/7 regimes: shuffled digits, permuted digits, mask/noise
  digits, wine, and breast-cancer.
- UPGD loses MLP on 2/7 regimes: class-blocked digits tracking loss and
  diabetes regression.
- Class-blocked digits is not a clean loss overall: UPGD loses recent-block
  tracking on final-window MSE/accuracy, but it strongly wins held-out test MSE
  and test accuracy. The MLP appears highly adapted to the current class block
  and generalizes poorly at evaluation time.
- Diabetes is a real negative result for the promoted default on tabular
  regression: MLP wins final-window MSE and test MSE, and accuracy is not a
  meaningful metric for this task.

## CE Check

Source:
`output/subagents/external_breadth_sklearn/promoted_step2_default_5seed/external_suite_ce_metrics.json`

The main suite does not natively record CE, so this was computed post-hoc by
rerunning the same classification benchmarks. MLP logits were softmaxed only
for evaluation; the MLP itself was still the suite's existing MSE-trained
baseline. Treat this as calibration/readout evidence, not a CE-trained MLP
baseline comparison.

| Benchmark | Final CE MLP | Final CE UPGD | Test CE MLP | Test CE UPGD | Outcome |
|---|---:|---:|---:|---:|---|
| digits_shuffled | 1.5957 | 0.0836 | 1.6078 | 0.1543 | UPGD lower CE |
| digits_class_blocked | 1.4874 | 0.1003 | 2.2661 | 0.5979 | UPGD lower CE |
| digits_permuted | 1.7689 | 0.6257 | 2.1083 | 1.2165 | UPGD lower CE |
| digits_mask_noise | 1.7392 | 0.3670 | 1.6262 | 0.2163 | UPGD lower CE |
| wine_shuffled | 0.5565 | 0.0011 | 0.5867 | 0.0264 | UPGD lower CE |
| breast_cancer_shuffled | 0.3398 | 0.0270 | 0.3499 | 0.0844 | UPGD lower CE |

## Conclusion

The promoted single Step 2 UPGD learner has meaningful external breadth against
the suite's MLP64 baseline on sklearn classification streams. It wins the
majority of final-window MSE comparisons, wins all six classification test-MSE
comparisons, and is especially strong under permuted and mask/noise digits.
The post-hoc CE readout check is also favorable across every classification
benchmark, consistent with the native softmax/CE branch doing real work.

The result is not universal. Class-blocked digits exposes a tracking tradeoff:
the MLP tracks the final class block better, while UPGD preserves much better
held-out performance. Diabetes regression is a straightforward promoted-UPGD
loss on both final-window and test MSE. This should be stated as a boundary,
not explained away.

## Recommended Next Patch

Make the external suite distinguish method names explicitly:
`upgd_step2_default` for the promoted learner and `upgd_legacy_raw` only when
old experiments need reproduction. At the same time, add native CE metrics and
per-method timing to the main runner so future reports do not require a
post-hoc CE pass and can make a real compute comparison.
