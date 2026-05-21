# Step 2 Combined UPGD Experiment Results

Date: 2026-05-04

## Goal

Test the proposed single-learner continuation:

```text
two_signal_upgd
+ native softmax/CE readout
+ less destructive rewiring
+ optional output margin adapter
+ retained-head follow-up if needed
```

The implementation kept all additions behind default-off UPGD flags so the
existing UPGD behavior remains comparable.

## Implemented

Core UPGD now supports:

- `readout_mode="softmax_ce"` for native softmax/cross-entropy readout;
- `readout_input_mode="hidden_plus_input"` so heads can use old raw features
  and new hidden features together;
- `readout_head_normalization="hidden_norm"`;
- output-only margin updates via `readout_margin` and
  `readout_margin_step_size`;
- outgoing-preserving recycling via `unit_replacement_outgoing_scale`;
- partial incoming rewiring via `unit_replacement_partial_fanin`;
- strict stale-score gates via `unit_replacement_score_threshold`;
- outgoing-aware unit utility via `unit_outgoing_utility_weight`.

Runner support was added to
`output/subagents/upgd_sweep/upgd_digits_sweep.py` for the new candidate
configs.

## Verification

Passed:

```bash
.venv/bin/ruff check src/alberta_framework/core/upgd.py tests/test_upgd.py output/subagents/upgd_sweep/upgd_digits_sweep.py
.venv/bin/python -m pytest tests/test_upgd.py -q
```

The full UPGD test run passed after all core changes, including the final
bias-normalization patch: `31 passed`.

## Result 1: Native Softmax/CE Readout Was Negative

Artifacts:

- `output/subagents/combined_upgd/label_drift_rate_probe/SUMMARY.md`
- `output/subagents/combined_upgd/label_drift_skip_probe/SUMMARY.md`
- `output/subagents/combined_upgd/label_drift_skip_normbias_probe/SUMMARY.md`

Label-drift, 1 seed, 1200 steps:

| Variant | Final MSE diff vs MLP | Test accuracy diff vs MLP |
|---|---:|---:|
| `upgd_sum_stale_fanin_3e_3` | -0.0021 | -0.0464 |
| `upgd_softmax_ce_margin` | -0.0189 | -0.2764 |
| `upgd_combo_softmax_stalepreserve_margin` | -0.0190 | -0.2579 |
| `upgd_softmax_skip_margin` | -0.0193 | -0.3061 |
| `upgd_softmax_ce_headx50` | -0.0209 | -0.3692 |

Read: raw softmax is strong, but native softmax on the UPGD trunk was not. A
skip readout that included raw input did not fix it. This suggests the current
UPGD trunk plus CE dynamics are not a good label-drift learner. Do not promote
softmax/CE UPGD.

## Result 2: Less Destructive Rewiring Helps MSE But Not Accuracy

Artifact:

- `output/subagents/combined_upgd/permuted_linear_probe_3seed/SUMMARY.md`

Permuted pixels, 3 seeds:

| Variant | Final MSE diff vs MLP | Final accuracy diff | Test accuracy diff |
|---|---:|---:|---:|
| `upgd_stale_preserve05_partial16_oututil` | +0.0012 | -0.1011 | -0.0495 |
| `upgd_stale_margin` | -0.0002 | -0.0878 | -0.0186 |
| `upgd_sum_stale_fanin_3e_3` | -0.0005 | -0.1133 | -0.0501 |

Read: outgoing/partial rewiring can preserve the MSE benefit, while margin can
improve held-out accuracy. The initial margin was too aggressive and spent the
MSE edge.

## Result 3: Gentle Margin Is The Best Combined Variant On Permuted Pixels

Artifacts:

- `output/subagents/combined_upgd/permuted_margin_tune_3seed/SUMMARY.md`
- `output/subagents/combined_upgd/permuted_best_10seed/SUMMARY.md`

Permuted pixels, 10 seeds:

| Variant | Final MSE diff vs MLP | Test MSE diff vs MLP | Final acc diff | Test acc diff |
|---|---:|---:|---:|---:|
| `upgd_sum_stale_fanin_3e_3` | +0.0031 | +0.0028 | -0.0687 | -0.0237 |
| `upgd_stale_margin_tiny` | +0.0024 | +0.0030 | -0.0757 | -0.0160 |
| `upgd_stale_margin_gentle` | +0.0022 | +0.0032 | -0.0670 | -0.0058 |
| `upgd_stale_preserve05_partial16_oututil` | +0.0019 | +0.0019 | -0.0780 | -0.0171 |

Read: `upgd_stale_margin_gentle` is a real improvement over current
stale/fan-in UPGD on the specific permuted-pixel failure:

- it keeps the final-window MSE win;
- it slightly improves held-out MSE;
- it cuts the held-out accuracy gap from `-0.0237` to `-0.0058`;
- it still does not beat MLP on final-window accuracy.

This is not full closure, but it is a stronger single-learner permuted-pixel
candidate than the prior two-signal UPGD.

## Result 4: The Candidate Is Not Universal Across Digits

Artifact:

- `output/subagents/combined_upgd/digits_universality_3seed/SUMMARY.md`

Across IID, class-blocked, permuted pixels, mask noise, and label drift:

| Variant | Overall final MSE diff | Overall test accuracy diff |
|---|---:|---:|
| `upgd_stale_margin_tiny` | -0.0043 | -0.0751 |
| `upgd_sum_stale_fanin_3e_3` | -0.0054 | -0.1054 |
| `upgd_stale_margin_gentle` | -0.0061 | -0.1060 |

The permuted improvement does not generalize. Always-on recycling/margin hurts
IID, mask-noise, and label-drift rows.

## Result 5: Stricter Gates Help Ordinary Rows But Blunt Permuted Adaptation

Artifact:

- `output/subagents/combined_upgd/gated_digits_3seed/SUMMARY.md`

Across IID, permuted pixels, mask noise, and label drift:

| Variant | Overall final MSE diff | Overall test MSE diff | Overall test acc diff |
|---|---:|---:|---:|
| `upgd_stale_rate3e_3_gate0_25_tiny` | -0.0046 | +0.0015 | -0.0073 |
| `upgd_sum_sigma0` | -0.0054 | +0.0009 | -0.0164 |
| `upgd_stale_rate1e_3_gate0_25_tiny` | -0.0057 | +0.0004 | -0.0196 |
| `upgd_sum_stale_fanin_3e_3` | -0.0086 | -0.0087 | -0.1303 |

By regime, `upgd_stale_rate3e_3_gate0_25_tiny` is encouraging on ordinary rows:

| Regime | Final MSE diff | Test acc diff |
|---|---:|---:|
| IID | +0.0014 | -0.0025 |
| Mask noise | +0.0052 | +0.0353 |
| Label drift | -0.0184 | -0.0575 |
| Permuted pixels | -0.0064 | -0.0043 |

Read: strict gates reduce damage on IID/mask-noise but lose the permuted-pixel
MSE win. The main unsolved problem is not how to make recycling work; it is how
to turn recycling on only for genuine feature-coordinate shifts.

## Scientific Conclusion

The combined candidate did not solve Step 2 universally.

What worked:

- two-signal stale/fan-in recycling is real for permuted-pixel MSE;
- a gentle output margin adapter improves the held-out accuracy gap while
  retaining most of the MSE edge;
- strict stale-score gates make UPGD much less harmful on IID and mask-noise.

What failed:

- native softmax/CE UPGD was worse than squared-error UPGD;
- skip readout did not rescue CE;
- always-on margin/recycling is not universal;
- class-blocked and label-drift are still readout/retention problems rather
  than hidden-feature recycling problems.

Best current single-learner candidate for permuted pixels:

```text
upgd_stale_margin_gentle
```

Best current broad digits UPGD-family candidate:

```text
upgd_stale_rate3e_3_gate0_25_tiny
```

But the best overall Step 2 system remains the strict live portfolio/resource
manager, not this single UPGD branch.

## Next Implementation Direction

The next high-value step is not another readout loss. It is a causal shift
detector for recycling pressure:

```text
if feature-coordinate shift is likely:
    allow stale/fan-in recycling + gentle margin
else:
    behave close to upgd_sum_sigma0
```

Candidate signals:

- fast/slow loss ratio;
- fast/slow input covariance or feature-gradient covariance;
- unit stale-score distribution tail mass;
- disagreement between no-recycle and recycle shadow heads;
- cumulative benefit of recycling from recent prequential loss.

This is effectively a resource manager inside a single learner. If this cannot
be made simple and causal, the honest conclusion is that the portfolio is the
right Alberta-Plan-friendly abstraction for Step 2.
