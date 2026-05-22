# Worker I: tanh-safe compositional rescue

Date: 2026-05-05

Scope: compositional/tanh-safe rescue using existing `step2_conclusive_learner.py`
flags and Worker I output directories only. No source edits were made for this
worker report.

## Question

The incumbent strict Step 2 portfolio had one remaining
`synthetic_compositional` seed loss. Worker C showed that adding
`tanh_random_features` to the MLP-safe route sources fixed compositional, but
could harm `synthetic_polynomial`. Worker I tested whether tanh-safe routing can
be made useful without broad side effects.

The route table supports this directly:

- `SAFE_SOURCE_NAMES` includes `tanh_random_features`.
- `ROUTE_DISABLE_GROUPS` includes `safe_tanh_random`.
- The CLI exposes `--safe-route-sources`, `--tanh-random-step-size`, and
  `--tanh-random-weight-scale`.

## Base Command

All runs used this base unless noted:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features,tanh_random_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

## Focused Compositional Sweep

Metric: final-window MSE versus each seed's best fair MLP. Positive diff favors
the conclusive learner.

| Output | Tanh step | Scale | Diff | W/L/T | Assessment |
|---|---:|---:|---:|---:|---|
| `outputs/step2_worker_i_rp_tanh_w100_s0p1_g1p0` | 0.1 | 1.0 | +0.000944 | 4/6/0 | Too weak; not a rescue. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p2_g1p0` | 0.2 | 1.0 | +0.011410 | 7/3/0 | Still loses seed 5. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p4_g0p5` | 0.4 | 0.5 | +0.046311 | 10/0/0 | Rescue, but polynomial side check worsened to 8/2. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p4_g1p0` | 0.4 | 1.0 | +0.050782 | 10/0/0 | Reproduces Worker C's rescue. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p4_g1p5` | 0.4 | 1.5 | +0.047369 | 10/0/0 | Rescue, lower margin than default. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p8_g1p0` | 0.8 | 1.0 | +0.055148 | 10/0/0 | Best compositional setting. |

The attempted `scale=2.0` sequential sweep was terminated before writing a
`results.json`; it is not used as evidence. Selector-window 120/150 variants
were not promoted because `selector-window=100, tanh-step=0.8` already fixed the
targeted compositional loss and then confirmed at all-suite scale.

## Side-Effect Checks

The strongest tanh setting was checked on the two regimes where harm would be
most plausible:

| Output | Benchmark | Diff | W/L/T | Assessment |
|---|---|---:|---:|---|
| `outputs/step2_worker_i_rp_tanh_w100_s0p8_g1p0_polycheck` | `synthetic_polynomial` | +0.060875 | 9/1/0 | Matches incumbent loss count; no broad tanh harm. |
| `outputs/step2_worker_i_rp_tanh_w100_s0p8_g1p0_rarecheck` | `controlled_rare` | +0.014437 | 8/2/0 | Tanh alone does not fix rare. |

The smaller-scale tanh variant was less benign on polynomial:

| Output | Benchmark | Diff | W/L/T |
|---|---|---:|---:|
| `outputs/step2_worker_i_rp_tanh_w100_s0p4_g0p5_polycheck` | `synthetic_polynomial` | +0.061229 | 8/2/0 |

## Combined Candidate

Worker G independently found that `--route-rare-active-step-weight 4.0` helped
`controlled_rare` but did not fix compositional. Combining it with Worker I's
best tanh-safe setting gives the best weak3 result found in this pass:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks controlled_rare,synthetic_compositional,synthetic_polynomial \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features,tanh_random_features \
  --tanh-random-step-size 0.8 \
  --tanh-random-weight-scale 1.0 \
  --route-rare-active-step-weight 4.0 \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_worker_i_tanh_s0p8_rareweight4_weak3_10seed \
  --note-path docs/research/step2_worker_i_tanh_s0p8_rareweight4_weak3_10seed.md
```

Weak3 result:

| Benchmark | Diff | W/L/T |
|---|---:|---:|
| `controlled_rare` | +0.015371 | 9/1/0 |
| `synthetic_compositional` | +0.055148 | 10/0/0 |
| `synthetic_polynomial` | +0.060876 | 9/1/0 |
| **Total** | | **28/2/0** |

This improves over:

- incumbent eta-1 safe-poly all-suite support: 126/4/10;
- Worker A eta-0.5 safe-poly all-suite support: 127/3/10.

## All-Suite Confirmation

Promoted confirmation command:

```bash
source .venv/bin/activate
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
  --safe-route-sources recursive_features,polynomial_features,tanh_random_features \
  --tanh-random-step-size 0.8 \
  --tanh-random-weight-scale 1.0 \
  --route-rare-active-step-weight 4.0 \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_worker_i_tanh_s0p8_rareweight4_all_10seed_rerun \
  --note-path docs/research/step2_worker_i_tanh_s0p8_rareweight4_all_10seed_rerun.md
```

Output:
`outputs/step2_worker_i_tanh_s0p8_rareweight4_all_10seed_rerun/results.json`

Final-window MSE versus best fair MLP:

| Benchmark | Diff | W/L/T |
|---|---:|---:|
| `controlled_frequency` | +0.155975 | 10/0/0 |
| `controlled_interaction` | +0.147745 | 10/0/0 |
| `controlled_nonlinear` | +0.020575 | 10/0/0 |
| `controlled_polynomial` | +0.532656 | 10/0/0 |
| `controlled_rare` | +0.015371 | 9/1/0 |
| `controlled_triple` | +0.579143 | 10/0/0 |
| `digits_class_blocked` | +0.000000 | 0/0/10 |
| `digits_iid` | +0.006411 | 10/0/0 |
| `digits_label_drift` | +0.006954 | 10/0/0 |
| `digits_mask_noise` | +0.006187 | 10/0/0 |
| `digits_permuted_pixels` | +0.008587 | 10/0/0 |
| `synthetic_compositional` | +0.055148 | 10/0/0 |
| `synthetic_frequency` | +0.806073 | 10/0/0 |
| `synthetic_polynomial` | +0.060876 | 9/1/0 |
| **Total** | | **128/2/10** |

Accuracy caveat: final-window MSE is the primary strict matrix here. Digits test
accuracy remains positive versus best fair MLP on all listed digits variants,
including `digits_class_blocked` (+0.349165, 10/0/0), but online mean accuracy is
still negative on the non-class-blocked digits variants. Do not claim universal
dominance on every metric.

## Interpretation

The result is promotable as the strongest current Step 2 strict portfolio
candidate, with two remaining final-window MSE seed losses:

- `controlled_rare` seed 9;
- `synthetic_polynomial` seed 6.

The mechanism is not "tanh everywhere." On compositional streams, the promoted
setting routes heavily through `safe_tanh_random_*` and fixes the former seed-5
loss. On polynomial and controlled algebraic streams, it mostly routes through
polynomial/recursive/mixture routes. On digits, it routes through MLP mixture or
retained MLP expert routes. That is the right qualitative behavior for a
portfolio.

The useful mathematical lesson is that two orthogonal scoring failures were
being mixed:

1. Compositional nonlinear features needed a stronger random tanh basis with
   fast enough online output adaptation.
2. Rare multi-head targets needed route scoring to give rare-active time steps
   enough influence in the causal route-loss window.

Combining them improves the all-suite support from 126/4/10 or 127/3/10 to
128/2/10 without adding new external-data failures.

## Recommendation

Promote `tanh-step=0.8`, `tanh-scale=1.0`, and
`route-rare-active-step-weight=4.0` to the next canonical-candidate review. It is
not yet an unqualified universal win because two final-window MSE seed losses
remain and online mean accuracy is still weaker on several digits variants, but
it is the best confirmed candidate from this tanh-safe rescue pass.
