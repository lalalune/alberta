# Step 2 UPGD Promotion Matrix

Worker A scope: compare `target_structure` against `target_density`, `mean`,
and `sum` UPGD loss normalization on the same fair MLP-paired protocols.

## Variants

Use the `step2_upgd_ablation.py` preset `promotion_matrix`:

- `upgd64_mean_sigma1e_4_kappa05`
- `upgd64_density_sigma1e_4_kappa05`
- `upgd64_sum_sigma1e_4_kappa05`
- `upgd64_structure_sigma1e_4_kappa05`

All four use `hidden_sizes=(64,)`, `perturbation_sigma=1e-4`,
`utility_decay=0.995`, `perturbation_beta=2.0`, `perturbation_interval=1`,
and `ObGDBounding(kappa=0.5)`.

## Full 30-Seed Commands

Synthetic polynomial/frequency/compositional:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite synthetic \
  --preset promotion_matrix \
  --streams polynomial,frequency,compositional \
  --steps 6000 \
  --n-seeds 30 \
  --final-window 2000 \
  --output-dir output/subagents/worker_a_upgd_promotion_synthetic_30seed
```

Digits IID/permuted/class-blocked/label-drift/mask-noise:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset promotion_matrix \
  --digits-regimes iid,permuted,class_blocked,label_drift,mask_noise \
  --steps 6000 \
  --n-seeds 30 \
  --final-window 2000 \
  --phase-length 500 \
  --mask-keep-fraction 0.5 \
  --mask-noise-std 0.05 \
  --output-dir output/subagents/worker_a_upgd_promotion_digits_30seed
```

## Smoke Command

These commands prove the full matrix path without paying the 30-seed cost:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite synthetic \
  --preset promotion_matrix \
  --streams polynomial,frequency,compositional \
  --steps 120 \
  --n-seeds 1 \
  --final-window 40 \
  --output-dir output/subagents/worker_a_upgd_promotion_synthetic_smoke
```

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset promotion_matrix \
  --digits-regimes iid,permuted,class_blocked,label_drift,mask_noise \
  --steps 120 \
  --n-seeds 1 \
  --final-window 40 \
  --phase-length 40 \
  --output-dir output/subagents/worker_a_upgd_promotion_digits_smoke
```

Promotion should be decided from the two full 30-seed summaries, not from the
smoke output.

Smoke outputs from the command path check:

- `output/subagents/worker_a_upgd_promotion_digits_smoke/digits_ablation_SUMMARY.md`
- `output/subagents/worker_a_upgd_promotion_synthetic_smoke/synthetic_ablation_SUMMARY.md`

## Completed 30-Seed Results

The full 30-seed runs expose a useful distinction between a normalization
promotion and an unconditional learner promotion.

The canonical out-of-class synthetic harness supports replacing
`target_density` with `target_structure`. In
`output/subagents/target_structure_canonical_out_of_class_30seed/out_of_class_SUMMARY.md`,
`structure_sigma1e4_kappa05` beats the best fair MLP on all three dense vector
streams:

| Stream | Best MLP final MSE | Structure UPGD final MSE | Paired diff | Wins |
|---|---:|---:|---:|---:|
| Polynomial | `1.1458 +/- 0.0641` | `0.5977 +/- 0.0325` | `+0.5481 +/- 0.0317` | `30/30` |
| Frequency | `1.1689 +/- 0.0787` | `0.5898 +/- 0.0409` | `+0.5790 +/- 0.0379` | `30/30` |
| Compositional | `0.1908 +/- 0.0081` | `0.0983 +/- 0.0043` | `+0.0925 +/- 0.0039` | `30/30` |

The alternate `step2_upgd_ablation.py` synthetic protocol is a stress harness,
not the canonical out-of-class protocol. Its full run at
`output/subagents/worker_a_upgd_promotion_synthetic_30seed/synthetic_ablation_SUMMARY.md`
does **not** support promoting the simple fixed-kappa learner: the
`target_structure` row loses to the best fair MLP on polynomial, frequency, and
compositional streams in that runner. This result should stay in the record as
a robustness blocker for broad "superiority" claims.

The five-regime digits promotion matrix at
`output/subagents/worker_a_upgd_promotion_digits_30seed/digits_ablation_SUMMARY.md`
also does **not** support promoting the simple fixed-kappa learner against the
stronger `mlp64_64` baseline. `structure_sigma1e_4_kappa05` wins final-window
MSE only on the permuted regime, while losing final-window MSE on IID,
class-blocked, label-drift, and mask-noise.

The promoted interpretation is therefore:

- `target_structure` replaces `target_density` as the canonical normalization
  rule because it is identical on one-hot simplex targets, matches dense mean
  behavior on non-simplex vector targets, and fixes the dense-zero/multilabel
  ambiguity.
- Existing one-hot digit density evidence transfers to target-structure
  branches by update equivalence on non-negative simplex targets.
- The serious digit default remains an adaptive/readout branch, not the simple
  fixed-kappa promotion-matrix row.

## Completed Adaptive Structure Rerun

The native target-structure adaptive retention preset is now implemented as
`structure_adaptive_retention` in `step2_upgd_ablation.py` and was run on the
five-regime digit matrix:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset structure_adaptive_retention \
  --digits-regimes iid,permuted,class_blocked,label_drift,mask_noise \
  --steps 6000 \
  --n-seeds 10 \
  --final-window 2000 \
  --phase-length 500 \
  --mask-keep-fraction 0.5 \
  --mask-noise-std 0.05 \
  --output-dir output/subagents/structure_adaptive_retention/allregime_10seed_6000
```

Summary:
`output/subagents/structure_adaptive_retention/allregime_10seed_6000/digits_ablation_SUMMARY.md`.

The 64-64 target-structure branch with repeated-target readout plasticity is
the strongest strict online-MSE digit row in this rerun:

| Regime | Result vs best MLP final-window MSE |
|---|---:|
| IID | `+0.0058`, `10/10` wins |
| Permuted | `+0.0037`, `10/10` wins |
| Label drift | `+0.0035`, `10/10` wins |
| Mask noise | `+0.0095`, `10/10` wins |
| Class blocked | `-0.0009`, `0/10` wins |

The 64-wide `structure_meta003_notrunk` branch gives the best class-blocked
retained test accuracy (`+0.0640`, `10/10`) but still loses class-blocked
final-window MSE. This means the adaptive target-structure branch is a better
promotion candidate than the simple fixed-kappa row, but the class-blocked
online/retention tradeoff remains open.
