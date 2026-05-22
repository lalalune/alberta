# Step 2 Target-Structure UPGD Stress Probe

Date: 2026-05-05.

## Purpose

Target-density UPGD fixed the mean-vs-sum loss conflict between dense synthetic
vector targets and sparse one-hot digit targets. The remaining obvious failure
mode is target structure ambiguity: exact-zero dense regression heads and
sparse multilabel targets can look sparse under a naive nonzero-count rule even
when they are not one-hot classification tasks.

This probe tests that boundary directly and introduces a simpler structural
normalizer:

- `target_density`: divide by the number of nonzero target components.
- `target_structure`: use sum-style loss only for non-negative simplex targets
  with total mass 1; otherwise use mean-style loss over active heads.

The structural rule preserves one-hot digit pressure while avoiding an
extra gradient boost for dense-zero and multilabel targets.

## Local Stress Protocol

Output:

- `outputs/step2_target_structure_stress_local/stress_results.json`

Protocol:

- 8 seeds.
- 2,500 online examples per seed.
- Final window: 700 examples.
- Baselines: fair `mlp64`, fair `mlp64_64`.
- UPGD variants: `mean`, `target_density`, `target_structure`.
- UPGD settings: hidden size 64, `perturbation_sigma=1e-4`, `ObGDBounding(kappa=0.5)`.

Stressors:

- `dense_zero`: four active regression heads; two heads are exact zero at every
  step, while the other two are nonlinear dense targets.
- `sparse_multilabel`: six active heads with multi-hot targets, occasional
  all-zero rows, and occasional dense-ish rows.

## Results

Positive paired diff means best fair MLP final-window MSE minus method
final-window MSE.

| Stressor | Method | Final-window MSE | Diff vs best MLP | Wins/losses |
|---|---:|---:|---:|---:|
| Dense-zero | `upgd_mean` | `0.028075 +/- 0.000439` | `+0.000232 +/- 0.000311` | `5/3` |
| Dense-zero | `upgd_target_density` | `0.028771 +/- 0.000364` | `-0.000464 +/- 0.000269` | `3/5` |
| Dense-zero | `upgd_target_structure` | `0.028075 +/- 0.000439` | `+0.000232 +/- 0.000311` | `5/3` |
| Sparse multilabel | `upgd_mean` | `0.119105 +/- 0.001179` | `+0.026938 +/- 0.000865` | `8/0` |
| Sparse multilabel | `upgd_target_density` | `0.139496 +/- 0.001426` | `+0.006547 +/- 0.000783` | `8/0` |
| Sparse multilabel | `upgd_target_structure` | `0.119105 +/- 0.001179` | `+0.026938 +/- 0.000865` | `8/0` |

## Original Synthetic Check

Output:

- `outputs/step2_target_structure_synthetic_5seed/out_of_class_results.json`
- `outputs/step2_target_structure_synthetic_5seed/out_of_class_SUMMARY.md`

Protocol:

- 5 seeds.
- 2,500 online examples per stream.
- Same out-of-hypothesis-class stream harness as the canonical Step 2
  synthetic benchmark.

The structural rule does not give back the original dense synthetic UPGD
advantage in this quick check:

| Stream | Structure variant | Diff vs best MLP | Wins/losses |
|---|---:|---:|---:|
| Polynomial | `structure_sigma1e4_kappa05` | `+0.4561 +/- 0.0653` | `5/0` |
| Frequency mismatch | `structure_sigma1e4_kappa05` | `+0.6610 +/- 0.1696` | `5/0` |
| Compositional | `structure_sigma1e4_kappa05` | `+0.0951 +/- 0.0047` | `5/0` |
| Polynomial | `structure_adaptk035_065_lr06` | `+0.4644 +/- 0.0656` | `5/0` |
| Frequency mismatch | `structure_adaptk035_065_lr06` | `+0.7023 +/- 0.1790` | `5/0` |
| Compositional | `structure_adaptk035_065_lr06` | `+0.0839 +/- 0.0036` | `5/0` |

## Interpretation

The naive target-density rule is strong on the current canonical dense/sparse
matrix, but it is not the simplest robust target normalizer. It treats
zero-valued dense heads and multilabel rows as if they were one-hot sparse
classification targets.

`target_structure` is the cleaner candidate for the next promoted UPGD default:

- It is identical to mean loss on these dense-zero and multilabel stressors.
- It is identical to sum-style loss on one-hot/simplex targets.
- It removes a target-count hyperparameter-like assumption without adding a
  dataset router or deployment portfolio.

This is a local stress probe, not yet a canonical replacement. The next required
check is a full synthetic/digits comparison showing that `target_structure`
matches target-density UPGD on the previous 30-seed wins while improving these
boundary cases.
