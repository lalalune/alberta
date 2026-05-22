# Step 2 Target-Density UPGD Closure

Date: 2026-05-05.

## Claim

The strongest current Step 2 answer in this repo is target-density UPGD: a
single continual supervised learner with nonlinear hidden features, explicit
hidden-feature utility, low-utility perturbation, optional unit recycling, and
causal meta-controls for plasticity.

This is not a portfolio, output router, MLP-width selector, or deployment-time
expert mixture. It is one learner with one active prediction path.

## Mechanism

The core fix is `loss_normalization="target_density"` in `UPGDLearner`.

For active output heads, the squared-error loss is divided by the number of
nonzero target components. This makes dense vector targets behave like mean
loss and sparse one-hot targets behave like sum loss:

- Dense synthetic Step 2 targets: all or most target components are nonzero,
  so target-density is essentially mean loss.
- One-hot digit targets: one target component is nonzero, so target-density is
  essentially sum loss.

That is the key bridge. Earlier runs showed a real conflict: mean-loss UPGD
won dense synthetic vector-target streams but lost too much on digits, while
raw sum-loss UPGD won digits but failed the synthetic streams. Target-density
is the simplest successful invariant found so far.

The promoted learner family uses:

- target-density multi-task MSE;
- low-utility UPGD perturbation, usually `perturbation_sigma=1e-4`;
- base ObGD `kappa=0.5`;
- optional fast/slow loss adaptive `kappa` clipped to `[0.35, 0.65]`;
- optional gradient-alignment readout plasticity;
- optional repeated-target readout plasticity for strict class-blocked online
  tracking.

## Evidence

### Dense Synthetic Vector Targets

The simple fixed bridge `density_sigma1e4_kappa05` beats the same-run best fair
MLP on all three 30-seed synthetic out-of-hypothesis-class streams:

| Stream | Diff vs best MLP | Wins |
|---|---:|---:|
| Polynomial | `+0.5473 +/- 0.0316` | `30/30` |
| Frequency mismatch | `+0.5761 +/- 0.0375` | `30/30` |
| Compositional | `+0.0924 +/- 0.0038` | `30/30` |

The adaptive/readout-control target-density variants still win all three
synthetic streams. They slightly reduce compositional margin relative to the
fixed bridge but improve digit behavior.

Sources:
`output/subagents/combined_upgd/synthetic_target_density_30seed_6000/out_of_class_SUMMARY.md`
and
`output/subagents/combined_upgd/synthetic_density_control_30seed_6000/out_of_class_SUMMARY.md`.

### Sklearn-Digits Online Matrix

The best average/test branch is:

`upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight`

It uses target-density loss, low perturbation, adaptive `kappa`, and learned
readout plasticity without learned trunk plasticity. On 30 seeds x 5 regimes:

| Metric | MLP | UPGD | Paired diff | Wins |
|---|---:|---:|---:|---:|
| Final-window MSE | `0.0405 +/- 0.0017` | `0.0341 +/- 0.0014` | `+0.0063 +/- 0.0003` | `120/150` |
| Test accuracy | `0.7143 +/- 0.0246` | `0.7442 +/- 0.0249` | `+0.0298 +/- 0.0017` | `140/150` |
| Final-window accuracy | `0.8272 +/- 0.0090` | `0.8550 +/- 0.0078` | `+0.0278 +/- 0.0020` | `119/150` |
| Test MSE | `0.0581 +/- 0.0032` | `0.0501 +/- 0.0032` | `+0.0080 +/- 0.0004` | `147/150` |

The strict online-MSE branch is:

`upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight`

It wins final-window MSE in all 150 digit seed/regime comparisons, at the cost
of weaker held-out test accuracy:

| Metric | Diff vs `mlp64` | Wins |
|---|---:|---:|
| Final-window MSE | `+0.0059 +/- 0.0003` | `150/150` |
| Test accuracy | `+0.0214 +/- 0.0019` | `119/150` |

The best class-blocked compromise in the latest 30-seed check is:

`upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight`

It is not the aggregate winner, but it is the cleanest class-blocked balance:
class-blocked final-window MSE `+0.0001`, class-blocked final-window accuracy
`+0.0006`, and class-blocked test accuracy `+0.0053`.

Sources:
`output/subagents/combined_upgd/digits_density_control_30seed/SUMMARY.md` and
`output/subagents/combined_upgd/digits_density_repetition_compromise_30seed/SUMMARY.md`.

## Critical Read

This solves the main empirical Step 2 blocker found in the repo: a single
feature-finding learner can now beat fair MLP baselines on dense vector-target
feature-construction streams and sparse online digit streams without selecting
among separate learners.

It also matches the Alberta Plan framing better than the earlier D18 and
portfolio results. UPGD directly assigns utility to hidden features, perturbs
or recycles low-utility features, and keeps learning temporally uniform.

The remaining limitations are real:

- Target-density uses target sparsity. It should be stress-tested on dense
  targets with exact zeros, sparse multilabel targets, and mixed dense/sparse
  heads.
- The class-blocked digit regime still exposes a tracking/retention tradeoff:
  the best average/test branch has negative class-blocked online MSE, while
  repetition branches fix online MSE but give back held-out accuracy.
- This is an empirical controlled solution, not a proof of universal recursive
  representation learning.

## Next Work

The highest-value next experiments are not broader local sweeps. They are:

1. Dense-zero and sparse-multilabel stress tests for target-density.
2. Longer-horizon class-blocked tests that separate online tracking from
   retained held-out performance.
3. External scale replication, especially the 800-task OPMNIST-style matrix.
4. A direct ablation of hidden-unit recycling criteria under target-density
   loss.
5. A combined Step 2/Step 3 transition test where target-density UPGD features
   feed continual GVF prediction.
