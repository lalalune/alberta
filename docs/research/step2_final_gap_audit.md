# Step 2 Final Gap Audit

## Verdict

The current supervised Step 2 promotion matrix is complete under the repo's
strict empirical acceptance bar:

- one online learner can beat the same-run best fair MLP without a deployment
  portfolio or output router;
- all methods update every timestep;
- the matrix includes controlled nonlinear/interaction/rare/polynomial/frequency
  probes, out-of-class synthetic probes, and five sklearn-digits regimes;
- hard digit risk rows and target-structure boundary rows have focused
  confirmation.

The promoted learner is target-structure UPGD. It is a single nonlinear online
learner with hidden feature utility, low-utility perturbation, ObGD-bounded
updates, and a structural target-loss normalizer. The normalizer uses sum-style
loss only for non-negative simplex targets with total mass 1, and mean-style
loss otherwise. This keeps the one-hot digit advantage of sum loss without the
exact-zero dense-target and sparse-multilabel ambiguity of the interim
`target_density` rule.

## Evidence

Canonical artifacts:

- `docs/research/step2_target_structure_upgd_stress.md`
- `output/subagents/upgd_simplification_structure_synthetic_30seed/out_of_class_results.json`
- `output/subagents/upgd_simplification_scale_digits_lr05e1_30seed/upgd_digits_sweep_results.json`
- `output/subagents/upgd_simplification_structure_digits_30seed/digits_ablation_results.json`
- `outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_results.json`
- `outputs/step2_canonical/opmnist_true_mnist_40block_mse_results.json`

Target-structure UPGD:

- Local 8-seed stress probe: `target_structure` matches mean loss on
  dense-zero (`+0.000232 +/- 0.000311`, `5/8`) and sparse multilabel
  (`+0.026938 +/- 0.000865`, `8/8`) targets, while `target_density` is negative
  on dense-zero and much weaker on sparse multilabel.
- 30-seed synthetic structure rerun: `structure_sigma1e4_kappa05` beats the
  same-run best fair MLP on polynomial (`+0.5473 +/- 0.0316`, `30/30`),
  frequency (`+0.5756 +/- 0.0376`, `30/30`), and compositional
  (`+0.0924 +/- 0.0038`, `30/30`).
- 30-seed digits simplification rerun: the no-meta density-equivalent
  `lr05_e1` branch has final-window MSE `+0.0062 +/- 0.0003` over `150`
  seed/regime comparisons and held-out accuracy `+0.0299 +/- 0.0019`. Because
  all active digit targets are one-hot simplex targets, `target_structure` has
  the same loss behavior on this matrix.
- Fixed-kappa UPGD remains a simpler control: density-equivalent `kappa05`
  gives final-window MSE `+0.0036 +/- 0.0002` and `147/150` wins on digits,
  while `structure_sigma1e4_kappa05` is the strongest compositional synthetic
  row among the tested structure variants.

D18 persistent trace is now superseded rather than promoted. It remains useful
historical evidence: the 10-seed 14-regime matrix had `138/2/0` seed-level
final-window MSE wins/losses/ties against the same-run best fair MLP, and the
30-seed hard digit rows were positive by mean under the projected-MLP check.
UPGD is preferred because it is a native nonlinear learner with online hidden
utility and direct vector-target heads, not a hand-assembled additive
resource-basis system.

External scale evidence:

- Latest-best true OpenML MNIST OPMNIST evaluated run: 800/800 task blocks,
  48,000,000 online examples, 800 random pixel permutations, canonical 60k/10k
  split, no task id, prediction before update, and same-run raw/sharpened fair
  MLP baselines.
- It is positive versus the best fair MLP for online MSE (`+0.003782`), online
  accuracy (`+0.012320`), and final-window MSE (`+0.002115`).
- It is negative versus the best fair MLP for final-window accuracy
  (`-0.004200`), all-permutation held-out test MSE (`-0.039517`), and
  all-permutation held-out test accuracy (`-0.017840`).
- `published_scale_external_claim_supported` can now be true only for the
  narrow online-tracking/MSE side of OPMNIST. A retained-view/generalization
  claim still needs a better learner or additional evidence.

## Alternate Closure Paths

Pure single recursive feature construction is now positive but not the promoted
Step 2 solution. The best current pure recursive candidate,
`single_mechanism_retention_tanh24_tanh_heavy_conservative`, is a single
learner with a tanh-heavy/product operation prior and conservative promotion.
It beats the best fair MLP by paired mean on all six controlled probes at 10
seeds: nonlinear `+0.008718` (`9/1`), interaction `+0.386067` (`10/0`),
triple `+0.387012` (`9/1`), rare `+0.025779` (`8/2`), polynomial `+0.081576`
(`6/4`), and frequency `+0.068532` (`10/0`). This is meaningful evidence for
recursive construction, but nonlinear still has one large failed seed and
polynomial is only `6/10`, so target-structure UPGD remains the cleaner
promoted Step 2 learner.

Native deep feature lifecycle is rejected as the promoted Step 2 solution. The
implementation is real and tested, including preserve-outgoing promotion,
active perturbation, soft-gated candidates, Net2Net-style variants, and the
new normalized candidate-update path. The latest shallow/native follow-up still
reaches only `2/6` positive hard probes as a single variant, so this remains a
diagnostic path rather than the promoted Step 2 mechanism.

TD/GVF feature discovery is not a Step 2 blocker. A 2026-05-06 Step 3
follow-up adds positive DoD-7 evidence on coupled-hidden AR(1) GVF prediction
and a separate clipped-IS off-policy TD probe, but it remains Step 3 research
because the target class changes from supervised targets to bootstrapped
GVF/question targets. The positive mechanisms use GVF feedback and off-policy
Bellman-error feature selection, not a supervised-to-TD bridge that would expand
the Step 2 claim.

## What Is Still Not Claimable

Do not claim:

- a theorem of universal feature construction;
- that a pure recursive generator is robustly universal;
- that native deep hidden-unit lifecycle is solved;
- that the completed one-seed Dohare-style 800-task OPMNIST run is an
  unqualified win, because the retained all-permutation test metrics are still
  MLP-favored.

The unqualified local claim that is now defensible is:

> The repo's current supervised Step 2 empirical acceptance matrix is closed by
> a single non-router learner, target-structure UPGD, against same-run best fair
> MLP baselines.
