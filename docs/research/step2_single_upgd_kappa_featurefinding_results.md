# Step 2 Single-UPGD Kappa/Feature-Finding Results

This note records the latest no-portfolio Step 2 result on the external
sklearn-digits universality matrix. The tested regimes were `iid`,
`permuted_pixels`, `mask_noise`, `label_drift`, and `class_blocked`, using the
existing 1200-step online protocol, 300-step final window, hidden size 64, and
the fair `mlp64` baseline.

## 2026-05-05 Update: Target-Density Supersedes Raw Sum Loss

The current UPGD result is no longer just a digits-only sum-loss tuning story.
The important change is `loss_normalization="target_density"` in
`UPGDLearner`. It divides the active squared-error loss by the number of
nonzero supervised target components. Dense vector targets therefore behave
like mean loss; sparse one-hot targets behave like sum loss. That removes the
previous conflict where mean-loss UPGD won the dense synthetic Step 2 streams
and raw sum-loss UPGD won sklearn-digits but failed the synthetic suite.

Promoted single-learner variants:

| Role | Variant |
|---|---|
| Simple bridge | `upgd_density_sigma1e_4_kappa05` |
| Best average/test default | `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight` |
| Strict online-MSE default | `upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight` |
| Conservative class-blocked compromise | `upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight` |

The best average/test default uses target-density loss, low-utility
perturbation `sigma=1e-4`, base `kappa=0.5`, fast/slow adaptive `kappa` in
`[0.35, 0.65]`, hidden-step multiplier `0.6`, and gradient-alignment
meta-plasticity for readout weights/biases only. It leaves trunk plasticity
fixed.

30-seed synthetic vector-target evidence:

| Stream | Target-density UPGD diff vs best MLP | Wins |
|---|---:|---:|
| Polynomial | `+0.5473 +/- 0.0316` | `30/30` |
| Frequency | `+0.5761 +/- 0.0375` | `30/30` |
| Compositional | `+0.0924 +/- 0.0038` | `30/30` |

30-seed sklearn-digits evidence:

| Metric | Best average/test default vs `mlp64` |
|---|---:|
| Final-window MSE | `+0.0063 +/- 0.0003`, `120/150` |
| Test accuracy | `+0.0298 +/- 0.0017`, `140/150` |
| Final-window accuracy | `+0.0278 +/- 0.0020`, `119/150` |
| Test MSE | `+0.0080 +/- 0.0004`, `147/150` |

The strict online-MSE branch, `repx075_meta001_notrunk_tight`, gives up some
held-out test accuracy but beats `mlp64` on final-window MSE in all `150/150`
digit seed/regime comparisons. The conservative compromise,
`repx025_meta001_notrunk_tight`, has weaker aggregate MSE than the default but
keeps the hard class-blocked row nonnegative on both online MSE and held-out
accuracy in the latest 30-seed run.

Sources:
`output/subagents/combined_upgd/synthetic_target_density_30seed_6000/out_of_class_SUMMARY.md`,
`output/subagents/combined_upgd/synthetic_density_control_30seed_6000/out_of_class_SUMMARY.md`,
`output/subagents/combined_upgd/digits_density_control_30seed/SUMMARY.md`, and
`output/subagents/combined_upgd/digits_density_repetition_compromise_30seed/SUMMARY.md`.

## Earlier Sum-Loss Digit Family

Before target-density loss closed the dense/sparse conflict, the strongest
digits-only result was not a single fixed `kappa` setting. The better
interpretation was a small raw-sum-loss UPGD family:

- Online/final-window MSE default:
  `upgd_sum_sigma1e_4_adaptk035_065_lr06`
- Strict no-loss online-MSE variant:
  `upgd_sum_sigma1e_4_adaptk035_065_lr06_repx075`
- Conservative class-blocked fix:
  `upgd_sum_sigma1e_4_adaptk035_065_lr06_repx025`
- Strong follow-up online-MSE variant:
  `upgd_sum_sigma1e_4_adaptk035_065_lr05_e1`
- Held-out test MSE/accuracy specialist:
  `upgd_sum_sigma1e_4_kappa05_lr045`

Shared configuration:

- Sum multi-task MSE.
- UPGD low-utility perturbation `sigma=1e-4`.
- ObGD bounding with base `kappa=0.5` instead of `kappa=2.0`.
- No portfolio, no Hedge router, no live MLP widths, no method switching.

For the adaptive variants, `kappa` is controlled online by a fast/slow loss
ratio. When fast loss rises above slow loss, effective `kappa` is reduced,
allowing larger plastic updates; when fast loss is calm, effective `kappa`
returns toward the base. This is a temporally uniform control law, not
meta-gradient learning.

Corrected 30 seeds x 5 regimes = 150 paired comparisons for the first adaptive
MSE default, using seeds 60-89:

| Metric | MLP | `upgd_sum_sigma1e_4_adaptk035_065_lr06` | Paired Diff | Wins |
|---|---:|---:|---:|---:|
| Final-window MSE | 0.0402 +/- 0.0017 | 0.0344 +/- 0.0015 | +0.0059 +/- 0.0003 | 127/150 |
| Final-window accuracy | 0.8304 +/- 0.0088 | 0.8562 +/- 0.0078 | +0.0258 +/- 0.0018 | 123/150 |
| Test MSE | 0.0578 +/- 0.0032 | 0.0517 +/- 0.0033 | +0.0061 +/- 0.0003 | 138/150 |
| Test accuracy | 0.7184 +/- 0.0244 | 0.7408 +/- 0.0253 | +0.0224 +/- 0.0020 | 119/150 |

Artifact:

`output/subagents/combined_upgd/kappa_lr_adaptive_top_30seed/SUMMARY.md`

The second-stage LR/adaptive sweep, using seeds 90-119, found a slightly
different frontier:

| Metric | Best Variant | MLP | Variant | Paired Diff | Wins |
|---|---|---:|---:|---:|---:|
| Final-window MSE | `adaptk035_065_lr05_e1` | 0.0403 +/- 0.0017 | 0.0343 +/- 0.0014 | +0.0059 +/- 0.0003 | 120/150 |
| Final-window accuracy | `adaptk035_065_lr06` | 0.8285 +/- 0.0090 | 0.8569 +/- 0.0077 | +0.0284 +/- 0.0020 | 121/150 |
| Test MSE | `kappa05_lr045` | 0.0576 +/- 0.0032 | 0.0505 +/- 0.0032 | +0.0071 +/- 0.0003 | 147/150 |
| Test accuracy | `kappa05_lr045` | 0.7189 +/- 0.0249 | 0.7455 +/- 0.0253 | +0.0265 +/- 0.0018 | 134/150 |

Artifact:

`output/subagents/combined_upgd/kappa_lr_adaptive_lr05_top_30seed/SUMMARY.md`

## Class-Blocked Follow-Up: Target-Repetition Head Plasticity

The original adaptive-kappa default still lost the `class_blocked` final-window
MSE row. The useful fix was not more trunk plasticity, hard recycling, softmax,
or a portfolio. It was a narrow online readout-plasticity signal: track an EMA
of whether the supervised target vector is locally repeated, and temporarily
increase only output-head update size when the stream has persistent target
blocks.

This is implemented in `UPGDLearner` as `head_repetition_multiplier` with
`head_repetition_decay`, `head_repetition_delta_threshold`, and optional
`head_repetition_pressure_threshold`. It uses only the current and previous
supervised target vectors, so it is causal and temporally uniform in the Step 2
supervised setting.

Scaled 30 seeds x 5 regimes = 150 paired comparisons for the best repetition
variants, using seeds 200-229:

| Metric | `adaptk035_065_lr06` | `repx025` | `repx075` |
|---|---:|---:|---:|
| Final-window MSE diff | +0.0058, 127/150 | +0.0057, 147/150 | +0.0055, 150/150 |
| Final-window accuracy diff | +0.0273, 123/150 | +0.0243, 123/150 | +0.0268, 135/150 |
| Test MSE diff | +0.0058, 137/150 | +0.0056, 134/150 | +0.0036, 123/150 |
| Test accuracy diff | +0.0236, 126/150 | +0.0209, 117/150 | +0.0181, 114/150 |

Artifact:

`output/subagents/combined_upgd/repetition_top_30seed/SUMMARY.md`

By-regime final-window MSE deltas in that run:

| Regime | `adaptk035_065_lr06` | `repx025` | `repx075` |
|---|---:|---:|---:|
| `iid` | +0.0063 | +0.0060 | +0.0056 |
| `class_blocked` | -0.0002 | +0.0004 | +0.0008 |
| `permuted_pixels` | +0.0078 | +0.0079 | +0.0074 |
| `mask_noise` | +0.0090 | +0.0081 | +0.0077 |
| `label_drift` | +0.0062 | +0.0060 | +0.0060 |

Interpretation: `repx075` is the first single-learner UPGD variant in this
thread to beat the fair MLP on final-window MSE in every one of the 150
seed/regime comparisons. It is the best strict online-MSE universality result.
The cost is weaker held-out retention than the old default. `repx025` is a more
conservative compromise: it fixes class-blocked final-window MSE in aggregate
with much smaller test-MSE/test-accuracy cost, but it is not perfectly no-loss.

A follow-up class-blocked-only bias ablation found that slowing or freezing
head biases can recover some held-out retention, but it gives back too much
online-MSE gain to replace the plain repetition variants. Source:

`output/subagents/combined_upgd/classblocked_repetition_bias_screen_10seed/SUMMARY.md`

## Learned Group Plasticity Follow-Up

The next useful simplification was to stop treating all of the readout/trunk
plasticity knobs as fixed constants. `UPGDLearner` now supports an optional
`meta_plasticity_mode="gradient_alignment"` controller. It learns bounded log
multipliers for four groups:

- shared trunk updates,
- output-head weight updates,
- output-head bias updates,
- the repeated-target head-plasticity gain.

The rule is deliberately small: if consecutive gradients for a group align,
the group's log multiplier increases; if they reverse, it decreases. The
multiplier is clipped, uses only past gradients and current gradients, and is
applied on the next update. This keeps it causal and temporally uniform while
learning the bias/readout/trunk plasticity rates rather than hand-setting every
one of them.

A 5-seed x 5-regime screen against the same fair `mlp64` baseline found that
the conservative learned-plasticity variant improved average final-window MSE
over the previous adaptive default:

| Variant | Final-window MSE diff | Wins | Test accuracy diff |
|---|---:|---:|---:|
| `adaptk035_065_lr06_meta001_tight` | +0.0055 +/- 0.0007 | 21/25 | +0.0220 +/- 0.0037 |
| `adaptk035_065_lr06_meta003_tight` | +0.0054 +/- 0.0008 | 20/25 | +0.0212 +/- 0.0031 |
| `adaptk035_065_lr06` | +0.0052 +/- 0.0006 | 21/25 | +0.0229 +/- 0.0049 |
| `repx075_meta001_tight` | +0.0050 +/- 0.0006 | 25/25 | +0.0150 +/- 0.0031 |
| `repx075` | +0.0049 +/- 0.0006 | 25/25 | +0.0119 +/- 0.0040 |

Source:

`output/subagents/combined_upgd/meta_plasticity_all_5seed/SUMMARY.md`

The class-blocked-only screen showed the complementary result: the best
class-blocked final-window MSE variant was `repx075_meta001_tight`, improving
the MSE diff to `+0.0006 +/- 0.0001` at `5/5` wins in that screen, compared
with `+0.0003 +/- 0.0002` at `4/5` for fixed `repx075`. Stronger
meta-plasticity (`0.003`) often helped held-out MSE but hurt online tracking.

Source:

`output/subagents/combined_upgd/meta_plasticity_classblocked_5seed/SUMMARY.md`

A stronger 10-seed x 5-regime top-candidate validation moved the frontier
slightly. The learned-plasticity variants became the best average-MSE and
held-out candidates, while fixed `repx075` remained the cleanest class-blocked
anchor:

| Variant | Final-window MSE diff | Wins | Test accuracy diff |
|---|---:|---:|---:|
| `repx025_meta003_tight` | +0.0062 +/- 0.0006 | 40/50 | +0.0210 +/- 0.0034 |
| `repx075_meta003_tight` | +0.0062 +/- 0.0005 | 45/50 | +0.0233 +/- 0.0031 |
| `meta003_tight` | +0.0062 +/- 0.0006 | 40/50 | +0.0239 +/- 0.0030 |
| `meta001_tight` | +0.0060 +/- 0.0006 | 40/50 | +0.0222 +/- 0.0033 |
| `repx075_meta001_tight` | +0.0058 +/- 0.0005 | 50/50 | +0.0219 +/- 0.0031 |
| `adaptk035_065_lr06` | +0.0057 +/- 0.0005 | 43/50 | +0.0164 +/- 0.0031 |
| `repx075` | +0.0056 +/- 0.0004 | 50/50 | +0.0171 +/- 0.0029 |

Source:

`output/subagents/combined_upgd/meta_plasticity_top_10seed/SUMMARY.md`

Interpretation: learned group plasticity is now the best average-MSE follow-up
candidate and appears to improve test accuracy as well. It is still not as
decisive as the 30-seed fixed `repx075` no-loss result because the average-MSE
winners have some seed/regime losses. The best strict learned variant in the
10-seed run is `repx075_meta001_tight` at `50/50` final-window MSE wins, but it
still needed a 30-seed matrix before replacing fixed `repx075`. The later
target-density repetition-compromise run supplies that missing confirmation for
`repx075_meta001_notrunk_tight`.

## Learned Kappa Follow-Up

`UPGDLearner` now also supports learning a bounded log multiplier on effective
ObGD `kappa`. The rule is intentionally parallel to the group-plasticity rule:
when consecutive global gradients align, the learned multiplier lowers
effective `kappa`, allowing larger bounded updates; when gradients reverse, it
raises `kappa`, making the update more conservative. The learned multiplier is
causal, clipped, and applied on the next update.

The 5-seed screen made learned kappa look promising, but the decisive 30-seed
validation is more conservative. On 30 fresh seeds x 5 regimes = 150 paired
comparisons, using seeds 600-629:

| Variant | Final-window MSE diff | Wins | Test MSE diff | Test accuracy diff |
|---|---:|---:|---:|---:|
| `meta003_tight` | +0.0059 +/- 0.0003 | 120/150 | +0.0084 +/- 0.0003 | +0.0301 +/- 0.0021 |
| `meta003_kmeta001_tight` | +0.0059 +/- 0.0003 | 120/150 | +0.0082 +/- 0.0003 | +0.0299 +/- 0.0019 |
| `repx075_meta001_tight` | +0.0058 +/- 0.0003 | 149/150 | +0.0053 +/- 0.0004 | +0.0250 +/- 0.0021 |
| `repx075` | +0.0054 +/- 0.0002 | 150/150 | +0.0038 +/- 0.0005 | +0.0198 +/- 0.0022 |

Source:

`output/subagents/combined_upgd/kappa_meta_top_30seed/SUMMARY.md`

By-regime final-window MSE on the hard `class_blocked` row:

| Variant | Class-blocked final-window MSE diff | Class-blocked test accuracy diff |
|---|---:|---:|
| `repx075` | +0.0008 | -0.0096 |
| `repx075_meta001_tight` | +0.0007 | +0.0028 |
| `meta003_tight` | -0.0009 | +0.0229 |
| `meta003_kmeta001_tight` | -0.0010 | +0.0194 |

Interpretation: learned kappa is implemented and scientifically sensible, but
it does not beat learned group plasticity by itself at 30 seeds. The best
average/test branch is `meta003_tight` with learned group plasticity and no
learned kappa. In that raw-sum kappa sweep, the best strict no-loss branch
remained fixed `repx075`, while `repx075_meta001_tight` nearly preserved strict
online-MSE behavior (`149/150`) and recovered held-out accuracy. The later
target-density run makes `repx075_meta001_notrunk_tight` the strict branch.
Learned kappa should stay as an ablation and future-control mechanism, not the
promoted default.

## Meta-Control Group Ablation

The group-plasticity result raised a mechanism question: is the win from
learning hidden-feature/trunk plasticity, or from simpler readout adaptation?
`UPGDLearner` now exposes switches for trunk, head-weight, head-bias, and
repetition-gain meta-control. A 10-seed screen showed that disabling trunk
meta-control was at least as good as learning all groups, so the top candidates
were scaled to 30 seeds.

On 30 fresh seeds x 5 regimes = 150 paired comparisons, using seeds 720-749:

| Variant | Final-window MSE diff | Wins | Test MSE diff | Test accuracy diff |
|---|---:|---:|---:|---:|
| `meta003_notrunk_tight` | +0.0061 +/- 0.0003 | 120/150 | +0.0078 +/- 0.0003 | +0.0293 +/- 0.0018 |
| `meta003_tight` | +0.0060 +/- 0.0003 | 120/150 | +0.0076 +/- 0.0003 | +0.0280 +/- 0.0018 |
| `meta003_noheadb_tight` | +0.0059 +/- 0.0003 | 120/150 | +0.0074 +/- 0.0003 | +0.0271 +/- 0.0017 |
| `repx075_meta001_notrunk_tight` | +0.0057 +/- 0.0002 | 149/150 | +0.0047 +/- 0.0004 | +0.0212 +/- 0.0016 |
| `repx075` | +0.0056 +/- 0.0002 | 150/150 | +0.0037 +/- 0.0006 | +0.0213 +/- 0.0020 |

Source:

`output/subagents/combined_upgd/meta_group_ablation_top_30seed/SUMMARY.md`

By-regime class-blocked read:

| Variant | Class-blocked final-window MSE diff | Class-blocked test accuracy diff |
|---|---:|---:|
| `repx075` | +0.0010 | -0.0080 |
| `repx075_meta001_notrunk_tight` | +0.0009 | +0.0011 |
| `meta003_tight` | -0.0009 | +0.0112 |
| `meta003_noheadb_tight` | -0.0009 | +0.0094 |
| `meta003_notrunk_tight` | -0.0009 | +0.0193 |

Interpretation: the best average/test branch is now
`meta003_notrunk_tight`, not the learn-every-group version. This is a useful
simplification: trunk plasticity should remain fixed in the current digits
learner, while readout plasticity is learned. The hard class-blocked tracking
row still requires target-repetition head plasticity. The later target-density
repetition-compromise run upgrades `repx075_meta001_notrunk_tight` from
near-strict to strict (`150/150`) while `repx025_meta001_notrunk_tight` becomes
the better class-blocked held-out compromise.

## Why Adaptive Kappa Is The MSE Default

The earlier breakthrough was lowering ObGD `kappa` from `2.0` to `0.5`, which
made UPGD plastic enough to beat the fair MLP. The next sweep showed that a
fixed `kappa=0.5,lr=0.6` was still not the MSE frontier. Letting effective
`kappa` move within `[0.35, 0.65]` from the fast/slow loss ratio improved
final-window MSE from `+0.0051` for the same-seed fixed `lr=0.6` control to
`+0.0059`.

The lower-LR sweep shows a second effect: if the goal is held-out
generalization, fixed `lr=0.045` is currently strongest. That does not replace
adaptive kappa as the online-MSE default because it sacrifices final-window
accuracy and is worse on class-blocked final-window MSE.

The previous fixed-kappa default remains important as a control:

| Metric | Fixed `kappa05_lr06` diff | Adaptive `adaptk035_065_lr06` diff |
|---|---:|---:|
| Final-window MSE | +0.0051 | +0.0059 |
| Test MSE | +0.0058 | +0.0061 |
| Final-window accuracy | +0.0189 | +0.0258 |
| Test accuracy | +0.0202 | +0.0224 |

This matters for Step 2: the useful control variable is not just whether
features are perturbed, but how strongly the hidden-feature updates are bounded
as non-stationary loss pressure changes.

## By-Regime Read

Final-window MSE improvements for `upgd_sum_sigma1e_4_adaptk035_065_lr06`
on seeds 60-89:

- `iid`: +0.0064
- `permuted_pixels`: +0.0080
- `mask_noise`: +0.0086
- `label_drift`: +0.0063
- `class_blocked`: -0.0001

Held-out accuracy improvements for `upgd_sum_sigma1e_4_adaptk035_065_lr06`
on seeds 60-89:

- `iid`: +0.0202
- `permuted_pixels`: +0.0332
- `mask_noise`: +0.0321
- `label_drift`: +0.0286
- `class_blocked`: -0.0019

The remaining weak row for the adaptive-kappa-only default is still
`class_blocked`. Repetition-gated head plasticity solves the class-blocked
online-MSE row, but class-blocked held-out retention remains an open tradeoff.

## Historical Sum-Loss Interpretation

The raw-sum-loss mechanism was simple:

1. Use sum loss so the multi-task target vector scales like the Step 2 problem.
2. Bound hidden updates weakly enough to permit plasticity.
3. Adapt the bound from recent-vs-background loss pressure.
4. Add tiny low-utility perturbations so unused features can search.

The result is not from softmax, margins, aggressive recycling, or a portfolio.
It is a single MLP-shaped learner with UPGD utility pressure and conservative
optimizer tuning.

## Negative Results

- Native softmax/CE readout was worse than linear MSE in this UPGD trunk.
- Strong readout margins did not close the accuracy gap; no margin or tiny
  margin was better.
- Negative-target downweighting improved held-out classification accuracy but
  worsened full-vector MSE, so it is a classifier specialization, not a faithful
  Step 2 default.
- Hard stale-unit recycling helped some `permuted_pixels` rows but was not
  universal.
- Unbounded UPGD improved some adaptation rows but destabilized more plastic
  variants. A weaker bounder is better than removing the bounder.

## Current Recommendation

Use `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight` as the
promoted average-MSE/test branch within the single UPGD family. Use
`upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight` when
the criterion is strict single-learner online universality against MLP, because
it has the clean 150/150 final-window MSE win matrix with better aggregate
held-out accuracy than fixed `repx075`. Use
`upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight` as
the best current class-blocked balance: it keeps class-blocked online MSE and
held-out accuracy nonnegative in the latest 30-seed matrix. Keep
`kappa05_lr045` as the older raw-sum held-out generalization
specialist/control.

Do not promote learned kappa yet. It is implemented and tested, but
`meta003_kmeta001_tight` did not beat `meta003_tight` at 30 seeds and it did
not fix class-blocked final-window MSE.

The next scientifically sensible work is to reduce the class-blocked retention
cost without changing the target-density loss semantics or falling back to a
portfolio. The most promising knob is now not global `kappa`; it is learned
target-persistence-aware readout plasticity coupled to feature-local retention
under the same single-learner objective.
