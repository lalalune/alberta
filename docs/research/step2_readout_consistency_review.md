# Step 2 Readout Consistency Review

Date: 2026-05-07.

## Question

Class-blocked digits is no longer the main blocker. A 64-64 softmax/CE UPGD
branch now closes strict class-blocked online MSE while preserving a large
retained-test advantage, but that same branch loses label-drift online MSE.
The earlier 64-64 linear-MSE target-structure branch wins label drift and the
random/noisy/permuted digit rows, but loses class-blocked online MSE.

The remaining question is therefore not "does UPGD have enough trunk
plasticity?" It is whether one causal readout rule can get both:

- softmax/probability behavior under persistent one-hot blocks;
- linear or robust behavior under label remapping/noisy target dynamics.

## Local Evidence

| Evidence | Read |
|---|---|
| `output/subagents/class_blocked_retention/softmax_lr07_allregime_10seed_6000/digits_ablation_SUMMARY.md` | The softmax lr07 64-64 branch beats MLP on IID, class-blocked, mask-noise, and permuted final-window MSE, but loses label-drift MSE. |
| `output/subagents/structure_adaptive_retention/allregime_10seed_6000/digits_ablation_SUMMARY.md` | The linear-MSE 64-64 structure branch beats MLP on IID, label-drift, mask-noise, and permuted final-window MSE, but loses class-blocked MSE. |
| `output/subagents/class_blocked_retention/softmax_closeout_label_drift_10seed_6000/digits_ablation_SUMMARY.md` | Softmax variants do not fix label drift by changing sigma, lr, repetition multiplier, or head scale. |
| `output/subagents/external_breadth_sklearn/passthrough_rescue_full_10seed_all/external_suite_SUMMARY.md` | Scalar regression and delayed temporal regression are readout/input bottlenecks, not evidence against UPGD feature utility; passthrough/deep UPGD rescues them but hurts classification/vector rows. |

Two new adaptive-readout probes were run after the softmax closeout:

| Probe | Result |
|---|---|
| `output/subagents/class_blocked_retention/readout_consistency_adaptive_sharp_3seed_6000/digits_ablation_SUMMARY.md` | A target-repeat-gated adaptive readout preserves label-drift, IID, mask-noise, and permuted wins, improves class-blocked relative to linear-MSE, but still does not beat `mlp64_64` on class-blocked final-window MSE. |
| `output/subagents/class_blocked_retention/readout_consistency_adaptive_cefloor_3seed_6000/digits_ablation_SUMMARY.md` | Adding a CE loss floor is rejected: class-blocked barely improves and label-drift collapses to about `0.062` final-window MSE. |

The adaptive readout is useful as a diagnostic but is not promoted.

The next implementation pass decoupled readout loss from prediction transform
and added robust generalized-cross-entropy simplex losses. It tested
linear-loss/softmax-prediction, CE-loss/identity-prediction,
CE-loss/clipped-prediction, GCE `q=0.7`/`q=0.3`, and adaptive GCE:

| Probe | Result |
|---|---|
| `output/subagents/class_blocked_retention/readout_consistency_decoupled_3seed_6000/digits_ablation_SUMMARY.md` | Linear-loss/softmax-prediction is rejected on both class-blocked and label-drift. CE-loss/identity-prediction has strong retained class-blocked accuracy but unusable online MSE because logits are unbounded. GCE improves over linear-softmax on class-blocked but does not close the gap and worsens label-drift. |
| `output/subagents/class_blocked_retention/readout_consistency_decoupled_clip_3seed_6000/digits_ablation_SUMMARY.md` | CE-loss/clipped-prediction is rejected: class-blocked final-window MSE is about `0.1035`, and label-drift final-window MSE is about `0.2132`. Bounded logits do not recover the softmax row. |

This closes the obvious decoupling and robust-loss ablations. There is still no
single fixed or simply gated readout branch that beats the best MLP on both
class-blocked and label-drift online MSE.

The next implementation pass tested a factorized simplex label-map readout:
`pred = softmax(logits) @ A_t`, where `A_t` is a row-stochastic causal adapter
stored inside the UPGD learner state. The base version trained through the
adapted probability vector; the gated version used target-repeat EMA to blend
between linear logits and the factorized simplex prediction.

| Probe | Result |
|---|---|
| `output/subagents/class_blocked_retention/factorized_simplex_preset_probe_3seed_6000/digits_ablation_SUMMARY.md` | Plain factorized readout is rejected. The best class-blocked factorized row still loses to `mlp64_64` (`-0.000295`, `0/3` wins), and all factorized rows lose label-drift badly. |
| `output/subagents/class_blocked_retention/adaptive_factorized_probe_3seed_6000/digits_ablation_SUMMARY.md` | Target-repeat-gated factorized readout preserves label-drift wins (`+0.003676`, `3/3` for the best gated row), but does not improve class-blocked beyond `adaptive_simplex` and remains below `mlp64_64`. |

This rejects the simplest label-map factorization as a promotion candidate. It
is useful as a negative result: target remapping alone does not solve the
readout conflict unless the learner also has genuinely separate readout
timescales or a better causal signal for when to protect the probability head.

The next implementation pass added a two-timescale simplex readout inside one
UPGD learner state. The slow head remains the original linear-MSE readout. The
fast head is a separate softmax/CE readout that shares the same trunk. The
emitted prediction uses the existing target-repeat gate:

`pred = gate * softmax(fast_logits) + (1 - gate) * slow_logits`

The first version updated the fast head only. It preserved label-drift wins but
did not close class-blocked because the trunk was still shaped only by the slow
linear-MSE head. The follow-up version adds a gated fast-head CE gradient into
the shared trunk with `readout_fast_trunk_gradient_multiplier`.

| Probe | Result |
|---|---|
| `output/subagents/class_blocked_retention/twotime_probe_3seed_6000/digits_ablation_SUMMARY.md` | Head-only two-timescale readout wins label-drift (`+0.0039` to `+0.0047`, `3/3`) but misses class-blocked final-window MSE. |
| `output/subagents/class_blocked_retention/fasttrunk_probe_3seed_6000/digits_ablation_SUMMARY.md` | Gated fast-trunk CE pressure gives the first positive bridge: `fastx2_trunk05` beats the best MLP on class-blocked by `+0.000028` with `2/3` wins and keeps label-drift positive by `+0.00398` with `3/3` wins. |
| `output/subagents/class_blocked_retention/fasttrunk_bound_patched_probe_5seed_6000/digits_ablation_SUMMARY.md` | The first bridge depended on an unfair implementation detail: the fast readout head was not in the ObGD bounder. After bounding it in the shared budget, the bridge is rejected. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_probe_5seed_6000/digits_ablation_SUMMARY.md` | A fairer bounded readout allocation restores the bridge: fast head separately bounded, fast trunk pressure retained, slow simplex MSE budget suppressed at a fully open gate. `fastx2_trunk2_slow0_sepbound` wins class-blocked and label drift `5/5`. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_allregime_10seed_6000/digits_ablation_SUMMARY.md` | The same row passes the 10-seed all-regime gate: positive final-window MSE on IID, class-blocked, label drift, mask-noise, and permuted digits. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_strict_30seed_6000/digits_ablation_SUMMARY.md` | The same row passes the 30-seed gate: final-window MSE paired diffs versus the best fair MLP are IID `+0.006172` `30/30`, class-blocked `+0.000189` `26/30`, label drift `+0.005352` `30/30`, mask-noise `+0.009692` `30/30`, and permuted `+0.007323` `30/30`. |
| `output/subagents/class_blocked_retention/fasttrunk_sepbound_x2_synthetic_30seed_6000/synthetic_ablation_SUMMARY.md` | The same row is not a broad synthetic default. It is positive on polynomial and frequency but loses compositional regression to `MLP(64)`, so it should be framed as the strict digit readout only. |

This promotes the separately bounded two-timescale readout as the strict
one-branch digit readout. The class-blocked margin remains the smallest row,
but it survived the larger seed block while label-drift kept the linear-MSE
tracking advantage. It does not replace the broad target-structure default.

## Literature Read

The plasticity side is well supported by the continual learning literature:

- Continual Backprop argues that persistent randomization can prevent loss of
  plasticity in nonstationary neural networks.
- UPGD formalizes utility-based perturbation: change low-utility weights more
  than useful weights, so exploration is concentrated where the current
  representation is least productive.

Those papers justify the feature-utility and perturbation side of this repo's
UPGD learner, but they do not settle output-target semantics.

The readout conflict matches a different literature: robust learning under
noisy labels. Classical cross-entropy can be brittle under label noise, while
MAE-like or generalized cross-entropy losses trade statistical efficiency for
robustness. Label-drift digits is not ordinary IID label noise, but from the
readout's perspective it repeatedly sees target remapping pressure. That makes
the local result plausible: fixed CE/probability geometry is excellent for
stable simplex blocks, but too brittle for the label-drift row.

Relevant sources:

- Dohare et al., "Continual Backprop: Stochastic Gradient Descent with
  Persistent Randomness", [arXiv:2108.06325](https://arxiv.org/abs/2108.06325).
- Elsayed et al., "Utility-based Perturbed Gradient Descent: An Optimizer for
  Continual Learning", [arXiv:2302.03281](https://arxiv.org/abs/2302.03281).
- Elsayed and Mahmood, "Addressing Loss of Plasticity and Catastrophic
  Forgetting in Continual Learning", [arXiv:2404.00781](https://arxiv.org/abs/2404.00781).
- Dohare et al., "Maintaining Plasticity in Deep Continual Learning",
  [arXiv:2306.13812](https://arxiv.org/abs/2306.13812).
- Farias and Jozefiak, "Self-Normalized Resets for Plasticity in Continual
  Learning", [arXiv:2410.20098](https://arxiv.org/abs/2410.20098).
- Lillo and Cheney, "Activation Function Design Sustains Plasticity in
  Continual Learning", [arXiv:2509.22562](https://arxiv.org/abs/2509.22562).
- Titsias et al., "Kalman Filter for Online Classification of Non-Stationary
  Data", [arXiv:2306.08448](https://arxiv.org/abs/2306.08448).
- Ghosh et al., "Robust Loss Functions under Label Noise for Deep Neural
  Networks", [arXiv:1712.09482](https://arxiv.org/abs/1712.09482).
- Zhang and Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural
  Networks with Noisy Labels", [arXiv:1805.07836](https://arxiv.org/abs/1805.07836).

## Diagnosis

The original implementation coupled two separate choices:

1. the training gradient geometry: linear MSE vs softmax CE;
2. the prediction space scored by the online MSE: logits vs probabilities.

That coupling has now been tested directly. The result is sharper:

- Label drift needs the linear-MSE/logit combination. Linear loss with softmax
  predictions is bad, and CE loss with identity predictions is catastrophic on
  MSE.
- Class-blocked needs CE/probability geometry. Linear loss with softmax
  predictions is bad, so the softmax output alone is not enough.
- Robust GCE and clipped logits are not sufficient bridges.

The adaptive-readout probe shows that target-repeat gating alone is
insufficient. Once the stream is recognized as block-like, the CE/probability
geometry has not been trained enough to fully match the always-softmax branch.
The CE-floor probe shows that blindly training CE all the time is worse: it
destroys the label-drift advantage.

## What Is Known

- UPGD's utility perturbation and target-structure normalization are still
  strong. The latest failures are not broad evidence against hidden-feature
  construction.
- A single fixed readout mode was the blocker; the current promoted strict
  digit readout uses one learner state with two readout timescales.
- Scalar/temporal regression should be handled by target-structure input
  passthrough or causal temporal context, not by changing the digit readout.
- Simple target-repeat CE/MSE blending, GCE robust losses, clipped-logit
  calibration, and a single factorized label-map adapter are not enough.
- A two-timescale readout with separately bounded fast head, gated fast-trunk
  CE pressure, and slow simplex MSE budget suppression is the first fair bridge
  to pass both the 10-seed and 30-seed all-regime digit gates.

## What Is Novel

The current repo result is not merely "UPGD plus CE". The novel part is the
combination of:

- online hidden-feature utility from weight magnitude and gradient demand;
- low-utility perturbation instead of global random mutation;
- target-structure normalization that distinguishes simplex, dense-zero, and
  multilabel targets without a dataset switch;
- a discovered readout failure mode where a single MLP-shaped UPGD learner
  needs different output semantics for persistent simplex targets versus
  remapped/noisy simplex targets.

That last point is publishable only if the paper is honest: the current
evidence supports a strong empirical mechanism, not universal representation
learning in the theorem sense.

## Recommended Next Work

1. Instrument the readout.

   Every digit readout run should log target-repeat EMA, selected readout gate,
   CE branch loss, MSE branch loss, readout gradient norms, and prediction
   transform. Without this, the next failure will be hard to interpret.

2. Keep `adaptive_simplex` experimental.

   It is a useful diagnostic and preserves the label-drift side in small
   probes, but it does not close class-blocked. Do not promote it without a
   positive 10-seed or 30-seed matrix.

3. Stress the two-timescale fast-trunk bridge outside digits.

   The current mechanism is `two_timescale_simplex` plus
   `readout_fast_trunk_gradient_multiplier`,
   `readout_fast_head_bounder_mode="separate"`, and
   `readout_slow_simplex_gradient_multiplier=0.0`. It passed the digit
   promotion gate but lost synthetic compositional regression. The next
   falsification checks are sparse multilabel behavior, scalar/temporal
   passthrough selection, and whether the readout budget multipliers can be
   learned or derived rather than hand-set.

4. Promotion criterion.

   The promoted one-branch digit readout has now beaten the best same-run MLP
   on all five digit final-window MSE rows at both 10 and 30 seeds. It also
   preserves nonnegative retained accuracy on IID, class-blocked, mask-noise,
   and permuted digits. Label-drift retained accuracy is treated as a
   diagnostic only because the retained label map is intentionally drifting.

## Current Recommendation

Do not promote the softmax row as the universal digit default, and do not
promote the linear row as the universal digit default. The honest current
status is:

- `softmax_ce` 64-64 is the class-blocked closeout branch.
- `linear_mse` 64-64 target-structure is the label-drift branch.
- `adaptive_simplex`, readout decoupling, robust GCE, clipped CE prediction,
  `factorized_simplex`, and `adaptive_factorized_simplex` are
  negative/diagnostic bridge attempts.
- `two_timescale_simplex` with separately bounded fast readout, gated
  fast-trunk CE pressure, and slow simplex MSE budget suppression is the
  promoted strict one-branch digit readout. It is canonical for the readout
  consistency problem, while the simpler target-structure UPGD branch remains
  the resource-efficient broad default.
