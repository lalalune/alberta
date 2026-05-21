# Step 2 Scaling And New Directions Synthesis

## Scaled Positive Result

The M01/M04 follow-up was deliberately stricter than the moonshot smoke tests:
candidates were compared against the best fair MLP in each suite rather than a
single fixed-width MLP.

Result: the current quadratic/centered candidate is not universal.

- Pair-product interaction stream: positive. `quad_linear_h128` beat the best
  fair MLP with a paired final-window loss difference of `+0.1298` over 3 seeds.
- Tanh-latent nonlinear control: negative. The best fair MLP beat the best
  quadratic candidate on all 3 seeds.
- IID digits: negative. `mlp_one_hot_h128` beat the centered and quadratic
  candidates.
- Label-drift digits: mixed but not promoted. Centered targets had higher mean
  final-window accuracy than one-hot h64, but did not beat the selected best MLP
  by the paired rule.
- Permuted-pixel digits: negative. One-hot MLPs beat centered/quadratic
  candidates on all paired final-window accuracy comparisons.

Artifact: `docs/research/step2_quadratic_centered_scale.md`.

## Hedge Follow-Up

The next variant used exponential-weights aggregation over fair MLP and
quadratic experts. This has a clear regret-style rationale: prefer quadratic
experts when the stream is pair-product structured, and fall back to MLP experts
when it is not.

Result: also not universal.

- Interaction stream: Hedge put essentially all final weight on
  `quad_linear_h512` and beat the best MLP in 2/3 seeds.
- Nonlinear control: Hedge put all final weight on MLP experts, but still trailed
  the best MLP by `-0.0017` final-window loss on average.

This is useful diagnostically: the mechanism can identify the right expert, but
mixing overhead and short-horizon adaptation are enough to prevent a universal
win.

Artifact: `docs/research/step2_quadratic_hedge_scale.md`.

## Current Assessment

The honest conclusion is conditional, not universal. Explicit pair-product
features are a real win when the target contains pair products. Centered targets
are a small classification tuning knob. Neither is a general replacement for a
fair MLP on external digits or non-quadratic synthetic streams.

The next Step 2 candidate should not be another static feature expansion. The
best new directions are those that preserve the fair MLP's broad competence
while improving adaptation, conditioning, or observability.

## D01-D10 Pilot Closure

The ten grounded directions have now been moved from paper specs to runnable
pilots in `step2_new_direction_pilots.py`. The 5-seed, 900-step scaled pass is
recorded in `docs/research/step2_new_direction_pilots_scaled.md`.

Result: dynamic sparse rewiring is the strongest new isolated candidate on the
two evaluated regression streams.

- Interaction stream: `d03_dynamic_sparse` beats fair MLP by
  `+0.5377 +/- 0.0951` final-window loss, `5/0` paired wins.
- Nonlinear stream: `d03_dynamic_sparse` beats fair MLP by
  `+0.0481 +/- 0.0039` final-window loss, `5/0` paired wins.
- Loss-space Hedge portfolios over the candidate set also win both suites, but
  mostly by tracking dynamic sparse. They are causal diagnostics, not yet
  prediction-space canonical learners.

The scaled pass also closes several attractive but weaker ideas:

- fixed splines flipped negative after scaling;
- input whitening flipped negative after scaling;
- RLS random features, history features, and precision-weighted linear readout
  are negative;
- KRLS and independent heads are interaction-only;
- high-leak homeostasis is a small nonlinear-only proxy signal.

Detailed closure assessment:
`docs/research/step2_candidate_closure_assessment.md`.

## Universal Portfolio Attempt

The follow-up prediction-space portfolio over `mlp_h64`, `mlp_h128`,
`mlp_h64_64`, `upgd_low_noise`, and `dynamic_sparse` was run on the full
synthetic/digits matrix.

Historical result: the first eta-8 portfolio was stronger but not strict
universality. A follow-up eta-1 version with a causal online class-imbalance
MSE guard closes the earlier eight-regime synthetic/digits benchmark matrix by
the predeclared mean criterion.

- The promoted portfolio beats or ties the best fair MLP width on final-window
  MSE across all eight current regimes.
- It has no negative mean held-out digit accuracy versus the best fair MLP
  width.
- The rows that carried risk were rerun at 30 seeds: compositional remains
  small-positive by mean, frequency remains positive by mean despite balanced
  sign counts, class-blocked online MSE ties the best current-block MLP, and
  non-blocked digits remain positive on final-window MSE and held-out accuracy.
- The result is an operational benchmark closure, not a proof of general
  feature construction: the winning mechanism is still a portfolio over
  existing learners plus hand-specified class-imbalance triggers.

This note is superseded for the broad all-suite claim by the conclusive
telemetry-gated learner in
`outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_results.json`,
which reaches `130/0/10` seed-level final-window MSE wins/losses/ties against
the same-run best fair MLP.

It is also superseded for the stricter non-router learner claim by D18
persistent trace:
`outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_results.json`.
That result reaches `138/2/0` seed-level final-window MSE wins/losses/ties over
14 regimes without an output portfolio or deployment router. The remaining
caveat is mechanism simplicity: D18 is still an additive resource-basis learner
with several fixed basis families, not a clean single recursive feature-growth
principle.

Artifact: `docs/research/step2_universal_portfolio_assessment.md`.

## Ten New Grounded Directions

The ten new specs are intentionally different from the first moonshot batch.

| ID | Direction | Why It Is Different | Why It Might Beat The Current Iteration |
|---|---|---|---|
| D01 | Online Newton/RLS last layer | Changes readout optimizer, not representation or target. | Faster adaptation over correlated MLP features. |
| D02 | Auxiliary self-supervised predictions | Adds predictive-state objectives at every tick. | Keeps useful trunk features alive before supervised relevance appears. |
| D03 | Dynamic sparse connectivity | Reallocates edge budget online under fixed capacity. | Tracks changing feature relevance without hand-coded features. |
| D04 | Adaptive spline basis | Learns local piecewise-linear knots. | Allocates resolution near persistent residual regions. |
| D05 | Homeostatic plasticity regularizers | Smoothly preserves active/sensitive hidden units. | Prevents dormancy without destructive resets. |
| D06 | Online whitening/natural-gradient preconditioning | Changes gradient metric, not weights directly. | Reduces covariance-induced interference. |
| D07 | Budgeted KRLS / ALD kernel dictionary | Main predictor is a novelty-gated kernel dictionary. | Combines online basis creation with second-order coefficient updates. |
| D08 | Head-conditioned trunk modulation | Learns per-head FiLM routing through shared trunk. | Reduces destructive sharing across heads. |
| D09 | Causal history-state features | Adds traces/delay embeddings, no new loss. | Fixes partial observability and latent phase aliasing. |
| D10 | Uncertainty-weighted targets | Learns precision weights for ordinary target errors. | Downweights unreliable drift-boundary or noisy updates. |

## Highest-Priority Next Experiments

1. **D01 RLS last layer**: smallest implementation, strongest conditioning
   argument, and directly compatible with fair MLP features.
2. **D06 online whitening**: the principled generalization of the failed
   orthogonalization idea; precondition updates without overwriting learned
   geometry.
3. **D08 head-conditioned modulation**: targeted at the multi-head interference
   structure that Step 2/3 depend on.
4. **D09 causal history features**: likely essential for partially observable
   and recurring-context streams; should be run with a shuffled-time negative
   control.
5. **D03 dynamic sparse connectivity**: higher implementation cost, but closest
   to a general feature lifecycle mechanism under a fixed budget.

Promotion rule remains unchanged: a candidate must beat the best fair MLP in at
least one non-co-designed external or negative-control setting, while preserving
or improving the interaction-stream result.
