# Delight and Kondo Gating for the Alberta Path

This note records the Step 2/3/4 implications of two March 2026 Osband papers:

- "Delightful Policy Gradient", arXiv:2603.14608.
- "Does This Gradient Spark Joy?", arXiv:2603.20526.

## Core Read

The key signal is not surprise by itself. It is useful surprise:
`delight = advantage * surprisal`, where surprisal is policy-relative
`-log pi(action)`. Delightful Policy Gradient uses this as a multiplicative
policy-gradient gate. The Kondo gate uses the same forward-pass signal to decide
whether an example deserves a backward pass at all.

For this framework, the important lesson is that raw novelty, prediction error,
or world-model surprise should not directly own the update budget. They should
be crossed with value, learning progress, or task utility. That matches the
local LeWM probe: surprise-only selection was weak, while utility-weighted
selection was useful.

## Direct Applications

1. Supervised OPMNIST: treat the true label as the useful action. Use
   `advantage = 1 - p_true`, `surprisal = -log(p_true)`, and gate updates by an
   adaptive compute price. This is now implemented as `DelightGatedLearner` in
   the OPMNIST runner.
2. Actor-critic control: gate actor updates by `A(s,a) * -log pi(a|s)`. This is
   the closest literal transfer from DG to Step 4.
3. Horde/world-model learning: gate auxiliary prediction-demon updates by
   `abs(TD error) * usefulness`, where usefulness can be downstream control
   sensitivity, GVF relevance, or retained-view improvement. This avoids the
   raw-surprise trap.
4. Dreaming/self-simulation: prioritize imagined rollouts whose model surprise
   is also policy- or value-relevant. The world model should propose unfamiliar
   futures; the agent learner should decide which imagined futures are worth
   learning from.

## Current Evidence

The 1% OpenML OPMNIST screen promoted delight gating:

- `step2_hybrid_memory_trace_delight_gate30` won final-window MSE in the focused
  ablation at `0.008143`.
- `upgd_structure_softmax_h128_delight_gate30` won held-out test accuracy at
  `0.737287`.
- All hybrid delight gates improved held-out accuracy over the ungated primary
  (`0.693912`) and ungated h128 UPGD (`0.707275`) in the same 1% protocol.

The tradeoff is objective-specific. Delight gating tends to improve
retention/generalization across held-out permutation views, while the ungated
h128 UPGD path still has the best 1% online accuracy.

## End-to-End Path

The practical path is progressive elimination, not an exhaustive Cartesian grid:

1. Keep the current primary hybrid, h128 UPGD, h128 delight gate, and hybrid
   delight gate as the Step 2 image-memory frontier.
2. Use 1% OPMNIST to prune rate/width choices.
3. Promote only the best one or two candidates to full 800-task OPMNIST.
4. If full OPMNIST confirms a delight-gated retention win, move the same
   signal into Step 4 actor updates and Step 3/4 dream replay selection.
5. For world-model experiments, compare four selectors: random replay,
   surprise-only replay, utility-only replay, and delight replay. The expected
   winner should be delight replay if the Osband mechanism transfers.

## Full-Scale Gate Result

The full 800-task run promoted two candidates:

- `upgd_structure_softmax_h128_delight_gate30`.
- `step2_hybrid_memory_trace_delight_gate30`.

The comparison bars are the existing canonical full OPMNIST runs:

- `upgd_structure_softmax_h128`: test accuracy `0.134712`.
- `mlp_h128_sharp`: test accuracy `0.139329`.
- `step2_hybrid_memory_trace`: online mean MSE `0.018542`, online mean accuracy
  `0.876835`, and final-window MSE `0.015042`.

The result is mixed rather than a replacement:

- `upgd_structure_softmax_h128_delight_gate30`: final-window MSE `0.013570`,
  final-window accuracy `0.909600`, test MSE `0.132588`, test accuracy
  `0.127438`, online MSE `0.019791`, online accuracy `0.863964`.
- `step2_hybrid_memory_trace_delight_gate30`: final-window MSE `0.016366`,
  final-window accuracy `0.895400`, test MSE `0.118151`, test accuracy
  `0.124635`, online MSE `0.023454`, online accuracy `0.852887`.

The h128 delight gate improved full h128 final-window MSE, final-window
accuracy, and test MSE, but lost test accuracy and online adaptation. The
hybrid delight gate improved primary held-out test MSE and barely improved test
accuracy, but lost the primary's online/final adaptation. This means the
current online hard-skip gate is not the full OPMNIST winner.

The next useful transfer is not "skip 70% of all online updates." It is:

1. Use delight as a soft weight for online supervised/control updates.
2. Use hard Kondo gating for auxiliary/dream/model updates where compute
   selection is the point and online adaptation is protected.
3. In Step 4, gate actor updates by delightful advantage, while leaving critic
   prediction demons temporally uniform unless their updates are auxiliary.
