# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `480000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008751 +/- 0.000000 | 0.944800 +/- 0.000000 | 0.044436 +/- 0.000000 | 0.693912 +/- 0.000000 |
| `upgd_structure_softmax_h128` | 0.008328 +/- 0.000000 | 0.948600 +/- 0.000000 | 0.047095 +/- 0.000000 | 0.707275 +/- 0.000000 |
| `step2_hybrid_memory_trace_delight_gate15` | 0.009632 +/- 0.000000 | 0.942600 +/- 0.000000 | 0.036965 +/- 0.000000 | 0.730575 +/- 0.000000 |
| `step2_hybrid_memory_trace_delight_gate30` | 0.008143 +/- 0.000000 | 0.947800 +/- 0.000000 | 0.038052 +/- 0.000000 | 0.720487 +/- 0.000000 |
| `step2_hybrid_memory_trace_delight_gate50` | 0.008468 +/- 0.000000 | 0.944600 +/- 0.000000 | 0.038824 +/- 0.000000 | 0.719325 +/- 0.000000 |
| `upgd_structure_softmax_h128_delight_gate30` | 0.008206 +/- 0.000000 | 0.945000 +/- 0.000000 | 0.039050 +/- 0.000000 | 0.737287 +/- 0.000000 |

## Paper-Informed Rationale

The March 2026 Delightful Policy Gradient paper argues that policy-gradient
updates should depend on both value and policy-relative surprisal, not
advantage alone. Its gate uses delight, `advantage * -log pi(action)`, so rare
successes are promoted and rare failures are suppressed. The Kondo gate paper
turns the same signal into a compute/update filter: pay for a backward pass
only when delight exceeds an adaptive price.

This OPMNIST probe implements the supervised analogue. The true label is the
useful action, `1 - p_true` is the advantage-like term, and `-log(p_true)` is
the surprisal term. This is deliberately not raw novelty: the earlier LeWM
probe showed surprise-only selection was weak, while utility-weighted surprise
was useful. The implementation now uses a conditional update branch, so skipped
examples skip the wrapped learner update semantically instead of computing an
update and masking it afterward.

## Ranking

On the 1% OpenML OPMNIST screen, the best candidates split by objective:

- Online mean MSE: `step2_hybrid_memory_trace_delight_gate50` at `0.012411`.
- Online mean accuracy: `upgd_structure_softmax_h128` at `0.920952`.
- Final-window MSE: `step2_hybrid_memory_trace_delight_gate30` at `0.008143`.
- Final-window accuracy: `upgd_structure_softmax_h128` at `0.948600`.
- Held-out test MSE: `step2_hybrid_memory_trace_delight_gate15` at `0.036965`.
- Held-out test accuracy: `upgd_structure_softmax_h128_delight_gate30` at
  `0.737287`.

The useful signal is that delight gating consistently improves held-out
permutation-view generalization over the ungated primary (`0.693912` test
accuracy) and ungated h128 UPGD (`0.707275` test accuracy). The tradeoff is
online adaptation: stronger gating can improve retention/generalization while
giving up some immediate online accuracy.

## Promotion Decision

Promote two candidates to the full 800-task OPMNIST gate:

- `upgd_structure_softmax_h128_delight_gate30`: best held-out test accuracy in
  the ablation and a direct comparison against the existing full h128 UPGD
  result.
- `step2_hybrid_memory_trace_delight_gate30`: best final-window MSE and strong
  held-out gain without the slower 15% gate's online penalty.

Keep `step2_hybrid_memory_trace_delight_gate50` as the online-MSE fallback if
the full 30% hybrid run loses too much online adaptation.

## References

- Osband, "Delightful Policy Gradient", arXiv:2603.14608,
  https://arxiv.org/abs/2603.14608.
- Osband, "Does This Gradient Spark Joy?", arXiv:2603.20526,
  https://arxiv.org/abs/2603.20526.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
