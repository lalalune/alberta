# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `48000000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace_delight_gate30` | 0.016366 +/- 0.000000 | 0.895400 +/- 0.000000 | 0.118151 +/- 0.000000 | 0.124635 +/- 0.000000 |
| `upgd_structure_softmax_h128_delight_gate30` | 0.013570 +/- 0.000000 | 0.909600 +/- 0.000000 | 0.132588 +/- 0.000000 | 0.127438 +/- 0.000000 |

## Full-Scale Comparison

This run satisfies the published-scale OPMNIST gate for one seed: true OpenML
MNIST, 800 random pixel permutations, 60,000-example task blocks, no task id,
and 48,000,000 online updates.

Against the existing full h128 UPGD result, the h128 delight gate improved
final-window MSE (`0.013570` vs `0.014868`), final-window accuracy (`0.909600`
vs `0.908600`), and held-out test MSE (`0.132588` vs `0.148280`). It did not
improve held-out test accuracy (`0.127438` vs `0.134712`) or online adaptation
(`0.019791` online MSE vs `0.018753`; `0.863964` online accuracy vs
`0.883623`).

Against the existing full primary hybrid, the hybrid delight gate improved
held-out test MSE (`0.118151` vs `0.136308`) and barely improved test accuracy
(`0.124635` vs `0.123512`), but it lost online and final-window adaptation
(`0.023454` online MSE vs `0.018542`; `0.852887` online accuracy vs
`0.876835`; `0.016366` final-window MSE vs `0.015042`).

The 1% promotion result therefore did not fully transfer to the 800-task
accuracy objective. Delight gating is useful as a retention/MSE regularizer,
but the current gate is too blunt for online OPMNIST: it spends less update
budget and loses adaptation faster than the ungated primary/h128 paths.

## Decision

Do not replace the current full-scale Step 2 OPMNIST frontier with the 30%
delight gates. Keep the h128 delight gate as an interesting final-MSE/test-MSE
variant, but the deployed path remains:

- Online/final adaptation: `upgd_structure_softmax_h128` and
  `step2_hybrid_memory_trace`.
- Held-out full OPMNIST accuracy: existing MLP/MLP-sharp baselines still hold
  the highest one-seed full test accuracy.
- Next delight work: use soft weighting or replay prioritization rather than
  hard online update skipping, and gate only auxiliary/dream updates until the
  online adaptation loss is controlled.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
