# Step 2 UPGD-Memory OPMNIST

Superseded on 2026-05-09: this rerun fixed post-target RLS weight scoring, but
the deployment gate could still turn on using the current label before returning
the current prediction. Use `step2_rls_calibrated_opmnist_28x28_5seed_5000_prequential.md`
for the fully causal result.

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `5000`
- Seeds: `5`
- Permutations: `10`
- Task block size: `500`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.028279 +/- 0.000706 | 0.802800 +/- 0.004477 | 0.017302 +/- 0.000469 | 0.886040 +/- 0.002855 |
| `step2_hybrid_memory_trace_rls_cal` | 0.028038 +/- 0.000689 | 0.805800 +/- 0.005911 | 0.017360 +/- 0.000411 | 0.886080 +/- 0.003530 |
| `mlp_h64` | 0.055425 +/- 0.000647 | 0.686400 +/- 0.009837 | 0.072815 +/- 0.001330 | 0.518040 +/- 0.009966 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.034141 +/- 0.000596; wins/losses/ties 5/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.162680 +/- 0.005891; wins/losses/ties 5/0/0.
- `final_window_mse` vs `mlp_h64`: +0.027146 +/- 0.000483; wins/losses/ties 5/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.116400 +/- 0.008328; wins/losses/ties 5/0/0.
- `test_mse` vs `mlp_h64`: +0.055514 +/- 0.001566; wins/losses/ties 5/0/0.
- `test_accuracy` vs `mlp_h64`: +0.368000 +/- 0.010232; wins/losses/ties 5/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.034433 +/- 0.000588; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.162600 +/- 0.006194; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.027387 +/- 0.000667; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.119400 +/- 0.010048; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: +0.055456 +/- 0.001531; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: +0.368040 +/- 0.010631; wins/losses/ties 5/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 Honest RLS Calibration Assessment

This run uses the corrected prequential RLS calibrator: predictions are scored
from the calibrator state before the current target is fitted, then the RLS
state is updated. An earlier implementation returned the post-target fitted
prediction and therefore overstated online/final-window gains.

Paired `step2_hybrid_memory_trace_rls_cal` minus
`step2_hybrid_memory_trace` deltas, where positive favors RLS calibration:

- Online mean MSE: `+0.000292`, `5/5` wins.
- Online mean accuracy: `-0.000080`, `2/5` wins.
- Final-window MSE: `+0.000241`, `4/5` wins.
- Final-window accuracy: `+0.003000`, `3/5` wins.
- Held-out test MSE: `-0.000058`, `2/5` wins.
- Held-out test accuracy: `+0.000040`, `3/5` wins.

Interpretation: prediction-space RLS calibration is safe under honest online
scoring and gives a small MSE tracking benefit, but the gain is not yet large
enough to justify making it the default. The prior large online gain was a
scoring bug, not a real mechanism. The current candidate should remain behind
`--include-rls-calibrated` until it shows a larger compute-adjusted advantage
on longer OPMNIST fractions or additional online digit regimes.
