# Step 2 UPGD-Memory OPMNIST

Superseded on 2026-05-09: this note was generated before the RLS deployment
gate was made fully prequential. The RLS calibrator weights were causal, but
the current-step gate could still be turned on by the current label. Use the
`*_honest.md` rerun after the gate-causality fix for promotion decisions.

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
| `step2_hybrid_memory_trace_rls_cal` | 0.024394 +/- 0.001218 | 0.841600 +/- 0.009092 | 0.082959 +/- 0.010625 | 0.375720 +/- 0.079730 |
| `mlp_h64` | 0.055425 +/- 0.000647 | 0.686400 +/- 0.009837 | 0.072815 +/- 0.001330 | 0.518040 +/- 0.009966 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.034141 +/- 0.000596; wins/losses/ties 5/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.162680 +/- 0.005891; wins/losses/ties 5/0/0.
- `final_window_mse` vs `mlp_h64`: +0.027146 +/- 0.000483; wins/losses/ties 5/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.116400 +/- 0.008328; wins/losses/ties 5/0/0.
- `test_mse` vs `mlp_h64`: +0.055514 +/- 0.001566; wins/losses/ties 5/0/0.
- `test_accuracy` vs `mlp_h64`: +0.368000 +/- 0.010232; wins/losses/ties 5/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.036071 +/- 0.000629; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.171560 +/- 0.006347; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.031030 +/- 0.001073; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.155200 +/- 0.011460; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: -0.010144 +/- 0.011474; wins/losses/ties 1/4/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: -0.142320 +/- 0.084055; wins/losses/ties 1/4/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
