# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_8x8`
- Steps: `12`
- Seeds: `1`
- Permutations: `2`
- Task block size: `6`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.115570 +/- 0.000000 | 0.250000 +/- 0.000000 | 0.100048 +/- 0.000000 | 0.325000 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.113799 +/- 0.000000 | 0.250000 +/- 0.000000 | 0.099784 +/- 0.000000 | 0.350000 +/- 0.000000 |
| `mlp_h64` | 0.191222 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.154497 +/- 0.000000 | 0.150000 +/- 0.000000 |
| `mlp_h128` | 0.535772 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.317704 +/- 0.000000 | 0.200000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.200000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.167602 +/- 0.000000 | 0.100000 +/- 0.000000 |
| `mlp_h128_sharp` | 0.200000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.174604 +/- 0.000000 | 0.175000 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64_sharp`: +0.075769 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.166667 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.075653 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.250000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.054450 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h128`: +0.125000 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h64_sharp`: +0.076385 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128`: +0.166667 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h64`: +0.077423 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h64`: +0.250000 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h64`: +0.054714 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h128`: +0.150000 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
