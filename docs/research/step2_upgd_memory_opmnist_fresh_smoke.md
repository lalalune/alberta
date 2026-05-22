# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `600`
- Seeds: `1`
- Permutations: `3`
- Task block size: `200`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.030636 +/- 0.000000 | 0.815000 +/- 0.000000 | 0.011654 +/- 0.000000 | 0.935000 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.030507 +/- 0.000000 | 0.810000 +/- 0.000000 | 0.011632 +/- 0.000000 | 0.931667 +/- 0.000000 |
| `mlp_h64` | 0.082044 +/- 0.000000 | 0.445000 +/- 0.000000 | 0.072160 +/- 0.000000 | 0.510000 +/- 0.000000 |
| `mlp_h128` | 0.086000 +/- 0.000000 | 0.530000 +/- 0.000000 | 0.076726 +/- 0.000000 | 0.480000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.085990 +/- 0.000000 | 0.495000 +/- 0.000000 | 0.071339 +/- 0.000000 | 0.525000 +/- 0.000000 |
| `mlp_h128_sharp` | 0.082192 +/- 0.000000 | 0.545000 +/- 0.000000 | 0.066680 +/- 0.000000 | 0.598333 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.058323 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128_sharp`: +0.321667 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.051408 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128_sharp`: +0.270000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h128_sharp`: +0.055026 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h128_sharp`: +0.336667 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h64`: +0.058376 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128_sharp`: +0.316667 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h64`: +0.051536 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128_sharp`: +0.265000 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h128_sharp`: +0.055048 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h128_sharp`: +0.333333 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
