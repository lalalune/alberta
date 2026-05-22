# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_8x8`
- Steps: `600`
- Seeds: `3`
- Permutations: `5`
- Task block size: `120`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.035045 +/- 0.001669 | 0.756667 +/- 0.012019 | 0.019039 +/- 0.000663 | 0.888333 +/- 0.008838 |
| `step2_hybrid_memory_trace_rls_cal` | 0.032647 +/- 0.001346 | 0.760000 +/- 0.013229 | 0.019940 +/- 0.000319 | 0.893000 +/- 0.009539 |
| `mlp_h64` | 0.079755 +/- 0.001068 | 0.435000 +/- 0.020207 | 0.084849 +/- 0.001812 | 0.348667 +/- 0.027205 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.050339 +/- 0.000629; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.353333 +/- 0.001925; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.044710 +/- 0.000661; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.321667 +/- 0.010138; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.065809 +/- 0.002173; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.539667 +/- 0.032405; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.053232 +/- 0.000727; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.356667 +/- 0.003469; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.047107 +/- 0.000344; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.325000 +/- 0.010408; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: +0.064909 +/- 0.001726; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: +0.544333 +/- 0.033278; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
