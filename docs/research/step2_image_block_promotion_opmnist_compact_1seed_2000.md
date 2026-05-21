# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `2000`
- Seeds: `1`
- Permutations: `5`
- Task block size: `400`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.043549 +/- 0.000000 | 0.717500 +/- 0.000000 | 0.034687 +/- 0.000000 | 0.774600 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.043388 +/- 0.000000 | 0.710000 +/- 0.000000 | 0.038821 +/- 0.000000 | 0.769600 +/- 0.000000 |
| `proto_mem_s20` | 0.047559 +/- 0.000000 | 0.652500 +/- 0.000000 | 0.040776 +/- 0.000000 | 0.718200 +/- 0.000000 |
| `proto_mem_s32` | 0.044059 +/- 0.000000 | 0.682500 +/- 0.000000 | 0.039611 +/- 0.000000 | 0.727000 +/- 0.000000 |
| `mlp_h64` | 0.070756 +/- 0.000000 | 0.537500 +/- 0.000000 | 0.077155 +/- 0.000000 | 0.378400 +/- 0.000000 |
| `mlp_h128` | 0.075977 +/- 0.000000 | 0.547500 +/- 0.000000 | 0.083658 +/- 0.000000 | 0.360800 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.030489 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.182000 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64`: +0.027207 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.170000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.042468 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.396200 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h64`: +0.030344 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `online_mean_mse` vs `mlp_h64`: +0.029842 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `online_mean_mse` vs `mlp_h64`: +0.031001 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h64`: +0.180000 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `online_mean_accuracy` vs `mlp_h64`: +0.154500 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `online_mean_accuracy` vs `mlp_h64`: +0.164000 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h64`: +0.027368 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `final_window_mse` vs `mlp_h64`: +0.023197 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `final_window_mse` vs `mlp_h64`: +0.026697 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128`: +0.162500 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `final_window_accuracy` vs `mlp_h128`: +0.105000 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `final_window_accuracy` vs `mlp_h128`: +0.135000 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h64`: +0.038334 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `test_mse` vs `mlp_h64`: +0.036379 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `test_mse` vs `mlp_h64`: +0.037545 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h64`: +0.391200 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `test_accuracy` vs `mlp_h64`: +0.339800 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `test_accuracy` vs `mlp_h64`: +0.348600 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
