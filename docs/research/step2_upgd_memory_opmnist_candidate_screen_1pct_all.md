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
| `step2_hybrid_memory_trace_sharp` | 0.010936 +/- 0.000000 | 0.944000 +/- 0.000000 | 0.060282 +/- 0.000000 | 0.684150 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.008735 +/- 0.000000 | 0.944200 +/- 0.000000 | 0.045235 +/- 0.000000 | 0.690275 +/- 0.000000 |
| `mlp_h64` | 0.017376 +/- 0.000000 | 0.931800 +/- 0.000000 | 0.072024 +/- 0.000000 | 0.424325 +/- 0.000000 |
| `mlp_h128` | 0.016759 +/- 0.000000 | 0.940200 +/- 0.000000 | 0.084707 +/- 0.000000 | 0.394175 +/- 0.000000 |
| `mlp_h64_sharp` | 0.011841 +/- 0.000000 | 0.933400 +/- 0.000000 | 0.093491 +/- 0.000000 | 0.427975 +/- 0.000000 |
| `mlp_h128_sharp` | 0.010863 +/- 0.000000 | 0.936200 +/- 0.000000 | 0.107797 +/- 0.000000 | 0.381212 +/- 0.000000 |
| `centroid_hysteretic64_center_c030` | 0.010668 +/- 0.000000 | 0.939800 +/- 0.000000 | 0.090787 +/- 0.000000 | 0.363587 +/- 0.000000 |
| `proto_mem_s20` | 0.074303 +/- 0.000000 | 0.461800 +/- 0.000000 | 0.075300 +/- 0.000000 | 0.473475 +/- 0.000000 |
| `proto_mem_s32` | 0.053823 +/- 0.000000 | 0.603200 +/- 0.000000 | 0.057363 +/- 0.000000 | 0.590837 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128_sharp`: +0.003521 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.011950 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128_sharp`: +0.002112 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.004600 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.027587 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64_sharp`: +0.265937 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_sharp` `online_mean_mse` vs `mlp_h128_sharp`: +0.000718 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h128_sharp`: +0.003623 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `online_mean_mse` vs `mlp_h128_sharp`: +0.001428 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `online_mean_mse` vs `mlp_h128_sharp`: -0.044175 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` `online_mean_mse` vs `mlp_h128_sharp`: -0.033634 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` `online_mean_accuracy` vs `mlp_h128`: +0.012540 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128`: +0.012473 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `online_mean_accuracy` vs `mlp_h128`: +0.008515 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `online_mean_accuracy` vs `mlp_h128`: -0.342208 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` `online_mean_accuracy` vs `mlp_h128`: -0.259423 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` `final_window_mse` vs `mlp_h128_sharp`: -0.000074 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h128_sharp`: +0.002128 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `final_window_mse` vs `mlp_h128_sharp`: +0.000195 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` `final_window_mse` vs `mlp_h128_sharp`: -0.063440 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` `final_window_mse` vs `mlp_h128_sharp`: -0.042961 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` `final_window_accuracy` vs `mlp_h128`: +0.003800 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128`: +0.004000 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `final_window_accuracy` vs `mlp_h128`: -0.000400 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s20` `final_window_accuracy` vs `mlp_h128`: -0.478400 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` `final_window_accuracy` vs `mlp_h128`: -0.337000 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` `test_mse` vs `mlp_h64`: +0.011741 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h64`: +0.026789 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `test_mse` vs `mlp_h64`: -0.018763 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s20` `test_mse` vs `mlp_h64`: -0.003277 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` `test_mse` vs `mlp_h64`: +0.014660 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `test_accuracy` vs `mlp_h64_sharp`: +0.256175 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h64_sharp`: +0.262300 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `test_accuracy` vs `mlp_h64_sharp`: -0.064387 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s20` `test_accuracy` vs `mlp_h64_sharp`: +0.045500 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` `test_accuracy` vs `mlp_h64_sharp`: +0.162862 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
