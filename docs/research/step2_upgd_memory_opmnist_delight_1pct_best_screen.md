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
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.008814 +/- 0.000000 | 0.944000 +/- 0.000000 | 0.045496 +/- 0.000000 | 0.684150 +/- 0.000000 |
| `mlp_h64` | 0.017483 +/- 0.000000 | 0.933400 +/- 0.000000 | 0.078316 +/- 0.000000 | 0.407112 +/- 0.000000 |
| `mlp_h128` | 0.016710 +/- 0.000000 | 0.935600 +/- 0.000000 | 0.087928 +/- 0.000000 | 0.383437 +/- 0.000000 |
| `upgd_structure_softmax_h128` | 0.008051 +/- 0.000000 | 0.949200 +/- 0.000000 | 0.048118 +/- 0.000000 | 0.699537 +/- 0.000000 |
| `step2_hybrid_memory_trace_delight_gate30` | 0.009107 +/- 0.000000 | 0.940600 +/- 0.000000 | 0.035264 +/- 0.000000 | 0.743362 +/- 0.000000 |
| `upgd_structure_softmax_h64_delight_gate30` | 0.008091 +/- 0.000000 | 0.945400 +/- 0.000000 | 0.048200 +/- 0.000000 | 0.686550 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128`: +0.010281 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.013315 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128`: +0.007959 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.009200 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.033880 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64`: +0.286800 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h128`: +0.010373 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `online_mean_mse` vs `mlp_h128`: +0.010359 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `online_mean_mse` vs `mlp_h128`: +0.009105 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `online_mean_mse` vs `mlp_h128`: +0.009452 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128`: +0.013904 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `online_mean_accuracy` vs `mlp_h128`: +0.015435 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `online_mean_accuracy` vs `mlp_h128`: +0.010702 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `online_mean_accuracy` vs `mlp_h128`: +0.008188 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h128`: +0.007896 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `final_window_mse` vs `mlp_h128`: +0.008659 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `final_window_mse` vs `mlp_h128`: +0.007603 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `final_window_mse` vs `mlp_h128`: +0.008619 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128`: +0.008400 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `final_window_accuracy` vs `mlp_h128`: +0.013600 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `final_window_accuracy` vs `mlp_h128`: +0.005000 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `final_window_accuracy` vs `mlp_h128`: +0.009800 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h64`: +0.032821 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `test_mse` vs `mlp_h64`: +0.030199 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `test_mse` vs `mlp_h64`: +0.043053 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `test_mse` vs `mlp_h64`: +0.030116 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h64`: +0.277037 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128` `test_accuracy` vs `mlp_h64`: +0.292425 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_delight_gate30` `test_accuracy` vs `mlp_h64`: +0.336250 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_delight_gate30` `test_accuracy` vs `mlp_h64`: +0.279437 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
