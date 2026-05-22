# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `48000000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`
- Combined result JSON:
  `outputs/step2_upgd_memory_opmnist_latest_best_split/combined_latest_best_800block_1seed_results.json`
- Canonical copy:
  `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`

## Command Provenance

This result was run as four method-filtered resumable splits from a common
latest-best checkpoint family, then merged by method name:

- `step2_hybrid_memory_trace`
- `step2_hybrid_memory_trace_adaptive_sharp`
- `mlp_h64`, `mlp_h128`
- `mlp_h64_sharp`, `mlp_h128_sharp`

Each split used true OpenML MNIST, the canonical 60,000/10,000 split, 800
random pixel permutations, 60,000-example task blocks, no task id, prediction
before update, and 48,000,000 online updates. The split outputs are:

- `outputs/step2_upgd_memory_opmnist_latest_best_split/primary_raw/primary_raw_800block_1seed_results.json`
- `outputs/step2_upgd_memory_opmnist_latest_best_split/primary_adaptive/primary_adaptive_800block_1seed_results.json`
- `outputs/step2_upgd_memory_opmnist_latest_best_split/mlp_raw/mlp_raw_800block_1seed_results.json`
- `outputs/step2_upgd_memory_opmnist_latest_best_split/mlp_sharp/mlp_sharp_800block_1seed_results.json`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.015042 +/- 0.000000 | 0.904000 +/- 0.000000 | 0.136308 +/- 0.000000 | 0.123512 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.014981 +/- 0.000000 | 0.902200 +/- 0.000000 | 0.138332 +/- 0.000000 | 0.121489 +/- 0.000000 |
| `mlp_h64` | 0.024625 +/- 0.000000 | 0.894000 +/- 0.000000 | 0.101397 +/- 0.000000 | 0.118140 +/- 0.000000 |
| `mlp_h128` | 0.021526 +/- 0.000000 | 0.906400 +/- 0.000000 | 0.098814 +/- 0.000000 | 0.130719 +/- 0.000000 |
| `mlp_h64_sharp` | 0.017096 +/- 0.000000 | 0.896800 +/- 0.000000 | 0.128641 +/- 0.000000 | 0.123515 +/- 0.000000 |
| `mlp_h128_sharp` | 0.017444 +/- 0.000000 | 0.897600 +/- 0.000000 | 0.127261 +/- 0.000000 | 0.139329 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128_sharp`: +0.003742 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128_sharp`: +0.011959 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h64_sharp`: +0.002054 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: -0.002400 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_mse` vs `mlp_h128`: -0.037494 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_accuracy` vs `mlp_h128_sharp`: -0.015816 +/- 0.000000; wins/losses/ties 0/1/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h128_sharp`: +0.003782 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128_sharp`: +0.012320 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h64_sharp`: +0.002115 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128`: -0.004200 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h128`: -0.039517 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h128_sharp`: -0.017840 +/- 0.000000; wins/losses/ties 0/1/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

This run satisfies the 800-task / 48,000,000-update scale gate for one seed.
It does not establish an unqualified OPMNIST win: UPGD-memory wins online MSE,
online accuracy, and final-window MSE, while fair MLP baselines still win
final-window accuracy and all-permutation held-out test MSE/accuracy.
