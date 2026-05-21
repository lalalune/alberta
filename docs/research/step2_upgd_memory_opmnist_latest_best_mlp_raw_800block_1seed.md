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
| `mlp_h64` | 0.024625 +/- 0.000000 | 0.894000 +/- 0.000000 | 0.101397 +/- 0.000000 | 0.118140 +/- 0.000000 |
| `mlp_h128` | 0.021526 +/- 0.000000 | 0.906400 +/- 0.000000 | 0.098814 +/- 0.000000 | 0.130719 +/- 0.000000 |

## Primary vs Best MLP


## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
