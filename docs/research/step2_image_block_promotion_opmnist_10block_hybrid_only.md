# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `600000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008136 +/- 0.000000 | 0.950200 +/- 0.000000 | 0.052534 +/- 0.000000 | 0.642940 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.008059 +/- 0.000000 | 0.949400 +/- 0.000000 | 0.054608 +/- 0.000000 | 0.630040 +/- 0.000000 |

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
