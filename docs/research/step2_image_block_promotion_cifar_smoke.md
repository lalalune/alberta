# Step 2 CIFAR Stream Benchmark

This note records the external CIFAR-style supervised stream probe for the current packaged single UPGD-memory trace learner.

Positive paired differences favor `step2_hybrid_memory_trace` in the primary comparison. For MSE, the difference is best MLP minus candidate; for accuracy, it is candidate minus best MLP.

## Dataset

- Source: `synthetic_cifar_smoke`
- Real CIFAR-10 evidence: `False`
- Train shape: `[40, 32]`
- Test shape: `[20, 32]`

This is not real CIFAR-10 evidence. The runner used the deterministic synthetic smoke fallback because local CIFAR-10 was unavailable without optional dependency/data setup.

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.044760 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.048015 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `upgd_step2_default` | 0.099289 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.082151 +/- 0.000000 | 0.600000 +/- 0.000000 |
| `mlp_h32` | 0.083744 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.070160 +/- 0.000000 | 0.750000 +/- 0.000000 |
| `mlp_h64` | 0.099976 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.068561 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `mlp_h128` | 0.291347 +/- 0.000000 | 0.250000 +/- 0.000000 | 0.225258 +/- 0.000000 | 0.400000 +/- 0.000000 |
| `mlp_h256` | 0.369375 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.376715 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `mlp_h128_128` | 0.259205 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.226099 +/- 0.000000 | 0.350000 +/- 0.000000 |
| `proto_mem_s20` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.055284 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `proto_mem_s32` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.055284 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `step2_hybrid_memory_trace_sharp` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.060000 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.044760 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.048015 +/- 0.000000 | 0.700000 +/- 0.000000 |
| `mlp_h32_sharp` | 0.062743 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.056465 +/- 0.000000 | 0.750000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.074155 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.052569 +/- 0.000000 | 0.700000 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32_sharp`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.017982 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` primary-vs-best-MLP diff: -0.050000 +/- 0.000000; wins/losses/ties 0/1/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: -0.036547 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: -0.150000 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s20` final-window MSE diff vs best MLP: +0.012743 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` held-out accuracy diff vs best MLP: -0.050000 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` final-window MSE diff vs best MLP: +0.012743 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s32` held-out accuracy diff vs best MLP: -0.050000 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: +0.012743 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: -0.050000 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.017982 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: -0.050000 +/- 0.000000; wins/losses/ties 0/1/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.043882 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.119311 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `upgd_step2_default` | 0.099153 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.121522 +/- 0.000000 | 0.600000 +/- 0.000000 |
| `mlp_h32` | 0.074716 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.116874 +/- 0.000000 | 0.550000 +/- 0.000000 |
| `mlp_h64` | 0.053125 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.114197 +/- 0.000000 | 0.450000 +/- 0.000000 |
| `mlp_h128` | 0.117698 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.152527 +/- 0.000000 | 0.200000 +/- 0.000000 |
| `mlp_h256` | 0.413774 +/- 0.000000 | 0.500000 +/- 0.000000 | 0.352216 +/- 0.000000 | 0.400000 +/- 0.000000 |
| `mlp_h128_128` | 0.113467 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.208942 +/- 0.000000 | 0.250000 +/- 0.000000 |
| `proto_mem_s20` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.138724 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `proto_mem_s32` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.138724 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `step2_hybrid_memory_trace_sharp` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.140000 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.043882 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.119311 +/- 0.000000 | 0.300000 +/- 0.000000 |
| `mlp_h32_sharp` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.102428 +/- 0.000000 | 0.550000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.050000 +/- 0.000000 | 0.750000 +/- 0.000000 | 0.107055 +/- 0.000000 | 0.450000 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32_sharp`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.006118 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` primary-vs-best-MLP diff: -0.250000 +/- 0.000000; wins/losses/ties 0/1/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: -0.049153 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: +0.050000 +/- 0.000000; wins/losses/ties 1/0/0.
- `proto_mem_s20` final-window MSE diff vs best MLP: +0.000000 +/- 0.000000; wins/losses/ties 0/0/1.
- `proto_mem_s20` held-out accuracy diff vs best MLP: -0.250000 +/- 0.000000; wins/losses/ties 0/1/0.
- `proto_mem_s32` final-window MSE diff vs best MLP: +0.000000 +/- 0.000000; wins/losses/ties 0/0/1.
- `proto_mem_s32` held-out accuracy diff vs best MLP: -0.250000 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: +0.000000 +/- 0.000000; wins/losses/ties 0/0/1.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: -0.250000 +/- 0.000000; wins/losses/ties 0/1/0.
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.006118 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: -0.250000 +/- 0.000000; wins/losses/ties 0/1/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional dependency behavior, and metric accounting. A stronger CIFAR claim requires real CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no test-set tuning.
