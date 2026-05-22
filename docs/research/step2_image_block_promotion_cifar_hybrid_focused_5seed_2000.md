# Step 2 CIFAR Stream Benchmark

This note records the external CIFAR-style supervised stream probe for the current packaged single UPGD-memory trace learner.

Positive paired differences favor `step2_hybrid_memory_trace` in the primary comparison. For MSE, the difference is best MLP minus candidate; for accuracy, it is candidate minus best MLP.

## Dataset

- Source: `cifar10_python_archive`
- Real CIFAR-10 evidence: `True`
- Train shape: `[3000, 3072]`
- Test shape: `[1000, 3072]`

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.077762 +/- 0.000852 | 0.355500 +/- 0.011816 | 0.085165 +/- 0.000760 | 0.256600 +/- 0.010889 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.077762 +/- 0.000852 | 0.355500 +/- 0.011816 | 0.085165 +/- 0.000760 | 0.256600 +/- 0.010889 |
| `step2_hybrid_memory_trace_sharp` | 0.117979 +/- 0.002202 | 0.355500 +/- 0.011816 | 0.135004 +/- 0.002283 | 0.256600 +/- 0.010889 |
| `mlp_h32` | 0.105412 +/- 0.000746 | 0.222500 +/- 0.008367 | 0.107502 +/- 0.002087 | 0.201600 +/- 0.008232 |
| `mlp_h64` | 0.154415 +/- 0.001440 | 0.166000 +/- 0.011281 | 0.156955 +/- 0.005368 | 0.174200 +/- 0.009630 |
| `mlp_h128` | 0.630729 +/- 0.010331 | 0.127000 +/- 0.005327 | 0.634277 +/- 0.024392 | 0.128600 +/- 0.003669 |
| `mlp_h256` | 0.985080 +/- 0.011916 | 0.136000 +/- 0.007483 | 1.042398 +/- 0.072068 | 0.123400 +/- 0.010053 |
| `mlp_h128_128` | 0.244049 +/- 0.005044 | 0.156000 +/- 0.005948 | 0.253891 +/- 0.018199 | 0.141400 +/- 0.007366 |
| `mlp_h32_sharp` | 0.131599 +/- 0.000843 | 0.222500 +/- 0.008367 | 0.136012 +/- 0.002907 | 0.201600 +/- 0.008232 |
| `mlp_h64_sharp` | 0.157988 +/- 0.001724 | 0.166000 +/- 0.011281 | 0.157371 +/- 0.002179 | 0.174200 +/- 0.009630 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.027650 +/- 0.000424; wins/losses/ties 5/0/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.055000 +/- 0.007759; wins/losses/ties 5/0/0.

Additional candidate comparisons:

- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.027650 +/- 0.000424; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.055000 +/- 0.007759; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: -0.012567 +/- 0.001629; wins/losses/ties 0/5/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.055000 +/- 0.007759; wins/losses/ties 5/0/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.004434 +/- 0.000073 | 0.997500 +/- 0.000000 | 0.112614 +/- 0.001637 | 0.188800 +/- 0.008108 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.156817 +/- 0.001606 | 0.188800 +/- 0.008108 |
| `step2_hybrid_memory_trace_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.156817 +/- 0.001606 | 0.188800 +/- 0.008108 |
| `mlp_h32` | 0.001893 +/- 0.000020 | 0.990500 +/- 0.000935 | 0.179328 +/- 0.001259 | 0.095000 +/- 0.000000 |
| `mlp_h64` | 0.001632 +/- 0.000031 | 0.993500 +/- 0.000612 | 0.178109 +/- 0.001079 | 0.095000 +/- 0.000000 |
| `mlp_h128` | 0.021809 +/- 0.001749 | 0.992000 +/- 0.001658 | 0.194117 +/- 0.005824 | 0.095000 +/- 0.000000 |
| `mlp_h256` | 0.711020 +/- 0.009097 | 0.389500 +/- 0.010012 | 0.938204 +/- 0.085186 | 0.110200 +/- 0.008714 |
| `mlp_h128_128` | 0.002340 +/- 0.000178 | 0.997000 +/- 0.000500 | 0.180894 +/- 0.002592 | 0.095000 +/- 0.000000 |
| `mlp_h32_sharp` | 0.001822 +/- 0.000171 | 0.990500 +/- 0.000935 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.001300 +/- 0.000122 | 0.993500 +/- 0.000612 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h64_sharp`
- Best MLP for held-out accuracy: `mlp_h256`
- `final_window_mse` primary-vs-best-MLP diff: -0.003134 +/- 0.000104; wins/losses/ties 0/5/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.078600 +/- 0.015332; wins/losses/ties 5/0/0.

Additional candidate comparisons:

- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.000800 +/- 0.000122; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.078600 +/- 0.015332; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: +0.000800 +/- 0.000122; wins/losses/ties 5/0/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.078600 +/- 0.015332; wins/losses/ties 5/0/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional dependency behavior, and metric accounting. A stronger CIFAR claim requires real CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no test-set tuning.
