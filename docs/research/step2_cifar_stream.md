# Step 2 CIFAR Stream Benchmark

This note records the external CIFAR-style supervised stream probe for the
current promoted non-router target-structure UPGD learner.

Positive paired differences favor `upgd_step2_default`. For MSE, the
difference is best MLP minus UPGD; for accuracy, it is UPGD minus best MLP.

## Dataset

- Source: `synthetic_cifar_smoke`
- Real CIFAR-10 evidence: `False`
- Train shape: `[60, 32]`
- Test shape: `[30, 32]`

This is not real CIFAR-10 evidence. The runner used the deterministic
synthetic smoke fallback because local CIFAR-10 was unavailable without
optional dependency/data setup.

The runner now has two real CIFAR-10 paths: optional `torchvision`, or direct
standard-library loading from the public CIFAR-10 Python archive when
`--allow-download` is set. A separate real-CIFAR smoke run has been recorded in
`docs/research/step2_cifar_stream_real_smoke.md`; it validates the data path
but is still one seed and only 200 online steps.

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `upgd_step2_default` | 0.110726 +/- 0.000000 | 0.200000 +/- 0.000000 | 0.106857 +/- 0.000000 | 0.533333 +/- 0.000000 |
| `mlp_h32` | 0.094877 +/- 0.000000 | 0.600000 +/- 0.000000 | 0.096411 +/- 0.000000 | 0.666667 +/- 0.000000 |
| `mlp_h64` | 0.059306 +/- 0.000000 | 0.800000 +/- 0.000000 | 0.082136 +/- 0.000000 | 0.666667 +/- 0.000000 |

- Best MLP for final-window MSE: `mlp_h64`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` UPGD-vs-best-MLP diff: -0.051420 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_accuracy` UPGD-vs-best-MLP diff: -0.133333 +/- 0.000000; wins/losses/ties 0/1/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `upgd_step2_default` | 0.084511 +/- 0.000000 | 0.600000 +/- 0.000000 | 0.161759 +/- 0.000000 | 0.333333 +/- 0.000000 |
| `mlp_h32` | 0.074786 +/- 0.000000 | 0.600000 +/- 0.000000 | 0.152980 +/- 0.000000 | 0.400000 +/- 0.000000 |
| `mlp_h64` | 0.036735 +/- 0.000000 | 0.800000 +/- 0.000000 | 0.112113 +/- 0.000000 | 0.366667 +/- 0.000000 |

- Best MLP for final-window MSE: `mlp_h64`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` UPGD-vs-best-MLP diff: -0.047776 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_accuracy` UPGD-vs-best-MLP diff: -0.066667 +/- 0.000000; wins/losses/ties 0/1/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional
dependency behavior, and metric accounting. This smoke run is negative for
UPGD versus the best MLP on both IID and class-blocked rows, so it must not be
used as image-scale support for Step 2. A stronger CIFAR claim requires real
CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no
test-set tuning.
