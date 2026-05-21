# Step 2 CIFAR Stream Benchmark

This note records a real CIFAR-10 smoke probe for the current promoted
non-router target-structure UPGD learner.

Positive paired differences favor `upgd_step2_default`. For MSE, the
difference is best MLP minus UPGD; for accuracy, it is UPGD minus best MLP.

## Dataset

- Source: `cifar10_python_archive`
- Real CIFAR-10 evidence: `True`
- Train shape: `[1000, 3072]`
- Test shape: `[500, 3072]`

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `upgd_step2_default` | 0.112482 +/- 0.000000 | 0.240000 +/- 0.000000 | 0.118405 +/- 0.000000 | 0.178000 +/- 0.000000 |
| `mlp_h32` | 0.122442 +/- 0.000000 | 0.160000 +/- 0.000000 | 0.112273 +/- 0.000000 | 0.166000 +/- 0.000000 |
| `mlp_h64` | 0.188047 +/- 0.000000 | 0.120000 +/- 0.000000 | 0.171645 +/- 0.000000 | 0.154000 +/- 0.000000 |

- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` UPGD-vs-best-MLP diff: +0.009960 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` UPGD-vs-best-MLP diff: +0.012000 +/- 0.000000; wins/losses/ties 1/0/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `upgd_step2_default` | 0.036151 +/- 0.000000 | 0.940000 +/- 0.000000 | 0.171626 +/- 0.000000 | 0.106000 +/- 0.000000 |
| `mlp_h32` | 0.006380 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.187281 +/- 0.000000 | 0.100000 +/- 0.000000 |
| `mlp_h64` | 0.011495 +/- 0.000000 | 1.000000 +/- 0.000000 | 0.187256 +/- 0.000000 | 0.100000 +/- 0.000000 |

- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` UPGD-vs-best-MLP diff: -0.029770 +/- 0.000000; wins/losses/ties 0/1/0.
- `test_accuracy` UPGD-vs-best-MLP diff: +0.006000 +/- 0.000000; wins/losses/ties 1/0/0.

## Interpretation

This is real CIFAR-10 evidence, but it is still only a smoke run: one seed,
200 online updates, 1,000 sampled training images, and 500 sampled test images.
It validates the direct CIFAR archive loader and benchmark wiring. It is
positive for UPGD on IID final-window MSE and held-out accuracy, mixed on the
class-blocked row (worse online MSE, slightly better held-out accuracy), and is
not a canonical image-scale Step 2 result. A stronger CIFAR claim requires
multiple seeds, longer streams, larger train/test coverage, and no test-set
tuning.
