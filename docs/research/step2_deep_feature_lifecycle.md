# Step 2 Deep Feature Lifecycle

Seeds: 1. Steps: 300. Final window: 100.

Positive paired differences mean the method beat the best fair MLP.

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.1230 +/- 0.0000 | 0.1968 +/- 0.0000 |
| `portfolio` | 0.2499 +/- 0.0000 | 0.3865 +/- 0.0000 |
| `compositional` | 0.2527 +/- 0.0000 | 0.3858 +/- 0.0000 |
| `deep_feature_lifecycle` | 0.2609 +/- 0.0000 | 0.4077 +/- 0.0000 |
| `mlp_64_64` | 0.2631 +/- 0.0000 | 0.4083 +/- 0.0000 |
| `cbp` | 0.2702 +/- 0.0000 | 0.4230 +/- 0.0000 |
| `mlp_64` | 0.2758 +/- 0.0000 | 0.4192 +/- 0.0000 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0127 +/- 0.0000 | 0/1 |
| `upgd` | +0.1401 +/- 0.0000 | 1/1 |
| `cbp` | -0.0071 +/- 0.0000 | 0/1 |
| `compositional` | +0.0104 +/- 0.0000 | 1/1 |
| `deep_feature_lifecycle` | +0.0022 +/- 0.0000 | 1/1 |
| `portfolio` | +0.0132 +/- 0.0000 | 1/1 |

Mean deep-feature promotions per run: 3.00.

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.1222 +/- 0.0000 | 0.1788 +/- 0.0000 |
| `mlp_64` | 0.2146 +/- 0.0000 | 0.3164 +/- 0.0000 |
| `portfolio` | 0.2146 +/- 0.0000 | 0.3185 +/- 0.0000 |
| `cbp` | 0.2309 +/- 0.0000 | 0.3492 +/- 0.0000 |
| `deep_feature_lifecycle` | 0.3027 +/- 0.0000 | 0.4052 +/- 0.0000 |
| `mlp_64_64` | 0.3626 +/- 0.0000 | 0.4741 +/- 0.0000 |
| `compositional` | 1.1781 +/- 0.0000 | 0.8781 +/- 0.0000 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.1480 +/- 0.0000 | 0/1 |
| `upgd` | +0.0924 +/- 0.0000 | 1/1 |
| `cbp` | -0.0163 +/- 0.0000 | 0/1 |
| `compositional` | -0.9635 +/- 0.0000 | 0/1 |
| `deep_feature_lifecycle` | -0.0882 +/- 0.0000 | 0/1 |
| `portfolio` | -0.0000 +/- 0.0000 | 0/1 |

Mean deep-feature promotions per run: 5.00.

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.0548 +/- 0.0000 | 0.0999 +/- 0.0000 |
| `mlp_64` | 0.1249 +/- 0.0000 | 0.1889 +/- 0.0000 |
| `portfolio` | 0.1255 +/- 0.0000 | 0.1790 +/- 0.0000 |
| `cbp` | 0.1268 +/- 0.0000 | 0.1789 +/- 0.0000 |
| `compositional` | 0.1293 +/- 0.0000 | 0.2348 +/- 0.0000 |
| `mlp_64_64` | 0.1402 +/- 0.0000 | 0.2122 +/- 0.0000 |
| `deep_feature_lifecycle` | 0.1430 +/- 0.0000 | 0.2225 +/- 0.0000 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0153 +/- 0.0000 | 0/1 |
| `upgd` | +0.0701 +/- 0.0000 | 1/1 |
| `cbp` | -0.0019 +/- 0.0000 | 0/1 |
| `compositional` | -0.0044 +/- 0.0000 | 0/1 |
| `deep_feature_lifecycle` | -0.0181 +/- 0.0000 | 0/1 |
| `portfolio` | -0.0006 +/- 0.0000 | 0/1 |

Mean deep-feature promotions per run: 4.00.

## Verdict

The native deep feature lifecycle beat the best fair MLP on 1/3 streams; the portfolio beat it on 1/3 streams.  A single general feature-construction algorithm should be considered empirically open unless the native method wins robustly across the full matrix.