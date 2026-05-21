# Step 2 Deep Feature Lifecycle

Seeds: 5. Steps: 800. Final window: 200.

Positive paired differences mean the method beat the best fair MLP.

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_nlms` | 0.0589 +/- 0.0092 | 0.0847 +/- 0.0081 |
| `mlp_64` | 0.0612 +/- 0.0093 | 0.0815 +/- 0.0091 |
| `deep_shallow_soft_gate` | 0.0613 +/- 0.0099 | 0.0802 +/- 0.0088 |
| `deep_shallow_net2net` | 0.0617 +/- 0.0090 | 0.0809 +/- 0.0086 |
| `mlp_64_64` | 0.0855 +/- 0.0121 | 0.1138 +/- 0.0107 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0243 +/- 0.0040 | 0/5 |
| `deep_shallow_nlms` | +0.0023 +/- 0.0003 | 5/5 |
| `deep_shallow_soft_gate` | -0.0002 +/- 0.0016 | 2/5 |
| `deep_shallow_net2net` | -0.0006 +/- 0.0016 | 2/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=1.00, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=2.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_nlms` | 771 | 52 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_shallow_soft_gate` | 771 | 104 | active + candidates update every step |
| `deep_shallow_net2net` | 771 | 52 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |

## `interaction`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_net2net` | 0.9285 +/- 0.1832 | 1.0011 +/- 0.1608 |
| `mlp_64` | 0.9502 +/- 0.1978 | 0.9909 +/- 0.1562 |
| `deep_shallow_soft_gate` | 0.9511 +/- 0.1737 | 1.0058 +/- 0.1515 |
| `mlp_64_64` | 0.9697 +/- 0.2254 | 1.0817 +/- 0.1502 |
| `deep_shallow_nlms` | 1.0728 +/- 0.2933 | 1.2883 +/- 0.2350 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0194 +/- 0.0333 | 2/5 |
| `deep_shallow_nlms` | -0.1225 +/- 0.0987 | 1/5 |
| `deep_shallow_soft_gate` | -0.0008 +/- 0.0410 | 2/5 |
| `deep_shallow_net2net` | +0.0218 +/- 0.0218 | 2/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=3.40, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=5.20.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_net2net` | 771 | 52 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_shallow_soft_gate` | 771 | 104 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_shallow_nlms` | 771 | 52 | active + candidates update every step |

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64_64` | 0.9517 +/- 0.2554 | 0.8938 +/- 0.1885 |
| `mlp_64` | 0.9904 +/- 0.2664 | 0.9185 +/- 0.1914 |
| `deep_shallow_net2net` | 0.9923 +/- 0.2652 | 0.9181 +/- 0.1919 |
| `deep_shallow_soft_gate` | 0.9959 +/- 0.2710 | 0.9195 +/- 0.1916 |
| `deep_shallow_nlms` | 1.6731 +/- 0.6976 | 1.2394 +/- 0.3174 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0387 +/- 0.0121 | 0/5 |
| `deep_shallow_nlms` | -0.7214 +/- 0.4713 | 0/5 |
| `deep_shallow_soft_gate` | -0.0442 +/- 0.0161 | 0/5 |
| `deep_shallow_net2net` | -0.0405 +/- 0.0109 | 0/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=5.80, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=6.60.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_shallow_net2net` | 771 | 52 | active + candidates update every step |
| `deep_shallow_soft_gate` | 771 | 104 | active + candidates update every step |
| `deep_shallow_nlms` | 771 | 52 | active + candidates update every step |

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_net2net` | 1.6789 +/- 0.6301 | 1.5536 +/- 0.7030 |
| `mlp_64` | 1.6993 +/- 0.6462 | 1.5567 +/- 0.7104 |
| `deep_shallow_soft_gate` | 1.7156 +/- 0.6466 | 1.5631 +/- 0.7080 |
| `mlp_64_64` | 1.7432 +/- 0.6474 | 1.5884 +/- 0.6933 |
| `deep_shallow_nlms` | 2.2959 +/- 0.6668 | 2.0772 +/- 0.8774 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0439 +/- 0.0225 | 1/5 |
| `deep_shallow_nlms` | -0.5966 +/- 0.1584 | 0/5 |
| `deep_shallow_soft_gate` | -0.0163 +/- 0.0192 | 3/5 |
| `deep_shallow_net2net` | +0.0204 +/- 0.0258 | 3/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=7.80, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=7.60.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_net2net` | 450 | 32 | active + candidates update every step |
| `mlp_64` | 450 | 0 | active learner updates every step |
| `deep_shallow_soft_gate` | 450 | 64 | active + candidates update every step |
| `mlp_64_64` | 4610 | 0 | active learner updates every step |
| `deep_shallow_nlms` | 450 | 32 | active + candidates update every step |

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_soft_gate` | 0.1376 +/- 0.0114 | 0.1635 +/- 0.0148 |
| `mlp_64` | 0.1395 +/- 0.0078 | 0.1677 +/- 0.0137 |
| `deep_shallow_net2net` | 0.1396 +/- 0.0116 | 0.1617 +/- 0.0129 |
| `mlp_64_64` | 0.1740 +/- 0.0158 | 0.2055 +/- 0.0159 |
| `deep_shallow_nlms` | 0.3110 +/- 0.0978 | 0.2122 +/- 0.0188 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0345 +/- 0.0083 | 0/5 |
| `deep_shallow_nlms` | -0.1715 +/- 0.0945 | 1/5 |
| `deep_shallow_soft_gate` | +0.0019 +/- 0.0060 | 4/5 |
| `deep_shallow_net2net` | -0.0001 +/- 0.0040 | 3/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=2.40, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=2.40.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_soft_gate` | 643 | 88 | active + candidates update every step |
| `mlp_64` | 643 | 0 | active learner updates every step |
| `deep_shallow_net2net` | 643 | 44 | active + candidates update every step |
| `mlp_64_64` | 4803 | 0 | active learner updates every step |
| `deep_shallow_nlms` | 643 | 44 | active + candidates update every step |

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0363 +/- 0.0007 | 0.0509 +/- 0.0005 |
| `deep_shallow_net2net` | 0.0373 +/- 0.0008 | 0.0513 +/- 0.0012 |
| `deep_shallow_soft_gate` | 0.0374 +/- 0.0007 | 0.0506 +/- 0.0008 |
| `deep_shallow_nlms` | 0.0383 +/- 0.0008 | 0.0512 +/- 0.0009 |
| `mlp_64_64` | 0.0392 +/- 0.0007 | 0.0554 +/- 0.0011 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0029 +/- 0.0007 | 0/5 |
| `deep_shallow_nlms` | -0.0020 +/- 0.0005 | 0/5 |
| `deep_shallow_soft_gate` | -0.0010 +/- 0.0003 | 1/5 |
| `deep_shallow_net2net` | -0.0010 +/- 0.0008 | 2/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=0.00, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=0.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 4810 | 0 | active learner updates every step |
| `deep_shallow_net2net` | 4810 | 304 | active + candidates update every step |
| `deep_shallow_soft_gate` | 4810 | 608 | active + candidates update every step |
| `deep_shallow_nlms` | 4810 | 304 | active + candidates update every step |
| `mlp_64_64` | 8970 | 0 | active learner updates every step |

## Verdict

The best native deep feature lifecycle variant was `deep_shallow_net2net`, which beat the best fair MLP on 2/6 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.