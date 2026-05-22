# Step 2 Deep Feature Lifecycle

Seeds: 5. Steps: 800. Final window: 200.

Positive paired differences mean the method beat the best fair MLP.

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_nlms` | 0.0589 +/- 0.0092 | 0.0847 +/- 0.0081 |
| `deep_shallow_soft_gate` | 0.0607 +/- 0.0089 | 0.0805 +/- 0.0085 |
| `mlp_64` | 0.0612 +/- 0.0093 | 0.0815 +/- 0.0091 |
| `deep_shallow_safe` | 0.0647 +/- 0.0118 | 0.0810 +/- 0.0095 |
| `deep_shallow_net2net` | 0.0652 +/- 0.0110 | 0.0833 +/- 0.0100 |
| `mlp_64_64` | 0.0855 +/- 0.0121 | 0.1138 +/- 0.0107 |
| `deep_active_perturb_low` | 0.0860 +/- 0.0138 | 0.1128 +/- 0.0139 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0243 +/- 0.0040 | 0/5 |
| `deep_shallow_nlms` | +0.0023 +/- 0.0003 | 5/5 |
| `deep_shallow_safe` | -0.0035 +/- 0.0044 | 2/5 |
| `deep_shallow_soft_gate` | +0.0005 +/- 0.0015 | 3/5 |
| `deep_shallow_net2net` | -0.0040 +/- 0.0019 | 1/5 |
| `deep_active_perturb_low` | -0.0248 +/- 0.0049 | 0/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=1.00, `deep_shallow_safe`=0.20, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=1.80, `deep_active_perturb_low`=6.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_nlms` | 771 | 52 | active + candidates update every step |
| `deep_shallow_soft_gate` | 771 | 104 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_shallow_safe` | 771 | 52 | active + candidates update every step |
| `deep_shallow_net2net` | 771 | 52 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_active_perturb_low` | 4931 | 328 | active + candidates update every step |

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_shallow_net2net` | 0.1355 +/- 0.0067 | 0.1631 +/- 0.0133 |
| `deep_shallow_soft_gate` | 0.1385 +/- 0.0110 | 0.1616 +/- 0.0129 |
| `mlp_64` | 0.1395 +/- 0.0078 | 0.1677 +/- 0.0137 |
| `deep_shallow_safe` | 0.1471 +/- 0.0124 | 0.1659 +/- 0.0133 |
| `deep_active_perturb_low` | 0.1727 +/- 0.0177 | 0.2074 +/- 0.0143 |
| `mlp_64_64` | 0.1740 +/- 0.0158 | 0.2055 +/- 0.0159 |
| `deep_shallow_nlms` | 0.3110 +/- 0.0978 | 0.2122 +/- 0.0188 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0345 +/- 0.0083 | 0/5 |
| `deep_shallow_nlms` | -0.1715 +/- 0.0945 | 1/5 |
| `deep_shallow_safe` | -0.0076 +/- 0.0068 | 2/5 |
| `deep_shallow_soft_gate` | +0.0010 +/- 0.0034 | 3/5 |
| `deep_shallow_net2net` | +0.0040 +/- 0.0026 | 3/5 |
| `deep_active_perturb_low` | -0.0332 +/- 0.0118 | 0/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=2.40, `deep_shallow_safe`=0.40, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=2.20, `deep_active_perturb_low`=6.20.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_shallow_net2net` | 643 | 44 | active + candidates update every step |
| `deep_shallow_soft_gate` | 643 | 88 | active + candidates update every step |
| `mlp_64` | 643 | 0 | active learner updates every step |
| `deep_shallow_safe` | 643 | 44 | active + candidates update every step |
| `deep_active_perturb_low` | 4803 | 320 | active + candidates update every step |
| `mlp_64_64` | 4803 | 0 | active learner updates every step |
| `deep_shallow_nlms` | 643 | 44 | active + candidates update every step |

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0363 +/- 0.0007 | 0.0509 +/- 0.0005 |
| `deep_shallow_net2net` | 0.0371 +/- 0.0006 | 0.0505 +/- 0.0009 |
| `deep_shallow_soft_gate` | 0.0373 +/- 0.0008 | 0.0513 +/- 0.0012 |
| `deep_shallow_safe` | 0.0374 +/- 0.0007 | 0.0506 +/- 0.0008 |
| `deep_shallow_nlms` | 0.0383 +/- 0.0008 | 0.0512 +/- 0.0009 |
| `mlp_64_64` | 0.0392 +/- 0.0007 | 0.0554 +/- 0.0011 |
| `deep_active_perturb_low` | 0.0413 +/- 0.0003 | 0.0580 +/- 0.0003 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0029 +/- 0.0007 | 0/5 |
| `deep_shallow_nlms` | -0.0020 +/- 0.0005 | 0/5 |
| `deep_shallow_safe` | -0.0010 +/- 0.0003 | 1/5 |
| `deep_shallow_soft_gate` | -0.0010 +/- 0.0008 | 2/5 |
| `deep_shallow_net2net` | -0.0007 +/- 0.0007 | 2/5 |
| `deep_active_perturb_low` | -0.0050 +/- 0.0005 | 0/5 |

Mean deep-feature promotions per run: `deep_shallow_nlms`=0.00, `deep_shallow_safe`=0.00, `deep_shallow_soft_gate`=0.00, `deep_shallow_net2net`=0.00, `deep_active_perturb_low`=0.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 4810 | 0 | active learner updates every step |
| `deep_shallow_net2net` | 4810 | 304 | active + candidates update every step |
| `deep_shallow_soft_gate` | 4810 | 608 | active + candidates update every step |
| `deep_shallow_safe` | 4810 | 304 | active + candidates update every step |
| `deep_shallow_nlms` | 4810 | 304 | active + candidates update every step |
| `mlp_64_64` | 8970 | 0 | active learner updates every step |
| `deep_active_perturb_low` | 8970 | 608 | active + candidates update every step |

## Verdict

The best native deep feature lifecycle variant was `deep_shallow_soft_gate`, which beat the best fair MLP on 2/3 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.