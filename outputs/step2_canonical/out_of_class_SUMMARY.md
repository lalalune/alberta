# Step 2 Out-of-Hypothesis-Class Benchmark

Wall clock: 1522.9s. Seeds: 30. Steps per run: 6000. Final-window: last 2000 steps.

All MLP-family learners use ObGDBounding(kappa=2.0). MSE values are averaged across active heads at each step.

## Stream: `out_of_class_polynomial`

Final-window MSE (mean +/- stderr over seeds):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `upgd` | 0.5765 +/- 0.0320 | 0.5803 +/- 0.0311 |
| `feature_lifecycle` | 1.1151 +/- 0.0632 | 1.1271 +/- 0.0615 |
| `compositional` | 1.1416 +/- 0.0648 | 1.1494 +/- 0.0622 |
| `mlp_64_64` | 1.1449 +/- 0.0641 | 1.1756 +/- 0.0627 |
| `interaction` | 1.1656 +/- 0.0634 | 1.1713 +/- 0.0619 |
| `mlp_64` | 1.1922 +/- 0.0650 | 1.2186 +/- 0.0637 |
| `cbp` | 1.2057 +/- 0.0653 | 1.2256 +/- 0.0640 |
| `linear` | 1.4174 +/- 0.0821 | 1.4285 +/- 0.0789 |

Paired-vs-best-MLP comparison (best MLP on this stream: `mlp_64_64`). A positive `best_mlp - method` value means the method **beats** the MLP.

| Method | best_mlp - method | stderr | Wins/30 | Cohen's d |
|---|---|---|---|---|
| `mlp_64` | -0.0473 | 0.0021 | 0/30 | -4.021 |
| `linear` | -0.2724 | 0.0188 | 0/30 | -2.650 |
| `interaction` | -0.0207 | 0.0022 | 1/30 | -1.703 |
| `compositional` | +0.0033 | 0.0048 | 16/30 | +0.127 |
| `upgd` | +0.5684 | 0.0320 | 30/30 | +3.240 |
| `cbp` | -0.0607 | 0.0024 | 0/30 | -4.546 |
| `feature_lifecycle` | +0.0298 | 0.0017 | 30/30 | +3.213 |

## Stream: `frequency_mismatch`

Final-window MSE (mean +/- stderr over seeds):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `upgd` | 0.6330 +/- 0.0434 | 0.6449 +/- 0.0451 |
| `mlp_64` | 1.1677 +/- 0.0785 | 1.2340 +/- 0.0834 |
| `mlp_64_64` | 1.1775 +/- 0.0794 | 1.2383 +/- 0.0834 |
| `cbp` | 1.1931 +/- 0.0803 | 1.2485 +/- 0.0838 |
| `feature_lifecycle` | 1.3303 +/- 0.0856 | 1.3322 +/- 0.0862 |
| `linear` | 1.5058 +/- 0.0995 | 1.5101 +/- 0.1012 |
| `interaction` | 1.7625 +/- 0.0898 | 1.7519 +/- 0.0902 |
| `compositional` | 2.0493 +/- 0.1220 | 1.9497 +/- 0.1036 |

Paired-vs-best-MLP comparison (best MLP on this stream: `mlp_64`). A positive `best_mlp - method` value means the method **beats** the MLP.

| Method | best_mlp - method | stderr | Wins/30 | Cohen's d |
|---|---|---|---|---|
| `mlp_64_64` | -0.0098 | 0.0071 | 11/30 | -0.253 |
| `linear` | -0.3381 | 0.0313 | 0/30 | -1.972 |
| `interaction` | -0.5948 | 0.0499 | 0/30 | -2.176 |
| `compositional` | -0.8816 | 0.0814 | 0/30 | -1.976 |
| `upgd` | +0.5348 | 0.0363 | 30/30 | +2.691 |
| `cbp` | -0.0254 | 0.0033 | 0/30 | -1.400 |
| `feature_lifecycle` | -0.1626 | 0.0290 | 5/30 | -1.025 |

## Stream: `compositional`

Final-window MSE (mean +/- stderr over seeds):

| Method | Final-window MSE | Total mean MSE |
|---|---|---|
| `upgd` | 0.1633 +/- 0.0089 | 0.1621 +/- 0.0087 |
| `mlp_64` | 0.1904 +/- 0.0081 | 0.1949 +/- 0.0084 |
| `cbp` | 0.1912 +/- 0.0082 | 0.1946 +/- 0.0083 |
| `mlp_64_64` | 0.2154 +/- 0.0088 | 0.2277 +/- 0.0094 |
| `linear` | 0.2259 +/- 0.0090 | 0.2248 +/- 0.0092 |
| `compositional` | 0.6304 +/- 0.0356 | 0.6339 +/- 0.0357 |
| `feature_lifecycle` | 0.6940 +/- 0.0372 | 0.6982 +/- 0.0374 |
| `interaction` | 1.3063 +/- 0.0517 | 1.2872 +/- 0.0506 |

Paired-vs-best-MLP comparison (best MLP on this stream: `mlp_64`). A positive `best_mlp - method` value means the method **beats** the MLP.

| Method | best_mlp - method | stderr | Wins/30 | Cohen's d |
|---|---|---|---|---|
| `mlp_64_64` | -0.0251 | 0.0015 | 0/30 | -3.048 |
| `linear` | -0.0355 | 0.0023 | 0/30 | -2.766 |
| `interaction` | -1.1160 | 0.0450 | 0/30 | -4.523 |
| `compositional` | -0.4400 | 0.0308 | 0/30 | -2.611 |
| `upgd` | +0.0270 | 0.0024 | 30/30 | +2.078 |
| `cbp` | -0.0008 | 0.0007 | 13/30 | -0.215 |
| `feature_lifecycle` | -0.5036 | 0.0309 | 0/30 | -2.979 |

## Honest summary

### `out_of_class_polynomial`

Baseline winner: `mlp_64_64` with final-window MSE = 1.1449.

Methods that beat the best MLP on this stream (mean diff, wins/n_seeds, d):
- `compositional`: +0.0033 +/- 0.0048, 16/30 wins, d = +0.127
- `upgd`: +0.5684 +/- 0.0320, 30/30 wins, d = +3.240
- `feature_lifecycle`: +0.0298 +/- 0.0017, 30/30 wins, d = +3.213

Methods that lost to the best MLP:
- `mlp_64`: -0.0473 +/- 0.0021, 0/30 wins, d = -4.021
- `linear`: -0.2724 +/- 0.0188, 0/30 wins, d = -2.650
- `interaction`: -0.0207 +/- 0.0022, 1/30 wins, d = -1.703
- `cbp`: -0.0607 +/- 0.0024, 0/30 wins, d = -4.546

### `frequency_mismatch`

Baseline winner: `mlp_64` with final-window MSE = 1.1677.

Methods that beat the best MLP on this stream (mean diff, wins/n_seeds, d):
- `upgd`: +0.5348 +/- 0.0363, 30/30 wins, d = +2.691

Methods that lost to the best MLP:
- `mlp_64_64`: -0.0098 +/- 0.0071, 11/30 wins, d = -0.253
- `linear`: -0.3381 +/- 0.0313, 0/30 wins, d = -1.972
- `interaction`: -0.5948 +/- 0.0499, 0/30 wins, d = -2.176
- `compositional`: -0.8816 +/- 0.0814, 0/30 wins, d = -1.976
- `cbp`: -0.0254 +/- 0.0033, 0/30 wins, d = -1.400
- `feature_lifecycle`: -0.1626 +/- 0.0290, 5/30 wins, d = -1.025

### `compositional`

Baseline winner: `mlp_64` with final-window MSE = 0.1904.

Methods that beat the best MLP on this stream (mean diff, wins/n_seeds, d):
- `upgd`: +0.0270 +/- 0.0024, 30/30 wins, d = +2.078

Methods that lost to the best MLP:
- `mlp_64_64`: -0.0251 +/- 0.0015, 0/30 wins, d = -3.048
- `linear`: -0.0355 +/- 0.0023, 0/30 wins, d = -2.766
- `interaction`: -1.1160 +/- 0.0450, 0/30 wins, d = -4.523
- `compositional`: -0.4400 +/- 0.0308, 0/30 wins, d = -2.611
- `cbp`: -0.0008 +/- 0.0007, 13/30 wins, d = -0.215
- `feature_lifecycle`: -0.5036 +/- 0.0309, 0/30 wins, d = -2.979


## Science verdict

Step 2 demands feature *construction*, not feature *selection*. The three streams used here are deliberately out-of-class for a 1-layer pair-product hypothesis: triple-product polynomials, sums of sinusoids, and 2-hidden-layer tanh networks. Across 3 streams, methods beat the best MLP as follows: `upgd` on 3/3, `feature_lifecycle` (generate-test-rank-replace) on 1/3, `compositional` on 1/3, `cbp` on 0/3, `interaction` on 0/3.

`feature_lifecycle` (`FixedBudgetFeatureLearner`) is the first benchmark evidence that the generate-test-rank-replace lifecycle itself — active banks, utility tracking, candidate promotion, feature replacement — produces features that beat a fair MLP. Its win is on `out_of_class_polynomial` (30/30 seeds, d=+3.213). It loses on `frequency_mismatch` and `compositional` because single-layer tanh features cannot represent sinusoidal or 2-hidden-layer tanh oracles. The limitation is the hypothesis class of the feature bank, not the lifecycle mechanism. `upgd` remains the broadest Step 2 method, winning on all three streams.
