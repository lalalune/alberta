# D13 Koopman/Reservoir RLS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Configs: koopman_tanh256, koopman_mixed384.

This is a single-predictor test. The candidate methods use fixed nonlinear observables and an online RLS readout; they do not route over MLP predictions or train an MLP trunk. Positive candidate-vs-MLP paired differences favor the reservoir learner.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Readout dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.8407 +/- 1.0670 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 2.2925 +/- 1.6092 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.5338 +/- 0.4240 |
| `koopman_tanh256` | 0.7744 +/- 0.2520 | 0.4723 +/- 0.0810 |  |  | 263.0000 +/- 0.0000 | 1.3958 +/- 0.3898 |
| `koopman_mixed384` | 1.0478 +/- 0.3584 | 0.5959 +/- 0.1155 |  |  | 503.0000 +/- 0.0000 | 9.1654 +/- 1.8222 |

`final_window_mse` best-candidate-vs-best-MLP diff: -0.5027 +/- 0.1578; wins/losses/ties 0/3/0; best-candidate counts {'koopman_tanh256': 3}.

## Interpretation Bar

A positive row here means the fixed-observable mechanism has headroom on that benchmark. A Step 2 closure claim still requires one canonical configuration to beat the best fair MLP across the full benchmark set without per-dataset selection.
