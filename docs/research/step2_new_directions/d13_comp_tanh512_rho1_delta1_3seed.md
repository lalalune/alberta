# D13 Koopman/Reservoir RLS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Configs: koopman_tanh512.

This is a single-predictor test. The candidate methods use fixed nonlinear observables and an online RLS readout; they do not route over MLP predictions or train an MLP trunk. Positive candidate-vs-MLP paired differences favor the reservoir learner.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Readout dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 2.0264 +/- 1.1092 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.8564 +/- 1.0621 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 1.5870 +/- 0.7155 |
| `koopman_tanh512` | 1.1476 +/- 0.4468 | 0.7086 +/- 0.1336 |  |  | 519.0000 +/- 0.0000 | 9.7571 +/- 0.4302 |

`final_window_mse` best-candidate-vs-best-MLP diff: -0.8759 +/- 0.3516; wins/losses/ties 0/3/0; best-candidate counts {'koopman_tanh512': 3}.

## Interpretation Bar

A positive row here means the fixed-observable mechanism has headroom on that benchmark. A Step 2 closure claim still requires one canonical configuration to beat the best fair MLP across the full benchmark set without per-dataset selection.
