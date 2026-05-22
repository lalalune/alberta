# D07 Budgeted Diffusion-KRLS Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Kernel=arccosine, updates=rls, budgets=[16], sigmas=[1.0], rhos=[0.99], novelty thresholds=[0.001].

This is a single-learner test. The kernel methods do not consume MLP predictions, routes, stacker weights, or offline labels. Positive kernel-vs-MLP paired differences favor the kernel method.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Active centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  | 0.8185 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  | 0.7881 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  | 0.4518 +/- 0.0000 |
| `arccosine_rls_b16_s1_r0p99_n0p001_arc2` | 0.2660 +/- 0.0000 | 0.2871 +/- 0.0000 |  |  | 16.0000 +/- 0.0000 | 0.0175 +/- 0.0000 |
| `hybrid_mlp_h64_arccosine_rls_b16_s1_r0p99_n0p001_arc2` | 0.1764 +/- 0.0000 | 0.2864 +/- 0.0000 |  |  | 16.0000 +/- 0.0000 | 0.1749 +/- 0.0000 |

`final_window_mse` best-kernel-vs-best-MLP diff: +0.0826 +/- 0.0000; wins/losses/ties 1/0/0; best-kernel counts {'hybrid_mlp_h64_arccosine_rls_b16_s1_r0p99_n0p001_arc2': 1}.

## Interpretation Bar

A positive result here is still a search result unless one fixed kernel configuration beats the best fair MLP across the broad suite. The `best_kernel_vs_best_mlp` rows are useful for detecting whether the mathematical mechanism has headroom; a universal learner claim requires promoting one canonical configuration and rerunning it without per-dataset selection.
