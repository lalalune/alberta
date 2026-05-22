# Step 2 Recursive Feature Utility Probe

Seeds: 10; steps: 5000; final-window: 1000.


## frequency

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0293 +/- 0.0014 | 0.0224 +/- 0.0036 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.0636 +/- 0.0028 | 0.0481 +/- 0.0016 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.0205 +/- 0.0013 | 0.0207 +/- 0.0035 | 27.50 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0088; `recursive_mlp_router` wins 9/10 seeds.

## interaction

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.1878 +/- 0.0076 | 0.1680 +/- 0.0128 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.4197 +/- 0.0196 | 0.3934 +/- 0.0310 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.0597 +/- 0.0028 | 0.0517 +/- 0.0054 | 28.90 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.1281; `recursive_mlp_router` wins 10/10 seeds.

## nonlinear

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0315 +/- 0.0015 | 0.0217 +/- 0.0023 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.0647 +/- 0.0034 | 0.0612 +/- 0.0046 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.0165 +/- 0.0004 | 0.0104 +/- 0.0012 | 28.20 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0150; `recursive_mlp_router` wins 10/10 seeds.

## polynomial

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4802 +/- 0.0547 | 0.3721 +/- 0.0316 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.7668 +/- 0.0636 | 0.7736 +/- 0.0905 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.2296 +/- 0.0354 | 0.1756 +/- 0.0134 | 29.10 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.2506; `recursive_mlp_router` wins 10/10 seeds.

## rare

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0552 +/- 0.0048 | 0.4214 +/- 0.0261 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.0637 +/- 0.0046 | 0.3977 +/- 0.0264 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.0481 +/- 0.0052 | 0.3617 +/- 0.0228 | 29.00 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0071; `recursive_mlp_router` wins 9/10 seeds.

## triple

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.5260 +/- 0.0407 | 0.5436 +/- 0.0416 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.6264 +/- 0.0341 | 0.6403 +/- 0.0308 | 0.00 | 0/10 |
| `recursive_mlp_router` | 0.1167 +/- 0.0149 | 0.1311 +/- 0.0212 | 29.40 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.4093; `recursive_mlp_router` wins 10/10 seeds.

## Suite summary

`recursive_mlp_router` beats best fair MLP on 6/6 tasks and ties within 0.02 MSE on 0/6 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. `recursive_mlp_router` is a causal resource router over that recursive mechanism and the fair MLP controls. Both are judged against the best fair MLP run for each task.