# Step 2 Recursive Feature Utility Probe

Seeds: 5; steps: 5000; final-window: 1000.


## frequency

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0292 +/- 0.0010 | 0.0191 +/- 0.0034 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0664 +/- 0.0028 | 0.0468 +/- 0.0023 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.0221 +/- 0.0020 | 0.0229 +/- 0.0059 | 20.40 | 5/5 |
| `single_mechanism` | 0.0836 +/- 0.0103 | 0.1164 +/- 0.0169 | 19.80 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0071; `recursive_mlp_router` wins 4/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.0543; `single_mechanism` wins 0/5 seeds.

## interaction

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.1984 +/- 0.0058 | 0.1669 +/- 0.0126 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.4666 +/- 0.0104 | 0.4264 +/- 0.0408 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.0576 +/- 0.0016 | 0.0586 +/- 0.0071 | 21.60 | 5/5 |
| `single_mechanism` | 0.0638 +/- 0.0042 | 0.0729 +/- 0.0083 | 21.80 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.1407; `recursive_mlp_router` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.1346; `single_mechanism` wins 5/5 seeds.

## nonlinear

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0320 +/- 0.0022 | 0.0240 +/- 0.0040 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0648 +/- 0.0057 | 0.0649 +/- 0.0079 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.0173 +/- 0.0001 | 0.0129 +/- 0.0013 | 21.40 | 5/5 |
| `single_mechanism` | 0.3066 +/- 0.1499 | 0.2736 +/- 0.0624 | 21.20 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0147; `recursive_mlp_router` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.2746; `single_mechanism` wins 0/5 seeds.

## polynomial

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4929 +/- 0.0611 | 0.4235 +/- 0.0283 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.7650 +/- 0.0836 | 0.9371 +/- 0.1432 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.2133 +/- 0.0207 | 0.1928 +/- 0.0218 | 21.60 | 5/5 |
| `single_mechanism` | 0.5674 +/- 0.0454 | 0.6137 +/- 0.1228 | 21.00 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.2796; `recursive_mlp_router` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.0746; `single_mechanism` wins 2/5 seeds.

## rare

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0595 +/- 0.0089 | 0.4489 +/- 0.0433 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0691 +/- 0.0078 | 0.4345 +/- 0.0371 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.0534 +/- 0.0081 | 0.3345 +/- 0.0840 | 21.20 | 5/5 |
| `single_mechanism` | 0.0997 +/- 0.0122 | 0.1249 +/- 0.0413 | 22.20 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0061; `recursive_mlp_router` wins 4/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.0402; `single_mechanism` wins 0/5 seeds.

## triple

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4750 +/- 0.0582 | 0.5064 +/- 0.0608 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.6284 +/- 0.0445 | 0.6709 +/- 0.0402 | 0.00 | 0/5 |
| `recursive_mlp_router` | 0.0834 +/- 0.0082 | 0.0760 +/- 0.0240 | 22.20 | 5/5 |
| `single_mechanism` | 0.0970 +/- 0.0221 | 0.0693 +/- 0.0167 | 21.80 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.3915; `recursive_mlp_router` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.3780; `single_mechanism` wins 5/5 seeds.

## Suite summary

`recursive_mlp_router` beats best fair MLP on 6/6 tasks and ties within 0.02 MSE on 0/6 tasks.
`single_mechanism` beats best fair MLP on 2/6 tasks and ties within 0.02 MSE on 0/6 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. `recursive_mlp_router` is a causal resource router over that recursive mechanism and the fair MLP controls. Both are judged against the best fair MLP run for each task.