# Step 2 Recursive Feature Utility Probe

Seeds: 5; steps: 5000; final-window: 1000.


## frequency

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0292 +/- 0.0010 | 0.0191 +/- 0.0034 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0664 +/- 0.0028 | 0.0468 +/- 0.0023 | 0.00 | 0/5 |
| `single_mechanism` | 0.0303 +/- 0.0064 | 0.0178 +/- 0.0061 | 20.00 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: -0.0011; single-mechanism wins 2/5 seeds.

## interaction

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.1984 +/- 0.0058 | 0.1669 +/- 0.0126 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.4666 +/- 0.0104 | 0.4264 +/- 0.0408 | 0.00 | 0/5 |
| `single_mechanism` | 0.0857 +/- 0.0168 | 0.0830 +/- 0.0413 | 21.20 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: 0.1127; single-mechanism wins 5/5 seeds.

## nonlinear

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0320 +/- 0.0022 | 0.0240 +/- 0.0040 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0648 +/- 0.0057 | 0.0649 +/- 0.0079 | 0.00 | 0/5 |
| `single_mechanism` | 0.3558 +/- 0.0742 | 0.2745 +/- 0.0659 | 21.20 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: -0.3238; single-mechanism wins 0/5 seeds.

## polynomial

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4929 +/- 0.0611 | 0.4235 +/- 0.0283 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.7650 +/- 0.0836 | 0.9371 +/- 0.1432 | 0.00 | 0/5 |
| `single_mechanism` | 6.2335 +/- 4.6672 | 1.0118 +/- 0.1788 | 22.60 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: -5.7406; single-mechanism wins 0/5 seeds.

## rare

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0595 +/- 0.0089 | 0.4489 +/- 0.0433 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0691 +/- 0.0078 | 0.4345 +/- 0.0371 | 0.00 | 0/5 |
| `single_mechanism` | 0.0711 +/- 0.0063 | 0.1231 +/- 0.0753 | 21.60 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: -0.0116; single-mechanism wins 1/5 seeds.

## triple

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4750 +/- 0.0582 | 0.5064 +/- 0.0608 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.6284 +/- 0.0445 | 0.6709 +/- 0.0402 | 0.00 | 0/5 |
| `single_mechanism` | 0.1160 +/- 0.0193 | 0.0752 +/- 0.0250 | 21.80 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: 0.3590; single-mechanism wins 5/5 seeds.

## Suite summary

Single mechanism beats best fair MLP on 2/6 tasks and ties within 0.02 MSE on 2/6 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. It is judged against the best fair MLP run for each task.