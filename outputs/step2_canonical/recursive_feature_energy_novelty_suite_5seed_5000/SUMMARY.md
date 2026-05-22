# Step 2 Recursive Feature Utility Probe

Seeds: 5; steps: 5000; final-window: 500.


## frequency

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0271 +/- 0.0009 | 0.0195 +/- 0.0032 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0631 +/- 0.0029 | 0.0459 +/- 0.0030 | 0.00 | 0/5 |
| `single_mechanism` | 0.0744 +/- 0.0122 | 0.0700 +/- 0.0229 | 27.40 | 5/5 |
| `single_mechanism_energy_novelty` | 0.1097 +/- 0.0728 | 0.1949 +/- 0.1549 | 25.80 | 5/5 |
| `single_mechanism_signed_tanh` | 0.0776 +/- 0.0224 | 0.0798 +/- 0.0364 | 23.40 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.0474; `single_mechanism` wins 0/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: -0.0826; `single_mechanism_energy_novelty` wins 2/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: -0.0505; `single_mechanism_signed_tanh` wins 1/5 seeds.

## interaction

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.1987 +/- 0.0060 | 0.1724 +/- 0.0132 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.4795 +/- 0.0162 | 0.4263 +/- 0.0353 | 0.00 | 0/5 |
| `single_mechanism` | 0.1102 +/- 0.0182 | 0.1197 +/- 0.0251 | 29.40 | 5/5 |
| `single_mechanism_energy_novelty` | 0.3693 +/- 0.0848 | 0.6858 +/- 0.2193 | 26.60 | 5/5 |
| `single_mechanism_signed_tanh` | 0.1055 +/- 0.0273 | 0.0828 +/- 0.0205 | 28.40 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.0886; `single_mechanism` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: -0.1706; `single_mechanism_energy_novelty` wins 1/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: 0.0933; `single_mechanism_signed_tanh` wins 4/5 seeds.

## nonlinear

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0301 +/- 0.0013 | 0.0236 +/- 0.0043 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0618 +/- 0.0041 | 0.0653 +/- 0.0071 | 0.00 | 0/5 |
| `single_mechanism` | 0.2209 +/- 0.0159 | 0.1962 +/- 0.0120 | 29.20 | 5/5 |
| `single_mechanism_energy_novelty` | 0.4029 +/- 0.0463 | 0.4212 +/- 0.0524 | 27.20 | 5/5 |
| `single_mechanism_signed_tanh` | 0.1124 +/- 0.0175 | 0.1112 +/- 0.0228 | 25.20 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.1909; `single_mechanism` wins 0/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: -0.3728; `single_mechanism_energy_novelty` wins 0/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: -0.0823; `single_mechanism_signed_tanh` wins 0/5 seeds.

## polynomial

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.4264 +/- 0.0518 | 0.4182 +/- 0.0667 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.7051 +/- 0.0925 | 0.8964 +/- 0.1613 | 0.00 | 0/5 |
| `single_mechanism` | 0.3719 +/- 0.0534 | 0.3544 +/- 0.0784 | 28.40 | 5/5 |
| `single_mechanism_energy_novelty` | 0.6867 +/- 0.1401 | 0.6768 +/- 0.1465 | 27.20 | 5/5 |
| `single_mechanism_signed_tanh` | 0.4548 +/- 0.0957 | 0.4753 +/- 0.1659 | 28.00 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.0545; `single_mechanism` wins 4/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: -0.2603; `single_mechanism_energy_novelty` wins 1/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: -0.0284; `single_mechanism_signed_tanh` wins 3/5 seeds.

## rare

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0661 +/- 0.0126 | 0.5687 +/- 0.0848 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.0751 +/- 0.0119 | 0.5513 +/- 0.0760 | 0.00 | 0/5 |
| `single_mechanism` | 0.1282 +/- 0.0329 | 0.1903 +/- 0.0852 | 28.60 | 5/5 |
| `single_mechanism_energy_novelty` | 0.0591 +/- 0.0123 | 0.0707 +/- 0.0285 | 26.20 | 5/5 |
| `single_mechanism_signed_tanh` | 0.1200 +/- 0.0242 | 0.2941 +/- 0.1745 | 28.00 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: -0.0621; `single_mechanism` wins 0/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: 0.0070; `single_mechanism_energy_novelty` wins 4/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: -0.0539; `single_mechanism_signed_tanh` wins 0/5 seeds.

## triple

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.3958 +/- 0.0371 | 0.5581 +/- 0.0892 | 0.00 | 0/5 |
| `mlp_64x64_no_ln` | 0.5484 +/- 0.0161 | 0.7358 +/- 0.0695 | 0.00 | 0/5 |
| `single_mechanism` | 0.0964 +/- 0.0192 | 0.0792 +/- 0.0126 | 28.80 | 5/5 |
| `single_mechanism_energy_novelty` | 0.0522 +/- 0.0078 | 0.0658 +/- 0.0142 | 27.20 | 5/5 |
| `single_mechanism_signed_tanh` | 0.0857 +/- 0.0124 | 0.1549 +/- 0.0700 | 29.40 | 5/5 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.2995; `single_mechanism` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_energy_novelty`: 0.3436; `single_mechanism_energy_novelty` wins 5/5 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism_signed_tanh`: 0.3101; `single_mechanism_signed_tanh` wins 5/5 seeds.

## Suite summary

`single_mechanism` beats best fair MLP on 3/6 tasks and ties within 0.02 MSE on 0/6 tasks.
`single_mechanism_energy_novelty` beats best fair MLP on 2/6 tasks and ties within 0.02 MSE on 0/6 tasks.
`single_mechanism_signed_tanh` beats best fair MLP on 2/6 tasks and ties within 0.02 MSE on 0/6 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. `single_mechanism_energy_novelty` keeps the same signed-tanh scaffold budget as `single_mechanism_signed_tanh` but changes candidate credit to energy-normalized residual alignment gated by active-feature correlation novelty. Each non-MLP candidate is judged against the best fair MLP run for each task.