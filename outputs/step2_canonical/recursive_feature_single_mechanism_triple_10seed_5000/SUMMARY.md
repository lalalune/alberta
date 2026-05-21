# Step 2 Recursive Feature Utility Probe

Seeds: 10; steps: 5000; final-window: 1000.


## triple

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.5260 +/- 0.0407 | 0.5436 +/- 0.0416 | 0.00 | 0/10 |
| `mlp_64x64_no_ln` | 0.6264 +/- 0.0341 | 0.6403 +/- 0.0308 | 0.00 | 0/10 |
| `single_mechanism` | 0.0839 +/- 0.0144 | 0.0582 +/- 0.0138 | 22.10 | 10/10 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus single mechanism: 0.4421; single-mechanism wins 10/10 seeds.

## Suite summary

Single mechanism beats best fair MLP on 1/1 tasks and ties within 0.02 MSE on 0/1 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. It is judged against the best fair MLP run for each task.