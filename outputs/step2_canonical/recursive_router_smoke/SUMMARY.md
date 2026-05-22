# Step 2 Recursive Feature Utility Probe

Seeds: 2; steps: 800; final-window: 200.


## rare

| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |
|---|---:|---:|---:|---:|
| `mlp_32x32_no_ln` | 0.0840 +/- 0.0360 | 0.4326 +/- 0.2238 | 0.00 | 0/2 |
| `recursive_mlp_router` | 0.0437 +/- 0.0318 | 0.4015 +/- 0.2469 | 19.50 | 2/2 |
| `single_mechanism` | 0.0645 +/- 0.0430 | 0.3764 +/- 0.2518 | 19.00 | 2/2 |

Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `recursive_mlp_router`: 0.0403; `recursive_mlp_router` wins 2/2 seeds.
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `single_mechanism`: 0.0195; `single_mechanism` wins 2/2 seeds.

## Suite summary

`recursive_mlp_router` beats best fair MLP on 1/1 tasks and ties within 0.02 MSE on 0/1 tasks.
`single_mechanism` beats best fair MLP on 1/1 tasks and ties within 0.02 MSE on 0/1 tasks.

Interpretation: `single_mechanism` is the robust recursive configuration: contribution-trace utility, residual imprint, product-biased operation priors, utility/novelty parent choice, and depth retention. `recursive_mlp_router` is a causal resource router over that recursive mechanism and the fair MLP controls. Both are judged against the best fair MLP run for each task.