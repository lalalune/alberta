# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                   17.6667  |                                                          1 |                                                       15   |                                                          1 |
| catch    |   3 | total_regret                 |                                                   -3.33333 |                                                          1 |                                                       -6   |                                                          1 |
| overall  |   6 | episode_return, total_regret |                                                    7.16667 |                                                          2 |                                                        4.5 |                                                          2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                103 |                199 |                                                   171 |                                                    75 |
|      1 | cartpole/0  | cartpole     | episode_return |                   119 |                114 |                 30 |                                                   -89 |                                                    -5 |
|      2 | cartpole/0  | cartpole     | episode_return |                   110 |                 85 |                 81 |                                                   -29 |                                                   -25 |
|      0 | catch/0     | catch        | total_regret   |                    54 |                 52 |                 52 |                                                     2 |                                                     2 |
|      1 | catch/0     | catch        | total_regret   |                    44 |                 50 |                 52 |                                                    -8 |                                                    -6 |
|      2 | catch/0     | catch        | total_regret   |                    38 |                 52 |                 42 |                                                    -4 |                                                   -14 |