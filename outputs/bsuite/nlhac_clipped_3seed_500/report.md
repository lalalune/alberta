# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_clipped`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_clipped_improvement_vs_autostep_bottleneck_mean |   nlhac_clipped_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|--------------------------------------------------------:|--------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                   26.6667  |                                                          2 |                                                -3.66667 |                                                       2 |
| catch    |   3 | total_regret                 |                                                   -9.33333 |                                                          0 |                                                -8       |                                                       0 |
| overall  |   6 | episode_return, total_regret |                                                    8.66667 |                                                          2 |                                                -5.83333 |                                                       2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_clipped |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_clipped_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|----------------:|-------------------:|------------------------------------------------------:|---------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |              53 |                104 |                                                    76 |                                                 25 |
|      1 | cartpole/0  | cartpole     | episode_return |                    82 |              27 |                 29 |                                                   -53 |                                                -55 |
|      2 | cartpole/0  | cartpole     | episode_return |                    83 |             102 |                140 |                                                    57 |                                                 19 |
|      0 | catch/0     | catch        | total_regret   |                    66 |              72 |                 76 |                                                   -10 |                                                 -6 |
|      1 | catch/0     | catch        | total_regret   |                    72 |              76 |                 82 |                                                   -10 |                                                 -4 |
|      2 | catch/0     | catch        | total_regret   |                    72 |              86 |                 80 |                                                    -8 |                                                -14 |