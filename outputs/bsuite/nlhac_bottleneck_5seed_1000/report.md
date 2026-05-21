# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   5 | episode_return               |                                                       31   |                                                          4 |                                                      -14.8 |                                                          1 |
| catch    |   5 | total_regret                 |                                                       20   |                                                          4 |                                                        5.2 |                                                          3 |
| overall  |  10 | episode_return, total_regret |                                                       25.5 |                                                          8 |                                                       -4.8 |                                                          4 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |                 28 |                 90 |                                                    60 |                                                    -2 |
|      1 | cartpole/0  | cartpole     | episode_return |                   111 |                103 |                 29 |                                                   -82 |                                                    -8 |
|      2 | cartpole/0  | cartpole     | episode_return |                   104 |                109 |                145 |                                                    41 |                                                     5 |
|      3 | cartpole/0  | cartpole     | episode_return |                    98 |                 30 |                137 |                                                    39 |                                                   -68 |
|      4 | cartpole/0  | cartpole     | episode_return |                    31 |                 30 |                128 |                                                    97 |                                                    -1 |
|      0 | catch/0     | catch        | total_regret   |                   156 |                152 |                122 |                                                    34 |                                                     4 |
|      1 | catch/0     | catch        | total_regret   |                   164 |                148 |                168 |                                                    -4 |                                                    16 |
|      2 | catch/0     | catch        | total_regret   |                   158 |                138 |                138 |                                                    20 |                                                    20 |
|      3 | catch/0     | catch        | total_regret   |                   146 |                158 |                138 |                                                     8 |                                                   -12 |
|      4 | catch/0     | catch        | total_regret   |                   164 |                166 |                122 |                                                    42 |                                                    -2 |