# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_t20_as001_clip1`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_t20_as001_clip1_improvement_vs_autostep_bottleneck_mean |   qhorde_t20_as001_clip1_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------------:|-----------------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                    22.3333 |                                                          2 |                                                        -12.6667  |                                                                1 |
| catch    |   3 | total_regret                 |                                                     2      |                                                          2 |                                                          0       |                                                                1 |
| overall  |   6 | episode_return, total_regret |                                                    12.1667 |                                                          4 |                                                         -6.33333 |                                                                2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_t20_as001_clip1 |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_t20_as001_clip1_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                       62 |                104 |                                                    76 |                                                          34 |
|      1 | cartpole/0  | cartpole     | episode_return |                   112 |                       74 |                 29 |                                                   -83 |                                                         -38 |
|      2 | cartpole/0  | cartpole     | episode_return |                    75 |                       41 |                149 |                                                    74 |                                                         -34 |
|      0 | catch/0     | catch        | total_regret   |                    46 |                       46 |                 48 |                                                    -2 |                                                           0 |
|      1 | catch/0     | catch        | total_regret   |                    46 |                       48 |                 42 |                                                     4 |                                                          -2 |
|      2 | catch/0     | catch        | total_regret   |                    50 |                       48 |                 46 |                                                     4 |                                                           2 |