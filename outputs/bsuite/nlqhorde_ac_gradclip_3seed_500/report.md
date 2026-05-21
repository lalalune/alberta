# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlqhorde_ac_gradclip`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlqhorde_ac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlqhorde_ac_gradclip_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------------:|---------------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                    2       |                                                          2 |                                                       -1.66667 |                                                              2 |
| catch    |   3 | total_regret                 |                                                   16.6667  |                                                          3 |                                                       -3.33333 |                                                              0 |
| overall  |   6 | episode_return, total_regret |                                                    9.33333 |                                                          5 |                                                       -2.5     |                                                              2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlqhorde_ac_gradclip |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlqhorde_ac_gradclip_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-----------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    27 |                     81 |                111 |                                                    84 |                                                        54 |
|      1 | cartpole/0  | cartpole     | episode_return |                   109 |                     28 |                 28 |                                                   -81 |                                                       -81 |
|      2 | cartpole/0  | cartpole     | episode_return |                    95 |                    117 |                 98 |                                                     3 |                                                        22 |
|      0 | catch/0     | catch        | total_regret   |                    82 |                     88 |                 62 |                                                    20 |                                                        -6 |
|      1 | catch/0     | catch        | total_regret   |                    78 |                     80 |                 74 |                                                     4 |                                                        -2 |
|      2 | catch/0     | catch        | total_regret   |                    74 |                     76 |                 48 |                                                    26 |                                                        -2 |