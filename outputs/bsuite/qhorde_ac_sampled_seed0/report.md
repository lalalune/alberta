# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac_sampled`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|------------------------------------------------------------:|------------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         84 |                                                          1 |                                                          78 |                                                           1 |
| catch    |   1 | total_regret                 |                                                         14 |                                                          1 |                                                           6 |                                                           1 |
| overall  |   2 | episode_return, total_regret |                                                         49 |                                                          2 |                                                          42 |                                                           2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac_sampled |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|--------------------:|-------------------:|------------------------------------------------------:|-------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                 106 |                112 |                                                    84 |                                                     78 |
|      0 | catch/0     | catch        | total_regret   |                    46 |                  40 |                 32 |                                                    14 |                                                      6 |