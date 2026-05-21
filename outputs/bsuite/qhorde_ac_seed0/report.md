# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|----------------------------------------------------:|----------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         48 |                                                          1 |                                                  86 |                                                   1 |
| catch    |   1 | total_regret                 |                                                          6 |                                                          1 |                                                  -6 |                                                   0 |
| overall  |   2 | episode_return, total_regret |                                                         27 |                                                          2 |                                                  40 |                                                   1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|------------:|-------------------:|------------------------------------------------------:|-----------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |         116 |                 78 |                                                    48 |                                             86 |
|      0 | catch/0     | catch        | total_regret   |                    48 |          54 |                 42 |                                                     6 |                                             -6 |