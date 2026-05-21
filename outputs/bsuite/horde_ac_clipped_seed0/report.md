# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, horde_ac_clipped, nlhac_clipped`

## Summary

| scope    |   n | metrics                      |   horde_ac_clipped_improvement_vs_autostep_bottleneck_mean |   horde_ac_clipped_improvement_vs_autostep_bottleneck_wins |   nlhac_clipped_improvement_vs_autostep_bottleneck_mean |   nlhac_clipped_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|--------------------------------------------------------:|--------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         80 |                                                          1 |                                                     158 |                                                       1 |
| catch    |   1 | total_regret                 |                                                          0 |                                                          0 |                                                      -6 |                                                       0 |
| overall  |   2 | episode_return, total_regret |                                                         40 |                                                          1 |                                                      76 |                                                       1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   horde_ac_clipped |   nlhac_clipped |   horde_ac_clipped_improvement_vs_autostep_bottleneck |   nlhac_clipped_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|----------------:|------------------------------------------------------:|---------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                108 |             186 |                                                    80 |                                                158 |
|      0 | catch/0     | catch        | total_regret   |                    48 |                 48 |              54 |                                                     0 |                                                 -6 |