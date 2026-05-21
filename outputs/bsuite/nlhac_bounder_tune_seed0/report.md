# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, nlhac_bottleneck, nlhac_as03, nlhac_as10`

## Summary

| scope    |   n | metrics                      |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_as03_improvement_vs_autostep_bottleneck_mean |   nlhac_as03_improvement_vs_autostep_bottleneck_wins |   nlhac_as10_improvement_vs_autostep_bottleneck_mean |   nlhac_as10_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         56 |                                                          1 |                                                 -1   |                                                    0 |                                                 -1   |                                                    0 |
| catch    |   1 | total_regret                 |                                                         -2 |                                                          0 |                                                 -6   |                                                    0 |                                                  6   |                                                    1 |
| overall  |   2 | episode_return, total_regret |                                                         27 |                                                          1 |                                                 -3.5 |                                                    0 |                                                  2.5 |                                                    1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_as03 |   nlhac_as10 |   nlhac_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_as03_improvement_vs_autostep_bottleneck |   nlhac_as10_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------:|-------------:|-------------------:|------------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    29 |           28 |           28 |                 85 |                                                    56 |                                              -1 |                                              -1 |
|      0 | catch/0     | catch        | total_regret   |                    50 |           56 |           44 |                 52 |                                                    -2 |                                              -6 |                                               6 |