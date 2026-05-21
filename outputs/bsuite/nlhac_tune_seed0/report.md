# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, nlhac_bottleneck, nlhac_as03, nlhac_as10`

## Summary

| scope    |   n | metrics                      |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_as03_improvement_vs_autostep_bottleneck_mean |   nlhac_as03_improvement_vs_autostep_bottleneck_wins |   nlhac_as10_improvement_vs_autostep_bottleneck_mean |   nlhac_as10_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|-----------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                       61   |                                                          1 |                                                    2 |                                                    1 |                                                    0 |                                                    0 |
| catch    |   1 | total_regret                 |                                                       -6   |                                                          0 |                                                   -8 |                                                    0 |                                                  -10 |                                                    0 |
| overall  |   2 | episode_return, total_regret |                                                       27.5 |                                                          1 |                                                   -3 |                                                    1 |                                                   -5 |                                                    0 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_as03 |   nlhac_as10 |   nlhac_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_as03_improvement_vs_autostep_bottleneck |   nlhac_as10_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------:|-------------:|-------------------:|------------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |           30 |           28 |                 89 |                                                    61 |                                               2 |                                               0 |
|      0 | catch/0     | catch        | total_regret   |                    40 |           48 |           50 |                 46 |                                                    -6 |                                              -8 |                                             -10 |