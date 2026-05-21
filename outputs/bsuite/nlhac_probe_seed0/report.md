# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                       65   |                                                          1 |                                                       61   |                                                          1 |
| catch    |   1 | total_regret                 |                                                        8   |                                                          1 |                                                        8   |                                                          1 |
| overall  |   2 | episode_return, total_regret |                                                       36.5 |                                                          2 |                                                       34.5 |                                                          2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                 89 |                 93 |                                                    65 |                                                    61 |
|      0 | catch/0     | catch        | total_regret   |                    52 |                 44 |                 44 |                                                     8 |                                                     8 |