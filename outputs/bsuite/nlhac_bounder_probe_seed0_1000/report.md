# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         -4 |                                                          0 |
| catch    |   1 | total_regret                 |                                                        -28 |                                                          0 |
| overall  |   2 | episode_return, total_regret |                                                        -16 |                                                          0 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    31 |                 27 |                                                    -4 |
|      0 | catch/0     | catch        | total_regret   |                   150 |                178 |                                                   -28 |