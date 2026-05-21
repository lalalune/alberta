# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                   48       |                                                          3 |
| catch    |   3 | total_regret                 |                                                    1.33333 |                                                          1 |
| overall  |   6 | episode_return, total_regret |                                                   24.6667  |                                                          4 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |                 97 |                                                    67 |
|      1 | cartpole/0  | cartpole     | episode_return |                    85 |                145 |                                                    60 |
|      2 | cartpole/0  | cartpole     | episode_return |                    98 |                115 |                                                    17 |
|      0 | catch/0     | catch        | total_regret   |                    44 |                 46 |                                                    -2 |
|      1 | catch/0     | catch        | total_regret   |                    52 |                 52 |                                                     0 |
|      2 | catch/0     | catch        | total_regret   |                    50 |                 44 |                                                     6 |