# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_bottleneck`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_mean |   nlhac_bottleneck_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |  10 | episode_return               |                                                        5   |                                                          5 |                                                     -14.1  |                                                          3 |
| catch    |  10 | total_regret                 |                                                        1.8 |                                                          6 |                                                      -1    |                                                          4 |
| overall  |  20 | episode_return, total_regret |                                                        3.4 |                                                         11 |                                                      -7.55 |                                                          7 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_bottleneck |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_bottleneck_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    29 |                 95 |                100 |                                                    71 |                                                    66 |
|      1 | cartpole/0  | cartpole     | episode_return |                    93 |                 28 |                 29 |                                                   -64 |                                                   -65 |
|      2 | cartpole/0  | cartpole     | episode_return |                   166 |                109 |                 82 |                                                   -84 |                                                   -57 |
|      3 | cartpole/0  | cartpole     | episode_return |                   122 |                 27 |                125 |                                                     3 |                                                   -95 |
|      4 | cartpole/0  | cartpole     | episode_return |                    30 |                 29 |                133 |                                                   103 |                                                    -1 |
|      5 | cartpole/0  | cartpole     | episode_return |                    30 |                 28 |                 95 |                                                    65 |                                                    -2 |
|      6 | cartpole/0  | cartpole     | episode_return |                    32 |                 31 |                 30 |                                                    -2 |                                                    -1 |
|      7 | cartpole/0  | cartpole     | episode_return |                    27 |                 29 |                 29 |                                                     2 |                                                     2 |
|      8 | cartpole/0  | cartpole     | episode_return |                    73 |                 27 |                 30 |                                                   -43 |                                                   -46 |
|      9 | cartpole/0  | cartpole     | episode_return |                    29 |                 87 |                 28 |                                                    -1 |                                                    58 |
|      0 | catch/0     | catch        | total_regret   |                    32 |                 30 |                 24 |                                                     8 |                                                     2 |
|      1 | catch/0     | catch        | total_regret   |                    28 |                 32 |                 36 |                                                    -8 |                                                    -4 |
|      2 | catch/0     | catch        | total_regret   |                    32 |                 32 |                 30 |                                                     2 |                                                     0 |
|      3 | catch/0     | catch        | total_regret   |                    30 |                 32 |                 28 |                                                     2 |                                                    -2 |
|      4 | catch/0     | catch        | total_regret   |                    36 |                 32 |                 24 |                                                    12 |                                                     4 |
|      5 | catch/0     | catch        | total_regret   |                    30 |                 24 |                 32 |                                                    -2 |                                                     6 |
|      6 | catch/0     | catch        | total_regret   |                    30 |                 32 |                 38 |                                                    -8 |                                                    -2 |
|      7 | catch/0     | catch        | total_regret   |                    20 |                 34 |                 26 |                                                    -6 |                                                   -14 |
|      8 | catch/0     | catch        | total_regret   |                    36 |                 34 |                 30 |                                                     6 |                                                     2 |
|      9 | catch/0     | catch        | total_regret   |                    36 |                 38 |                 24 |                                                    12 |                                                    -2 |