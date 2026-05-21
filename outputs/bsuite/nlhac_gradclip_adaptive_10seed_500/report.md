# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip_adaptive`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|------------------------------------------------------------------:|------------------------------------------------------------------:|
| cartpole |  10 | episode_return               |                                                        5.8 |                                                          7 |                                                              10.6 |                                                                 5 |
| catch    |  10 | total_regret                 |                                                        6.6 |                                                          7 |                                                              -1.4 |                                                                 5 |
| overall  |  20 | episode_return, total_regret |                                                        6.2 |                                                         14 |                                                               4.6 |                                                                10 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_gradclip_adaptive |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|--------------------------:|-------------------:|------------------------------------------------------:|-------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                       150 |                 74 |                                                    46 |                                                          122 |
|      1 | cartpole/0  | cartpole     | episode_return |                   109 |                        28 |                 30 |                                                   -79 |                                                          -81 |
|      2 | cartpole/0  | cartpole     | episode_return |                   104 |                       130 |                 84 |                                                   -20 |                                                           26 |
|      3 | cartpole/0  | cartpole     | episode_return |                    81 |                        28 |                 86 |                                                     5 |                                                          -53 |
|      4 | cartpole/0  | cartpole     | episode_return |                    30 |                        96 |                146 |                                                   116 |                                                           66 |
|      5 | cartpole/0  | cartpole     | episode_return |                    29 |                       116 |                 85 |                                                    56 |                                                           87 |
|      6 | cartpole/0  | cartpole     | episode_return |                    28 |                        27 |                 29 |                                                     1 |                                                           -1 |
|      7 | cartpole/0  | cartpole     | episode_return |                    29 |                        29 |                 32 |                                                     3 |                                                            0 |
|      8 | cartpole/0  | cartpole     | episode_return |                   107 |                        27 |                 31 |                                                   -76 |                                                          -80 |
|      9 | cartpole/0  | cartpole     | episode_return |                    27 |                        47 |                 33 |                                                     6 |                                                           20 |
|      0 | catch/0     | catch        | total_regret   |                    78 |                        72 |                 62 |                                                    16 |                                                            6 |
|      1 | catch/0     | catch        | total_regret   |                    74 |                        78 |                 86 |                                                   -12 |                                                           -4 |
|      2 | catch/0     | catch        | total_regret   |                    90 |                        78 |                 78 |                                                    12 |                                                           12 |
|      3 | catch/0     | catch        | total_regret   |                    74 |                        64 |                 50 |                                                    24 |                                                           10 |
|      4 | catch/0     | catch        | total_regret   |                    80 |                        84 |                 72 |                                                     8 |                                                           -4 |
|      5 | catch/0     | catch        | total_regret   |                    88 |                        78 |                 74 |                                                    14 |                                                           10 |
|      6 | catch/0     | catch        | total_regret   |                    78 |                        76 |                 74 |                                                     4 |                                                            2 |
|      7 | catch/0     | catch        | total_regret   |                    56 |                        80 |                 54 |                                                     2 |                                                          -24 |
|      8 | catch/0     | catch        | total_regret   |                    74 |                        82 |                 76 |                                                    -2 |                                                           -8 |
|      9 | catch/0     | catch        | total_regret   |                    70 |                        84 |                 70 |                                                     0 |                                                          -14 |