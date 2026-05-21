# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip_adaptive`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|------------------------------------------------------------------:|------------------------------------------------------------------:|
| cartpole |  10 | episode_return               |                                                      17.9  |                                                          5 |                                                              22.2 |                                                                 5 |
| catch    |  10 | total_regret                 |                                                      13.4  |                                                          7 |                                                              -6   |                                                                 2 |
| overall  |  20 | episode_return, total_regret |                                                      15.65 |                                                         12 |                                                               8.1 |                                                                 7 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_gradclip_adaptive |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|--------------------------:|-------------------:|------------------------------------------------------:|-------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |                       219 |                 93 |                                                    63 |                                                          189 |
|      1 | cartpole/0  | cartpole     | episode_return |                    72 |                        28 |                 31 |                                                   -41 |                                                          -44 |
|      2 | cartpole/0  | cartpole     | episode_return |                    78 |                        95 |                130 |                                                    52 |                                                           17 |
|      3 | cartpole/0  | cartpole     | episode_return |                   111 |                        28 |                109 |                                                    -2 |                                                          -83 |
|      4 | cartpole/0  | cartpole     | episode_return |                    31 |                        81 |                 72 |                                                    41 |                                                           50 |
|      5 | cartpole/0  | cartpole     | episode_return |                    30 |                       111 |                145 |                                                   115 |                                                           81 |
|      6 | cartpole/0  | cartpole     | episode_return |                    30 |                        30 |                 28 |                                                    -2 |                                                            0 |
|      7 | cartpole/0  | cartpole     | episode_return |                    31 |                        28 |                 32 |                                                     1 |                                                           -3 |
|      8 | cartpole/0  | cartpole     | episode_return |                    76 |                        33 |                 29 |                                                   -47 |                                                          -43 |
|      9 | cartpole/0  | cartpole     | episode_return |                    30 |                        88 |                 29 |                                                    -1 |                                                           58 |
|      0 | catch/0     | catch        | total_regret   |                   152 |                       158 |                118 |                                                    34 |                                                           -6 |
|      1 | catch/0     | catch        | total_regret   |                   144 |                       146 |                134 |                                                    10 |                                                           -2 |
|      2 | catch/0     | catch        | total_regret   |                   144 |                       168 |                132 |                                                    12 |                                                          -24 |
|      3 | catch/0     | catch        | total_regret   |                   150 |                       166 |                152 |                                                    -2 |                                                          -16 |
|      4 | catch/0     | catch        | total_regret   |                   138 |                       168 |                138 |                                                     0 |                                                          -30 |
|      5 | catch/0     | catch        | total_regret   |                   170 |                       140 |                138 |                                                    32 |                                                           30 |
|      6 | catch/0     | catch        | total_regret   |                   140 |                       154 |                166 |                                                   -26 |                                                          -14 |
|      7 | catch/0     | catch        | total_regret   |                   138 |                       158 |                126 |                                                    12 |                                                          -20 |
|      8 | catch/0     | catch        | total_regret   |                   158 |                       164 |                134 |                                                    24 |                                                           -6 |
|      9 | catch/0     | catch        | total_regret   |                   174 |                       146 |                136 |                                                    38 |                                                           28 |