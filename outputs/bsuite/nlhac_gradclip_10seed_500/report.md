# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|
| cartpole |  10 | episode_return               |                                                      17.1  |                                                          6 |                                                      6.4 |                                                        7 |
| catch    |  10 | total_regret                 |                                                      11.8  |                                                         10 |                                                      2.2 |                                                        5 |
| overall  |  20 | episode_return, total_regret |                                                      14.45 |                                                         16 |                                                      4.3 |                                                       12 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_gradclip |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-----------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |               60 |                106 |                                                    76 |                                                  30 |
|      1 | cartpole/0  | cartpole     | episode_return |                   110 |               27 |                 28 |                                                   -82 |                                                 -83 |
|      2 | cartpole/0  | cartpole     | episode_return |                    89 |              101 |                104 |                                                    15 |                                                  12 |
|      3 | cartpole/0  | cartpole     | episode_return |                   103 |               27 |                105 |                                                     2 |                                                 -76 |
|      4 | cartpole/0  | cartpole     | episode_return |                    30 |               87 |                180 |                                                   150 |                                                  57 |
|      5 | cartpole/0  | cartpole     | episode_return |                    31 |               96 |                102 |                                                    71 |                                                  65 |
|      6 | cartpole/0  | cartpole     | episode_return |                    29 |               30 |                 29 |                                                     0 |                                                   1 |
|      7 | cartpole/0  | cartpole     | episode_return |                    27 |               30 |                 33 |                                                     6 |                                                   3 |
|      8 | cartpole/0  | cartpole     | episode_return |                    96 |               30 |                 29 |                                                   -67 |                                                 -66 |
|      9 | cartpole/0  | cartpole     | episode_return |                    29 |              150 |                 29 |                                                     0 |                                                 121 |
|      0 | catch/0     | catch        | total_regret   |                    78 |               80 |                 68 |                                                    10 |                                                  -2 |
|      1 | catch/0     | catch        | total_regret   |                    84 |               72 |                 82 |                                                     2 |                                                  12 |
|      2 | catch/0     | catch        | total_regret   |                    74 |               80 |                 60 |                                                    14 |                                                  -6 |
|      3 | catch/0     | catch        | total_regret   |                    74 |               74 |                 58 |                                                    16 |                                                   0 |
|      4 | catch/0     | catch        | total_regret   |                    82 |               94 |                 64 |                                                    18 |                                                 -12 |
|      5 | catch/0     | catch        | total_regret   |                    86 |               68 |                 62 |                                                    24 |                                                  18 |
|      6 | catch/0     | catch        | total_regret   |                    76 |               80 |                 74 |                                                     2 |                                                  -4 |
|      7 | catch/0     | catch        | total_regret   |                    86 |               80 |                 68 |                                                    18 |                                                   6 |
|      8 | catch/0     | catch        | total_regret   |                    86 |               80 |                 80 |                                                     6 |                                                   6 |
|      9 | catch/0     | catch        | total_regret   |                    82 |               78 |                 74 |                                                     8 |                                                   4 |