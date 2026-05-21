# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|
| cartpole |  10 | episode_return               |                                                      -2.1  |                                                          5 |                                                      4.2 |                                                        5 |
| catch    |  10 | total_regret                 |                                                      13.8  |                                                          9 |                                                      2.6 |                                                        5 |
| overall  |  20 | episode_return, total_regret |                                                       5.85 |                                                         14 |                                                      3.4 |                                                       10 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_gradclip |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-----------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    31 |               94 |                102 |                                                    71 |                                                  63 |
|      1 | cartpole/0  | cartpole     | episode_return |                    83 |               29 |                 30 |                                                   -53 |                                                 -54 |
|      2 | cartpole/0  | cartpole     | episode_return |                   181 |              151 |                 97 |                                                   -84 |                                                 -30 |
|      3 | cartpole/0  | cartpole     | episode_return |                    91 |               30 |                 91 |                                                     0 |                                                 -61 |
|      4 | cartpole/0  | cartpole     | episode_return |                    29 |               95 |                 73 |                                                    44 |                                                  66 |
|      5 | cartpole/0  | cartpole     | episode_return |                    29 |               95 |                127 |                                                    98 |                                                  66 |
|      6 | cartpole/0  | cartpole     | episode_return |                    31 |               27 |                 28 |                                                    -3 |                                                  -4 |
|      7 | cartpole/0  | cartpole     | episode_return |                    27 |               30 |                 32 |                                                     5 |                                                   3 |
|      8 | cartpole/0  | cartpole     | episode_return |                   129 |               35 |                 29 |                                                  -100 |                                                 -94 |
|      9 | cartpole/0  | cartpole     | episode_return |                    28 |              115 |                 29 |                                                     1 |                                                  87 |
|      0 | catch/0     | catch        | total_regret   |                   150 |              166 |                126 |                                                    24 |                                                 -16 |
|      1 | catch/0     | catch        | total_regret   |                   176 |              152 |                148 |                                                    28 |                                                  24 |
|      2 | catch/0     | catch        | total_regret   |                   158 |              168 |                142 |                                                    16 |                                                 -10 |
|      3 | catch/0     | catch        | total_regret   |                   144 |              160 |                136 |                                                     8 |                                                 -16 |
|      4 | catch/0     | catch        | total_regret   |                   174 |              150 |                148 |                                                    26 |                                                  24 |
|      5 | catch/0     | catch        | total_regret   |                   140 |              154 |                140 |                                                     0 |                                                 -14 |
|      6 | catch/0     | catch        | total_regret   |                   146 |              144 |                140 |                                                     6 |                                                   2 |
|      7 | catch/0     | catch        | total_regret   |                   140 |              150 |                134 |                                                     6 |                                                 -10 |
|      8 | catch/0     | catch        | total_regret   |                   164 |              146 |                142 |                                                    22 |                                                  18 |
|      9 | catch/0     | catch        | total_regret   |                   166 |              142 |                164 |                                                     2 |                                                  24 |