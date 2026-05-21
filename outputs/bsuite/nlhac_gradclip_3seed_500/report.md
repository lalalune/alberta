# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                    11      |                                                          2 |                                                25.6667   |                                                        2 |
| catch    |   3 | total_regret                 |                                                    23.3333 |                                                          3 |                                                -0.666667 |                                                        1 |
| overall  |   6 | episode_return, total_regret |                                                    17.1667 |                                                          5 |                                                12.5      |                                                        3 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   nlhac_gradclip |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-----------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    29 |              106 |                 90 |                                                    61 |                                                  77 |
|      1 | cartpole/0  | cartpole     | episode_return |                    72 |               29 |                 30 |                                                   -42 |                                                 -43 |
|      2 | cartpole/0  | cartpole     | episode_return |                    68 |              111 |                 82 |                                                    14 |                                                  43 |
|      0 | catch/0     | catch        | total_regret   |                    88 |               88 |                 64 |                                                    24 |                                                   0 |
|      1 | catch/0     | catch        | total_regret   |                    80 |               86 |                 68 |                                                    12 |                                                  -6 |
|      2 | catch/0     | catch        | total_regret   |                    86 |               82 |                 52 |                                                    34 |                                                   4 |