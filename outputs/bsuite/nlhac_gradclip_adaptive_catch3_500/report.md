# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip, nlhac_gradclip_adaptive`

## Summary

| scope   |   n | metrics      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck_wins |
|:--------|----:|:-------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|------------------------------------------------------------------:|------------------------------------------------------------------:|
| catch   |   3 | total_regret |                                                         10 |                                                          3 |                                                 0.666667 |                                                        2 |                                                           5.33333 |                                                                 1 |
| overall |   3 | total_regret |                                                         10 |                                                          3 |                                                 0.666667 |                                                        2 |                                                           5.33333 |                                                                 1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric       |   autostep_bottleneck |   nlhac_gradclip |   nlhac_gradclip_adaptive |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |   nlhac_gradclip_adaptive_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:-------------|----------------------:|-----------------:|--------------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|-------------------------------------------------------------:|
|      0 | catch/0     | catch        | total_regret |                    82 |               80 |                        84 |                 74 |                                                     8 |                                                   2 |                                                           -2 |
|      1 | catch/0     | catch        | total_regret |                    88 |               82 |                        70 |                 70 |                                                    18 |                                                   6 |                                                           18 |
|      2 | catch/0     | catch        | total_regret |                    78 |               84 |                        78 |                 74 |                                                     4 |                                                  -6 |                                                            0 |