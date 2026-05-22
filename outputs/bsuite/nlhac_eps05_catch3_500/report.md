# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip, nlhac_gradclip_eps05`

## Summary

| scope   |   n | metrics      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_eps05_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_eps05_improvement_vs_autostep_bottleneck_wins |
|:--------|----:|:-------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------------:|---------------------------------------------------------------:|
| catch   |   3 | total_regret |                                                    3.33333 |                                                          2 |                                                      -10 |                                                        0 |                                                             -8 |                                                              0 |
| overall |   3 | total_regret |                                                    3.33333 |                                                          2 |                                                      -10 |                                                        0 |                                                             -8 |                                                              0 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric       |   autostep_bottleneck |   nlhac_gradclip |   nlhac_gradclip_eps05 |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |   nlhac_gradclip_eps05_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:-------------|----------------------:|-----------------:|-----------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|----------------------------------------------------------:|
|      0 | catch/0     | catch        | total_regret |                    74 |               82 |                     76 |                 64 |                                                    10 |                                                  -8 |                                                        -2 |
|      1 | catch/0     | catch        | total_regret |                    68 |               74 |                     82 |                 74 |                                                    -6 |                                                  -6 |                                                       -14 |
|      2 | catch/0     | catch        | total_regret |                    76 |               92 |                     84 |                 70 |                                                     6 |                                                 -16 |                                                        -8 |