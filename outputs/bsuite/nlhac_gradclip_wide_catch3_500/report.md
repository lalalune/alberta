# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip, nlhac_gradclip_wide`

## Summary

| scope   |   n | metrics      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_wide_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_wide_improvement_vs_autostep_bottleneck_wins |
|:--------|----:|:-------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|--------------------------------------------------------------:|--------------------------------------------------------------:|
| catch   |   3 | total_regret |                                                    9.33333 |                                                          3 |                                                -0.666667 |                                                        1 |                                                            -2 |                                                             1 |
| overall |   3 | total_regret |                                                    9.33333 |                                                          3 |                                                -0.666667 |                                                        1 |                                                            -2 |                                                             1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric       |   autostep_bottleneck |   nlhac_gradclip |   nlhac_gradclip_wide |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |   nlhac_gradclip_wide_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:-------------|----------------------:|-----------------:|----------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|---------------------------------------------------------:|
|      0 | catch/0     | catch        | total_regret |                    78 |               72 |                    86 |                 72 |                                                     6 |                                                   6 |                                                       -8 |
|      1 | catch/0     | catch        | total_regret |                    86 |               90 |                    82 |                 80 |                                                     6 |                                                  -4 |                                                        4 |
|      2 | catch/0     | catch        | total_regret |                    76 |               80 |                    78 |                 60 |                                                    16 |                                                  -4 |                                                       -2 |