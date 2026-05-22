# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip, nlhac_gradclip_tdnorm`

## Summary

| scope   |   n | metrics      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_tdnorm_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_tdnorm_improvement_vs_autostep_bottleneck_wins |
|:--------|----:|:-------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|----------------------------------------------------------------:|----------------------------------------------------------------:|
| catch   |   3 | total_regret |                                                    5.33333 |                                                          3 |                                                        4 |                                                        3 |                                                        0.666667 |                                                               1 |
| overall |   3 | total_regret |                                                    5.33333 |                                                          3 |                                                        4 |                                                        3 |                                                        0.666667 |                                                               1 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric       |   autostep_bottleneck |   nlhac_gradclip |   nlhac_gradclip_tdnorm |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |   nlhac_gradclip_tdnorm_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:-------------|----------------------:|-----------------:|------------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|-----------------------------------------------------------:|
|      0 | catch/0     | catch        | total_regret |                    82 |               76 |                      86 |                 78 |                                                     4 |                                                   6 |                                                         -4 |
|      1 | catch/0     | catch        | total_regret |                    82 |               78 |                      84 |                 76 |                                                     6 |                                                   4 |                                                         -2 |
|      2 | catch/0     | catch        | total_regret |                    84 |               82 |                      76 |                 78 |                                                     6 |                                                   2 |                                                          8 |