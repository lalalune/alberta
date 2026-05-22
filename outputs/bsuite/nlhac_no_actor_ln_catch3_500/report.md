# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, nlhac_gradclip, nlhac_gradclip_no_actor_ln`

## Summary

| scope   |   n | metrics      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_improvement_vs_autostep_bottleneck_wins |   nlhac_gradclip_no_actor_ln_improvement_vs_autostep_bottleneck_mean |   nlhac_gradclip_no_actor_ln_improvement_vs_autostep_bottleneck_wins |
|:--------|----:|:-------------|-----------------------------------------------------------:|-----------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------:|---------------------------------------------------------------------:|---------------------------------------------------------------------:|
| catch   |   3 | total_regret |                                                          2 |                                                          2 |                                                  1.33333 |                                                        1 |                                                                  -14 |                                                                    0 |
| overall |   3 | total_regret |                                                          2 |                                                          2 |                                                  1.33333 |                                                        1 |                                                                  -14 |                                                                    0 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric       |   autostep_bottleneck |   nlhac_gradclip |   nlhac_gradclip_no_actor_ln |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   nlhac_gradclip_improvement_vs_autostep_bottleneck |   nlhac_gradclip_no_actor_ln_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:-------------|----------------------:|-----------------:|-----------------------------:|-------------------:|------------------------------------------------------:|----------------------------------------------------:|----------------------------------------------------------------:|
|      0 | catch/0     | catch        | total_regret |                    68 |               70 |                           86 |                 80 |                                                   -12 |                                                  -2 |                                                             -18 |
|      1 | catch/0     | catch        | total_regret |                    72 |               78 |                           88 |                 68 |                                                     4 |                                                  -6 |                                                             -16 |
|      2 | catch/0     | catch        | total_regret |                    80 |               68 |                           88 |                 66 |                                                    14 |                                                  12 |                                                              -8 |