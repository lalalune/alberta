# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac_expected_adv_pairwise`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_expected_adv_pairwise_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_expected_adv_pairwise_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|--------------------------------------------------------------------------:|--------------------------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                         60 |                                                          1 |                                                                      -3   |                                                                         0 |
| catch    |   1 | total_regret                 |                                                         -2 |                                                          0 |                                                                       0   |                                                                         0 |
| overall  |   2 | episode_return, total_regret |                                                         29 |                                                          1 |                                                                      -1.5 |                                                                         0 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac_expected_adv_pairwise |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_expected_adv_pairwise_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|----------------------------------:|-------------------:|------------------------------------------------------:|---------------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    30 |                                27 |                 90 |                                                    60 |                                                                   -3 |
|      0 | catch/0     | catch        | total_regret   |                    44 |                                44 |                 46 |                                                    -2 |                                                                    0 |