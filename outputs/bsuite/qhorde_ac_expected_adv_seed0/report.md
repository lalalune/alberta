# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac_expected_adv`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------------:|-----------------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                       55   |                                                          1 |                                                             75   |                                                                1 |
| catch    |   1 | total_regret                 |                                                       10   |                                                          1 |                                                             18   |                                                                1 |
| overall  |   2 | episode_return, total_regret |                                                       32.5 |                                                          2 |                                                             46.5 |                                                                2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac_expected_adv |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                      103 |                 83 |                                                    55 |                                                          75 |
|      0 | catch/0     | catch        | total_regret   |                    56 |                       38 |                 46 |                                                    10 |                                                          18 |