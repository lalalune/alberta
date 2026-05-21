# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac_expected_adv`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------------:|-----------------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                  -0.333333 |                                                          2 |                                                       -28        |                                                                1 |
| catch    |   3 | total_regret                 |                                                   0        |                                                          1 |                                                         0.666667 |                                                                1 |
| overall  |   6 | episode_return, total_regret |                                                  -0.166667 |                                                          3 |                                                       -13.6667   |                                                                2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac_expected_adv |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_expected_adv_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------------:|-------------------:|------------------------------------------------------:|------------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                      116 |                117 |                                                    89 |                                                          88 |
|      1 | cartpole/0  | cartpole     | episode_return |                   121 |                       27 |                 27 |                                                   -94 |                                                         -94 |
|      2 | cartpole/0  | cartpole     | episode_return |                   106 |                       28 |                110 |                                                     4 |                                                         -78 |
|      0 | catch/0     | catch        | total_regret   |                    46 |                       48 |                 52 |                                                    -6 |                                                          -2 |
|      1 | catch/0     | catch        | total_regret   |                    52 |                       52 |                 46 |                                                     6 |                                                           0 |
|      2 | catch/0     | catch        | total_regret   |                    46 |                       42 |                 46 |                                                     0 |                                                           4 |