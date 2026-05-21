# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, horde_ac_tuned, horde_ac_pairwise, horde_ac_history`

## Summary

| scope    |   n | metrics                      |   horde_ac_tuned_improvement_vs_autostep_bottleneck_mean |   horde_ac_tuned_improvement_vs_autostep_bottleneck_wins |   horde_ac_pairwise_improvement_vs_autostep_bottleneck_mean |   horde_ac_pairwise_improvement_vs_autostep_bottleneck_wins |   horde_ac_history_improvement_vs_autostep_bottleneck_mean |   horde_ac_history_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|---------------------------------------------------------:|---------------------------------------------------------:|------------------------------------------------------------:|------------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   1 | episode_return               |                                                       52 |                                                        1 |                                                          42 |                                                           1 |                                                          2 |                                                          1 |
| catch    |   1 | total_regret                 |                                                        2 |                                                        1 |                                                           0 |                                                           0 |                                                          6 |                                                          1 |
| overall  |   2 | episode_return, total_regret |                                                       27 |                                                        2 |                                                          21 |                                                           1 |                                                          4 |                                                          2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   horde_ac_history |   horde_ac_pairwise |   horde_ac_tuned |   horde_ac_tuned_improvement_vs_autostep_bottleneck |   horde_ac_pairwise_improvement_vs_autostep_bottleneck |   horde_ac_history_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|--------------------:|-----------------:|----------------------------------------------------:|-------------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    28 |                 30 |                  70 |               80 |                                                  52 |                                                     42 |                                                     2 |
|      0 | catch/0     | catch        | total_regret   |                    52 |                 46 |                  52 |               50 |                                                   2 |                                                      0 |                                                     6 |