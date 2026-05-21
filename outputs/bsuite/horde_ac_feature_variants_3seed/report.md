# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, horde_ac_tuned, horde_ac_history`

## Summary

| scope    |   n | metrics                      |   horde_ac_tuned_improvement_vs_autostep_bottleneck_mean |   horde_ac_tuned_improvement_vs_autostep_bottleneck_wins |   horde_ac_history_improvement_vs_autostep_bottleneck_mean |   horde_ac_history_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|---------------------------------------------------------:|---------------------------------------------------------:|-----------------------------------------------------------:|-----------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                -18.3333  |                                                        1 |                                                 -37.6667   |                                                          1 |
| catch    |   3 | total_regret                 |                                                 -2.66667 |                                                        0 |                                                   0.666667 |                                                          1 |
| overall  |   6 | episode_return, total_regret |                                                -10.5     |                                                        1 |                                                 -18.5      |                                                          2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   horde_ac_history |   horde_ac_tuned |   horde_ac_tuned_improvement_vs_autostep_bottleneck |   horde_ac_history_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|-------------------:|-----------------:|----------------------------------------------------:|------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    27 |                 29 |               65 |                                                  38 |                                                     2 |
|      1 | cartpole/0  | cartpole     | episode_return |                   117 |                 87 |               77 |                                                 -40 |                                                   -30 |
|      2 | cartpole/0  | cartpole     | episode_return |                   112 |                 27 |               59 |                                                 -53 |                                                   -85 |
|      0 | catch/0     | catch        | total_regret   |                    42 |                 48 |               48 |                                                  -6 |                                                    -6 |
|      1 | catch/0     | catch        | total_regret   |                    54 |                 42 |               54 |                                                   0 |                                                    12 |
|      2 | catch/0     | catch        | total_regret   |                    46 |                 50 |               48 |                                                  -2 |                                                    -4 |