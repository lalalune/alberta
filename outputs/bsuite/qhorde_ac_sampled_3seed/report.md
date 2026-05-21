# Step 4 Control Comparison

Metric: `auto`
Baseline agent: `autostep_bottleneck`
Control agents: `autostep_bottleneck, sarsa_bottleneck, qhorde_ac_sampled`

## Summary

| scope    |   n | metrics                      |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_mean |   sarsa_bottleneck_improvement_vs_autostep_bottleneck_wins |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck_mean |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck_wins |
|:---------|----:|:-----------------------------|-----------------------------------------------------------:|-----------------------------------------------------------:|------------------------------------------------------------:|------------------------------------------------------------:|
| cartpole |   3 | episode_return               |                                                  -12.6667  |                                                          1 |                                                   -21       |                                                           1 |
| catch    |   3 | total_regret                 |                                                    6       |                                                          2 |                                                    -2.66667 |                                                           1 |
| overall  |   6 | episode_return, total_regret |                                                   -3.33333 |                                                          3 |                                                   -11.8333  |                                                           2 |

## Paired Final Metrics

|   seed | bsuite_id   | experiment   | metric         |   autostep_bottleneck |   qhorde_ac_sampled |   sarsa_bottleneck |   sarsa_bottleneck_improvement_vs_autostep_bottleneck |   qhorde_ac_sampled_improvement_vs_autostep_bottleneck |
|-------:|:------------|:-------------|:---------------|----------------------:|--------------------:|-------------------:|------------------------------------------------------:|-------------------------------------------------------:|
|      0 | cartpole/0  | cartpole     | episode_return |                    27 |                 104 |                 97 |                                                    70 |                                                     77 |
|      1 | cartpole/0  | cartpole     | episode_return |                    88 |                  88 |                 28 |                                                   -60 |                                                      0 |
|      2 | cartpole/0  | cartpole     | episode_return |                   169 |                  29 |                121 |                                                   -48 |                                                   -140 |
|      0 | catch/0     | catch        | total_regret   |                    42 |                  52 |                 42 |                                                     0 |                                                    -10 |
|      1 | catch/0     | catch        | total_regret   |                    46 |                  50 |                 34 |                                                    12 |                                                     -4 |
|      2 | catch/0     | catch        | total_regret   |                    54 |                  48 |                 48 |                                                     6 |                                                      6 |