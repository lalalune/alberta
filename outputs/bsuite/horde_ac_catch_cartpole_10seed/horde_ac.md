# Step 4 bsuite control comparison

Metric: `auto` (task-specific: lower `total_regret`, higher returns/rewards).
Pairing: same seed and same `bsuite_id`.
Baseline: `autostep_bottleneck`.
Positive improvement means the agent beat the baseline.

## Summary

| experiment | metric | n_pairs | autostep_bottleneck_mean | sarsa_bottleneck_mean | actor_critic_mean | horde_ac_mean | mean_sarsa_bottleneck_improvement_vs_autostep_bottleneck | mean_actor_critic_improvement_vs_autostep_bottleneck | mean_horde_ac_improvement_vs_autostep_bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole | episode_return | 10 | 52.9000 | 61.6000 | 44.8000 | 44.6000 | 8.7000 | -8.1000 | -8.3000 |
| catch | total_regret | 10 | 31.6000 | 30.0000 | 33.2000 | 34.2000 | 1.6000 | -1.6000 | -2.6000 |
| overall | mixed | 20 | 42.2500 | 45.8000 | 39.0000 | 39.4000 | 5.1500 | -4.8500 | -5.4500 |

## Paired Final Metrics

| seed | bsuite_id | experiment | metric | autostep_bottleneck | sarsa_bottleneck | actor_critic | horde_ac | sarsa_bottleneck_improvement_vs_autostep_bottleneck | actor_critic_improvement_vs_autostep_bottleneck | horde_ac_improvement_vs_autostep_bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole/0 | cartpole | episode_return | 33.0000 | 77.0000 | 30.0000 | 29.0000 | 44.0000 | -3.0000 | -4.0000 |
| 1 | cartpole/0 | cartpole | episode_return | 28.0000 | 31.0000 | 32.0000 | 28.0000 | 3.0000 | 4.0000 | 0.0000 |
| 2 | cartpole/0 | cartpole | episode_return | 78.0000 | 121.0000 | 32.0000 | 66.0000 | 43.0000 | -46.0000 | -12.0000 |
| 3 | cartpole/0 | cartpole | episode_return | 29.0000 | 28.0000 | 56.0000 | 54.0000 | -1.0000 | 27.0000 | 25.0000 |
| 4 | cartpole/0 | cartpole | episode_return | 86.0000 | 68.0000 | 67.0000 | 81.0000 | -18.0000 | -19.0000 | -5.0000 |
| 5 | cartpole/0 | cartpole | episode_return | 27.0000 | 177.0000 | 68.0000 | 64.0000 | 150.0000 | 41.0000 | 37.0000 |
| 6 | cartpole/0 | cartpole | episode_return | 101.0000 | 27.0000 | 31.0000 | 34.0000 | -74.0000 | -70.0000 | -67.0000 |
| 7 | cartpole/0 | cartpole | episode_return | 31.0000 | 30.0000 | 28.0000 | 31.0000 | -1.0000 | -3.0000 | 0.0000 |
| 8 | cartpole/0 | cartpole | episode_return | 27.0000 | 29.0000 | 31.0000 | 29.0000 | 2.0000 | 4.0000 | 2.0000 |
| 9 | cartpole/0 | cartpole | episode_return | 89.0000 | 28.0000 | 73.0000 | 30.0000 | -61.0000 | -16.0000 | -59.0000 |
| 0 | catch/0 | catch | total_regret | 32.0000 | 30.0000 | 34.0000 | 34.0000 | 2.0000 | -2.0000 | -2.0000 |
| 1 | catch/0 | catch | total_regret | 26.0000 | 28.0000 | 32.0000 | 34.0000 | -2.0000 | -6.0000 | -8.0000 |
| 2 | catch/0 | catch | total_regret | 22.0000 | 28.0000 | 34.0000 | 32.0000 | -6.0000 | -12.0000 | -10.0000 |
| 3 | catch/0 | catch | total_regret | 38.0000 | 24.0000 | 34.0000 | 32.0000 | 14.0000 | 4.0000 | 6.0000 |
| 4 | catch/0 | catch | total_regret | 26.0000 | 26.0000 | 34.0000 | 34.0000 | 0.0000 | -8.0000 | -8.0000 |
| 5 | catch/0 | catch | total_regret | 38.0000 | 36.0000 | 28.0000 | 38.0000 | 2.0000 | 10.0000 | 0.0000 |
| 6 | catch/0 | catch | total_regret | 34.0000 | 32.0000 | 32.0000 | 34.0000 | 2.0000 | 2.0000 | 0.0000 |
| 7 | catch/0 | catch | total_regret | 32.0000 | 32.0000 | 36.0000 | 36.0000 | 0.0000 | -4.0000 | -4.0000 |
| 8 | catch/0 | catch | total_regret | 32.0000 | 34.0000 | 32.0000 | 38.0000 | -2.0000 | 0.0000 | -6.0000 |
| 9 | catch/0 | catch | total_regret | 36.0000 | 30.0000 | 36.0000 | 30.0000 | 6.0000 | 0.0000 | 6.0000 |
