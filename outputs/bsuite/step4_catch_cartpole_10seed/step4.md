# Step 4 bsuite control comparison

Metric: `auto` (task-specific: lower `total_regret`, higher returns/rewards).
Pairing: same seed and same `bsuite_id`.
Baseline: `autostep`.
Positive improvement means the agent beat the baseline.

## Summary

| experiment | metric | n_pairs | autostep_mean | sarsa_mean | actor_critic_mean | mean_sarsa_improvement_vs_autostep | mean_actor_critic_improvement_vs_autostep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole | episode_return | 10 | 74.5000 | 67.7000 | 35.3000 | -6.8000 | -39.2000 |
| catch | total_regret | 10 | 233.8000 | 246.6000 | 310.6000 | -12.8000 | -76.8000 |
| overall | mixed | 20 | 154.1500 | 157.1500 | 172.9500 | -9.8000 | -58.0000 |

## Paired Final Metrics

| seed | bsuite_id | experiment | metric | autostep | sarsa | actor_critic | sarsa_improvement_vs_autostep | actor_critic_improvement_vs_autostep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole/0 | cartpole | episode_return | 115.0000 | 95.0000 | 28.0000 | -20.0000 | -87.0000 |
| 1 | cartpole/0 | cartpole | episode_return | 34.0000 | 32.0000 | 97.0000 | -2.0000 | 63.0000 |
| 2 | cartpole/0 | cartpole | episode_return | 131.0000 | 93.0000 | 29.0000 | -38.0000 | -102.0000 |
| 3 | cartpole/0 | cartpole | episode_return | 82.0000 | 101.0000 | 28.0000 | 19.0000 | -54.0000 |
| 4 | cartpole/0 | cartpole | episode_return | 32.0000 | 38.0000 | 27.0000 | 6.0000 | -5.0000 |
| 5 | cartpole/0 | cartpole | episode_return | 107.0000 | 122.0000 | 28.0000 | 15.0000 | -79.0000 |
| 6 | cartpole/0 | cartpole | episode_return | 30.0000 | 98.0000 | 30.0000 | 68.0000 | 0.0000 |
| 7 | cartpole/0 | cartpole | episode_return | 34.0000 | 29.0000 | 28.0000 | -5.0000 | -6.0000 |
| 8 | cartpole/0 | cartpole | episode_return | 107.0000 | 28.0000 | 30.0000 | -79.0000 | -77.0000 |
| 9 | cartpole/0 | cartpole | episode_return | 73.0000 | 41.0000 | 28.0000 | -32.0000 | -45.0000 |
| 0 | catch/0 | catch | total_regret | 236.0000 | 278.0000 | 314.0000 | -42.0000 | -78.0000 |
| 1 | catch/0 | catch | total_regret | 274.0000 | 246.0000 | 302.0000 | 28.0000 | -28.0000 |
| 2 | catch/0 | catch | total_regret | 228.0000 | 162.0000 | 306.0000 | 66.0000 | -78.0000 |
| 3 | catch/0 | catch | total_regret | 150.0000 | 332.0000 | 312.0000 | -182.0000 | -162.0000 |
| 4 | catch/0 | catch | total_regret | 296.0000 | 194.0000 | 318.0000 | 102.0000 | -22.0000 |
| 5 | catch/0 | catch | total_regret | 270.0000 | 278.0000 | 318.0000 | -8.0000 | -48.0000 |
| 6 | catch/0 | catch | total_regret | 232.0000 | 246.0000 | 310.0000 | -14.0000 | -78.0000 |
| 7 | catch/0 | catch | total_regret | 196.0000 | 214.0000 | 310.0000 | -18.0000 | -114.0000 |
| 8 | catch/0 | catch | total_regret | 210.0000 | 296.0000 | 302.0000 | -86.0000 | -92.0000 |
| 9 | catch/0 | catch | total_regret | 246.0000 | 220.0000 | 314.0000 | 26.0000 | -68.0000 |
