# Step 4 bsuite control comparison

Metric: `episode_return` (higher is better).
Pairing: same seed and same `bsuite_id`.
Baseline: `autostep`.

## Summary

| experiment | n_pairs | autostep_mean | sarsa_mean | actor_critic_mean | mean_sarsa_improvement_vs_autostep | mean_actor_critic_improvement_vs_autostep |
| --- | --- | --- | --- | --- | --- | --- |
| cartpole | 10 | 74.5000 | 67.7000 | 35.3000 | -6.8000 | -39.2000 |
| overall | 10 | 74.5000 | 67.7000 | 35.3000 | -6.8000 | -39.2000 |

## Paired Final Metrics

| seed | bsuite_id | experiment | autostep | sarsa | actor_critic | sarsa_improvement_vs_autostep | actor_critic_improvement_vs_autostep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole/0 | cartpole | 115.0000 | 95.0000 | 28.0000 | -20.0000 | -87.0000 |
| 1 | cartpole/0 | cartpole | 34.0000 | 32.0000 | 97.0000 | -2.0000 | 63.0000 |
| 2 | cartpole/0 | cartpole | 131.0000 | 93.0000 | 29.0000 | -38.0000 | -102.0000 |
| 3 | cartpole/0 | cartpole | 82.0000 | 101.0000 | 28.0000 | 19.0000 | -54.0000 |
| 4 | cartpole/0 | cartpole | 32.0000 | 38.0000 | 27.0000 | 6.0000 | -5.0000 |
| 5 | cartpole/0 | cartpole | 107.0000 | 122.0000 | 28.0000 | 15.0000 | -79.0000 |
| 6 | cartpole/0 | cartpole | 30.0000 | 98.0000 | 30.0000 | 68.0000 | 0.0000 |
| 7 | cartpole/0 | cartpole | 34.0000 | 29.0000 | 28.0000 | -5.0000 | -6.0000 |
| 8 | cartpole/0 | cartpole | 107.0000 | 28.0000 | 30.0000 | -79.0000 | -77.0000 |
| 9 | cartpole/0 | cartpole | 73.0000 | 41.0000 | 28.0000 | -32.0000 | -45.0000 |
