# SARSA vs Q-learning bsuite comparison

Metric: `episode_return` (higher is better).
Pairing: same seed and same `bsuite_id`.
Positive improvement means SARSA beat the Q-learning agent.

## Summary

| experiment | n_pairs | q_mean | sarsa_mean | mean_improvement_vs_q | ci95_improvement | sarsa_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| cartpole | 10 | 72.9000 | 60.2000 | -12.7000 | 33.2598 | 0.3000 |
| overall | 10 | 72.9000 | 60.2000 | -12.7000 | 33.2598 | 0.3000 |

## Paired Final Metrics

| seed | bsuite_id | experiment | q_agent | sarsa_agent | q_value | sarsa_value | delta_sarsa_minus_q | improvement_vs_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole/0 | cartpole | autostep | sarsa | 77.0000 | 70.0000 | -7.0000 | -7.0000 |
| 1 | cartpole/0 | cartpole | autostep | sarsa | 113.0000 | 32.0000 | -81.0000 | -81.0000 |
| 2 | cartpole/0 | cartpole | autostep | sarsa | 75.0000 | 73.0000 | -2.0000 | -2.0000 |
| 3 | cartpole/0 | cartpole | autostep | sarsa | 109.0000 | 79.0000 | -30.0000 | -30.0000 |
| 4 | cartpole/0 | cartpole | autostep | sarsa | 31.0000 | 41.0000 | 10.0000 | 10.0000 |
| 5 | cartpole/0 | cartpole | autostep | sarsa | 82.0000 | 87.0000 | 5.0000 | 5.0000 |
| 6 | cartpole/0 | cartpole | autostep | sarsa | 31.0000 | 133.0000 | 102.0000 | 102.0000 |
| 7 | cartpole/0 | cartpole | autostep | sarsa | 53.0000 | 28.0000 | -25.0000 | -25.0000 |
| 8 | cartpole/0 | cartpole | autostep | sarsa | 35.0000 | 30.0000 | -5.0000 | -5.0000 |
| 9 | cartpole/0 | cartpole | autostep | sarsa | 123.0000 | 29.0000 | -94.0000 | -94.0000 |
