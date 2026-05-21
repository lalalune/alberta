# SARSA vs Q-learning bsuite comparison

Metric: `episode_return` (higher is better).
Pairing: same seed and same `bsuite_id`.
Positive improvement means SARSA beat the Q-learning agent.

## Summary

| experiment | n_pairs | q_mean | sarsa_mean | mean_improvement_vs_q | ci95_improvement | sarsa_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| cartpole | 3 | 47.6667 | 95.0000 | 47.3333 | 45.0421 | 1.0000 |
| overall | 3 | 47.6667 | 95.0000 | 47.3333 | 45.0421 | 1.0000 |

## Paired Final Metrics

| seed | bsuite_id | experiment | q_agent | sarsa_agent | q_value | sarsa_value | delta_sarsa_minus_q | improvement_vs_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole/0 | cartpole | autostep | sarsa | 28.0000 | 87.0000 | 59.0000 | 59.0000 |
| 1 | cartpole/0 | cartpole | autostep | sarsa | 29.0000 | 32.0000 | 3.0000 | 3.0000 |
| 2 | cartpole/0 | cartpole | autostep | sarsa | 86.0000 | 166.0000 | 80.0000 | 80.0000 |
