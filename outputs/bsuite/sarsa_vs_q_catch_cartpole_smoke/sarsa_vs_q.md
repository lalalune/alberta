# SARSA vs Q-learning bsuite comparison

Metric: `total_regret` (lower is better).
Pairing: same seed and same `bsuite_id`.
Positive improvement means SARSA beat the Q-learning agent.

## Summary

| experiment | n_pairs | q_mean | sarsa_mean | mean_improvement_vs_q | ci95_improvement | sarsa_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| catch | 3 | 29.3333 | 30.6667 | -1.3333 | 6.9142 | 0.3333 |
| overall | 3 | 29.3333 | 30.6667 | -1.3333 | 6.9142 | 0.3333 |

## Paired Final Metrics

| seed | bsuite_id | experiment | q_agent | sarsa_agent | q_value | sarsa_value | delta_sarsa_minus_q | improvement_vs_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | catch/0 | catch | autostep | sarsa | 28.0000 | 36.0000 | 8.0000 | -8.0000 |
| 1 | catch/0 | catch | autostep | sarsa | 30.0000 | 30.0000 | 0.0000 | 0.0000 |
| 2 | catch/0 | catch | autostep | sarsa | 30.0000 | 26.0000 | -4.0000 | 4.0000 |
