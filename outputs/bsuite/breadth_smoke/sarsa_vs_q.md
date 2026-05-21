# SARSA vs Q-learning bsuite comparison

Metric: `total_regret` (lower is better).
Pairing: same seed and same `bsuite_id`.
Positive improvement means SARSA beat the Q-learning agent.

## Summary

| experiment | n_pairs | q_mean | sarsa_mean | mean_improvement_vs_q | ci95_improvement | sarsa_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| bandit | 2 | 8.2500 | 10.6500 | -2.4000 | 21.9520 | 0.5000 |
| memory_len | 2 | 9.0000 | 4.0000 | 5.0000 | 5.8800 | 1.0000 |
| overall | 4 | 8.6250 | 7.3250 | 1.3000 | 10.1788 | 0.7500 |

## Paired Final Metrics

| seed | bsuite_id | experiment | q_agent | sarsa_agent | q_value | sarsa_value | delta_sarsa_minus_q | improvement_vs_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | bandit/0 | bandit | autostep_bottleneck | sarsa_bottleneck | 12.2000 | 3.4000 | -8.8000 | 8.8000 |
| 1 | bandit/0 | bandit | autostep_bottleneck | sarsa_bottleneck | 4.3000 | 17.9000 | 13.6000 | -13.6000 |
| 0 | memory_len/0 | memory_len | autostep_bottleneck | sarsa_bottleneck | 8.0000 | 6.0000 | -2.0000 | 2.0000 |
| 1 | memory_len/0 | memory_len | autostep_bottleneck | sarsa_bottleneck | 10.0000 | 2.0000 | -8.0000 | 8.0000 |
