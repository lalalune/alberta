# SARSA vs Q-learning bsuite comparison

Metric: `episode_return` (higher is better).
Pairing: same seed and same `bsuite_id`.
Positive improvement means SARSA beat the Q-learning agent.

## Summary

| experiment | n_pairs | q_mean | sarsa_mean | mean_improvement_vs_q | ci95_improvement | sarsa_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| cartpole_noise | 10 | 63.2947 | 65.8042 | 2.5095 | 24.6859 | 0.4000 |
| cartpole_scale | 10 | 0.0556 | 0.0862 | 0.0306 | 0.0278 | 0.7000 |
| overall | 20 | 31.6752 | 32.9452 | 1.2700 | 12.0267 | 0.5500 |

## Paired Final Metrics

| seed | bsuite_id | experiment | q_agent | sarsa_agent | q_value | sarsa_value | delta_sarsa_minus_q | improvement_vs_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 84.8534 | 70.3622 | -14.4911 | -14.4911 |
| 1 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 59.4525 | 29.7936 | -29.6588 | -29.6588 |
| 2 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 82.1489 | 148.0658 | 65.9170 | 65.9170 |
| 3 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 95.4833 | 101.0902 | 5.6070 | 5.6070 |
| 4 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 30.1067 | 27.1639 | -2.9428 | -2.9428 |
| 5 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 85.7668 | 118.5333 | 32.7666 | 32.7666 |
| 6 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 29.8794 | 75.7590 | 45.8797 | 45.8797 |
| 7 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 30.6877 | 30.0338 | -0.6539 | -0.6539 |
| 8 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 28.8421 | 26.8931 | -1.9490 | -1.9490 |
| 9 | cartpole_noise/0 | cartpole_noise | autostep | sarsa | 105.7269 | 30.3471 | -75.3798 | -75.3798 |
| 0 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0400 | 0.0970 | 0.0570 | 0.0570 |
| 1 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0770 | 0.0900 | 0.0130 | 0.0130 |
| 2 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0560 | 0.0590 | 0.0030 | 0.0030 |
| 3 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0380 | 0.0320 | -0.0060 | -0.0060 |
| 4 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0590 | 0.1310 | 0.0720 | 0.0720 |
| 5 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0340 | 0.0970 | 0.0630 | 0.0630 |
| 6 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.1000 | 0.0840 | -0.0160 | -0.0160 |
| 7 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0650 | 0.0620 | -0.0030 | -0.0030 |
| 8 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0380 | 0.1590 | 0.1210 | 0.1210 |
| 9 | cartpole_scale/0 | cartpole_scale | autostep | sarsa | 0.0490 | 0.0510 | 0.0020 | 0.0020 |
