# D19 Adaptive Fast/Slow D18 Results

Protocol: 1 paired seeds, 600 online steps, final window 150. Candidate configs: d19_learned_step2_canonical, d19_fixed_step2_canonical.

D19 is one D18 additive learner with a causal fast/slow controller over target-persistence threshold and basis readout decay. There is no MLP expert and no prediction router.

## opmnist

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Gate | Threshold | Basis Decay | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0660 +/- 0.0000 | 0.0803 +/- 0.0000 | 0.6133 +/- 0.0000 | 0.5383 +/- 0.0000 |  |  |  | 12.3489 +/- 0.0000 |
| `mlp_h128` | 0.0683 +/- 0.0000 | 0.0819 +/- 0.0000 | 0.6133 +/- 0.0000 | 0.6533 +/- 0.0000 |  |  |  | 6.0816 +/- 0.0000 |
| `mlp_h64_64` | 0.0688 +/- 0.0000 | 0.0826 +/- 0.0000 | 0.6067 +/- 0.0000 | 0.5250 +/- 0.0000 |  |  |  | 4.1336 +/- 0.0000 |
| `d19_learned_step2_canonical` | 0.0708 +/- 0.0000 | 0.0792 +/- 0.0000 | 0.4800 +/- 0.0000 | 0.4883 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.9975 +/- 0.0000 | 19.0589 +/- 0.0000 |
| `d19_fixed_step2_canonical` | 0.0696 +/- 0.0000 | 0.0788 +/- 0.0000 | 0.5200 +/- 0.0000 | 0.5550 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9975 +/- 0.0000 | 14.7257 +/- 0.0000 |

`final_window_mse` best-D19-vs-best-MLP diff: -0.003589 +/- 0.000000; wins/losses/ties 0/1/0; best-D19 counts {'d19_fixed_step2_canonical': 1}.
`test_accuracy` best-D19-vs-best-MLP diff: -0.098333 +/- 0.000000; wins/losses/ties 0/1/0.
