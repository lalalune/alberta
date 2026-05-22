# D19 Adaptive Fast/Slow D18 Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: d19_learned_step2_canonical, d19_fixed_step2_canonical.

D19 is one D18 additive learner with a causal fast/slow controller over target-persistence threshold and basis readout decay. There is no MLP expert and no prediction router.

## opmnist

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Gate | Threshold | Basis Decay | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.1082 +/- 0.0000 | 0.1103 +/- 0.0000 | 0.2500 +/- 0.0000 | 0.3833 +/- 0.0000 |  |  |  | 11.0942 +/- 0.0000 |
| `mlp_h128` | 0.1160 +/- 0.0000 | 0.1183 +/- 0.0000 | 0.2750 +/- 0.0000 | 0.3933 +/- 0.0000 |  |  |  | 9.9932 +/- 0.0000 |
| `mlp_h64_64` | 0.1038 +/- 0.0000 | 0.1103 +/- 0.0000 | 0.2250 +/- 0.0000 | 0.2300 +/- 0.0000 |  |  |  | 6.2370 +/- 0.0000 |
| `d19_learned_step2_canonical` | 0.0942 +/- 0.0000 | 0.0966 +/- 0.0000 | 0.1750 +/- 0.0000 | 0.3067 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.9975 +/- 0.0000 | 1.9815 +/- 0.0000 |
| `d19_fixed_step2_canonical` | 0.0951 +/- 0.0000 | 0.0966 +/- 0.0000 | 0.1750 +/- 0.0000 | 0.3533 +/- 0.0000 | 0.0001 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9975 +/- 0.0000 | 2.1536 +/- 0.0000 |

`final_window_mse` best-D19-vs-best-MLP diff: +0.009653 +/- 0.000000; wins/losses/ties 1/0/0; best-D19 counts {'d19_learned_step2_canonical': 1}.
`test_accuracy` best-D19-vs-best-MLP diff: -0.040000 +/- 0.000000; wins/losses/ties 0/1/0.
