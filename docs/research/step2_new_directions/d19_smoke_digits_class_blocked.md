# D19 Adaptive Fast/Slow D18 Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: d19_learned_step2_canonical, d19_fixed_step2_canonical.

D19 is one D18 additive learner with a causal fast/slow controller over target-persistence threshold and basis readout decay. There is no MLP expert and no prediction router.

## digits_class_blocked

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Gate | Threshold | Basis Decay | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0117 +/- 0.0000 | 0.0238 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0000 |  |  |  | 6.2723 +/- 0.0000 |
| `mlp_h128` | 0.0158 +/- 0.0000 | 0.0254 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0000 |  |  |  | 3.0799 +/- 0.0000 |
| `mlp_h64_64` | 0.0096 +/- 0.0000 | 0.0206 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0000 |  |  |  | 1.7525 +/- 0.0000 |
| `d19_learned_step2_canonical` | 0.0000 +/- 0.0000 | 0.0020 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0000 | 0.9038 +/- 0.0000 | 0.9500 +/- 0.0000 | 0.9995 +/- 0.0000 | 1.0790 +/- 0.0000 |
| `d19_fixed_step2_canonical` | 0.0000 +/- 0.0000 | 0.0018 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0000 | 0.9023 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.9975 +/- 0.0000 | 2.1704 +/- 0.0000 |

`final_window_mse` best-D19-vs-best-MLP diff: +0.009575 +/- 0.000000; wins/losses/ties 1/0/0; best-D19 counts {'d19_learned_step2_canonical': 1}.
`test_accuracy` best-D19-vs-best-MLP diff: +0.000000 +/- 0.000000; wins/losses/ties 1/0/0.
