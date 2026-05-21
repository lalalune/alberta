# D19 Adaptive Fast/Slow D18 Results

Protocol: 1 paired seeds, 600 online steps, final window 150. Candidate configs: d19_learned_step2_canonical.

D19 is one D18 additive learner with a causal fast/slow controller over target-persistence threshold and basis readout decay. There is no MLP expert and no prediction router.

## opmnist

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Gate | Threshold | Basis Decay | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0659 +/- 0.0000 | 0.0808 +/- 0.0000 | 0.6067 +/- 0.0000 | 0.5400 +/- 0.0000 |  |  |  | 11.6677 +/- 0.0000 |
| `mlp_h128` | 0.0678 +/- 0.0000 | 0.0817 +/- 0.0000 | 0.6133 +/- 0.0000 | 0.7183 +/- 0.0000 |  |  |  | 9.0122 +/- 0.0000 |
| `mlp_h64_64` | 0.0682 +/- 0.0000 | 0.0846 +/- 0.0000 | 0.5600 +/- 0.0000 | 0.4900 +/- 0.0000 |  |  |  | 7.0862 +/- 0.0000 |
| `d19_learned_step2_canonical` | 0.0684 +/- 0.0000 | 0.0792 +/- 0.0000 | 0.4800 +/- 0.0000 | 0.5250 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.9950 +/- 0.0000 | 20.7318 +/- 0.0000 |

`final_window_mse` best-D19-vs-best-MLP diff: -0.002549 +/- 0.000000; wins/losses/ties 0/1/0; best-D19 counts {'d19_learned_step2_canonical': 1}.
`test_accuracy` best-D19-vs-best-MLP diff: -0.193333 +/- 0.000000; wins/losses/ties 0/1/0.
