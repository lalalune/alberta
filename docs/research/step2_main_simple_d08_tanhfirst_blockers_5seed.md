# D08 Multi-Bank Kernel Learner

Protocol: 5 paired seeds, 1200 online steps, final window 300. Datasets: controlled_frequency, synthetic_compositional, synthetic_frequency, digits_label_drift, digits_mask_noise, digits_permuted_pixels.

Mechanism: one additive predictor over complementary feature banks. Dictionary banks allocate centers by their own ALD novelty rules; the active bank features are concatenated and one shared output matrix is updated from the single prequential prediction error. There is no MLP inside the D08 learner and no prediction router.

Positive paired differences favor the D08 candidate when comparing MSE against the best fair MLP baseline; for accuracy, positive differences mean D08 has higher accuracy.

## Configurations

### `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128`

Update: `bank_hybrid_seq`; total width 449; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `random_tanh_compositional` | random_tanh | 257 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `raw_poly_d3` | dictionary | 64 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |
| `algebraic_green_memory` | dictionary | 128 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |

## Results

### controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3923 +/- 0.0336 | 0.5588 +/- 0.0157 |  |  |  |  | 0.3126 +/- 0.1882 |
| `mlp_h128` | 0.4017 +/- 0.0187 | 0.5643 +/- 0.0162 |  |  |  |  | 0.2879 +/- 0.1619 |
| `mlp_h64_64` | 0.1692 +/- 0.0165 | 0.3877 +/- 0.0172 |  |  |  |  | 0.2418 +/- 0.0707 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.0187 +/- 0.0023 | 0.3899 +/- 0.0145 |  |  |  | 449.0000 +/- 0.0000 | 0.5212 +/- 0.0051 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.1505 +/- 0.0175; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: -0.0022 +/- 0.0298; wins/losses/ties 2/3/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

### digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0384 +/- 0.0006 | 0.0557 +/- 0.0003 | 0.8627 +/- 0.0085 | 0.8946 +/- 0.0057 | 0.0340 +/- 0.0006 |  | 0.4270 +/- 0.0359 |
| `mlp_h128` | 0.0441 +/- 0.0005 | 0.0594 +/- 0.0005 | 0.8500 +/- 0.0084 | 0.8965 +/- 0.0054 | 0.0357 +/- 0.0009 |  | 0.5181 +/- 0.0639 |
| `mlp_h64_64` | 0.0407 +/- 0.0010 | 0.0601 +/- 0.0008 | 0.8393 +/- 0.0109 | 0.8727 +/- 0.0073 | 0.0352 +/- 0.0007 |  | 0.4566 +/- 0.0281 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.0347 +/- 0.0006 | 0.0456 +/- 0.0005 | 0.9000 +/- 0.0038 | 0.9317 +/- 0.0082 | 0.0286 +/- 0.0013 | 449.0000 +/- 0.0000 | 0.7483 +/- 0.0168 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0037 +/- 0.0005; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0101 +/- 0.0004; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: +0.0313 +/- 0.0036; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0323 +/- 0.0098; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_mse` best-D08-vs-best-MLP diff: +0.0046 +/- 0.0011; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0486 +/- 0.0009 | 0.0601 +/- 0.0008 | 0.7773 +/- 0.0136 | 0.8037 +/- 0.0121 | 0.0449 +/- 0.0017 |  | 0.4511 +/- 0.0334 |
| `mlp_h128` | 0.0486 +/- 0.0008 | 0.0601 +/- 0.0009 | 0.7973 +/- 0.0090 | 0.8137 +/- 0.0058 | 0.0441 +/- 0.0010 |  | 0.4570 +/- 0.0573 |
| `mlp_h64_64` | 0.0531 +/- 0.0014 | 0.0650 +/- 0.0005 | 0.7393 +/- 0.0105 | 0.7774 +/- 0.0148 | 0.0464 +/- 0.0022 |  | 0.4973 +/- 0.0290 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.0480 +/- 0.0007 | 0.0545 +/- 0.0012 | 0.7873 +/- 0.0084 | 0.8215 +/- 0.0176 | 0.0462 +/- 0.0026 | 449.0000 +/- 0.0000 | 0.7472 +/- 0.0233 |

`final_window_mse` best-D08-vs-best-MLP diff: -0.0005 +/- 0.0006; wins/losses/ties 2/3/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0051 +/- 0.0006; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: -0.0100 +/- 0.0089; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0004 +/- 0.0132; wins/losses/ties 3/2/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_mse` best-D08-vs-best-MLP diff: -0.0034 +/- 0.0015; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

### digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0495 +/- 0.0011 | 0.0610 +/- 0.0003 | 0.7987 +/- 0.0122 | 0.8334 +/- 0.0125 | 0.0442 +/- 0.0016 |  | 0.3187 +/- 0.0186 |
| `mlp_h128` | 0.0489 +/- 0.0014 | 0.0599 +/- 0.0003 | 0.8107 +/- 0.0135 | 0.8716 +/- 0.0094 | 0.0395 +/- 0.0011 |  | 0.3694 +/- 0.0343 |
| `mlp_h64_64` | 0.0567 +/- 0.0014 | 0.0679 +/- 0.0008 | 0.7360 +/- 0.0206 | 0.7948 +/- 0.0142 | 0.0500 +/- 0.0032 |  | 0.3804 +/- 0.0116 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.0433 +/- 0.0010 | 0.0473 +/- 0.0003 | 0.8627 +/- 0.0079 | 0.8931 +/- 0.0070 | 0.0398 +/- 0.0008 | 449.0000 +/- 0.0000 | 0.6927 +/- 0.0117 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0054 +/- 0.0006; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0126 +/- 0.0004; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: +0.0487 +/- 0.0117; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0189 +/- 0.0091; wins/losses/ties 4/1/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`test_mse` best-D08-vs-best-MLP diff: -0.0003 +/- 0.0012; wins/losses/ties 3/2/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

### synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2617 +/- 0.0584 | 0.1862 +/- 0.0191 |  |  |  |  | 0.2216 +/- 0.0733 |
| `mlp_h128` | 0.2628 +/- 0.0544 | 0.1865 +/- 0.0186 |  |  |  |  | 0.2381 +/- 0.0839 |
| `mlp_h64_64` | 0.3101 +/- 0.0606 | 0.2257 +/- 0.0193 |  |  |  |  | 0.1919 +/- 0.0032 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.2246 +/- 0.0339 | 0.1585 +/- 0.0135 |  |  |  | 449.0000 +/- 0.0000 | 0.5032 +/- 0.0081 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0332 +/- 0.0224; wins/losses/ties 4/1/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0253 +/- 0.0060; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

### synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.1828 | 1.2698 +/- 0.1685 |  |  |  |  | 0.1393 +/- 0.0089 |
| `mlp_h128` | 1.0436 +/- 0.1798 | 1.2843 +/- 0.1661 |  |  |  |  | 0.1314 +/- 0.0024 |
| `mlp_h64_64` | 1.0247 +/- 0.1810 | 1.2631 +/- 0.1611 |  |  |  |  | 0.1806 +/- 0.0102 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 1.0524 +/- 0.2375 | 1.3703 +/- 0.2204 |  |  |  | 449.0000 +/- 0.0000 | 0.5024 +/- 0.0097 |

`final_window_mse` best-D08-vs-best-MLP diff: -0.0336 +/- 0.0759; wins/losses/ties 2/3/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: -0.1116 +/- 0.0730; wins/losses/ties 2/3/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.

## Assessment

Using the primary metric per dataset, D08 has a positive mean paired difference on 5/6 configured datasets.

Promotion bar: a Step 2 closure claim requires one fixed D08 method, not best-of-config selection, to beat the best fair MLP on the full benchmark set and retain that advantage under more seeds. The table above reports both individual methods and the best-D08 row so the search headroom and the canonical-config result can be separated.
