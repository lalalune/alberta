# D08 Multi-Bank Kernel Learner

Protocol: 5 paired seeds, 1200 online steps, final window 300. Datasets: controlled_frequency, synthetic_compositional, synthetic_frequency, digits_label_drift, digits_mask_noise, digits_permuted_pixels.

Mechanism: one additive predictor over complementary feature banks. Dictionary banks allocate centers by their own ALD novelty rules; the active bank features are concatenated and one shared output matrix is updated from the single prequential prediction error. There is no MLP inside the D08 learner and no prediction router.

Positive paired differences favor the D08 candidate when comparing MSE against the best fair MLP baseline; for accuracy, positive differences mean D08 has higher accuracy.

## Configurations

### `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364`

Update: `bank_hybrid_seq`; total width 554; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `algebraic_green_memory` | dictionary | 128 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |
| `random_tanh_compositional` | random_tanh | 257 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `fourier_frequency` | fourier | 105 | fixed Fourier features, max dim 8, frequencies (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) | 1.000 | 0.300 |
| `raw_poly_d3` | dictionary | 64 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |

## Results

### controlled_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.3923 +/- 0.0336 | 0.5588 +/- 0.0157 |  |  |  |  | 0.2217 +/- 0.1302 |
| `mlp_h128` | 0.4017 +/- 0.0187 | 0.5643 +/- 0.0162 |  |  |  |  | 0.1966 +/- 0.1047 |
| `mlp_h64_64` | 0.1692 +/- 0.0165 | 0.3877 +/- 0.0172 |  |  |  |  | 0.1767 +/- 0.0535 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0144 +/- 0.0004 | 0.1963 +/- 0.0022 |  |  |  | 502.0000 +/- 0.0000 | 0.5093 +/- 0.0012 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.1549 +/- 0.0161; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.1913 +/- 0.0176; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

### digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0384 +/- 0.0006 | 0.0557 +/- 0.0003 | 0.8627 +/- 0.0085 | 0.8946 +/- 0.0057 | 0.0340 +/- 0.0006 |  | 0.2578 +/- 0.0118 |
| `mlp_h128` | 0.0441 +/- 0.0005 | 0.0594 +/- 0.0005 | 0.8500 +/- 0.0084 | 0.8965 +/- 0.0054 | 0.0357 +/- 0.0009 |  | 0.3073 +/- 0.0510 |
| `mlp_h64_64` | 0.0407 +/- 0.0010 | 0.0601 +/- 0.0008 | 0.8393 +/- 0.0109 | 0.8727 +/- 0.0073 | 0.0352 +/- 0.0007 |  | 0.2881 +/- 0.0069 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0355 +/- 0.0008 | 0.0484 +/- 0.0005 | 0.8967 +/- 0.0079 | 0.9377 +/- 0.0078 | 0.0272 +/- 0.0012 | 554.0000 +/- 0.0000 | 0.7096 +/- 0.0065 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0029 +/- 0.0009; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0073 +/- 0.0004; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: +0.0280 +/- 0.0086; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0382 +/- 0.0085; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_mse` best-D08-vs-best-MLP diff: +0.0060 +/- 0.0010; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0486 +/- 0.0009 | 0.0601 +/- 0.0008 | 0.7773 +/- 0.0136 | 0.8037 +/- 0.0121 | 0.0449 +/- 0.0017 |  | 0.2458 +/- 0.0016 |
| `mlp_h128` | 0.0486 +/- 0.0008 | 0.0601 +/- 0.0009 | 0.7973 +/- 0.0090 | 0.8137 +/- 0.0058 | 0.0441 +/- 0.0010 |  | 0.2731 +/- 0.0175 |
| `mlp_h64_64` | 0.0531 +/- 0.0014 | 0.0650 +/- 0.0005 | 0.7393 +/- 0.0105 | 0.7774 +/- 0.0148 | 0.0464 +/- 0.0022 |  | 0.3029 +/- 0.0171 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0498 +/- 0.0011 | 0.0589 +/- 0.0012 | 0.7833 +/- 0.0091 | 0.8286 +/- 0.0166 | 0.0465 +/- 0.0028 | 554.0000 +/- 0.0000 | 0.7001 +/- 0.0059 |

`final_window_mse` best-D08-vs-best-MLP diff: -0.0022 +/- 0.0009; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0006 +/- 0.0006; wins/losses/ties 3/2/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: -0.0140 +/- 0.0154; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0074 +/- 0.0112; wins/losses/ties 3/2/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_mse` best-D08-vs-best-MLP diff: -0.0037 +/- 0.0017; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

### digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0495 +/- 0.0011 | 0.0610 +/- 0.0003 | 0.7987 +/- 0.0122 | 0.8334 +/- 0.0125 | 0.0442 +/- 0.0016 |  | 0.2474 +/- 0.0013 |
| `mlp_h128` | 0.0489 +/- 0.0014 | 0.0599 +/- 0.0003 | 0.8107 +/- 0.0135 | 0.8716 +/- 0.0094 | 0.0395 +/- 0.0011 |  | 0.2778 +/- 0.0195 |
| `mlp_h64_64` | 0.0567 +/- 0.0014 | 0.0679 +/- 0.0008 | 0.7360 +/- 0.0206 | 0.7948 +/- 0.0142 | 0.0500 +/- 0.0032 |  | 0.2848 +/- 0.0019 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0429 +/- 0.0009 | 0.0493 +/- 0.0002 | 0.8640 +/- 0.0089 | 0.9065 +/- 0.0090 | 0.0363 +/- 0.0011 | 554.0000 +/- 0.0000 | 0.7083 +/- 0.0097 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0058 +/- 0.0007; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0106 +/- 0.0004; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: +0.0500 +/- 0.0121; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0323 +/- 0.0095; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`test_mse` best-D08-vs-best-MLP diff: +0.0032 +/- 0.0015; wins/losses/ties 4/1/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

### synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2617 +/- 0.0584 | 0.1862 +/- 0.0191 |  |  |  |  | 0.2150 +/- 0.0621 |
| `mlp_h128` | 0.2628 +/- 0.0544 | 0.1865 +/- 0.0186 |  |  |  |  | 0.2130 +/- 0.0682 |
| `mlp_h64_64` | 0.3101 +/- 0.0606 | 0.2257 +/- 0.0193 |  |  |  |  | 0.1861 +/- 0.0118 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.2152 +/- 0.0340 | 0.1534 +/- 0.0136 |  |  |  | 528.0000 +/- 0.0000 | 0.5576 +/- 0.0025 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0427 +/- 0.0218; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0305 +/- 0.0056; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

### synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.0350 +/- 0.1828 | 1.2698 +/- 0.1685 |  |  |  |  | 0.1107 +/- 0.0013 |
| `mlp_h128` | 1.0436 +/- 0.1798 | 1.2843 +/- 0.1661 |  |  |  |  | 0.1120 +/- 0.0023 |
| `mlp_h64_64` | 1.0247 +/- 0.1810 | 1.2631 +/- 0.1611 |  |  |  |  | 0.1530 +/- 0.0089 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.8161 +/- 0.1338 | 0.9182 +/- 0.1566 |  |  |  | 502.0000 +/- 0.0000 | 0.5153 +/- 0.0026 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.2026 +/- 0.0640; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.3405 +/- 0.0539; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 5}.

## Assessment

Using the primary metric per dataset, D08 has a positive mean paired difference on 6/6 configured datasets.

Promotion bar: a Step 2 closure claim requires one fixed D08 method, not best-of-config selection, to beat the best fair MLP on the full benchmark set and retain that advantage under more seeds. The table above reports both individual methods and the best-D08 row so the search headroom and the canonical-config result can be separated.
