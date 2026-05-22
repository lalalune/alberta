# D08 Multi-Bank Kernel Learner

Protocol: 3 paired seeds, 1200 online steps, final window 300. Datasets: synthetic_compositional, synthetic_frequency, digits_mask_noise, digits_permuted_pixels.

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

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0492 +/- 0.0013 | 0.0603 +/- 0.0008 | 0.7922 +/- 0.0185 | 0.8089 +/- 0.0214 | 0.0448 +/- 0.0029 |  | 2.9715 +/- 0.4657 |
| `mlp_h128` | 0.0478 +/- 0.0009 | 0.0595 +/- 0.0013 | 0.8067 +/- 0.0102 | 0.8077 +/- 0.0073 | 0.0447 +/- 0.0012 |  | 4.1189 +/- 0.5393 |
| `mlp_h64_64` | 0.0533 +/- 0.0024 | 0.0650 +/- 0.0006 | 0.7356 +/- 0.0175 | 0.7737 +/- 0.0245 | 0.0481 +/- 0.0028 |  | 5.1395 +/- 1.1793 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0514 +/- 0.0005 | 0.0596 +/- 0.0018 | 0.7722 +/- 0.0078 | 0.8287 +/- 0.0209 | 0.0487 +/- 0.0025 | 554.0000 +/- 0.0000 | 10.1234 +/- 0.2798 |

`final_window_mse` best-D08-vs-best-MLP diff: -0.0036 +/- 0.0006; wins/losses/ties 0/3/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`online_mean_mse` best-D08-vs-best-MLP diff: -0.0002 +/- 0.0007; wins/losses/ties 1/2/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`final_window_accuracy` best-D08-vs-best-MLP diff: -0.0344 +/- 0.0056; wins/losses/ties 0/3/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0087 +/- 0.0091; wins/losses/ties 2/1/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`test_mse` best-D08-vs-best-MLP diff: -0.0057 +/- 0.0005; wins/losses/ties 0/3/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.

### digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0505 +/- 0.0014 | 0.0612 +/- 0.0004 | 0.7867 +/- 0.0168 | 0.8194 +/- 0.0163 | 0.0464 +/- 0.0013 |  | 3.6864 +/- 0.5517 |
| `mlp_h128` | 0.0493 +/- 0.0024 | 0.0599 +/- 0.0006 | 0.8100 +/- 0.0212 | 0.8813 +/- 0.0124 | 0.0397 +/- 0.0017 |  | 4.0339 +/- 0.8760 |
| `mlp_h64_64` | 0.0566 +/- 0.0019 | 0.0676 +/- 0.0010 | 0.7489 +/- 0.0161 | 0.7972 +/- 0.0236 | 0.0523 +/- 0.0050 |  | 3.5713 +/- 0.6617 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0431 +/- 0.0017 | 0.0491 +/- 0.0003 | 0.8678 +/- 0.0142 | 0.9103 +/- 0.0033 | 0.0377 +/- 0.0008 | 554.0000 +/- 0.0000 | 9.4201 +/- 1.9619 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0060 +/- 0.0012; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0108 +/- 0.0006; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`final_window_accuracy` best-D08-vs-best-MLP diff: +0.0578 +/- 0.0142; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0291 +/- 0.0096; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`test_mse` best-D08-vs-best-MLP diff: +0.0020 +/- 0.0024; wins/losses/ties 2/1/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.

### synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  |  | 5.3564 +/- 3.9895 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  |  | 4.5700 +/- 3.3847 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  |  | 3.5114 +/- 2.0982 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.2157 +/- 0.0594 | 0.1457 +/- 0.0221 |  |  |  | 528.0000 +/- 0.0000 | 8.4761 +/- 1.5784 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.0560 +/- 0.0360; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0330 +/- 0.0090; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.

### synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  |  | 2.3098 +/- 0.5573 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  |  | 1.9961 +/- 0.9363 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  |  | 1.8717 +/- 0.3843 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.9112 +/- 0.1841 | 1.0207 +/- 0.2432 |  |  |  | 502.0000 +/- 0.0000 | 7.3921 +/- 0.4405 |

`final_window_mse` best-D08-vs-best-MLP diff: +0.2373 +/- 0.1074; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.2944 +/- 0.0331; wins/losses/ties 3/0/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3}.

## Assessment

Using the primary metric per dataset, D08 has a positive mean paired difference on 4/4 configured datasets.

Promotion bar: a Step 2 closure claim requires one fixed D08 method, not best-of-config selection, to beat the best fair MLP on the full benchmark set and retain that advantage under more seeds. The table above reports both individual methods and the best-D08 row so the search headroom and the canonical-config result can be separated.
