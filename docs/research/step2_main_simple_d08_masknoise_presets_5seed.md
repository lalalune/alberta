# D08 Multi-Bank Kernel Learner

Protocol: 5 paired seeds, 1200 online steps, final window 300. Datasets: digits_mask_noise.

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

### `multibank_canonical_full_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_fourier_frequency105_raw_poly_d364_algebraic_green_memory128`

Update: `bank_hybrid_seq`; total width 554; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `random_tanh_compositional` | random_tanh | 257 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `fourier_frequency` | fourier | 105 | fixed Fourier features, max dim 8, frequencies (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) | 1.000 | 0.300 |
| `raw_poly_d3` | dictionary | 64 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |
| `algebraic_green_memory` | dictionary | 128 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |

### `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128`

Update: `bank_hybrid_seq`; total width 449; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `random_tanh_compositional` | random_tanh | 257 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `raw_poly_d3` | dictionary | 64 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |
| `algebraic_green_memory` | dictionary | 128 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |

### `multibank_compact_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory64_random_tanh_compositional129_fourier_frequency105_raw_poly_d332`

Update: `bank_hybrid_seq`; total width 330; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `algebraic_green_memory` | dictionary | 64 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |
| `random_tanh_compositional` | random_tanh | 129 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `fourier_frequency` | fourier | 105 | fixed Fourier features, max dim 8, frequencies (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) | 1.000 | 0.300 |
| `raw_poly_d3` | dictionary | 32 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |

### `multibank_compact_full_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional129_fourier_frequency105_raw_poly_d332_algebraic_green_memory64`

Update: `bank_hybrid_seq`; total width 330; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `random_tanh_compositional` | random_tanh | 129 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |
| `fourier_frequency` | fourier | 105 | fixed Fourier features, max dim 8, frequencies (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) | 1.000 | 0.300 |
| `raw_poly_d3` | dictionary | 32 | `polynomial`, novelty 0.001, interval 1, replace False | 1.000 | 1.000 |
| `algebraic_green_memory` | dictionary | 64 | `algebraic_green`, novelty 0.001, interval 8, replace False | 1.000 | 0.400 |

### `multibank_tanh_only_nlms_nlms_eta0p4_random_tanh_compositional257`

Update: `nlms`; total width 257; rho 0.99; RLS delta 100.0; NLMS step 0.4.

| Bank | Kind | Width | Allocation | Feature Scale | Step |
|---|---|---:|---|---:|---:|
| `random_tanh_compositional` | random_tanh | 257 | fixed tanh random features, weight scale 1.0, bias True | 1.000 | 0.400 |

## Results

### digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Test MSE | Active Features | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0486 +/- 0.0009 | 0.0601 +/- 0.0008 | 0.7773 +/- 0.0136 | 0.8037 +/- 0.0121 | 0.0449 +/- 0.0017 |  | 0.4537 +/- 0.1494 |
| `mlp_h128` | 0.0486 +/- 0.0008 | 0.0601 +/- 0.0009 | 0.7973 +/- 0.0090 | 0.8137 +/- 0.0058 | 0.0441 +/- 0.0010 |  | 0.4569 +/- 0.1190 |
| `mlp_h64_64` | 0.0531 +/- 0.0014 | 0.0650 +/- 0.0005 | 0.7393 +/- 0.0105 | 0.7774 +/- 0.0148 | 0.0464 +/- 0.0022 |  | 0.3308 +/- 0.0143 |
| `multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364` | 0.0498 +/- 0.0011 | 0.0589 +/- 0.0012 | 0.7833 +/- 0.0091 | 0.8286 +/- 0.0166 | 0.0465 +/- 0.0028 | 554.0000 +/- 0.0000 | 0.7961 +/- 0.0208 |
| `multibank_canonical_full_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_fourier_frequency105_raw_poly_d364_algebraic_green_memory128` | 0.0517 +/- 0.0011 | 0.0594 +/- 0.0012 | 0.7727 +/- 0.0095 | 0.8130 +/- 0.0214 | 0.0497 +/- 0.0035 | 554.0000 +/- 0.0000 | 0.7854 +/- 0.0101 |
| `multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128` | 0.0480 +/- 0.0007 | 0.0545 +/- 0.0012 | 0.7873 +/- 0.0084 | 0.8215 +/- 0.0176 | 0.0462 +/- 0.0026 | 449.0000 +/- 0.0000 | 0.7273 +/- 0.0185 |
| `multibank_compact_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory64_random_tanh_compositional129_fourier_frequency105_raw_poly_d332` | 0.0518 +/- 0.0013 | 0.0608 +/- 0.0012 | 0.7607 +/- 0.0101 | 0.8045 +/- 0.0178 | 0.0486 +/- 0.0023 | 330.0000 +/- 0.0000 | 0.5239 +/- 0.0055 |
| `multibank_compact_full_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional129_fourier_frequency105_raw_poly_d332_algebraic_green_memory64` | 0.0538 +/- 0.0013 | 0.0618 +/- 0.0011 | 0.7567 +/- 0.0101 | 0.7922 +/- 0.0190 | 0.0504 +/- 0.0024 | 330.0000 +/- 0.0000 | 0.5298 +/- 0.0051 |
| `multibank_tanh_only_nlms_nlms_eta0p4_random_tanh_compositional257` | 0.0525 +/- 0.0008 | 0.0575 +/- 0.0005 | 0.7660 +/- 0.0164 | 0.7807 +/- 0.0081 | 0.0512 +/- 0.0011 | 257.0000 +/- 0.0000 | 0.0620 +/- 0.0005 |

`final_window_mse` best-D08-vs-best-MLP diff: -0.0005 +/- 0.0006; wins/losses/ties 2/3/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`online_mean_mse` best-D08-vs-best-MLP diff: +0.0051 +/- 0.0006; wins/losses/ties 5/0/0; best-D08 counts {'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 5}.
`final_window_accuracy` best-D08-vs-best-MLP diff: -0.0053 +/- 0.0126; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3, 'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 2}.
`test_accuracy` best-D08-vs-best-MLP diff: +0.0093 +/- 0.0125; wins/losses/ties 3/2/0; best-D08 counts {'multibank_canonical_full_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_fourier_frequency105_raw_poly_d364_algebraic_green_memory128': 1, 'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 4}.
`test_mse` best-D08-vs-best-MLP diff: -0.0030 +/- 0.0016; wins/losses/ties 1/4/0; best-D08 counts {'multibank_canonical_green_first_full_seq_bank_hybrid_seq_seqrls_rho0p99_algebraic_green_memory128_random_tanh_compositional257_fourier_frequency105_raw_poly_d364': 3, 'multibank_canonical_tanh_first_seq_bank_hybrid_seq_seqrls_rho0p99_random_tanh_compositional257_raw_poly_d364_algebraic_green_memory128': 1, 'multibank_tanh_only_nlms_nlms_eta0p4_random_tanh_compositional257': 1}.

## Assessment

Using the primary metric per dataset, D08 has a positive mean paired difference on 1/1 configured datasets.

Promotion bar: a Step 2 closure claim requires one fixed D08 method, not best-of-config selection, to beat the best fair MLP on the full benchmark set and retain that advantage under more seeds. The table above reports both individual methods and the best-D08 row so the search headroom and the canonical-config result can be separated.
