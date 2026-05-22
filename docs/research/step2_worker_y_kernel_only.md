# Worker Y Kernel-Only Adaptive Filter

Protocol: 3 paired seeds, 1200 online steps, final window 300. Datasets: `digits_class_blocked, digits_mask_noise, synthetic_polynomial, synthetic_frequency, synthetic_compositional, controlled_rare`.

Learner constraint: each kernel candidate is one additive predictor `K(x, centers) @ alpha`; centers are observations admitted by one ALD/budget rule; coefficients use one residual KRLS/RLS update. The MLP widths are baselines only and are not consumed by the kernels.

Command:

```bash
source .venv/bin/activate
python output/subagents/step2_worker_y_kernel_only.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --final-window 300 --candidates screen --output-dir outputs/step2_worker_y_kernel_only_blockers_3seed --note-path docs/research/step2_worker_y_kernel_only.md
```

Artifacts:

- `outputs/step2_worker_y_kernel_only_blockers_3seed/results.json`
- `outputs/step2_worker_y_kernel_only_blockers_3seed/SUMMARY.md`
- `docs/research/step2_worker_y_kernel_only.md`
- Follow-up prior run:
  `outputs/step2_worker_y_kernel_only_prior_3seed/results.json`,
  `outputs/step2_worker_y_kernel_only_prior_3seed/SUMMARY.md`,
  `outputs/step2_worker_y_kernel_only_prior_3seed/REPORT.md`

## Candidates

- `green_rbf_b256_ai4`: single diffusion/RBF kernel, three fixed bandwidth multipliers, ALD centers, RLS residual update.
- `poly_heat_arc_equal_b256_ai4`: one fixed equal-weight composite kernel: diffusion heat + normalized degree-3 polynomial + ReLU arccosine; composite weights=[1.0, 1.0, 1.0].
- `poly_heat_arc_polyheavy_b256_ai4`: one fixed composite kernel biased toward algebraic structure: 0.25 heat + 0.50 polynomial + 0.25 arccosine; composite weights=[0.25, 0.5, 0.25].

## Blocker Matrix

Margin is paired best-MLP final-window MSE minus candidate final-window MSE, so positive favors the kernel. W/L/T is candidate versus best MLP at the seed level.

| Candidate | Dataset | Candidate MSE | Best MLP MSE | Margin | W/L/T |
|---|---|---:|---:|---:|---:|
| `green_rbf_b256_ai4` | `controlled_rare` | 0.088061 | 0.073166 | -0.014894 | 1/2/0 |
| `green_rbf_b256_ai4` | `digits_class_blocked` | 0.004569 | 0.002992 | -0.001577 | 0/3/0 |
| `green_rbf_b256_ai4` | `digits_mask_noise` | 0.086153 | 0.047837 | -0.038315 | 0/3/0 |
| `green_rbf_b256_ai4` | `synthetic_compositional` | 0.586329 | 0.275842 | -0.314627 | 0/3/0 |
| `green_rbf_b256_ai4` | `synthetic_frequency` | 2.778116 | 1.149273 | -1.629654 | 0/3/0 |
| `green_rbf_b256_ai4` | `synthetic_polynomial` | 2.440054 | 0.947464 | -1.492590 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `controlled_rare` | 0.076158 | 0.073166 | -0.002992 | 1/2/0 |
| `poly_heat_arc_equal_b256_ai4` | `digits_class_blocked` | 0.004264 | 0.002992 | -0.001271 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `digits_mask_noise` | 0.088680 | 0.047837 | -0.040843 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `synthetic_compositional` | 0.471668 | 0.275842 | -0.199965 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `synthetic_frequency` | 3.223274 | 1.149273 | -2.074812 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `synthetic_polynomial` | 1.931346 | 0.947464 | -0.983882 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `controlled_rare` | 0.078682 | 0.073166 | -0.005516 | 1/2/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `digits_class_blocked` | 0.004736 | 0.002992 | -0.001744 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `digits_mask_noise` | 0.085547 | 0.047837 | -0.037709 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `synthetic_compositional` | 0.451251 | 0.275842 | -0.179549 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `synthetic_frequency` | 3.012011 | 1.149273 | -1.863550 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `synthetic_polynomial` | 1.913810 | 0.947464 | -0.966345 | 0/3/0 |

## Digit Held-Out Check

Final-window MSE remains the gate above. Held-out accuracy is shown because class-blocked streams can reward current-block specialization while hiding retention failure.

| Candidate | Dataset | Candidate Test Acc | Best MLP Test Acc | Margin | W/L/T |
|---|---|---:|---:|---:|---:|
| `green_rbf_b256_ai4` | `digits_class_blocked` | 0.100186 | 0.152752 | -0.065553 | 0/3/0 |
| `green_rbf_b256_ai4` | `digits_mask_noise` | 0.601732 | 0.808905 | -0.218306 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `digits_class_blocked` | 0.100186 | 0.152752 | -0.065553 | 0/3/0 |
| `poly_heat_arc_equal_b256_ai4` | `digits_mask_noise` | 0.583797 | 0.808905 | -0.236240 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `digits_class_blocked` | 0.100186 | 0.152752 | -0.065553 | 0/3/0 |
| `poly_heat_arc_polyheavy_b256_ai4` | `digits_mask_noise` | 0.598021 | 0.808905 | -0.222016 | 0/3/0 |

## Green Check

No fixed kernel candidate was strict green across this blocker matrix by final-window MSE.

| Candidate | All Mean Margins Positive | No Seed-Level MLP Losses | Strict Green |
|---|---:|---:|---:|
| `green_rbf_b256_ai4` | `False` | `False` | `False` |
| `poly_heat_arc_equal_b256_ai4` | `False` | `False` | `False` |
| `poly_heat_arc_polyheavy_b256_ai4` | `False` | `False` | `False` |

## Strong Prior Follow-Up

I also ran the strongest current simple KRLS prior as one fixed candidate. This
is still one additive predictor with one algebraic-Green kernel, not a bank or a
router.

```bash
source .venv/bin/activate
python output/subagents/step2_worker_y_kernel_only.py --datasets digits_class_blocked,digits_mask_noise,synthetic_polynomial,synthetic_frequency,synthetic_compositional,controlled_rare --steps 1200 --n-seeds 3 --final-window 300 --candidates prior --output-dir outputs/step2_worker_y_kernel_only_prior_3seed --note-path outputs/step2_worker_y_kernel_only_prior_3seed/REPORT.md
```

| Candidate | Dataset | Candidate MSE | Best MLP MSE | Margin | W/L/T |
|---|---|---:|---:|---:|---:|
| `algebraic_green_prior_b128` | `controlled_rare` | 0.167623 | 0.073166 | -0.094457 | 0/3/0 |
| `algebraic_green_prior_b128` | `digits_class_blocked` | 0.007786 | 0.002992 | -0.004794 | 0/3/0 |
| `algebraic_green_prior_b128` | `digits_mask_noise` | 0.118425 | 0.047837 | -0.070588 | 0/3/0 |
| `algebraic_green_prior_b128` | `synthetic_compositional` | 0.460013 | 0.275842 | -0.188311 | 0/3/0 |
| `algebraic_green_prior_b128` | `synthetic_frequency` | 1.603308 | 1.149273 | -0.454846 | 0/3/0 |
| `algebraic_green_prior_b128` | `synthetic_polynomial` | 2.044616 | 0.947464 | -1.097151 | 0/3/0 |

Held-out digit accuracy also lost to the best MLP on both digit rows:
`digits_class_blocked` 0.100186 vs 0.152752 and `digits_mask_noise` 0.179344
vs 0.808905.

## Critical Assessment

The tested kernels are real single learners in the narrow mechanistic sense: one dictionary, one coefficient matrix, one prediction, and one residual update. There is no expert selector or output router.

The fixed composite kernel is less clean conceptually than a plain Gaussian RKHS. It is still one mathematically uniform kernel, but the polynomial/arccosine/heat choice is informed by known blocker structure. That makes it a compact hand-chosen kernel, not evidence of a discovered universal principle. The algebraic-Green prior is cleaner than D18-style block systems, but it is still a hand-chosen kernel family.

Because no fixed candidate cleared the blocker matrix by final-window MSE, I did not run a broader all-14 or 10-seed risk pass. The negative result is useful: at this budget and update rule, a single fixed kernel adaptive filter still cannot replace the more engineered additive-bank/D18-style solutions.
