# Step 2 Worker G: Rare-Aware Route Selection

This worker tested whether emphasizing multi-head active steps in route-selection
state can close the remaining `controlled_rare` failures without disturbing the
synthetic or external-digit regimes.

## Baseline

The starting conclusive command was:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

That artifact has total all-suite final-window MSE support of `126/4/10` versus
the same-run best fair MLP. Its losses are `controlled_rare` seeds 8 and 9,
`synthetic_compositional` seed 5, and `synthetic_polynomial` seed 6.

## Focused Sweep

All focused runs used `controlled_rare,synthetic_compositional,synthetic_polynomial`,
1200 steps, 10 seeds, final window 300, warmup 250, and the baseline command
above except for `--route-rare-active-step-weight`.

| Rare-active weight | controlled_rare | synthetic_compositional | synthetic_polynomial | Total |
| --- | ---: | ---: | ---: | ---: |
| `0.5` | `+0.013307`, `8/2/0` | `+0.043445`, `9/1/0` | `+0.061280`, `9/1/0` | `26/4/0` |
| `1.0` | `+0.012702`, `8/2/0` | `+0.043445`, `9/1/0` | `+0.061279`, `9/1/0` | `26/4/0` |
| `2.0` | `+0.015055`, `9/1/0` | `+0.043445`, `9/1/0` | `+0.061280`, `9/1/0` | `27/3/0` |
| `4.0` | `+0.015390`, `9/1/0` | `+0.043445`, `9/1/0` | `+0.061282`, `9/1/0` | `27/3/0` |
| `8.0` | `+0.013180`, `9/1/0` | `+0.043450`, `9/1/0` | `+0.061281`, `9/1/0` | `27/3/0` |

Weights `2.0`, `4.0`, and `8.0` consistently remove `controlled_rare` seed 8.
They do not solve `controlled_rare` seed 9 or `synthetic_polynomial` seed 6.

## Confirmation Runs

`--route-rare-active-step-weight 4.0` confirmed on the all-suite at `127/3/10`.
It improved the baseline by one seed loss:

| Candidate | All-suite total | Remaining losses |
| --- | ---: | --- |
| Baseline eta `1.0` | `126/4/10` | `controlled_rare` seeds 8/9, `synthetic_compositional` seed 5, `synthetic_polynomial` seed 6 |
| Rare-active `4.0` | `127/3/10` | `controlled_rare` seed 9, `synthetic_compositional` seed 5, `synthetic_polynomial` seed 6 |

Combining the rare-aware route score with Worker A's stronger hedge setting,
`--hedge-eta 0.5 --route-rare-active-step-weight 4.0`, gave the strongest
result:

| Benchmark | Final-window MSE delta vs best MLP | W/L/T |
| --- | ---: | ---: |
| `controlled_frequency` | `+0.156008` | `10/0/0` |
| `controlled_interaction` | `+0.147744` | `10/0/0` |
| `controlled_nonlinear` | `+0.020609` | `10/0/0` |
| `controlled_polynomial` | `+0.532656` | `10/0/0` |
| `controlled_rare` | `+0.015089` | `9/1/0` |
| `controlled_triple` | `+0.579204` | `10/0/0` |
| `digits_class_blocked` | `+0.000000` | `0/0/10` |
| `digits_iid` | `+0.006486` | `10/0/0` |
| `digits_label_drift` | `+0.006999` | `10/0/0` |
| `digits_mask_noise` | `+0.006320` | `10/0/0` |
| `digits_permuted_pixels` | `+0.008483` | `10/0/0` |
| `synthetic_compositional` | `+0.046028` | `10/0/0` |
| `synthetic_frequency` | `+0.804476` | `10/0/0` |
| `synthetic_polynomial` | `+0.061337` | `9/1/0` |

The combined all-suite total is `128/2/10`.

## Conclusion

Rare-active route weighting is promotable as a route-selection improvement, but
not by itself as a universal closure. The best candidate from this worker is:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --route-rare-active-step-weight 4.0
```

It is a strict improvement over the current all-suite artifact, but it still
leaves two localized losses: `controlled_rare` seed 9 and
`synthetic_polynomial` seed 6.

## Artifacts

- `outputs/step2_worker_g_rareweight_0p5_weak3_10seed/results.json`
- `outputs/step2_worker_g_rareweight_1p0_weak3_10seed/results.json`
- `outputs/step2_worker_g_rareweight_2p0_weak3_10seed_rerun/results.json`
- `outputs/step2_worker_g_rareweight_4p0_weak3_10seed/results.json`
- `outputs/step2_worker_g_rareweight_8p0_poly_10seed/results.json`
- `outputs/step2_worker_g_rareweight_4p0_all_10seed/results.json`
- `outputs/step2_worker_g_eta0p5_rare4p0_weak3_10seed/results.json`
- `outputs/step2_worker_g_eta0p5_rare4p0_all_10seed/results.json`
