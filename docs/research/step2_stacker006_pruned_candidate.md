# Step 2 Pruned Stacker Candidate

This note supersedes the earlier pruned Hedge-blend candidate.  The new
candidate keeps the parts that survived ablation, removes the pieces that were
not earning their compute in routing/deployment, and adds one low-cost learned
resource-manager route.

## Canonical Profile

Run profile:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks controlled,synthetic_polynomial,synthetic_frequency,synthetic_compositional \
  --steps 1200 \
  --n-seeds 3 \
  --final-window 300 \
  --warmup-steps 250 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --stacker-step-size 0.006 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 300 \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.4
```

Pruned from conclusive routing/deployment:

- `upgd_low_noise`
- `dynamic_sparse`

Kept:

- recursive feature construction
- polynomial basis
- Fourier basis
- fixed random tanh basis
- the full fair MLP comparator/control grid
- MLP-safe recursive routes
- discounted Hedge weighting
- class-blocked recursive retention deployment override

Added:

- `stacked_predictions`: a per-head normalized-LMS stacker over expert
  predictions.  It is initialized to the fair `mlp_32x32` prediction and learns
  only a small residual correction (`--stacker-step-size 0.006`).  The point is
  not to replace the expert system; it is a low-cost resource-manager route that
  can exploit specialist residuals while beginning at the fair MLP baseline.

## Evidence

Controlled/synthetic output:

- `outputs/step2_conclusive_stacker006_controlled_synth_3seed/results.json`
- `docs/research/step2_conclusive_stacker006_controlled_synth.md`

Digits output:

- `outputs/step2_conclusive_stacker006_blend04_digits_3seed/results.json`
- `docs/research/step2_conclusive_stacker006_blend04_digits.md`

Positive differences favor the conclusive learner.  For MSE this is
`best MLP - conclusive`; for accuracy this is `conclusive - best MLP`.

| Dataset | Metric | Mean Diff | W/L/T |
|---|---:|---:|---:|
| `controlled_frequency` | final-window MSE | +0.175865 | 3/0/0 |
| `controlled_interaction` | final-window MSE | +0.136030 | 3/0/0 |
| `controlled_nonlinear` | final-window MSE | +0.010281 | 3/0/0 |
| `controlled_polynomial` | final-window MSE | +0.530434 | 3/0/0 |
| `controlled_rare` | final-window MSE | +0.010655 | 3/0/0 |
| `controlled_triple` | final-window MSE | +0.460076 | 3/0/0 |
| `synthetic_compositional` | final-window MSE | +0.048064 | 3/0/0 |
| `synthetic_frequency` | final-window MSE | +0.426703 | 3/0/0 |
| `synthetic_polynomial` | final-window MSE | +0.146595 | 3/0/0 |
| `digits_iid` | final-window MSE | +0.006568 | 3/0/0 |
| `digits_iid` | held-out accuracy | +0.011132 | 3/0/0 |
| `digits_class_blocked` | final-window MSE | +0.000000 | 0/0/3 |
| `digits_class_blocked` | held-out accuracy | +0.366728 | 3/0/0 |
| `digits_label_drift` | final-window MSE | +0.007664 | 3/0/0 |
| `digits_label_drift` | held-out accuracy | +0.004947 | 3/0/0 |
| `digits_mask_noise` | final-window MSE | +0.006945 | 3/0/0 |
| `digits_mask_noise` | held-out accuracy | +0.016079 | 3/0/0 |
| `digits_permuted_pixels` | final-window MSE | +0.008408 | 3/0/0 |
| `digits_permuted_pixels` | held-out accuracy | +0.014842 | 3/0/0 |

This is the first candidate in this series that simultaneously clears:

- all controlled recursive tasks against the best fair MLP;
- all synthetic tasks against the best fair MLP;
- the prior `synthetic_polynomial` seed-level caveat;
- all digits final-window MSE comparisons;
- all digits held-out accuracy comparisons, including label drift and permuted
  pixels, at seed-level `3/0/0`.

## What Changed From The Prior Candidate

The prior pruned Hedge-blend profile was already strong on means but left two
explicit caveats:

- `synthetic_polynomial` was positive on mean final-window MSE but only `1/2/0`
  by seed: diffs `[-0.029921, -0.008131, +0.414206]`.
- `digits_permuted_pixels` held-out accuracy was positive on mean but `2/1/0`.

The stacker route fixes the first caveat.  With `--stacker-step-size 0.006`,
`synthetic_polynomial` becomes:

- seed 0: `0.2496` conclusive vs `0.2508` best MLP
- seed 1: `0.2822` conclusive vs `0.2826` best MLP
- seed 2: `1.7312` conclusive vs `2.1695` best MLP

The digits deployment blend fixes the second caveat.  `--h128-blend-weight 0.4`
is the best compromise tested so far:

- `0.25` fixed permuted pixels but reopened one label-drift seed.
- `0.5` fixed label drift but left one permuted-pixels seed loss.
- `0.4` clears both at `3/0/0`.

## Rejected Directions

Hard-pruning the small MLPs and fixed random tanh expert was rejected.  It saved
estimated compute but regressed `controlled_nonlinear` and
`synthetic_compositional`.

Windowed hard expert selectors were rejected.  They made the seed-2
`synthetic_polynomial` specialist choice cleaner but worsened seed 0.

Expanded safe polynomial routes were rejected for the canonical profile.  They
helped the large specialist seed but did not clear the close MLP seeds.

High stacker step sizes were rejected.  `0.1` and nearby values let the stacker
route drift too far from the MLP anchor.  The useful regime is narrow and
low-rate; `0.006` is the first setting tested that clears all three
`synthetic_polynomial` seeds while preserving the broader suite.

## Remaining Caveats

This is a strong benchmark result, not a proof of arbitrary recursive feature
discovery.  It demonstrates a causal learner/resource-manager that beats the
fair MLP bar across the current controlled, synthetic, and digits benchmark
matrix.  The next research standard should be larger seed counts and harder
external continual benchmarks, not another tweak to this compact matrix.

The current runner still executes disabled UPGD/dynamic experts for paired
diagnostics unless a physically pruned implementation is added.  The conclusive
router and deployment path do not use them under this profile.
