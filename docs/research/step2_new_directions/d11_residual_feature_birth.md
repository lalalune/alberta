# D11 Residual Feature Birth Assessment

## Purpose

D11 tested a non-router universal-learner candidate for Step 2.  The learner is
a single predictor: a bounded linear/RLS readout over a growing feature library.
It predicts before updating, updates the readout every timestep, stores recent
prequential residuals, and periodically creates new feature candidates whose
activations correlate with those residuals.

The candidate was meant to answer a specific concern: a router over MLPs is not
a universal learner.  D11 therefore does not route among MLP predictions, does
not stack MLP outputs, and does not use an MLP residual expert.  All prediction
comes from one linear readout over born features.

## Implemented Algorithm

Script:

```bash
examples/The Alberta Plan/Step2/new_directions/d11_residual_feature_birth.py
```

State:

- fixed base features: bias, raw coordinates, squares, and pair products on
  low-dimensional streams;
- born features in stable slots up to a fixed budget;
- one multi-head RLS readout matrix;
- a residual evidence buffer containing recent observations, active target
  masks, residuals, and online feature activations;
- utility traces for activation magnitude and readout coefficient magnitude;
- protected parent slots for composed features.

Online loop:

1. evaluate active features;
2. predict with one linear readout;
3. compute masked residuals from active targets only;
4. update the readout with RLS;
5. store observation, feature vector, residual, and target mask;
6. at allocation intervals, generate candidate features;
7. normalize each candidate on the residual buffer;
8. score candidates by normalized residual covariance;
9. install the top candidates, or replace weak mature unprotected features when
   the config allows replacement.

Feature families tested:

- `poly`: degree-2/3/4 input monomials;
- `tanh`: random tanh ridge features;
- `fourier`: random sine/cosine features;
- `bump`: local radial features centered on high-residual samples;
- `compose`: products or tanh combinations of already useful feature slots;
- `deep`: born two-layer tanh operator features;
- `imprint`: residual-imprinted directions from recent `X^T residual`, with
  bias and frequency sweeps.

One important engineering fix was added during the run: composition scoring
originally recomputed the whole feature library over the residual buffer, which
made wide configs too slow.  The final implementation stores feature
activations in the residual buffer at the time they are produced online, then
uses those stored activations for composition candidates.  This is both cheaper
and more temporally faithful.

## Protocol

The promotion gate was the known blocker:

```text
synthetic_compositional
3 paired seeds
1200 online steps
final window 300
fair MLP grid: mlp_h64, mlp_h128, mlp_h64_64
```

The fair MLP baseline matched prior D07 results:

- best fair MLP: `mlp_h128`;
- final-window MSE: `0.2758 +/- 0.0936`;
- online mean MSE: `0.1787`.

Because no D11 method beat best fair MLP on this gate, the conditional
follow-up to controlled frequency and digits was not run.  Running those would
not change the promotion decision: this candidate fails before the first
external expansion gate.

## Ablation Results

### Aggressive Feature Birth

Output:

```text
outputs/step2_new_directions/d11_synthetic_comp_3seed/results.json
```

| Method | Families | Final-window MSE | Runtime/seed |
|---|---|---:|---:|
| `rfb_canonical_ptfbc_b256_i20_add8` | poly, tanh, Fourier, bump, compose | `17.2919 +/- 5.1299` | `5.57s` |
| `rfb_tanh_comp_tc_b256_i20_add10` | tanh, compose | `79.6910 +/- 57.0730` | `6.30s` |
| `rfb_no_compose_ptfb_b256_i20_add8` | poly, tanh, Fourier, bump | `4.7166 +/- 2.3995` | `9.03s` |
| `rfb_poly_only_p_b192_i20_add8` | poly | `14.8727 +/- 4.6781` | `7.00s` |
| `rfb_fourier_tanh_tf_b256_i20_add10` | tanh, Fourier | `2.3816 +/- 0.4450` | `8.43s` |

Best D11 vs best MLP:

- paired diff positive favors D11: `-1.9129 +/- 0.3985`;
- wins/losses/ties: `0/3/0`.

Interpretation:

- aggressive birth plus replacement is unstable;
- composition amplifies bad parent features when the readout is not yet stable;
- polynomial-only is a poor fit for the two-layer tanh target;
- tanh/Fourier is the least bad aggressive family, but still far from MLP.

### Stabilized Feature Birth

Output:

```text
outputs/step2_new_directions/d11_synthetic_comp_stable_3seed/results.json
```

Stabilization changes:

- `rho=1.0`;
- lower RLS initial covariance;
- feature clipping at `3.0`;
- slower allocation;
- fewer births per event;
- no replacement when full.

| Method | Families | Final-window MSE | Runtime/seed |
|---|---|---:|---:|
| `rfb_stable_tanh_tf_b256_i30_add4` | tanh, Fourier | `0.7324 +/- 0.2395` | `5.70s` |
| `rfb_stable_comp_tfc_b256_i30_add4` | tanh, Fourier, compose | `0.7321 +/- 0.1473` | `6.00s` |
| `rfb_conservative_ptfc_b192_i50_add3` | poly, tanh, Fourier, compose | `0.6699 +/- 0.1357` | `2.68s` |

Best D11 vs best MLP:

- paired diff positive favors D11: `-0.3678 +/- 0.0539`;
- wins/losses/ties: `0/3/0`.

Interpretation:

- stabilization helps dramatically;
- conservative allocation is the best compute/accuracy point;
- composition is no longer catastrophic, but it does not help enough;
- the remaining gap is representational/adaptation quality, not only numerical
  instability.

### Deep-Born Operator Features

Output:

```text
outputs/step2_new_directions/d11_synthetic_comp_deep_3seed/results.json
```

This pass added born features that are themselves small two-layer tanh
operators.  This is closer to the benchmark oracle while still remaining a
single residual-selected feature library with one linear readout.

| Method | Families | Final-window MSE | Runtime/seed |
|---|---|---:|---:|
| `rfb_deep_birth_dtf_b384_i15_add8` | deep, tanh, Fourier | `1.1948 +/- 0.2992` | `16.02s` |
| `rfb_deep_comp_dtfc_b384_i15_add8` | deep, tanh, Fourier, compose | `1.1888 +/- 0.2912` | `16.38s` |

Best D11 vs best MLP:

- paired diff positive favors D11: `-0.9115 +/- 0.1976`;
- wins/losses/ties: `0/3/0`.

Interpretation:

- matching the oracle architecture with random born operators did not help;
- the cost increased about `12-18x` over MLP;
- residual correlation over a short buffer is not sufficient to select useful
  random deep operators from this candidate pool.

### Residual-Imprinted Candidate Directions

Output:

```text
outputs/step2_new_directions/d11_synthetic_comp_imprint_3seed/results.json
```

This pass generated candidate directions from recent `X^T residual` vectors,
then swept tanh biases and Fourier frequencies before applying the same
residual-correlation admission rule.

| Method | Families | Final-window MSE | Runtime/seed |
|---|---|---:|---:|
| `rfb_imprint_itfc_b256_i20_add5` | imprint, tanh, Fourier, compose | `0.6859 +/- 0.1918` | `4.78s` |
| `rfb_imprint_fast_itf_b320_i15_add6` | imprint, tanh, Fourier | `0.9892 +/- 0.2761` | `6.48s` |

Best D11 vs best MLP:

- paired diff positive favors D11: `-0.4142 +/- 0.0968`;
- wins/losses/ties: `0/3/0`.

Interpretation:

- imprinting improves over aggressive random features but not over the
  conservative stabilized config;
- fast imprinting hurts, consistent with over-admitting residual-correlated
  but low-generalization features;
- residual-gradient directions are too shallow for the two-layer compositional
  target.

## What Improved Accuracy

The following choices improved D11:

- stronger readout regularization;
- `rho=1.0` rather than forgetting in the compositional stream;
- no replacement when full;
- fewer births per event;
- slower allocation;
- feature clipping;
- storing online feature activations for composition scoring;
- mixing tanh/Fourier with conservative polynomial/compose options.

The best D11 point was:

```text
rfb_conservative_ptfc_b192_i50_add3
final-window MSE 0.6699 +/- 0.1357
```

The best residual-imprinted point was close but worse:

```text
rfb_imprint_itfc_b256_i20_add5
final-window MSE 0.6859 +/- 0.1918
```

## What Made Accuracy Worse

The following choices hurt:

- aggressive feature birth;
- replacement under high feature churn;
- large initial RLS covariance;
- composing unstable parent features;
- polynomial-only features on a tanh-compositional target;
- wide/fast random deep operators;
- fast residual-imprinted allocation.

The largest failure mode was not failure to add features.  All runs added many
features.  The failure was that residual-local feature selection admitted
features that reduced short-window residual covariance but did not generalize
well enough prequentially.

## Compute Cost

Mean runtime per seed:

- MLP baselines: roughly `0.7s` to `1.5s`;
- conservative D11: `2.68s`;
- stabilized D11: `5.70s` to `6.00s`;
- imprint D11: `4.78s` to `6.48s`;
- deep D11: `16.02s` to `16.38s`.

The best D11 method is roughly `2-4x` the fair MLP runtime while still losing by
`0.3678` final-window MSE.  The expensive deep-born variants are much slower
and less accurate.

## Verification

Commands run:

```bash
source .venv/bin/activate && ruff check "examples/The Alberta Plan/Step2/new_directions/d11_residual_feature_birth.py"

source .venv/bin/activate && MYPYPATH="src:examples/The Alberta Plan/Step2:examples/The Alberta Plan/Step2/new_directions" \
  mypy "examples/The Alberta Plan/Step2/new_directions/d11_residual_feature_birth.py"

source .venv/bin/activate && python -u "examples/The Alberta Plan/Step2/new_directions/d11_residual_feature_birth.py" \
  --smoke --output-dir outputs/step2_new_directions/d11_smoke_final \
  --note-path docs/research/step2_new_directions/d11_residual_feature_birth.md
```

Results:

- ruff passed;
- targeted mypy passed;
- final smoke run passed.

## Decision

D11 is not a canonical Step 2 candidate.

The strongest version does not beat fair MLP on the first required
synthetic-compositional gate:

```text
best D11 final-window MSE: 0.6699 +/- 0.1357
best MLP final-window MSE: 0.2758 +/- 0.0936
paired diff: -0.3678 +/- 0.0539
wins/losses/ties: 0/3/0
```

What remains useful from D11:

- residual-buffer scoring is a clean non-router feature-birth mechanism;
- conservative resource management is essential;
- composition must be gated by stable parent utility;
- cheap residual-imprinting has some signal but not enough;
- short-window residual correlation is not a sufficient objective for recursive
  feature construction on this benchmark.

The main research implication is negative but clear: the Step 2 solution likely
needs either learned feature parameters with a stronger credit-assignment
objective, a richer online second-order/operator learner, or a principled
resource manager that estimates future utility rather than immediate residual
correlation.
