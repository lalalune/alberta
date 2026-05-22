# Step 2 Universal Portfolio Assessment

This note records the strict Step 2 closure pass after the first universal
portfolio run exposed three remaining weaknesses:

- synthetic compositional final-window MSE versus the best fair MLP width;
- held-out digits accuracy on mask-noise and label-drift;
- class-blocked retained accuracy despite strong current-block tracking.

Supersession note: this remains the canonical record for the earlier
eight-regime universal-portfolio result. The broader controlled/synthetic/digits
all-suite claim is now carried by
`outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_results.json`,
which reaches `130/0/10` seed-level final-window MSE wins/losses/ties against
the same-run best fair MLP.

## Canonical Candidate

`step2_universal_portfolio.py` now promotes a live prediction-space portfolio
over five temporally uniform experts:

- `mlp_h64`;
- `mlp_h128`;
- `mlp_h64_64`;
- `upgd_low_noise`;
- `dynamic_sparse`.

At every time step, every expert predicts before update, the portfolio emits a
convex prediction, every expert then updates on the same example, and the
router state updates from the observed loss. There is no replay, no batch
training, and no post-hoc per-step oracle selection.

Two changes closed the strict gaps:

1. **Lower Hedge eta (`1.0`).** The previous eta (`8.0`) was too
   winner-take-all. Lower eta preserves useful convex averaging between fair
   MLP widths and the plasticity experts. This specifically closes the
   compositional strict gap.
2. **Online class-imbalance MSE guard.** Current-block class-blocked digits
   need `mlp_h64_64` for final-window tracking MSE, while retained held-out
   accuracy needs UPGD. The online guard uses only previously observed target
   classes: once lifetime class coverage is broad but the recent window covers
   a narrow class subset, online tracking routes to `mlp_h64_64`. The existing
   held-out deployment router still shifts deployment weight to UPGD for
   retained accuracy.

The promoted command is:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_universal_portfolio.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --dynamic-rewire-interval 240 \
  --output-dir outputs/step2_universal_portfolio_strict_10seed \
  --note-path docs/research/step2_universal_portfolio_strict_10seed.md
```

The canonical copy is stored at:

- `outputs/step2_canonical/universal_portfolio_strict_results.json`;
- `outputs/step2_canonical/universal_portfolio_strict_SUMMARY.md`.

The 30-seed risk-check JSONs are also copied into `outputs/step2_canonical/`
so the regression tests do not depend on ignored local output folders:

- `outputs/step2_canonical/universal_portfolio_compositional_30seed_results.json`;
- `outputs/step2_canonical/universal_portfolio_risk_30seed_results.json`;
- `outputs/step2_canonical/universal_portfolio_digits_30seed_results.json`.

## Strict 10-Seed Result

Positive differences favor the portfolio. For MSE, the difference is best MLP
minus portfolio. For accuracy, it is portfolio minus best MLP.

| Regime | Final-window MSE vs best MLP | Wins/losses/ties | Test accuracy vs best MLP |
|---|---:|---:|---:|
| `synthetic_polynomial` | `+0.0414 +/- 0.0072` | `10/0/0` | n/a |
| `synthetic_frequency` | `+0.0168 +/- 0.0128` | `6/4/0` | n/a |
| `synthetic_compositional` | `+0.0018 +/- 0.0008` | `7/3/0` | n/a |
| `digits_iid` | `+0.0071 +/- 0.0001` | `10/0/0` | `+0.0128 +/- 0.0023` |
| `digits_class_blocked` | `+0.0000 +/- 0.0000` | `0/0/10` | `+0.0970 +/- 0.0114` |
| `digits_permuted_pixels` | `+0.0075 +/- 0.0004` | `10/0/0` | `+0.0156 +/- 0.0027` |
| `digits_mask_noise` | `+0.0094 +/- 0.0002` | `10/0/0` | `+0.0158 +/- 0.0054` |
| `digits_label_drift` | `+0.0055 +/- 0.0002` | `10/0/0` | `+0.0128 +/- 0.0035` |

This closes the strict current matrix:

- no negative mean final-window MSE versus best fair MLP width;
- no negative mean held-out digit accuracy versus best fair MLP width;
- class-blocked tracking is tied to the best current-block MLP while retained
  held-out accuracy beats the MLP grid by deploying UPGD.

## 30-Seed Risk Checks

The rows most likely to fail under more seeds were scaled separately.

### Synthetic Compositional

`outputs/step2_canonical/universal_portfolio_compositional_30seed_results.json`

- Final-window MSE vs best MLP: `+0.0007 +/- 0.0007`.
- Paired wins/losses/ties: `22/8/0`.

The effect is small but positive by mean. This is the exact row that failed the
previous eta-8 portfolio.

### Synthetic Frequency

`outputs/step2_canonical/universal_portfolio_risk_30seed_results.json`

- Final-window MSE vs best MLP: `+0.0187 +/- 0.0137`.
- Paired wins/losses/ties: `15/15/0`.

The sign-count is balanced, but the mean remains positive because the portfolio
captures larger wins on high-loss seeds. This is weaker than the digits result,
but it satisfies the predeclared mean strict bar.

### Digits Class-Blocked

`outputs/step2_canonical/universal_portfolio_risk_30seed_results.json`

- Final-window MSE vs best MLP: `+0.0000 +/- 0.0000`.
- Paired wins/losses/ties: `0/0/30`.
- Held-out test accuracy vs best MLP: `+0.1203 +/- 0.0085`.
- Held-out paired wins/losses/ties: `30/0/0`.

This row is now deliberately two-objective: online tracking ties the best
current-block MLP, while held-out retained accuracy uses the UPGD deployment
route.

### Non-Blocked Digits

`outputs/step2_canonical/universal_portfolio_digits_30seed_results.json`

| Regime | Final-window MSE vs best MLP | Wins/losses/ties | Test accuracy vs best MLP | Wins/losses/ties |
|---|---:|---:|---:|---:|
| `digits_iid` | `+0.0073 +/- 0.0001` | `30/0/0` | `+0.0106 +/- 0.0013` | `26/2/2` |
| `digits_mask_noise` | `+0.0094 +/- 0.0002` | `30/0/0` | `+0.0169 +/- 0.0025` | `25/4/1` |
| `digits_label_drift` | `+0.0056 +/- 0.0001` | `30/0/0` | `+0.0116 +/- 0.0019` | `25/4/1` |
| `digits_permuted_pixels` | `+0.0075 +/- 0.0002` | `30/0/0` | `+0.0152 +/- 0.0019` | `27/2/1` |

The previous held-out accuracy regressions on mask-noise and label-drift are
closed at 30 seeds.

## Critical Interpretation

The strict operational Step 2 bar is now met for the current repository matrix:

- vector supervised prediction with nonlinear approximation;
- fair MLP width grid comparator;
- synthetic out-of-class streams;
- external digits streams with held-out retained evaluation;
- temporally uniform online updates;
- no mean regression on final-window MSE or held-out digit accuracy.

This should not be over-read as a proof of arbitrary recursive feature
construction. The mechanism that closes the original matrix is a conservative
portfolio over existing Step 2 learners, plus a hand-specified class-imbalance
guard.

A follow-up pass adds a learned contextual resource manager and harder
stateful external digits streams in
`outputs/step2_canonical/resource_manager_stateful_external_results.json`. The
tracking manager improves final-window MSE versus the static fair MLP on
recurrent pixel permutations, recurrent feature-mask/noise states, and
class-blocked retention (`10/10` wins on each). A separate prototype-balanced
retention manager learns a deployment allocation from online class prototypes
and improves held-out accuracy on all three harder streams, including
class-blocked retention (`10/10` wins).

The remaining caveat is narrower: this still is not a proof of arbitrary
recursive construction or a reproduction of Dohare-style Slowly-Changing
Regression / Permuted MNIST numbers. It is, however, no longer accurate to list
"learned resource manager" or "harder stateful external digits benchmark" as
unfilled Step 2 gaps for the current repo evidence matrix.

## Published-Style Stressor Follow-Up

`step2_published_stressors.py` now makes that caveat executable. It reuses the
same strict portfolio and adds compact plus scalable Dohare-style stressors:

- a local Online Permuted MNIST analogue using sklearn digits expanded to
  28x28 and permuted across recurring task blocks;
- a true OpenML MNIST path with canonical 60k/10k split, 60,000-example task
  blocks, deterministic permutations, chunked checkpoints, resume validation,
  and status/ETA reporting;
- a lightweight Slowly-Changing Regression analogue plus a Dohare-public
  million-step SCR preset with causal router variants.

Canonical-ish local output:

- `outputs/step2_canonical/published_stressors_results.json`;
- `outputs/step2_canonical/published_stressors_SUMMARY.md`.

5 seeds x 1500 steps:

| Stressor | Primary comparison vs best fair MLP | Wins/losses/ties |
|---|---:|---:|
| 28x28 sklearn-digits permuted pixels, final-window MSE | `+0.0071 +/- 0.0007` | `5/0/0` |
| 28x28 sklearn-digits permuted pixels, held-out test accuracy | `+0.0289 +/- 0.0183` | `3/2/0` |
| Slowly-Changing Regression analogue, final-window MSE | `-0.0003 +/- 0.0004` | `3/2/0` |

Follow-up evidence narrows, but does not fully close, the published-scale
external gap. Compact true OpenML MNIST is positive, the forty-block
full-source/full-task OpenML MNIST run is positive on final-window MSE
(`+0.002250` versus best fair MLP) and held-out test accuracy over observed
permutation views (`+0.011013`), and the runner can resume toward the full
800-task protocol from an atomic checkpoint. That still leaves OPMNIST
task-count scale unclosed: 40/800 task blocks have completed, not 800/800.
Million-step SCR is
closed for the narrowed `slow_meta` causal router at 3 seeds
(`+0.00006156 +/- 0.00001598`, `3/0/0`) with the fair MLP grid preserved.

The honest status is therefore:

**Step 2 is resolved for the current strict supervised benchmark acceptance
matrix. The broader Alberta Plan research problem of general smart recursive
feature construction, native deep feature lifecycle, TD/GVF-target feature
finding, and full published-scale OPMNIST reproduction remains open research
work. The published-style pass now includes true MNIST and million-step SCR
evidence, but it does not yet complete the 800-task OPMNIST run.**
