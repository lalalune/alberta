# Worker AB Calibrated D18 Simplex

Scope: simple target-geometry/calibration experiments for single learners.  The
candidate side is always one D18-style additive learner with a calibrated
readout; there is no output router, no portfolio, and no MLP expert inside the
candidate.

## Commands

Smoke:

```bash
source .venv/bin/activate
python output/subagents/step2_worker_ab_calibrated_simplex.py --smoke --output-dir outputs/step2_worker_ab_smoke --note-path outputs/step2_worker_ab_smoke_NOTE.md
```

Blocker rows:

```bash
source .venv/bin/activate
python output/subagents/step2_worker_ab_calibrated_simplex.py --datasets digits_class_blocked,digits_mask_noise --n-seeds 10 --steps 1200 --final-window 300 --output-dir outputs/step2_worker_ab_blockers_10seed --note-path docs/research/step2_worker_ab_calibrated_simplex.md
```

The blocker run completed and wrote
`outputs/step2_worker_ab_blockers_10seed/results.json` plus
`outputs/step2_worker_ab_blockers_10seed/SUMMARY.md`.

## Existing D18 Diagnosis

The previous D18 simplex rows are raw-positive mainly because the fair MLP is
judged by raw squared error, while D18 is already emitting one-hot simplex
predictions after target-geometry detection.  A fair projected-MLP check changes
the comparator to hard one-hot MSE computed from MLP final-window accuracy:
`2 / 10 * (1 - accuracy)`.

On `digits_class_blocked`, prior D18 simplex/gain-l2 has final MSE `0.0029`
with final accuracy `0.9857`.  The projected MLP bar is `0.0017`, from
`mlp_h64_64` final-window accuracy `0.9917`, so D18 loses all 10 paired seeds
against the projected bar despite tying the raw MLP MSE bar.

On `digits_mask_noise`, D18 is much closer: raw D18 simplex is `0.0397` versus
projected MLP `0.0395` (mean diff `-0.0002`, 5/5 seeds), while gain-l2 is
`0.0389` versus projected MLP `0.0395` (mean diff `+0.0006`, 6/4 seeds).  This
is a borderline calibration/accuracy issue rather than a large modeling gap.

The prior context-trace runs explain the failure mode.  `context_trace_s4_p3`
beats projected MLP on `digits_class_blocked` online MSE (`0.0015` vs `0.0017`,
9/1 seeds), but its held-out test accuracy is only `0.1006`, matching the weak
blocked-class MLP test result.  It is using causal block-label context, not
learning reusable digit geometry.

## Blocker Results

Positive diffs favor the AB candidate.  Projected-MLP MSE is computed from the
same-run fair MLP final-window accuracy, not from a newly trained baseline.

| Dataset | Method | Final MSE | Final Acc | Test Acc | Diff vs raw MLP | Diff vs projected MLP | Test Acc Diff |
|---|---|---:|---:|---:|---:|---:|---:|
| digits_class_blocked | `ab_hard_update` | 0.0043 +/- 0.0002 | 0.9787 +/- 0.0012 | 0.3620 +/- 0.0268 | -0.0014 +/- 0.0002 | -0.0026 +/- 0.0003 | +0.2262 +/- 0.0216 |
| digits_class_blocked | `ab_softmax_t010_update` | 0.0029 +/- 0.0002 | 0.9817 +/- 0.0013 | 0.3894 +/- 0.0173 | -0.0000 +/- 0.0002 | -0.0012 +/- 0.0002 | +0.2536 +/- 0.0170 |
| digits_class_blocked | `ab_softmax_t020_update` | 0.0036 +/- 0.0001 | 0.9770 +/- 0.0014 | 0.3904 +/- 0.0135 | -0.0007 +/- 0.0002 | -0.0020 +/- 0.0002 | +0.2545 +/- 0.0154 |
| digits_class_blocked | `ab_margin_t010_m025_update` | 0.0028 +/- 0.0001 | 0.9820 +/- 0.0010 | 0.3935 +/- 0.0198 | +0.0001 +/- 0.0001 | -0.0011 +/- 0.0002 | +0.2577 +/- 0.0208 |
| digits_class_blocked | `ab_prior_boost_hard_update` | 0.0051 +/- 0.0003 | 0.9743 +/- 0.0016 | 0.1006 +/- 0.0004 | -0.0023 +/- 0.0003 | -0.0035 +/- 0.0003 | -0.0353 +/- 0.0119 |
| digits_class_blocked | `ab_prior_balance_softmax_t010_update` | 0.0151 +/- 0.0005 | 0.9010 +/- 0.0039 | 0.6631 +/- 0.0131 | -0.0122 +/- 0.0005 | -0.0134 +/- 0.0005 | +0.5273 +/- 0.0203 |
| digits_mask_noise | `ab_hard_update` | 0.0543 +/- 0.0025 | 0.7287 +/- 0.0124 | 0.7833 +/- 0.0176 | -0.0063 +/- 0.0019 | -0.0148 +/- 0.0019 | -0.0527 +/- 0.0125 |
| digits_mask_noise | `ab_softmax_t010_update` | 0.0329 +/- 0.0018 | 0.7847 +/- 0.0117 | 0.8226 +/- 0.0144 | +0.0152 +/- 0.0013 | +0.0066 +/- 0.0018 | -0.0134 +/- 0.0130 |
| digits_mask_noise | `ab_softmax_t020_update` | 0.0277 +/- 0.0017 | 0.8097 +/- 0.0139 | 0.8458 +/- 0.0130 | +0.0204 +/- 0.0012 | +0.0118 +/- 0.0018 | +0.0098 +/- 0.0102 |
| digits_mask_noise | `ab_margin_t010_m025_update` | 0.0343 +/- 0.0019 | 0.7757 +/- 0.0119 | 0.8030 +/- 0.0162 | +0.0137 +/- 0.0015 | +0.0052 +/- 0.0020 | -0.0330 +/- 0.0160 |
| digits_mask_noise | `ab_prior_boost_hard_update` | 0.0685 +/- 0.0031 | 0.6577 +/- 0.0154 | 0.7167 +/- 0.0170 | -0.0205 +/- 0.0024 | -0.0290 +/- 0.0023 | -0.1193 +/- 0.0168 |
| digits_mask_noise | `ab_prior_balance_softmax_t010_update` | 0.0312 +/- 0.0017 | 0.7880 +/- 0.0107 | 0.8315 +/- 0.0135 | +0.0169 +/- 0.0013 | +0.0083 +/- 0.0018 | -0.0045 +/- 0.0125 |

## Verdict

No fixed single-learner calibration is green on both blocker rows.  The all-14
run was not launched because every AB variant loses `digits_class_blocked`
against the projected-MLP bar.

The principled result is `ab_softmax_t020_update` on `digits_mask_noise`: a
probability-simplex readout is trained by squared error and beats both raw and
projected MLP bars while also slightly improving held-out test accuracy.  That
is real target-geometry alignment, not routing.

The class-blocked result remains unresolved.  Balanced prior correction greatly
improves held-out accuracy (`+0.5273` vs best MLP) but sacrifices online
final-window MSE because it deliberately refuses to exploit the current blocked
label prior.  Prior boosting is the opposite and is best viewed as a classifier
patch; here it fails both online MSE and held-out accuracy.  The earlier
context-trace positive is also a classifier/block-prior patch rather than an
honest single-learner closure.
