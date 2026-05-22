# Worker T: Simple Single-UPGD Continuation

Date: 2026-05-05

## Scope

This note evaluates a simple fixed single learner, not a portfolio, router,
selector, or stacker. The candidate is UPGD-family and uses no hand-engineered
feature search.

Owned outputs:

- `outputs/step2_worker_t_simple_upgd_repx025_10seed/`
- helper: `output/subagents/step2_worker_t_simple_upgd_matrix.py`

## Existing Artifact Mining

Sources mined:

- `docs/research/step2_near_closure_report.md`
- `output/subagents/combined_upgd/*/upgd_digits_sweep_results.json`

The near-closure report says the current state plainly: UPGD is the strongest
single synthetic Step 2 learner, but no single online learner yet beats fair MLP
across digits IID, mask noise, permuted pixels, label drift, class-blocked
retention, and synthetic feature-construction rows.

I ranked fixed UPGD variants by the five-regime digits matrix rather than by a
single row. Positive diffs favor UPGD. Existing combined artifacts compare to
`MLP(64)`.

| Candidate | Source | Overall FW MSE diff | FW MSE W/L/T | Row FW-MSE losses | Overall test acc diff | Test acc W/L/T |
|---|---|---:|---:|---|---:|---:|
| `upgd_sum_sigma1e_4_adaptk035_065_lr05_e1` | `kappa_lr_adaptive_lr05_top_30seed` | +0.0059 | 120/30/0 | class_blocked -0.0008 | +0.0247 | 129/9/12 |
| `upgd_sum_sigma1e_4_adaptk035_065_lr05_w0` | `kappa_lr_adaptive_lr05_top_30seed` | +0.0059 | 120/30/0 | class_blocked -0.0008 | +0.0253 | 132/6/12 |
| `upgd_sum_sigma1e_4_adaptk035_065_lr06` | `kappa_lr_adaptive_top_30seed` | +0.0059 | 127/23/0 | class_blocked -0.0001 | +0.0224 | 119/22/9 |
| `upgd_sum_sigma1e_4_adaptk035_065_lr06_repx025` | `repetition_top_30seed` | +0.0057 | 147/3/0 | none | +0.0209 | 117/20/13 |
| `upgd_sum_sigma1e_4_kappa04_lr075` | `fine_tune_local_top_30seed` | +0.0042 | 146/4/0 | none | +0.0148 | 101/36/13 |

Selection: `upgd_sum_sigma1e_4_adaptk035_065_lr06_repx025`.

Reason: it is not the largest aggregate mean, but it is the strongest
30-seed fixed variant by matrix discipline: positive final-window MSE on all
five digit regimes versus `MLP(64)`, 147/150 paired FW-MSE wins, and no
row-level FW-MSE loss. The tradeoff already visible in the mined artifacts is
class-blocked retained/test accuracy: it loses that row despite winning current
tracking MSE.

## Confirmation Command

Syntax check:

```bash
source .venv/bin/activate && python -m py_compile output/subagents/step2_worker_t_simple_upgd_matrix.py
```

10-seed fixed matrix:

```bash
source .venv/bin/activate && python output/subagents/step2_worker_t_simple_upgd_matrix.py --output-dir outputs/step2_worker_t_simple_upgd_repx025_10seed --upgd-config upgd_sum_sigma1e_4_adaptk035_065_lr06_repx025 --n-seeds 10 --seed 0 --digits-steps 1200 --digits-final-window 300 --synthetic-steps 1600 --synthetic-final-window 400 2>&1 | tee outputs/step2_worker_t_simple_upgd_repx025_10seed/run.log
```

Protocol:

- Digits: `iid`, `class_blocked`, `permuted_pixels`, `mask_noise`, `label_drift`;
  1200 steps, final window 300.
- Synthetic: `polynomial`, `frequency`, `compositional`; 1600 steps, final
  window 400.
- Baselines: `MLP(64)` and `MLP(64,64)`, same step size, sparsity, layer norm,
  and ObGD.
- "Best fair MLP" means the better of `MLP(64)` and `MLP(64,64)` for each
  scenario and metric.

## Confirmation Results

Positive paired differences favor UPGD. W/L/T is paired over 10 seeds.

| Scenario | Metric | Diff vs MLP64 | W/L/T | Diff vs best fair MLP | W/L/T |
|---|---|---:|---:|---:|---:|
| digits_iid | final_window_mse | +0.0062 | 10/0/0 | +0.0062 vs `mlp64` | 10/0/0 |
| digits_iid | final_window_accuracy | +0.0207 | 10/0/0 | +0.0207 vs `mlp64` | 10/0/0 |
| digits_iid | test_mse | +0.0045 | 9/1/0 | +0.0045 vs `mlp64` | 9/1/0 |
| digits_iid | test_accuracy | +0.0199 | 10/0/0 | +0.0199 vs `mlp64` | 10/0/0 |
| digits_class_blocked | final_window_mse | +0.0003 | 7/3/0 | -0.0017 vs `mlp64_64` | 0/10/0 |
| digits_class_blocked | final_window_accuracy | +0.0003 | 4/3/3 | -0.0047 vs `mlp64_64` | 0/9/1 |
| digits_class_blocked | test_mse | -0.0044 | 3/7/0 | -0.0044 vs `mlp64` | 3/7/0 |
| digits_class_blocked | test_accuracy | -0.0128 | 2/5/3 | -0.0128 vs `mlp64` | 2/5/3 |
| digits_permuted_pixels | final_window_mse | +0.0068 | 10/0/0 | +0.0068 vs `mlp64` | 10/0/0 |
| digits_permuted_pixels | final_window_accuracy | +0.0233 | 9/1/0 | +0.0233 vs `mlp64` | 9/1/0 |
| digits_permuted_pixels | test_mse | +0.0081 | 10/0/0 | +0.0081 vs `mlp64` | 10/0/0 |
| digits_permuted_pixels | test_accuracy | +0.0384 | 9/0/1 | +0.0384 vs `mlp64` | 9/0/1 |
| digits_mask_noise | final_window_mse | +0.0085 | 10/0/0 | +0.0085 vs `mlp64` | 10/0/0 |
| digits_mask_noise | final_window_accuracy | +0.0430 | 10/0/0 | +0.0430 vs `mlp64` | 10/0/0 |
| digits_mask_noise | test_mse | +0.0077 | 10/0/0 | +0.0077 vs `mlp64` | 10/0/0 |
| digits_mask_noise | test_accuracy | +0.0391 | 10/0/0 | +0.0391 vs `mlp64` | 10/0/0 |
| digits_label_drift | final_window_mse | +0.0076 | 10/0/0 | +0.0076 vs `mlp64` | 10/0/0 |
| digits_label_drift | final_window_accuracy | +0.0400 | 9/1/0 | +0.0400 vs `mlp64` | 9/1/0 |
| digits_label_drift | test_mse | +0.0073 | 10/0/0 | +0.0073 vs `mlp64` | 10/0/0 |
| digits_label_drift | test_accuracy | +0.0278 | 10/0/0 | +0.0278 vs `mlp64` | 10/0/0 |
| synthetic_polynomial | final_window_mse | -0.0077 | 1/9/0 | -0.0461 vs `mlp64_64` | 0/10/0 |
| synthetic_frequency | final_window_mse | -0.0208 | 1/9/0 | -0.0349 vs `mlp64_64` | 2/8/0 |
| synthetic_compositional | final_window_mse | +0.0157 | 10/0/0 | +0.0157 vs `mlp64` | 10/0/0 |

Means from the generated summary:

- Class-blocked final-window MSE: `MLP64=0.0049`, `MLP64x64=0.0030`,
  `UPGD=0.0046`.
- Class-blocked test accuracy: `MLP64=0.1384`, `MLP64x64=0.1007`,
  `UPGD=0.1256`.
- Synthetic polynomial final-window MSE: `MLP64=1.0183`, `MLP64x64=0.9799`,
  `UPGD=1.0260`.
- Synthetic frequency final-window MSE: `MLP64=0.8719`, `MLP64x64=0.8578`,
  `UPGD=0.8927`.
- Synthetic compositional final-window MSE: `MLP64=0.2212`, `MLP64x64=0.2606`,
  `UPGD=0.2055`.

## Verdict

`upgd_sum_sigma1e_4_adaptk035_065_lr06_repx025` is the strongest simple fixed
single-UPGD digit candidate I found by matrix discipline. It is a real positive
result on IID, permuted pixels, mask noise, label drift, and current
class-blocked tracking versus `MLP(64)`.

It is not universal enough to replace the broader operational system:

- class-blocked loses to the best fair MLP on final-window tracking and loses
  to `MLP(64)` on held-out retained/test accuracy;
- synthetic polynomial loses 9/10 vs `MLP(64)` and 10/10 vs best fair MLP;
- synthetic frequency loses 9/10 vs `MLP(64)` and 8/10 vs best fair MLP;
- only synthetic compositional is green.

So the honest conclusion is narrower: the fixed UPGD-family variant is a strong
digits-tracking learner, not a universal Step 2 single-learner replacement. The
next simple single-learner continuation should preserve this digit behavior
while recovering the known standalone UPGD synthetic strength, especially on
polynomial and frequency rows, without adding routing or portfolio machinery.

## Verification

Commands run:

```bash
source .venv/bin/activate && python -m py_compile output/subagents/step2_worker_t_simple_upgd_matrix.py
source .venv/bin/activate && ruff check .
source .venv/bin/activate && mypy
source .venv/bin/activate && mypy src/
source .venv/bin/activate && MYPYPATH="src:examples/The Alberta Plan/Step2:output/subagents/upgd_sweep" mypy --follow-imports=silent output/subagents/step2_worker_t_simple_upgd_matrix.py
source .venv/bin/activate && pytest tests/ -v
```

Results:

- helper syntax check passed;
- `ruff check .` passed;
- bare `mypy` exits with `Missing target module, package, files, or command`;
- `mypy src/` passed;
- isolated helper mypy passed with imports silenced;
- full pytest failed: 25 failed, 1048 passed, 36 warnings.

The pytest failures are UPGD-state failures outside Worker T's owned files:
`UPGDState.__init__()` is missing `meta_trunk_log_scale`,
`meta_head_weight_log_scale`, `meta_head_bias_log_scale`,
`meta_repetition_log_scale`, and previous-gradient fields. The failure cluster
is `tests/test_upgd.py` plus two Step 2 scripts that call UPGD
(`test_step2_expert_mixture.py` and `test_step2_scr_router_search.py`).
