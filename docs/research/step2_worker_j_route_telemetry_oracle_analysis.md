# Worker J: Route Telemetry And Oracle Deployment Analysis

Date: 2026-05-05.

Scope: route telemetry/oracle analysis only. No source files were edited. The
reproducible analysis script and machine-readable summary are:

- `outputs/step2_worker_j_oracle_analysis/analyze.py`
- `outputs/step2_worker_j_oracle_analysis/analysis.json`

## Inputs Compared

The analysis loaded these full-suite candidate outputs:

| Candidate | Path | Status |
|---|---:|---|
| current best | `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json` | present |
| Worker B selector-window150 + switch0.010 | `outputs/step2_worker_b_w150_switch010_all_10seed/results.json` | present |
| Worker A eta-0.5 full | `outputs/step2_worker_a_full_hedge_eta_0p5_all_10seed/results.json` | missing |
| Worker A eta-0.5 full rerun | `outputs/step2_worker_a_full_hedge_eta_0p5_all_10seed_rerun/results.json` | present |

It also loaded the already-present weak3/focused outputs under:

- `outputs/step2_worker_a_focus_*`
- `outputs/step2_main_*weak3_10seed`
- `outputs/step2_worker_c_*synth2_w100`

`weak3` here means `controlled_rare`, `synthetic_compositional`, and
`synthetic_polynomial`.

## Full-Suite Comparison

Positive diff means the conclusive learner beats the same-run best fair MLP on
final-window MSE.

| Candidate | Records | Seed W/L/T | Mean diff vs best MLP | Remaining losses |
|---|---:|---:|---:|---|
| current best | 140 | 126/4/10 | +0.170501 | `controlled_rare` seeds 8,9; `synthetic_compositional` seed 5; `synthetic_polynomial` seed 6 |
| Worker B w150+switch010 | 140 | 126/4/10 | +0.171668 | `synthetic_frequency` seed 1; `synthetic_polynomial` seeds 0,6,7 |
| Worker A eta-0.5 rerun | 140 | 127/3/10 | +0.170769 | `controlled_rare` seeds 8,9; `synthetic_polynomial` seed 6 |

The eta-0.5 rerun is the best single full-suite candidate by seed count, but it
does not close Step 2. Worker B fixes the rare/compositional losses but moves
the failures into frequency/polynomial regimes. The current best remains the
best single full-suite candidate for `synthetic_frequency`, while Worker B is
better for `controlled_rare` and `synthetic_compositional`.

## Weak3 Focused Comparison

Top focused candidates by seed-loss count on their covered weak datasets:

| Candidate | Coverage | Weak W/L/T | Mean dataset diff | Comment |
|---|---:|---:|---:|---|
| `step2_worker_c_rec_poly_tanh_synth2_w100` | 20 seeds, synth2 only | 18/2/0 | +0.055946 | good on synthetic streams, no rare coverage |
| `step2_worker_c_baseline_synth2_w100` | 20 seeds, synth2 only | 18/2/0 | +0.052362 | similar to rec/poly/tanh |
| `step2_worker_c_all_safe_synth2_w100` | 20 seeds, synth2 only | 17/3/0 | +0.054749 | broader safe routes hurt some seeds |
| `step2_main_combo_discount099_w150_switch010_weak3_10seed` | 30 seeds | 27/3/0 | +0.043307 | fixes rare/compositional, still polynomial losses |
| `step2_main_combo_eta05_w150_switch010_weak3_10seed` | 30 seeds | 27/3/0 | +0.043263 | same profile |
| Worker B w150+switch010 | 30 seeds | 27/3/0 | +0.043084 | full-suite validated |
| Worker A eta-0.5 rerun | 30 seeds | 27/3/0 | +0.040595 | full-suite validated |
| current best | 30 seeds | 26/4/0 | +0.039714 | baseline for comparison |

Important exception: `step2_worker_a_focus_selector_window_60` is not a good
global weak3 setting (24/6/0), but it is the focused candidate that fixes the
remaining `synthetic_polynomial` seed 6 loss:

| Dataset/seed | Best focused candidate | Conclusive MSE | Strict best MLP MSE | Diff |
|---|---|---:|---:|---:|
| `synthetic_polynomial` seed 6 | `step2_worker_a_focus_selector_window_60` | 0.762178 | 0.768844 | +0.006666 |

That is the strongest evidence that seed 6 is a routing-memory issue rather
than a missing feature-capacity issue.

## Oracle Upper Bounds

### Candidate Oracle

| Oracle | Candidate set | Result | Residual losses |
|---|---|---:|---|
| Full per-seed fixed candidate | current, Worker B, eta-0.5 rerun | 129/1/10 | `synthetic_polynomial` seed 6 |
| Full per-dataset fixed candidate, optimized for loss count | current, Worker B, eta-0.5 rerun | 129/1/10 | `synthetic_polynomial` seed 6 |
| Named weak3 per-seed candidate | current, Worker B, eta-0.5, rec/poly/tanh variants | 29/1/0 | `synthetic_polynomial` seed 6 |
| All present weak3 focused per-seed candidate | all weak3 focused outputs loaded | 30/0/0 | none |

Interpretation: choosing among the validated full-suite candidates can remove
three of the four current seed losses. The last one requires a different
focused route-memory setting, not a new representation family.

### Expert Oracle Inside Current Best

Using current-best curves only:

| Oracle | Result vs best MLP |
|---|---:|
| Best fixed expert per dataset/seed | 83/0/57 |
| Per-step final-window expert oracle | 140/0/0 |

For the four current losing seeds:

| Dataset/seed | Current issue | Best fixed expert | Fixed-expert diff | Per-step expert oracle diff |
|---|---|---|---:|---:|
| `controlled_rare` seed 8 | route picked bad MLP-convex/safe mixture | `polynomial_features` | +0.036699 | +0.056842 |
| `controlled_rare` seed 9 | safe-route churn | `mlp_64x64_s01_no_ln` | +0.000000 | +0.032239 |
| `synthetic_compositional` seed 5 | small dilution loss | `mlp_h128` | +0.000000 | +0.044373 |
| `synthetic_polynomial` seed 6 | wrong route-memory timescale | `mlp_32x32_s01_no_ln` | +0.000000 | +0.313686 |

There is no evidence that the expert pool lacks the capacity to match best MLP.
The per-step expert oracle is positive everywhere, and the focused candidate
pool can beat MLP on every weak3 seed. The remaining blocker is causal route
selection, especially temporal switching among already-existing experts/routes.

## Causal Telemetry Policy Test

I tested a simple, hand-checkable rule that chooses between current best and
Worker B using only final-window prequential telemetry from the current-best
run. It does not use dataset names or seed ids.

Choose Worker B if any of these are true:

1. `all_selector_tanh > 0.90`
2. `all_selector_poly > 0.95` and `route_mlp_convex > 0.05`
3. `route_safe_poly + route_safe_rec > 0.75`, `route_expert_mlp < 0.10`,
   `all_selector_fourier < 0.50`, and `all_selector_recursive < 0.10`

Otherwise choose current best.

Result over the 140 full-suite records: 129/1/10. The only remaining loss is
`synthetic_polynomial` seed 6, where the rule stays with current best:

| Dataset/seed | Chosen | Diff | Telemetry signature |
|---|---|---:|---|
| `synthetic_polynomial` seed 6 | current best | -0.011224 | `route_safe_poly=0.493`, `route_safe_rec=0.247`, `route_expert_mlp=0.200`, `all_selector_recursive=0.163`, `all_selector_poly=0.013` |

The rule chooses Worker B on:

- `controlled_rare` seeds 5,8,9;
- all `synthetic_compositional` seeds;
- `synthetic_polynomial` seed 4;
- several harmless controlled/digits cases where both candidates remain
  positive or tied.

This is not a finished result because the policy has not been implemented and
rerun as part of the learner. It is, however, a strong diagnostic: a simple
causal telemetry rule recovers the full-candidate oracle exactly.

## Critical Assessment

The remaining failures are mostly policy-selection failures, not feature
capacity failures.

Evidence:

- Current best loses four seeds, but the full candidate oracle loses only one.
- A telemetry-only current-vs-Worker-B rule also loses only one.
- The current-run fixed expert oracle never loses to best MLP.
- The current-run per-step expert oracle beats best MLP on every seed.
- The all-focused weak3 per-seed candidate oracle closes all 30 weak3 seeds.

The hard blocker is narrower: no single validated full-suite candidate or
simple current-vs-Worker-B selector closes `synthetic_polynomial` seed 6. That
seed is characterized by mixed safe-recursive/safe-polynomial routing,
meaningful direct expert-MLP exposure, low polynomial selector mass, and
nonzero recursive selector mass. The global selector-window-60 setting fixes
this seed but damages other weak3 seeds. Therefore the next learner should not
adopt window 60 globally; it should maintain multiple route-memory timescales
and let telemetry select among them.

## Recommendations Ranked By Expected Payoff

1. Implement a causal multi-window route selector.
   Maintain route scores at windows 60, 100, and 150 in parallel, then expose
   these as deployment routes or route-score candidates. The empirical target is
   to keep current/Worker-B behavior on rare/compositional/frequency while
   letting the window-60 path rescue `synthetic_polynomial` seed 6. This is the
   most direct route to 130/0/10 because the all-focused weak3 oracle already
   shows the needed behavior exists.

2. Implement the telemetry current-vs-Worker-B gate as a real causal deployment
   policy and rerun the full suite.
   The post-hoc telemetry rule achieves 129/1/10 without dataset or seed ids.
   This should be implemented as a learned or fixed gate over route fractions
   and selector fractions, then rerun end-to-end to establish whether the signal
   survives without hindsight.

3. Add a synthetic-polynomial seed-6 rescue route focused on no-LN MLP exposure.
   Seed 6's signature is `route_expert_mlp=0.200`, low polynomial selector
   mass, and mixed safe-recursive/safe-polynomial routing. A candidate route
   that follows the no-LN MLP selector under this signature, or a short-window
   route-memory branch specific to high expert-MLP exposure, is more promising
   than adding another static feature family.

4. Do not prioritize new representation families for this blocker.
   The expert oracle says the existing expert set already contains enough
   information to match or beat best MLP. New feature families may improve mean
   MSE, but they are not the shortest path to universal Step 2 closure on the
   current matrix.
