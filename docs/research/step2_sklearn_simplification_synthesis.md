# Step 2 Sklearn-Digits Simplification Synthesis

Date: 2026-05-04

## Bottom Line

The best current sklearn-digits Step 2 result is not "UPGD beats MLP." UPGD alone loses to the fair MLP on external digits. The robust win comes from prediction-space averaging across live learners, and the simplest tracking explanation is even sharper:

- On sklearn-digits, most of the online tracking gain comes from a convex ensemble over MLP widths.
- A uniform MLP-width ensemble is already competitive with discounted Hedge.
- UPGD is still useful for class-blocked held-out retention and as a hedge against out-of-class synthetic structure.
- `dynamic_sparse` is not needed for the digits deployment candidate; the 10-seed cost/resource replay says `no_dynamic_convex` preserves the strict win pattern at lower static cost.

The simplest credible digits system is therefore:

1. Track with a uniform or slowly discounted convex ensemble over `mlp_h64`, `mlp_h128`, and `mlp_h64_64`.
2. Keep `upgd_low_noise` as a retention/synthetic hedge, not as the primary digits tracker.
3. Drop `dynamic_sparse` from the digits-facing deployment candidate unless synthetic feature-discovery experiments need it.
4. Use retention routing only for explicit class-imbalance hazards; do not use selector routing for normal tracking.

## Core Evidence

### Strict Portfolio

The strict 10-seed portfolio beats the best MLP on every sklearn-digits regime:

| Regime | Mix Final MSE | Best MLP Final MSE | Diff | Mix Test Acc | Best MLP Test Acc | Diff |
|---|---:|---:|---:|---:|---:|---:|
| `digits_class_blocked` | 0.0029 | 0.0029 | +0.0000 | 0.2245 | 0.1006 | +0.1239 |
| `digits_iid` | 0.0228 | 0.0302 | +0.0074 | 0.9449 | 0.9310 | +0.0139 |
| `digits_label_drift` | 0.0334 | 0.0396 | +0.0062 | 0.9163 | 0.8946 | +0.0217 |
| `digits_mask_noise` | 0.0385 | 0.0483 | +0.0098 | 0.8616 | 0.8453 | +0.0163 |
| `digits_permuted_pixels` | 0.0407 | 0.0484 | +0.0078 | 0.8931 | 0.8755 | +0.0176 |

Source: `outputs/step2_canonical/universal_portfolio_strict_results.json`.

### MLP-Only Convex Is Almost Enough

Three-seed local router ablations show that the MLP-width ensemble keeps most of the digits win:

| Variant | Mean Final-MSE Diff vs Best MLP | Mean Test-Acc Diff vs Best MLP |
|---|---:|---:|
| `all_convex` | +0.0061 | +0.0296 |
| `mlp_uniform` | +0.0053 | +0.0266 |
| `mlp_hedge_discount99` | +0.0053 | +0.0272 |
| `mlp_hedge_eta1` | +0.0049 | +0.0259 |
| `mlp_hedge_eta3` | +0.0031 | +0.0224 |
| `mlp_hedge_eta10` | +0.0011 | +0.0165 |

Interpretation: aggressive Hedge collapses toward selector behavior and loses ensemble benefit. Uniform or very slow adaptation is the simplest good tracker.

Sources:

- `output/local_simplify/portfolio_mlp_uniform_digits_3seed/results.json`
- `output/local_simplify/portfolio_mlp_convex_digits_3seed/results.json`
- `output/local_simplify/portfolio_mlp_convex_discount99_digits_3seed/results.json`
- `output/local_simplify/portfolio_convex_digits_3seed/results.json`

### Selector Routing Is A Negative Control

`all_selector` nearly collapses to the best single MLP and loses the mixture advantage on normal digits regimes. This is exactly what theory predicts: hard selection throws away variance reduction and complementary errors.

Source: `output/local_simplify/portfolio_all_selector_digits_3seed/results.json`.

### Cost/Resource Replay

The 10-seed cost/resource replay ranked `no_dynamic_convex` third overall and effectively tied strict:

- `strict_5_convex`: 5/5 datasets not losing best MLP, macro final-MSE diff +0.0059, macro test-accuracy diff +0.0308, static cost 7.46.
- `no_dynamic_convex`: 5/5 datasets not losing best MLP, macro final-MSE diff +0.0058, macro test-accuracy diff +0.0307, static cost 6.11.
- `mlp_only_convex`: 4/5 datasets not losing best MLP, macro final-MSE diff +0.0048, macro test-accuracy diff +0.0031, static cost 4.86.

The missing dataset for MLP-only is class-blocked retention: without UPGD deployment, held-out class-blocked accuracy falls from 0.2245 to 0.1006.

Source: `output/subagents/cost_resource_allocation/REPORT.md`.

## Ten Directions Assessed

| Direction | Result | Keep/Drop |
|---|---|---|
| Width ensemble | Strong positive. Width diversity plus convex averaging explains most digits tracking gains. | Keep; make this the simple baseline. |
| UPGD noise/schedule | Negative on digits as standalone, including low/no perturbation smoke. | Keep only as hedge/retention/synthetic expert. |
| Fair MLP sweep | Best single MLP remains below ensemble; portfolio win is not just a weak baseline artifact. | Keep fair MLP grid as comparator. |
| Router simplification | Convex works; selectors fail; high eta hurts. | Use uniform or slow Hedge. |
| Dynamic sparse | Not needed for digits deployment; no-dynamic retains 5/5 win pattern. | Drop from digits path; keep for synthetic-only research. |
| Target coding | Centered labels give tiny test-accuracy nudges but worse code MSE and no robust final-window gain. | Drop for now. |
| Preprocessing/whitening | Existing train-split z-score is already the fair baseline. No evidence whitening fixes UPGD. | Keep z-score only. |
| Last-layer/fixed features | Random fixed-feature readout probe underperformed badly and was unstable at wider banks. | Drop as explanation for MLP win. |
| Retention/tracking split | Slow/EMA memory can improve class-blocked held-out accuracy but destroys tracking; simple guard failed. | Research item, not current replacement. |
| Cost/resource allocation | Mild cost penalty and no-dynamic replay preserve strict behavior; hard pruning to MLP-only loses retention. | Promote no-dynamic digits candidate. |

## Paper 2603.21852 Read

The paper "All elementary functions from a single binary operator" proposes `eml(x,y)=exp(x)-ln(y)` plus constant `1` as a universal elementary-function grammar. This is not evidence that EML should replace MLPs or UPGD in Step 2. It is a symbolic representation result, not an online continual-learning result.

The only scientifically clean transfer is a feature-construction prior:

- test a bounded real `safe_eml(a,b)=clip(exp(clip(a))-log(softplus(b)+eps))` as a candidate feature op;
- quarantine it inside the feature generator;
- judge it by causal utility, NaN/saturation rate, and held-out digits accuracy;
- compare against a matched-budget multi-op feature generator.

The stronger, cleaner idea from the paper is not EML itself but single-primitive topology learning: use one stable binary primitive and learn parent/terminal routing under a fixed feature budget. The right ablation is:

| Factor | Variant |
|---|---|
| Operation vocabulary | current multi-op vs one stable binary op |
| Routing/resource allocation | learned vs uniform |

This tests whether universality comes from operation diversity or from learned wiring/resource allocation.

Sources:

- arXiv abstract: https://arxiv.org/abs/2603.21852
- arXiv PDF: https://arxiv.org/pdf/2603.21852
- HF paper metadata: https://huggingface.co/papers/2603.21852
- Reviewer notes: `output/subagents/paper_2603_21852/`

## Critical Recommendation

For the sklearn benchmark, stop presenting the full strict portfolio as a mysterious complex system. The causal story is simple:

1. MLPs of different widths make correlated but not identical errors.
2. Convex averaging reduces error and stabilizes non-stationary tracking.
3. Hard selection loses that averaging benefit.
4. UPGD is not a better digits learner, but it carries a retention/synthetic option the MLP ensemble lacks.
5. Dynamic sparse is not paying rent on digits.

The next implementation target should be a named "simple digits candidate":

`simple_digits_step2 = uniform_mlp_width_ensemble + upgd_retention_guard`

Then evaluate it against:

- strict 5-expert portfolio;
- best single fair MLP;
- MLP-only uniform ensemble;
- no-dynamic convex;
- published external stressors.

Acceptance rule: keep the simpler system only if it matches strict on the 10-seed digits matrix and does not regress synthetic out-of-class results enough to break the broader Step 2 claim.
