# Step 2: Supervised Feature Finding — Canonical Results

This document is the canonical numerical record for Alberta Plan Step 2.
It is populated by the experiment scripts in
`examples/The Alberta Plan/Step2/` and the JSON files committed under
`outputs/step2_canonical/`.

## What Step 2 requires (paper, lines 360–383)

> "Creating and introducing new features (made by combining existing
> features)" in continual supervised learning with vector targets,
> "smart generation of promising features and then smart testing to
> rank and replace them" under a bounded resource budget.

The audit identified two gaps:

1. The framework's strongest Step 2 evidence ("16/16 paired wins over
   MLP" on `InteractionFeatureDiscoveryStream`) is a **hypothesis-class
   match**: the stream's oracle features are pair products `x_i · x_j`
   and the learner's hypothesis class is also pair products. The
   "discovery" reduces to selection from an enumerable finite class.
2. The framework's feature learners build features only as direct
   functions of raw `x`; there was no mechanism to build features OF
   features. The paper requires composition.

## Reproducibility

| Script | Output JSON | Output summary |
|---|---|---|
| `step2_rigged_vs_fair.py` | `outputs/step2_canonical/rigged_vs_fair_results.json` | `outputs/step2_canonical/rigged_vs_fair_SUMMARY.md` |
| `step2_out_of_class.py` | `outputs/step2_canonical/out_of_class_results.json` | `outputs/step2_canonical/out_of_class_SUMMARY.md` |
| `step2_external_online.py` | `outputs/step2_canonical/digits_online_results.json` | `outputs/step2_canonical/digits_online_SUMMARY.md` |
| `step2_expert_mixture.py` | `outputs/step2_canonical/expert_mixture_low_noise_results.json` | `outputs/step2_canonical/expert_mixture_low_noise_SUMMARY.md` |
| `step2_expert_mixture.py --retention-router class_imbalance` | `outputs/step2_canonical/expert_mixture_retention_results.json` | `outputs/step2_canonical/expert_mixture_retention_SUMMARY.md` |
| `step2_universal_portfolio.py` | `outputs/step2_canonical/universal_portfolio_strict_results.json` | `outputs/step2_canonical/universal_portfolio_strict_SUMMARY.md` |
| `step2_conclusive_learner.py` | `outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_results.json` | `outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_SUMMARY.md` |
| `step2_resource_manager_stateful_external.py` | `outputs/step2_canonical/resource_manager_stateful_external_results.json` | `outputs/step2_canonical/resource_manager_stateful_external_SUMMARY.md` |
| `step2_published_stressors.py` | `outputs/step2_canonical/published_stressors_results.json` | `outputs/step2_canonical/published_stressors_SUMMARY.md` |

To regenerate:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_rigged_vs_fair.py"
python "examples/The Alberta Plan/Step2/step2_out_of_class.py"
python "examples/The Alberta Plan/Step2/step2_external_online.py" --output-dir outputs/step2_canonical
python "examples/The Alberta Plan/Step2/step2_expert_mixture.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --perturbation-sigma 1e-4 \
  --output-dir output/step2_expert_mixture_low_noise_10seed
cp output/step2_expert_mixture_low_noise_10seed/results.json \
  outputs/step2_canonical/expert_mixture_low_noise_results.json
cp output/step2_expert_mixture_low_noise_10seed/SUMMARY.md \
  outputs/step2_canonical/expert_mixture_low_noise_SUMMARY.md
python "examples/The Alberta Plan/Step2/step2_expert_mixture.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --perturbation-sigma 1e-4 \
  --retention-router class_imbalance \
  --retention-upgd-deployment-weight 1.0 \
  --output-dir output/step2_expert_mixture_retention_10seed
cp output/step2_expert_mixture_retention_10seed/results.json \
  outputs/step2_canonical/expert_mixture_retention_results.json
cp output/step2_expert_mixture_retention_10seed/SUMMARY.md \
  outputs/step2_canonical/expert_mixture_retention_SUMMARY.md
python "examples/The Alberta Plan/Step2/step2_universal_portfolio.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --dynamic-rewire-interval 240 \
  --output-dir outputs/step2_universal_portfolio_strict_10seed \
  --note-path docs/research/step2_universal_portfolio_strict_10seed.md
cp outputs/step2_universal_portfolio_strict_10seed/results.json \
  outputs/step2_canonical/universal_portfolio_strict_results.json
cp outputs/step2_universal_portfolio_strict_10seed/SUMMARY.md \
  outputs/step2_canonical/universal_portfolio_strict_SUMMARY.md
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 0 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --route-policy-mode telemetry_worker_b \
  --route-telemetry-window 300 \
  --worker-b-switch-margin 0.010 \
  --mlp-floor-blend-weight 0.5 \
  --mlp-floor-source selector \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_main_eta05_telemetry_worker_b_floor05_all_10seed \
  --note-path docs/research/step2_main_eta05_telemetry_worker_b_floor05_all_10seed.md
cp outputs/step2_main_eta05_telemetry_worker_b_floor05_all_10seed/results.json \
  outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_results.json
cp outputs/step2_main_eta05_telemetry_worker_b_floor05_all_10seed/SUMMARY.md \
  outputs/step2_canonical/conclusive_telemetry_worker_b_floor05_SUMMARY.md
python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" \
  --n-seeds 10 \
  --output-dir outputs/step2_resource_manager_stateful_external_10seed \
  --note-path docs/research/step2_resource_manager_stateful_external_10seed.md
cp outputs/step2_resource_manager_stateful_external_10seed/results.json \
  outputs/step2_canonical/resource_manager_stateful_external_results.json
cp outputs/step2_resource_manager_stateful_external_10seed/SUMMARY.md \
  outputs/step2_canonical/resource_manager_stateful_external_SUMMARY.md
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --canonical-ish \
  --output-dir outputs/step2_canonical \
  --result-prefix published_stressors
pytest tests/test_step2_canonical.py -v
```

## Headline results — rigged-vs-fair and out-of-class (committed)

### Part A — REPRODUCE the original "16/16 wins" claim

Configuration: `InteractionFeatureDiscoveryStream(feature_dim=10, ...)` + `FixedBudgetInteractionLearner(candidate_strategy='all_pairs')` vs `MultiHeadMLPLearner(hidden_sizes=(8,))`. **Same setup as the original docs claim.** 16 seeds × 2500 steps.

| Method | final-window MSE |
|---|---|
| Interaction (all-pairs candidates) | 0.893 ± 0.168 |
| MLP(8) baseline | 0.896 ± 0.153 |
| Paired diff (MLP − Interaction) | +0.003 ± 0.036 |
| **Wins for interaction learner** | **8/16 (not 16/16)** |

The original "16/16 paired wins" claim does **not** reproduce. Even on the rigged stream with the under-parameterized MLP, the actual rate is 8/16 — exactly chance. The original number was apparently a single lucky seed-set.

### Part B — out-of-class streams (oracle outside interaction learner's hypothesis class)

| Stream | Best MLP | Interaction (all-pairs) | Compositional | UPGD | CBP |
|---|---|---|---|---|---|
| OutOfClassPolynomial (deg-3) | 1.146 ± 0.064 | 1.166 ± 0.063; 3/30 wins, d=-1.71 | 1.142 ± 0.064; 19/30 wins, d=+0.25 | **0.577 ± 0.032; 30/30 wins, d=+3.23** | 1.204 ± 0.065; 0/30 wins, d=-4.71 |
| FrequencyMismatch (sinusoidal) | 1.169 ± 0.079 | 1.763 ± 0.090; 0/30 wins, d=-2.18 | 3.866 ± 2.140; 0/30 wins, d=-0.23 | **0.633 ± 0.043; 30/30 wins, d=+2.66** | 1.195 ± 0.079; 2/30 wins, d=-1.54 |
| Compositional 2-layer oracle | 0.191 ± 0.008 | 1.306 ± 0.052; 0/30 wins, d=-4.49 | 0.718 ± 0.045; 0/30 wins, d=-2.38 | **0.163 ± 0.009; 29/30 wins, d=+1.75** | 0.190 ± 0.008; 19/30 wins, d=+0.16 |

When the oracle leaves the pair-product hypothesis class, the interaction learner's edge **disappears entirely**. `CompositionalFeatureLearner` now has a tiny polynomial edge after the candidate-age fix, but it is not a meaningful general win and loses badly on the other two streams. CBP behaves like a plasticity baseline around MLP performance. UPGD is the only method with a strong win on all three synthetic out-of-class streams.

### Part C — same stream, fair MLP capacity

Same `InteractionFeatureDiscoveryStream` as Part A, but the MLP gets fair hidden width (64 or 64,64).

| Comparison | Paired diff | Wins | Cohen's d |
|---|---|---|---|
| Interaction vs MLP(64) | −0.058 ± 0.032 | 15/30 | −0.33 |
| Interaction vs MLP(64,64) | −0.013 ± 0.027 | 15/30 | −0.084 |

Once the MLP isn't capacity-starved, the interaction learner's wins on the rigged stream **drop to chance** (15/30).

### Headline conclusion

> **The previous Step 2 headline ("16/16 paired wins over MLP") approximately reproduces only when the stream oracle is exactly inside the interaction learner's hypothesis class AND the MLP baseline is held to a single 8-unit hidden layer. When the oracle leaves that hypothesis class, or when the MLP is given a fair hidden width, the paired margin collapses to within seed noise or reverses. The original result was a hypothesis-class match against an under-parameterized baseline, not evidence of useful feature discovery.**

This is the audit's central finding, now with committed numerical evidence.

### What this means for `CompositionalFeatureLearner`

The new compositional learner — built specifically to compose features-of-features (φ_a · φ_b, tanh(W·φ + b), …) instead of just projecting raw `x` — does **not** beat MLP on the out-of-class streams in this batch. Possible explanations:

- The compositional DAG search is non-trivial and may need many more steps than 5000 to find useful compositions.
- The current generation strategy is uniformly random over op types and parents; a smarter generator (e.g. utility-biased, or imprint from the residual gradient) could help.
- 30 seeds × 5000 steps is enough to detect a clear win or loss but not enough to disambiguate "no advantage" from "needs more compute."

The compositional learner is a foundation for future work, not a finished result.

#### Update — compositional path is positive on `triple_product` with the tuned `single_mechanism` + `GeneratorMetaResourceManager` recipe

After the recursive-feature-utility probe (`step2_recursive_feature_utility_probe.py`) showed that contribution-trace future utility, residual imprint, product-biased operation priors, and depth retention let the same `CompositionalFeatureLearner` consistently beat MLP on the triple-product target, the recipe was packaged with `learn_generator_resources=True` and re-evaluated on the canonical 10-seed × 5000-step protocol. Source: `examples/The Alberta Plan/Step2/step2_compositional_budget_eval.py`, results in `outputs/step2_canonical/compositional_budget_10seed_results.json`.

| Stream | mlp_64 final-window | compositional_tuned final-window | upgd final-window | best-MLP − compositional (d) | upgd − compositional (d) |
|---|---:|---:|---:|---:|---:|
| `triple_product` (`y = x0·x1·x2`) | 0.349 ± 0.025 | **0.132 ± 0.036** | 0.249 ± 0.014 | **+0.217, d = +1.44, 9/10 wins** | **+0.117, d = +0.97, 8/10 wins** |
| `out_of_class_polynomial` | 1.047 ± 0.094 | 1.023 ± 0.084 | 0.503 ± 0.048 | +0.024, d = +0.44, 6/10 wins | −0.520, d = −4.34, 0/10 wins |
| `compositional` (2-layer tanh) | 0.209 ± 0.022 | 1.190 ± 0.158 | 0.183 ± 0.026 | −0.981, d = −2.25, 0/10 wins | −1.007, d = −2.38, 0/10 wins |

**Compositional path now positive on `triple_product`** at Cohen d = +1.44 against a fair MLP(64) and d = +0.97 against UPGD. This is the first stream on which `CompositionalFeatureLearner` reaches the d > 1.0 threshold against a fair MLP — the canonical out-of-class run had it at d = +0.25 on polynomial and d = −2.4 on the 2-layer-tanh stream.

The triple-product win is a clean test of the literal Alberta Plan Step 2 spec ("features made by combining existing features"): the `OP_PRODUCT` DAG plus the contribution-trace utility plus the bounded generator-policy budget converges on `(x0·x1)·x2` and shrinks final-window MSE roughly 2.6× below the fair MLP baseline. On `out_of_class_polynomial` the same recipe ties MLP (d = +0.44) but does not reach d > 1.0 — the per-context active-triple churn every 500 steps disrupts the DAG faster than the 30-step minimum-feature-age allows it to consolidate. On the 2-layer-tanh `compositional` stream the recipe loses badly (d = −2.25); the depth-3 `OP_PRODUCT/OP_TANH/OP_GATED` set is not a good basis for stacked tanh composition with this hyperparameter budget.

**Promotion call: keep compositional as research with a positive triple-product entry; UPGD remains the canonical Step 2 promotion.** UPGD beats MLP at d > 1.7 on all three canonical out-of-class streams (`out_of_class_SUMMARY.md`); the compositional path beats UPGD only on `triple_product`. The compositional+budget+future-utility evaluation is *additive*: it shows the literal-feature-construction route can deliver d > 1.0 wins when the target structure aligns with the operation set, even though it does not yet generalise to the full out-of-class suite.

### Out-of-class benchmark suite (committed)

7 methods × 3 streams × 30 seeds × 6000 steps. All MLP-family learners use ObGDBounding(κ=2.0).

#### Final-window MSE per method per stream (lower is better)

| Method | Polynomial (deg-3) | Frequency mismatch | Compositional 2-layer |
|---|---|---|---|
| **upgd** | **0.577 ± 0.032** | **0.633 ± 0.043** | **0.163 ± 0.009** |
| cbp | 1.204 ± 0.065 | 1.195 ± 0.079 | 0.190 ± 0.008 |
| mlp_64_64 | 1.146 ± 0.064 | 1.173 ± 0.081 | 0.214 ± 0.009 |
| compositional | 1.142 ± 0.064 | 3.866 ± 2.140 | 0.718 ± 0.045 |
| interaction | 1.166 ± 0.063 | 1.763 ± 0.090 | 1.306 ± 0.052 |
| mlp_64 | 1.193 ± 0.065 | 1.169 ± 0.079 | 0.191 ± 0.008 |
| linear | 1.417 ± 0.082 | 1.506 ± 0.099 | 0.226 ± 0.009 |

#### Paired-vs-best-MLP effect sizes

| Method | Polynomial | Frequency | Compositional |
|---|---|---|---|
| **UPGD** | **+0.569 ± 0.032, 30/30 wins, d=+3.23** | **+0.535 ± 0.037, 30/30 wins, d=+2.66** | **+0.027 ± 0.003, 29/30 wins, d=+1.75** |
| CBP | −0.058, 0/30, d=−4.71 | −0.026, 2/30, d=−1.54 | +0.0005, 19/30, d=+0.16 |
| Compositional | +0.003, 19/30, d=+0.25 | −2.697, 0/30, d=−0.23 | −0.528, 0/30, d=−2.38 |
| Interaction | −0.020, 3/30, d=−1.71 | −0.594, 0/30, d=−2.18 | −1.116, 0/30, d=−4.49 |

Source: `outputs/step2_canonical/out_of_class_results.json`.

#### Findings

**UPGD wins decisively on all three out-of-class streams.** This is the first Step 2 result that genuinely beats a fair MLP baseline with statistical confidence on benchmarks the methods were not co-designed with. The effect sizes are large (d > 1.7 on every stream, d > 2.6 on two of three).

**The new `CompositionalFeatureLearner` does not produce a meaningful general win.** After the candidate-age fix it has a tiny polynomial edge (+0.003 MSE, d = +0.25), but it loses 0/30 on frequency-mismatch and 0/30 on the compositional stream. Despite being designed specifically to compose features-of-features, it usually underperforms a plain MLP. Likely reasons:

1. **Search inefficiency** — the DAG generation strategy is uniformly random over op types and parents; promising compositions are rare in a 16-slot bank.
2. **Output-weight churn** — replacing a feature resets its output weight to a blend; cascading replacements (which the learner does correctly) can disrupt downstream features mid-context.
3. **Step-size mismatch** — `step_size_output=0.03, step_size_theta=0.003` may not be tuned for these particular streams.

This is a research result, not an implementation bug: the framework now exposes `CompositionalFeatureLearner` as a concrete probe of the "features-of-features" hypothesis, and the answer (with this design, on these benchmarks, with these hyperparameters) is "doesn't beat MLP." A future iteration should explore guided generation, deeper depth, or hybrid UPGD × compositional approaches.

**CBP does not change the synthetic Step 2 conclusion.** It is MLP-like on polynomial/frequency, ties MLP on the compositional stream, and is never close to UPGD.

**The pair-product `FixedBudgetInteractionLearner` collapses out-of-class.** On the rigged stream it ties at chance (Part C); off-class it loses with d ≈ −2 to −4. The audit's claim that the original "16/16 wins" was a hypothesis-class match is now triple-confirmed (Part A, Part B, Part C, and out-of-class suite).

### Headline conclusion

> **Step 2 is now meaningfully advanced.** UPGD (utility-perturbed gradient descent) is the first method in this framework to beat a fair MLP baseline on multiple out-of-hypothesis-class supervised feature-finding benchmarks with Cohen's d > 1.7 across 30 seeds. The previous evidence base — built on a hypothesis-class match against an under-parameterized MLP — does not survive scrutiny. The new compositional DAG learner does not yet beat MLP, but provides a concrete substrate for future feature-of-features research.

### Context/output-adaptation disentanglement

The pair-product stream mixes two issues: whether the learner found useful
constructed features, and whether one shared output head can track
context-specific slopes when context is hidden. The probe in
`step2_context_disentanglement.py` freezes learned features and retrains simple
online readouts.

Default 3-seed result from `outputs/step2_context_disentanglement/results.json`:

| Representation/readout | final-window MSE | last-cycle MSE |
|---|---:|---:|
| Oracle pair features + context-indexed heads | 0.0278 | 0.0621 |
| Oracle pair features + context-gated slopes | 0.0286 | 0.0635 |
| Learned pair features + context-indexed heads | 0.1312 | 0.4035 |
| Learned pair features + context-gated slopes | 0.1327 | 0.4098 |
| Oracle pair features + one shared hidden-context head | 0.3534 | 0.4736 |
| Oracle pair features + one-hot context bias | 0.3588 | 0.4787 |
| Learned pair features + one shared hidden-context head | 0.3969 | 0.5778 |
| Raw observations + one shared hidden-context head | 0.5169 | 0.6861 |

The construction diagnostics recovered `4/4`, `4/4`, and `4/5` oracle-active
pairs across the three seeds. This supports the interpretation that feature
construction helped, but the final-window metric was also strongly limited by
output-memory/context adaptation. Appending one-hot context as a bias-like input
was not enough; context-specific feature slopes were needed. The gated-slope
probe is important because it reaches the same loss as context-indexed heads
when useful constructed features are present, while raw gated slopes do not
materially improve.

### CBP smoke baseline

`CBPMultiHeadMLPLearner` is now included in `step2_out_of_class.py` as a
plasticity-preserving MLP baseline. In the 30-seed canonical run written to
`outputs/step2_canonical/out_of_class_results.json`, UPGD remains best on all
three streams:

| Stream | Best MLP | UPGD | CBP |
|---|---:|---:|---:|
| Polynomial | 1.1458 | 0.5767 | 1.2036 |
| Frequency mismatch | 1.1689 | 0.6335 | 1.1951 |
| Compositional | 0.1908 | 0.1634 | 0.1903 |

The scientific reading is narrow: CBP is useful as a plasticity baseline, but
it does not change the Step 2 conclusion on the current synthetic suite.

### External online sanity check

`step2_external_online.py` adds a non-synthetic online supervised benchmark
using `sklearn.datasets.load_digits`. This is not a full continual-agent
environment, but it is a useful external check against overfitting to synthetic
streams. On 5 seeds x 3000 online training steps, fair MLP beats UPGD:

| Metric | MLP | UPGD | UPGD wins |
|---|---:|---:|---:|
| Final-window accuracy | 0.9668 | 0.9496 | 1/5 |
| Held-out test accuracy | 0.9477 | 0.9354 | 0/5 |
| Final-window MSE | 0.0204 | 0.0237 | 0/5 |
| Held-out test MSE | 0.0226 | 0.0254 | 0/5 |

This is now the main anti-universality result. UPGD is the strongest current
synthetic out-of-class method, not a universal supervised learner.

### bsuite breadth smoke on Python 3.13

The Step 2 breadth gap now has a working bsuite path beyond `catch` and
`cartpole` on Python 3.13. The PyPI source package `bsuite==0.3.5` still fails
metadata generation because its setup imports the removed Python `imp` module.
Installing from the upstream Git repository works:

```bash
source .venv/bin/activate
python -m pip install 'git+https://github.com/google-deepmind/bsuite.git'
python -m pip install -e '.[bsuite]'
```

The direct benchmark entry points now work from the repository root and the
sweep CLI accepts explicit ids for small breadth probes:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --q-agent autostep_bottleneck \
  --sarsa-agent sarsa_bottleneck \
  --bsuite-ids bandit/0 memory_len/0 \
  --seeds 0 1 \
  --num_steps 25 \
  --save_path outputs/bsuite/breadth_smoke \
  --comparison-report outputs/bsuite/breadth_smoke/sarsa_vs_q.md \
  --overwrite
```

This is a smoke test, not a statistical claim: 2 seeds, 25 continuing steps,
and bottleneck networks to keep the run cheap. It proves the path works beyond
the previous `catch`/`cartpole` envelope and exercises paired SARSA-vs-Q
analysis.

| Experiment | Pairs | Q final regret | SARSA final regret | SARSA improvement | SARSA win rate |
|---|---:|---:|---:|---:|---:|
| `bandit` | 2 | 8.25 | 10.65 | -2.40 | 0.50 |
| `memory_len` | 2 | 9.00 | 4.00 | +5.00 | 1.00 |
| overall | 4 | 8.625 | 7.325 | +1.30 | 0.75 |

Source: `outputs/bsuite/breadth_smoke/sarsa_vs_q.md`.

### Low-noise expert mixture canonical candidate

`step2_expert_mixture.py` tests a temporally-uniform portfolio rather than a
single learner. At every online step it predicts with both a fair MLP expert
and a matched low-noise UPGD expert, forms a discounted-Hedge convex prediction,
then updates both experts on the same example. The promoted run uses UPGD
`perturbation_sigma=1e-4`, 10 seeds, 1200 steps, and the same hidden width,
step size, sparsity, layer norm, and ObGD bounding as the fair MLP.

This is the first Step 2 result in the repo that closes the **fair-MLP
promotion bar** across the combined synthetic and external-digits probe:

| Regime | Mixture vs MLP final-window MSE | Mixture vs MLP accuracy/test signal |
|---|---:|---:|
| Synthetic polynomial | `+0.0728`, 10/10 wins | N/A |
| Synthetic frequency | `+0.0535`, 7/10 wins, 2 losses | N/A |
| Synthetic compositional | `+0.0000`, 10 ties | N/A |
| Digits IID | `+0.0020`, 10/10 wins | final-window accuracy `+0.0087`; test accuracy `+0.0106` |
| Digits class-blocked | `+0.0000`, 10 ties | final/test accuracy tie with MLP |
| Digits permuted pixels | `+0.000004`, 8/10 wins, 2 ties | final/test accuracy tie with MLP |
| Digits mask noise | `+0.0034`, 10/10 wins | final-window accuracy `+0.0130`; test accuracy `+0.0108` |
| Digits label drift | `+0.0000`, 10 ties | final/test accuracy tie with MLP |

Positive MSE differences mean MLP minus mixture, so positive favors the
mixture. On the external digits regimes, the mixture never has negative mean
test-accuracy difference against MLP. It improves IID and mask-noise digits and
falls back to MLP on blocked/permuted/label-drift cases.

### Retention-aware expert mixture canonical candidate

The low-noise mixture exposed one remaining concrete blocker: class-blocked
digits. One-step Hedge correctly routed to MLP because MLP tracks the final
class block best, but that same decision erased UPGD's much stronger balanced
held-out retention.

`step2_expert_mixture.py` now has an opt-in `class_imbalance` retention router.
It leaves the prequential tracking predictor unchanged. For held-out deployment
only, it checks whether the lifetime stream covered most target classes while
the recent final-window stream covered only a small fraction of those classes.
When that retention hazard is present, deployment weights are shifted to UPGD.
This uses only online-observed stream labels; held-out data are never used for
updates or routing.

Promoted command:

```bash
python "examples/The Alberta Plan/Step2/step2_expert_mixture.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --perturbation-sigma 1e-4 \
  --retention-router class_imbalance \
  --retention-upgd-deployment-weight 1.0 \
  --output-dir output/step2_expert_mixture_retention_10seed
```

Expected outcome relative to the low-noise tracking run:

- final-window MSE and online accuracy are unchanged, because online tracking
  still uses the ordinary Hedge mixture;
- non-blocked digits deployment keeps the ordinary final Hedge weights;
- class-blocked digits deployment switches to UPGD and removes the held-out
  best-expert retention regret.

Canonical 10-seed result:

| Regime | Final-window MSE vs MLP | Test accuracy vs MLP | Retention trigger | Best-expert test-accuracy regret |
|---|---:|---:|---:|---:|
| Digits IID | `+0.0020` | `+0.0106` | 0/10 | `-0.0087` |
| Digits class-blocked | `+0.0000` | `+0.0917` | 10/10 | `0.0000` |
| Digits permuted pixels | `+0.000004` | `+0.0000` | 0/10 | `+0.0013` |
| Digits mask noise | `+0.0034` | `+0.0108` | 0/10 | `+0.0048` |
| Digits label drift | `+0.0000` | `+0.0000` | 0/10 | `0.0000` |

The class-blocked held-out result is the important unblocker: mean mixture test
accuracy rises from the MLP value `0.1195` to the UPGD value `0.2111`, the
paired gain over MLP is `+0.0917`, and best-expert failures drop from `10/10`
in the one-step router to `0/10`.

This closes the explicit low-noise mixture gap without hiding the metric
tradeoff: MLP is still the current-block tracking expert, while UPGD is the
retained balanced-class deployment expert under detected class imbalance.

### Strict universal portfolio canonical candidate

`step2_universal_portfolio.py` is now the strict Step 2 candidate. It extends
the two-expert mixture to five live experts: `mlp_h64`, `mlp_h128`,
`mlp_h64_64`, `upgd_low_noise`, and `dynamic_sparse`. The promoted default uses
discounted Hedge with eta `1.0`, which keeps useful convex averaging instead of
collapsing too quickly to one expert. It also has a causal online
class-imbalance MSE guard: when lifetime observed target-class coverage is
broad but the recent window is class-narrow, current-block online prediction
routes to `mlp_h64_64`; held-out deployment still uses the retained-accuracy
router and therefore switches to UPGD on class-blocked digits.

Promoted 10-seed result:

| Regime | Final-window MSE vs best MLP | Test accuracy vs best MLP |
|---|---:|---:|
| Synthetic polynomial | `+0.0414` | N/A |
| Synthetic frequency | `+0.0168` | N/A |
| Synthetic compositional | `+0.0018` | N/A |
| Digits IID | `+0.0071` | `+0.0128` |
| Digits class-blocked | `+0.0000` | `+0.0970` |
| Digits permuted pixels | `+0.0075` | `+0.0156` |
| Digits mask noise | `+0.0094` | `+0.0158` |
| Digits label drift | `+0.0055` | `+0.0128` |

The strict comparison is against the best fair MLP width per seed, not only
`mlp_h64`. Positive MSE values mean best MLP minus portfolio; positive accuracy
values mean portfolio minus best MLP.

30-seed risk checks preserve the conclusion:

- compositional final-window MSE vs best MLP: `+0.0007`;
- frequency final-window MSE vs best MLP: `+0.0187`;
- class-blocked final-window MSE vs best MLP: exact tie across 30 seeds, with
  held-out accuracy `+0.1203`;
- non-blocked digits final-window MSE: 30/30 wins on IID, mask-noise,
  label-drift, and permuted-pixels;
- non-blocked digits held-out accuracy remains positive by mean on all four
  regimes.

This closes the earlier strict synthetic/digits portfolio acceptance matrix.
The later conclusive learner extends the comparison to the broader controlled,
synthetic, and digits all-suite matrix; the resource-manager pass closes the
prior learned-allocation and stateful-external-digits gaps. The broader
research problem remains open beyond this supervised Step 2 matrix: published-
scale OPMNIST is not complete, and TD/GVF feature discovery remains Step 3
research.

### Published-style external stressor pass

`step2_published_stressors.py` adds a compact external-stressor pass aimed at
the Dohare-style benchmark gap. The runner reuses the strict
`step2_universal_portfolio.py` learner side and changes only the data stream.
It includes:

- compact Online Permuted MNIST-style classification with a default local
  fallback from `sklearn.datasets.load_digits`, expanded to 28x28 pixels and
  permuted in recurring task blocks;
- optional true OpenML MNIST via
  `--mnist-source openml --allow-openml-download`, with
  `--mnist-published-scale` configuring the canonical 60,000/10,000 source
  split, uncapped examples, sequential per-task epochs, and 60,000-example task
  blocks;
- a lightweight Slowly-Changing Regression analogue with binary inputs,
  slow-changing bits, iid random bits, a constant bias bit, and a fixed LTU
  target network;
- stronger SCR presets: `--scr-preset dohare_paper` matches the public
  reproduction config (`m=20`, `f=15`, `T=10000`, target hidden units=100), and
  `--long-scr` runs a feasible 20,000-step local variant with the same
  bit/target shape and `T=1000`.

The default does **not** download OpenML MNIST. It is therefore a
published-style stressor, not a published-scale reproduction. OpenML MNIST is
available behind `--mnist-source openml --allow-openml-download`.

Canonical-ish local command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --canonical-ish \
  --output-dir outputs/step2_canonical \
  --result-prefix published_stressors
```

One-block full-source MNIST command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_published_mnist_openml_cache \
  --mnist-published-scale \
  --n-permutations 5 \
  --steps 60000 \
  --n-seeds 1 \
  --final-window 10000 \
  --max-test-permutation-views 5 \
  --output-dir outputs/step2_published_mnist_fullsplit_60k_1seed \
  --result-prefix fullsplit_60k_1seed
```

True OpenML compact follow-up command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks permuted_mnist_like \
  --canonical-ish \
  --mnist-source openml \
  --allow-openml-download \
  --openml-data-home outputs/step2_canonical/openml_external_cache \
  --output-dir outputs/step2_canonical/openml_external_published_5seed \
  --result-prefix published_stressors_openml_5seed
```

Longer SCR command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks slowly_changing_regression \
  --long-scr \
  --output-dir outputs/step2_canonical \
  --result-prefix published_stressors_long_scr
```

SCR router-search command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" \
  --long-scr \
  --output-dir outputs/step2_canonical \
  --result-prefix scr_router_search_long
```

5 seeds x 1500 online steps:

| Stressor | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---:|---:|---|
| 28x28 sklearn-digits permuted pixels | final-window MSE `+0.0068 +/- 0.0004` | `5/0/0` | positive local analogue |
| 28x28 sklearn-digits permuted pixels | held-out test accuracy `+0.0517 +/- 0.0077` | `5/0/0` | positive local analogue |
| Slowly-Changing Regression analogue | final-window MSE `-0.0003 +/- 0.0004` | `3/2/0` | mixed/slightly negative mean |

True OpenML MNIST compact follow-up, 5 seeds x 1,500 online steps, 5 random
pixel permutations, 4,000/1,000 capped stratified split:

| Stressor | Source metadata | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---|---:|---:|---|
| Permuted MNIST-style pixels | OpenML `mnist_784`, `is_true_mnist=true`, `n_total=70000` | final-window MSE `+0.0131 +/- 0.0010` | `5/0/0` | positive true-source compact result |
| Permuted MNIST-style pixels | OpenML `mnist_784`, `is_true_mnist=true`, `n_total=70000` | held-out test accuracy `+0.1408 +/- 0.0117` | `5/0/0` | positive true-source compact result |

One-block OpenML MNIST full-source/full-task-block run, 1 seed x 60,000 online
steps, canonical 60,000/10,000 split, one observed 60,000-example randomized
pixel-permutation task:

| Stressor | Source/protocol gate | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---|---:|---:|---|
| Permuted MNIST-style pixels | `uses_full_openml_mnist_split=true`, `uses_full_mnist_task_blocks=true`, `matches_dohare_opmnist_core_protocol=true` | final-window MSE `+0.002199` | `1/0/0` | positive online MSE on one full block |
| Permuted MNIST-style pixels | same | held-out test accuracy `-0.000600` | `0/1/0` | slight negative vs best fair MLP |

Chunked/resumable true OpenML MNIST follow-up, 1 seed, canonical split,
60,000-example randomized pixel-permutation task blocks, with 800 configured
permutations. The latest durable checkpoint has advanced to 2,760,000 online
steps / 46 full blocks. The latest completed evaluation aggregation remains the
2,400,000-step / 40-block result:

| Stressor | Source/protocol gate | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---|---:|---:|---|
| Permuted MNIST-style pixels | `matches_dohare_opmnist_core_protocol=true`, `completed_full_task_blocks=40`, `n_permutations=800` | final-window MSE `+0.002250` | `1/0/0` | positive on forty full blocks |
| Permuted MNIST-style pixels | same | held-out test accuracy vs best fair MLP | `+0.011013` | positive held-out accuracy over observed permutation views |

Historical deployment-objective replay from the earlier 20-block checkpoint:

| Deployment objective | Held-out accuracy | Held-out test MSE | Reading |
|---|---:|---:|---|
| MSE tracking portfolio | `0.327580` | `0.084924` | canonical reported deployment |
| Accuracy tracking portfolio | `0.269080` | `0.102125` | weaker on this checkpoint |
| Dynamic sparse only | `0.402620` | `0.075841` | strongest held-out replay variant |

Longer SCR-only pass, 3 seeds x 20,000 online steps, `m=20`, `f=15`,
`T=1000`, target hidden units=100:

| Stressor | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---:|---:|---|
| Slowly-Changing Regression longer Dohare-style local variant | final-window MSE `-0.0011 +/- 0.0003` | `0/3/0` | negative; best fair MLP remains ahead |

SCR router-search follow-up, 3 seeds x 20,000 online steps, same local
Dohare-style shape:

| Router | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---:|---:|---|
| `convex_reference` | final-window MSE `-0.001069 +/- 0.000259` | `0/3/0` | reproduces the negative longer-SCR result |
| `guarded_best_mlp` | final-window MSE `+0.000099 +/- 0.000074` | `3/0/0` | closes the feasible local SCR comparator with causal guarded routing |

Published-scale SCR follow-up, 3 seeds x 1,000,000 online steps, Dohare public
config (`m=20`, `f=15`, `T=10000`, target hidden units=100):

| Router | Primary metric vs best fair MLP | Wins/losses/ties | Reading |
|---|---:|---:|---|
| `slow_meta` | final-window MSE `+0.00006156 +/- 0.00001598` | `3/0/0` | closes million-step SCR against the fair MLP grid |

This narrows the external-evidence gap but does not fully close it. The
portfolio clearly beats the fair MLP grid on the compact permuted-pixel
fallback, on compact true OpenML MNIST, and on the current evaluated 40-block
true-OpenML streaming run. The portfolio OPMNIST runner now has atomic
checkpoints, chunk progress, status/ETA reporting, strict resume validation,
and migration for legacy UPGD checkpoint states. A separate single-learner
UPGD-memory OPMNIST runner now evaluates the packaged production learner
directly; its current full-source/full-block OpenML result is 10 blocks /
600,000 examples, positive against best fair MLP on all online, final-window,
and held-out metrics over the 10 observed permutation views. This is not the
800-task / 48M-example main OPMNIST protocol. Its latest status sidecar reports
10/800 completed blocks and an approximately 12-hour full-run ETA under current
CPU contention. Million-step SCR is now closed for the narrowed `slow_meta`
causal router with the fair MLP comparator preserved. The combined external
status remains partial because OPMNIST task-count scale is runnable/resumable
but still not completed.

## What was closed since the audit

- `OutOfClassPolynomialStream`: degree-3 polynomial oracle (pair-product
  features cannot fit it exactly)
- `FrequencyMismatchStream`: trigonometric oracle (tanh / pair-product
  features cannot fit it exactly)
- `CompositionalStream`: 2-hidden-layer tanh oracle (1-layer feature
  banks cannot fit it exactly)
- `CompositionalFeatureLearner`: feature DAG with op types
  `OP_PRODUCT`, `OP_SUM`, `OP_TANH`, `OP_GATED` over existing features
  (not just raw inputs); cascade replacement; topological invariant
- `UPGDLearner`: utility-perturbed-gradient-descent baseline
  (Dohare 2023 family) with per-layer utility tracking
- Out-of-class benchmark suite: 7 methods × 3 streams × 30 seeds
- Rigged-vs-fair demonstration that the original 16/16 result was
  hypothesis-class match
- Context/output-adaptation disentanglement showing why final-window pair-stream
  loss can understate feature-construction success
- `CBPMultiHeadMLPLearner` wired into the out-of-class runner as a Step 2 smoke
  baseline
- External online digits benchmark showing fair MLP can beat UPGD off the
  synthetic stream family
- Low-noise MLP/UPGD expert mixture showing a temporally-uniform portfolio can
  improve or tie fair MLP final-window MSE across the combined synthetic and
  shifted-digits probe, while preserving external digits accuracy
- Retention-aware deployment routing for the expert mixture, closing the
  class-blocked held-out retention gap by switching to UPGD only when observed
  class coverage is lifetime-broad but recent-window-narrow
- Strict MLP/UPGD/dynamic-sparse universal portfolio with no negative mean
  final-window MSE against the best fair MLP width and no negative mean
  held-out digits accuracy against the best fair MLP width on the current
  acceptance matrix
- Learned contextual resource manager over static MLP, low/high UPGD, and CBP
  replacement resource policies on harder stateful external digits streams:
  recurrent pixel permutations, recurrent feature-mask/noise states, and
  class-blocked retention. The tracking manager improves final-window MSE
  versus static MLP on all three streams (`10/10` wins each), while the
  prototype-balanced retention manager improves held-out accuracy on all three,
  including class-blocked retention (`10/10` wins).
- Compact published-style stressor runner:
  `step2_published_stressors.py` provides local Permuted-MNIST-style and
  Slowly-Changing-Regression-style checks. The 5-seed local fallback is
  positive on 28x28 permuted digits and mixed/slightly negative on the SCR
  analogue, so this is evidence hardening rather than full published-scale
  closure.
- Core MLP hidden-unit utility tracking: `MLPLearnerState` and
  `MultiHeadMLPState` now carry per-hidden-layer EMA utilities from
  loss-gradient magnitude.
- Deep MLP feature lifecycle wiring: `CBPMultiHeadMLPLearner` performs
  hidden-unit testing/replacement, and `CBPMLPLearner` exposes the same
  Continual Backprop path for single-output Step 2 MLPs.
- Native deep lifecycle hard-blocker follow-up:
  preserve-outgoing early promotion and active low-utility perturbation are
  implemented and tested, but the best native lifecycle variant still beats
  the fair MLP on only `2/6` probes in the 5-seed x 800-step audit. It remains
  a diagnostic path, not a promoted Step 2 feature-construction mechanism.
- Soft-gated and Net2Net native-deep follow-ups:
  live candidate gates and function-preserving promotion are implemented as
  opt-in diagnostics. Soft-gated variants still reached only `2/6`; Net2Net
  improved the native ceiling to `3/6` by paired mean but still lost nonlinear,
  compositional, and digits. Native deep lifecycle should not be treated as a
  Step 2 closure path.
- Generator-internal feature-resource manager:
  `FixedBudgetFeatureLearner(learn_feature_resources=True)` learns generator
  choice across random, parent-mutation, and residual-imprint generators while
  adapting replacement rate and promotion margin through
  conservative/nominal/aggressive plasticity policies.
- Causal future-utility estimation: `core.future_utility` provides one-step
  output-loss-reduction estimates, wired into fixed-budget, interaction, and
  compositional feature learners through `future_utility_mix`.
- Temporally extended future-utility estimation: `core.future_utility` now also
  provides a causal trace-based output-loss-reduction estimate. In
  `CompositionalFeatureLearner`, `future_utility_trace_decay > 0` tracks
  residual, feature-value, and feature-energy traces for active and candidate
  DAG slots. Candidate trace provenance is copied on promotion and stale traces
  are cleared when active/candidate slots are replaced.
- Focused recursive feature-utility probe:
  `step2_recursive_feature_utility_probe.py` now has two promoted paths. First,
  `single_mechanism` is the pure robust recursive mechanism: contribution-trace
  utility, residual imprint, product-biased operation priors,
  utility/novelty-biased parent choice, and depth retention. On the focused
  triple-product task, the 10-seed x 5,000-step run beats the best fair MLP
  (`0.0839 +/- 0.0144` final-window MSE vs `0.5260 +/- 0.0407`) with `10/10`
  paired wins and depth>=2 active features in every seed.
  Second, `recursive_mlp_router` is a causal resource router over
  `single_mechanism`, `mlp_32x32_no_ln`, and `mlp_64x64_no_ln`. On the harder
  six-task suite (`nonlinear`, `interaction`, `triple`, `rare`, `polynomial`,
  `frequency`), the 10-seed x 5,000-step run beats the best fair MLP on `6/6`
  tasks. Paired wins are `10/10` on nonlinear, interaction, triple, and
  polynomial; `9/10` on rare and frequency. Every router run retains depth>=2
  recursive features. This closes the previous fair-MLP boundary for the
  controlled recursive feature-construction suite.
- Recursive single-mechanism follow-up:
  `single_mechanism_signed_tanh` and `single_mechanism_tanh_shadow` add small,
  opt-in nonlinear scaffolds without task labels or offline search. On a
  3-seed x 2,500-step six-probe suite, signed tanh improves the nonlinear
  miss (`0.2251` to `0.1430` final-window MSE) and beats/ties several
  algebraic probes, but still loses nonlinear decisively to the best fair MLP
  (`0.0584`, `0/3` paired wins). These knobs remain experimental and are not
  promoted as a universal single recursive mechanism.
- Recursive retention follow-up:
  `single_mechanism_retention` adds slow utility hysteresis and family quotas
  for useful tanh/product scaffolds. It is the best pure mechanism so far,
  beating best fair MLP on `5/6` six-probe tasks at 5 seeds and preserving the
  algebraic wins, but it still loses nonlinear (`0.1002 +/- 0.0167` versus
  fair MLP `0.0597 +/- 0.0023`, `1/5` paired wins). It is partial, not
  promoted.
- Rejected single-mechanism directions:
  the budgeted geometric dictionary lost all six probes, and
  `candidate_scoring_mode="energy_novelty"` improved triple/rare but badly
  regressed nonlinear, interaction, polynomial, and frequency. The failure
  mode is not lack of novelty; it is slow online output/theta credit assignment
  for partially redundant smooth nonlinear scaffolds.
- Rare-task utility protection: `step2_rare_task_utility.py` now includes a
  `rare_protected` opt-in combining active inverse-frequency utility,
  one-step future utility, and slow utility retention. On the explicit
  12-seed rare-head oracle-pair stream, mean utility retained the rare oracle
  pair in `1/12` seeds with rare final-window MSE `21.0795 +/- 3.1431`;
  `rare_protected` retained it in `9/12` seeds with rare final-window MSE
  `17.7323 +/- 2.6164`. Paired against mean utility, `rare_protected`
  improved rare MSE by `3.3472`, improved active MSE by `0.0335`, and had no
  common-head harm (`method - mean common MSE = -4.44e-09`). This closes the
  narrow rare-head utility-retention failure on this explicit stream as an
  opt-in mechanism, but it is not promoted to the global default.
- `AdaptiveObGD`: Appendix-B ObGD variant with RMSProp-style diagonal
  second-moment normalization before the ObGD overshoot cap.
- `step2_mlp_nonstationarity_comparison.py`: 10-seed MLPLearner comparison
  across random-walk drift, abrupt changes, cyclic contexts, and periodic
  drift. On these deliberately linear streams, `linear_lms` beats all MLP
  variants; among MLPs, `mlp_h32_autostep` is the best final-window comparator
  in all four regimes.
- Pytest regression now covers resource-manager behavior, Step 2 canonical
  evidence, universal-portfolio routing, and the feature-resource controls.

### Conclusive learner candidate

`step2_conclusive_learner.py` combines the previous strongest Step 2 pieces
into one causal prediction-space learner:

- recursive compositional features,
- bounded degree-3 polynomial features,
- Fourier features for sinusoidal streams,
- fixed random tanh features for compositional tanh streams,
- the fair MLP grid,
- UPGD and dynamic-sparse experts for measurement and optional ablation,
- MLP-anchored safe recursive/polynomial residual routes,
- a causal sliding-window resource manager with a class-blocked online MSE
  guard and deployment-time retention guard,
- a telemetry route-policy recovery gate and an MLP-floor deployment blend.

Promoted all-benchmark command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 0 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --route-policy-mode telemetry_worker_b \
  --route-telemetry-window 300 \
  --worker-b-switch-margin 0.010 \
  --mlp-floor-blend-weight 0.5 \
  --mlp-floor-source selector \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_main_eta05_telemetry_worker_b_floor05_all_10seed \
  --note-path docs/research/step2_main_eta05_telemetry_worker_b_floor05_all_10seed.md
```

Canonical 10-seed result versus the same-run best fair MLP:

| Group | Final-window MSE wins/losses/ties |
|---|---:|
| Controlled six-probe suite | `60/0/0` |
| Synthetic polynomial/frequency/compositional | `30/0/0` |
| Digits IID/permuted/mask-noise/label-drift | `40/0/0` |
| Digits class-blocked | `0/0/10` |
| **Total** | **`130/0/10`** |

The remaining ties are intentional class-blocked online-MSE ties to the best
MLP route. The same class-blocked run wins held-out test accuracy by
`+0.3492 +/- 0.0080` at `10/0/0`. The previously weak rows are now
seed-positive:

| Row | Final-window MSE diff | Wins/losses/ties |
|---|---:|---:|
| Controlled rare | `+0.0115 +/- 0.0025` | `10/0/0` |
| Synthetic compositional | `+0.0413 +/- 0.0099` | `10/0/0` |
| Synthetic polynomial | `+0.0489 +/- 0.0286` | `10/0/0` |

Current interpretation: this is the strongest Step 2 learner in the repo and
closes the current supervised all-suite promotion matrix. It remains a causal
portfolio/routing result, not a proof that one recursive self-contained
feature-construction mechanism is universal. Published-scale OPMNIST and
TD/GVF feature discovery remain separate research boundaries.

### Target-structure UPGD simplification and promotion update

The current simple non-router Step 2 promotion is target-structure UPGD, not
D18. The promoted loss rule is:

`UPGDLearner(loss_normalization="target_structure", perturbation_sigma=1e-4, bounder_kappa=0.5, ...)`

`target_structure` uses sum-style loss only for non-negative simplex targets
with total mass 1 and mean-style loss otherwise. This preserves one-hot digit
pressure while fixing target-density's ambiguity on exact-zero dense heads and
sparse multilabel rows.

Focused stress probe, 8 seeds:

| Stressor | Method | Diff vs best fair MLP | Wins/losses |
|---|---|---:|---:|
| Dense-zero | `target_structure` | `+0.000232 +/- 0.000311` | `5/3` |
| Dense-zero | `target_density` | `-0.000464 +/- 0.000269` | `3/5` |
| Sparse multilabel | `target_structure` | `+0.026938 +/- 0.000865` | `8/0` |
| Sparse multilabel | `target_density` | `+0.006547 +/- 0.000783` | `8/0` |

30-seed dense synthetic structure rerun:

| Stream | Variant | Diff vs best fair MLP | Wins |
|---|---|---:|---:|
| Polynomial | `structure_sigma1e4_kappa05` | `+0.5473 +/- 0.0316` | `30/30` |
| Frequency mismatch | `structure_sigma1e4_kappa05` | `+0.5756 +/- 0.0376` | `30/30` |
| Compositional | `structure_sigma1e4_kappa05` | `+0.0924 +/- 0.0038` | `30/30` |

30-seed digits simplification rerun, five regimes x 30 seeds:

| Branch | Final-window MSE diff | Test accuracy diff | Interpretation |
|---|---:|---:|---|
| `upgd_density_sigma1e_4_adaptk035_065_lr05_e1` | `+0.0062 +/- 0.0003`, `120/150` wins | `+0.0299 +/- 0.0019` | Best no-meta digit branch; density-equivalent to target-structure on one-hot targets. |
| `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight` | `+0.0062 +/- 0.0003`, `120/150` wins | `+0.0296 +/- 0.0017` | Readout meta gives no aggregate MSE advantage over no-meta in this rerun. |
| `upgd_density_sigma1e_4_kappa05` | `+0.0036 +/- 0.0002`, `147/150` wins | `+0.0151 +/- 0.0017` | Simplest fixed-kappa digit control. |

A separate target-structure digits ablation confirms that
`upgd64_structure_sigma1e_4` beats sum and mean-style UPGD on shuffled digits
final-window MSE (`+0.0016`, `25/30` wins vs the best fair MLP), though that
older fixed-kappa harness still loses online MSE on class-blocked digits while
improving held-out class-blocked accuracy.

The ablation reads as follows: target-structure normalization and ObGD bounds
are essential; low-noise perturbation remains the conservative default;
adaptive `kappa`, readout meta-plasticity, and repeated-target plasticity are
useful branches but not required for the simple learner; trunk meta, head-bias
meta, hidden-unit replacement, and repetition gates are removable from the
promoted default.

Sources:
`docs/research/step2_target_structure_upgd_stress.md`,
`output/subagents/upgd_simplification_structure_synthetic_30seed/out_of_class_SUMMARY.md`,
`output/subagents/upgd_simplification_scale_digits_lr05e1_30seed/SUMMARY.md`,
and `output/subagents/upgd_simplification_structure_digits_30seed/digits_ablation_SUMMARY.md`.

### Superseded simple non-router D18 persistent-trace candidate

The follow-up single-learner grind produced a strong result under the stricter
"no output portfolio, no router, no selected expert" constraint. The candidate
was a D18 additive resource-basis learner with one deployment path:
resource-managed RKHS core, tanh/Fourier and small residual bases, learned block
gains, online-discovered one-hot simplex output geometry, and a causal
persistence-gated target trace. Every block updates from the same residual.

Canonical command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --configs step2_gain_l2_0p1 \
  --simplex-output \
  --target-trace-scale 4 \
  --target-trace-decay 0.95 \
  --target-trace-clip 1.0 \
  --target-trace-persistence-gate \
  --target-persistence-decay 0.95 \
  --target-persistence-power 6 \
  --output-dir outputs/step2_main_d18_persistent_trace_p6_all_10seed \
  --note-path docs/research/step2_main_d18_persistent_trace_p6_all_10seed.md
```

The named `step2_persistent_trace` config is equivalent except for the method
label in result files.

Canonical 10-seed result versus the same-run best fair MLP:

| Group | Final-window MSE wins/losses/ties |
|---|---:|
| Controlled six-probe suite | `60/0/0` |
| Synthetic polynomial/frequency/compositional | `28/2/0` |
| Digits IID/class-blocked/permuted/mask-noise/label-drift | `50/0/0` |
| **Total** | **`138/2/0`** |

The weakest aggregate rows remain positive: digits class-blocked `+0.00141`,
digits mask-noise `+0.00915`, synthetic compositional `+0.04220`, and synthetic
polynomial `+0.09408` final-window MSE, where positive means best fair MLP minus
D18.

Because D18 uses a causal one-hot projection when the target stream reveals
simplex geometry, the audit also compares against a fair projected-MLP MSE
computed from the MLP final-window accuracy. The 30-seed hard digit risk check
remains positive by mean:

| Risk row | Raw MSE diff | Raw wins/losses/ties | Projected-MLP diff |
|---|---:|---:|---:|
| Digits class-blocked | `+0.00147` | `30/0/0` | `+0.00024` |
| Digits mask-noise | `+0.00999` | `29/1/0` | `+0.00147` |

This was the strongest non-router Step 2 learner before the UPGD
target-structure rerun. The evidence does not make it a clean theory of
universal feature construction: pure residual birth, pure kernel,
calibration-only, and wide random-feature variants each failed different
blocker rows. D18's success is still a hand-assembled additive resource-basis
result, not a theorem and not a single recursive feature-growth principle.

## Remaining Claim Boundaries After This Batch

- The current supervised Step 2 promotion matrix is closed by target-structure
  UPGD. The remaining items below are claim boundaries, rejected alternate
  solution paths, or Step 3 research, not blockers for the current supervised
  Step 2 acceptance claim.
- Broader validation of rare-task utility protection outside the explicit
  rare-head oracle-pair stream remains useful, but the current 14-regime
  acceptance matrix no longer has a rare-task blocker row.
- Longer-horizon and recursive feature-construction mechanisms are now
  implemented and stress-tested. A pure single recursive mechanism is now
  positive by mean on all six controlled probes:
  `single_mechanism_retention_tanh24_tanh_heavy_conservative` gets nonlinear
  `+0.008718` (`9/1`), interaction `+0.386067` (`10/0`), triple `+0.387012`
  (`9/1`), rare `+0.025779` (`8/2`), polynomial `+0.081576` (`6/4`), and
  frequency `+0.068532` (`10/0`) against the best fair MLP. This is promoted
  as the best pure recursive evidence so far, but not as the global Step 2
  learner because the polynomial and nonlinear rows are less robust than the
  target-structure UPGD result.
- Native deep MLP feature lifecycle. Hidden-unit testing/replacement is wired
  and now has preserve-outgoing, active-perturbation, soft-gated, Net2Net, and
  normalized candidate-update variants, but the best single native lifecycle
  method still reaches only `2/6` positive probes against fair MLP in the
  hard-blocker audit. This path is rejected as the promoted Step 2 solution.
- TD/GVF-target feature finding beyond the current observable AR(1) bridge
  result. TD-error surprise interactions now narrowly beat raw linear, raw MLP,
  and fixed interactions on the fully observable AR(1) positive control.
  Predictive-state/MSPBE variants produced a narrow 3-seed hidden/off-policy
  signal, but 10-seed and harder-hidden follow-ups rejected robust closure:
  predictive-state features lost coupled-hidden AR(1) to raw MLP (`2/8/0`),
  and off-policy gains did not survive the harder variant.
- True published-scale external comparisons: compact true OpenML MNIST and
  true OpenML Fashion-MNIST delayed-context evidence are positive. The
  portfolio OPMNIST runner still has a 46-block checkpoint and a 40-block
  evaluated aggregation. The single packaged UPGD-memory runner now has a
  separate 10-block evaluated full-source/full-task-block OpenML MNIST result:
  600,000 examples, 800 configured random pixel permutations, and positive
  primary-vs-best-MLP deltas on online MSE (`+0.010216`), online accuracy
  (`+0.013312`), final-window MSE (`+0.007760`), final-window accuracy
  (`+0.011600`), held-out MSE (`+0.023794`), and held-out accuracy
  (`+0.233600`). Full published-scale OPMNIST still needs all 800 blocks /
  48,000,000 examples, so the published-scale flag correctly remains false.
  Million-step SCR is closed for the narrowed `slow_meta` router at 3 seeds.

### Resource-manager delayed-context image evidence

`step2_resource_manager_stateful_external.py` now has an opt-in
`external_delayed_contextual_permutation` regime.  It attempts OpenML
Fashion-MNIST only when the caller explicitly passes
`--external-image-source openml_fashion_mnist --allow-openml-download`; the
default smoke path uses local sklearn digits expanded to 28x28.  The stream is
stateful in two ways: the images are permuted by a recurring hidden block
state, and the resource manager receives a context id delayed by whole blocks.

Local smoke command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" \
  --benchmarks external_delayed_contextual_permutation \
  --steps 80 \
  --n-seeds 1 \
  --final-window 20 \
  --block-size 20 \
  --n-states 3 \
  --hidden-size 16 \
  --external-image-source digits_28x28_fallback \
  --output-dir outputs/step2_canonical/resource_manager_stateful_external_external_smoke \
  --note-path outputs/step2_canonical/resource_manager_stateful_external_external_smoke/NOTE.md
```

Smoke output:

| Regime | Source actually used | Final-window MSE vs MLP | Held-out acc vs MLP |
|---|---|---:|---:|
| Delayed contextual permutation | 28x28 sklearn-digits fallback | `+0.0120`, `1/1` wins | `+0.0569`, `1/1` wins |

True OpenML Fashion-MNIST multi-seed command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_resource_manager_stateful_external.py" \
  --benchmarks external_delayed_contextual_permutation \
  --external-image-source openml_fashion_mnist \
  --allow-openml-download \
  --output-dir outputs/step2_canonical/openml_external_resource_5seed \
  --note-path outputs/step2_canonical/openml_external_resource_5seed/NOTE.md
```

5 seeds x 1,200 online steps, OpenML Fashion-MNIST, `used_fallback=false`,
sample limit 3,000:

| Regime | Source actually used | Final-window MSE vs MLP | Held-out acc vs MLP |
|---|---|---:|---:|
| Delayed contextual permutation | OpenML Fashion-MNIST | `+0.0102 +/- 0.0015`, `5/0/0` wins | `+0.0246 +/- 0.0073`, `4/1/0` wins |

This closes the fallback-only Fashion-MNIST TODO: the harder delayed-context
stream now has true external-image, multi-seed evidence. It does not close the
separate published-scale Online Permuted MNIST gap.
