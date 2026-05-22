# Step 2 Universality Matrix

This audit summarizes committed Step 2 outputs only. Source JSON:

- `outputs/step2_canonical/rigged_vs_fair_results.json`
- `outputs/step2_canonical/out_of_class_results.json`
- `outputs/step2_canonical/digits_online_results.json`
- `outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_results.json`
- `outputs/step2_canonical/simple_d18_persistent_trace_risk_digits_30seed_results.json`
- `output/subagents/upgd_simplification_structure_synthetic_30seed/out_of_class_results.json`
- `output/subagents/upgd_simplification_scale_digits_lr05e1_30seed/upgd_digits_sweep_results.json`
- `outputs/step2_interaction_*/results.json`
- `outputs/step2_context_disentanglement/results.json`

Second-wave local outputs add smaller follow-up evidence:

- `outputs/step2_context_disentanglement_workerB_default/results.json`
- `output/workerD_rare_task/smoke_results.json`

## Benchmark Matrix

`+` means the method beats the relevant MLP comparator in the committed JSON;
`~` means tied or seed-noisy; `-` means worse; `n/t` means not tried in audited
outputs.

| Method / family | Pair-product interaction stream | Same pair stream, fair MLP | Polynomial deg-3 | Frequency mismatch | 2-layer compositional oracle | Main failure mode |
|---|---:|---:|---:|---:|---:|---|
| Linear baseline | `-` | `-` | `-` final 1.417 | `-` final 1.506 | `-` final 0.226 | Cannot fit nonlinear targets. |
| MLP(8)+ObGD | Comparator; loses mean-loss to exhaustive interaction | n/t | n/t | n/t | n/t | Capacity-starved baseline; not load-bearing evidence. |
| MLP(64)+ObGD | n/t in old interaction sweeps | Comparator; final-window beats/ties interaction | `-` vs MLP(64,64), final 1.193 | best MLP, final 1.169 | best MLP, final 0.191 | Strong baseline; still beaten by UPGD out-of-class. |
| MLP(64,64)+ObGD | n/t in old interaction sweeps | Comparator; final-window ties interaction | best MLP, final 1.146 | `~` vs MLP(64), final 1.172 | `-` vs MLP(64), final 0.214 | More depth is not uniformly better under these settings. |
| Fixed-budget interaction products | Strong only when oracle is pair products: Part A mean-loss diff +0.104, 15/16 wins; final diff +0.003, 8/16 wins | Mean-loss edge remains, final-window ties: vs MLP64 final diff -0.058, 15/30 wins | `-` final 1.166, 3/30 vs best MLP, d=-1.71 | `-` final 1.763, 0/30, d=-2.18 | `-` final 1.306, 0/30, d=-4.49 | Hypothesis-class match; collapses when oracle is not pairwise products. |
| Interaction generate-and-test | Tried on pair streams; sparse exact final 0.829 vs MLP 0.875 | n/t | n/t | n/t | n/t | Weak margins; benefit depends on pair-product oracle. |
| Interaction shadow candidates | Strong stationary pair result: shadow-safe final 0.0175 vs MLP 0.441 | n/t | n/t | n/t | n/t | Can discover exact products, but nonstationary final-window gains are small/noisy. |
| Interaction recurrent utility | Best final in 16-seed recurrent pair run: 0.8446 vs MLP 0.9083, 11/16 wins | n/t | n/t | n/t | n/t | Helps retain recurring pair features; not tested off-class. |
| Interaction exhaustive candidates | Strongest interim pair result: mean-loss diff +0.1116, 16/16 wins; final diff +0.0223, 10/16 wins | Same rigged family only | n/t | n/t | n/t | Enumerates finite pair class; selection, not universal construction. |
| Generic `FixedBudgetFeatureLearner` | Scripted for `NonlinearFeatureDiscoveryStream`, but no committed audited JSON here | n/t | n/t | n/t | n/t | Unsupported in this audit. |
| `CompositionalFeatureLearner` | n/t in interaction output dirs | n/t | `~` final 1.142, 19/30 vs best MLP, d=+0.25 | `-` final 3.866, 0/30, d=-0.23 with extreme variance | `-` final 0.718, 0/30, d=-2.38 | Feature-of-feature DAG exists, but current random search/hyperparameters do not produce robust wins. |
| UPGD | n/t in interaction output dirs | n/t | `+` final 0.577, 30/30, d=+3.23 | `+` final 0.633, 30/30, d=+2.66 | `+` final 0.163, 29/30, d=+1.75 | Best synthetic evidence so far, but MLP beats it on external online digits. |
| Target-structure UPGD | n/t in interaction output dirs | n/t | `+` synthetic polynomial diff +0.547, 30/30 | `+` synthetic frequency diff +0.576, 30/30 | `+` synthetic compositional diff +0.092, 30/30 | Promoted single nonlinear learner; also closes one-hot digits through the same simplex target-structure rule. |
| D18 persistent trace | n/t in old interaction output dirs | n/t | `+` synthetic polynomial diff +0.094, 9/10; controlled polynomial diff +0.816, 10/10 | `+` synthetic frequency diff +0.527, 10/10; controlled frequency diff +0.146, 10/10 | `+` synthetic compositional diff +0.042, 9/10 | Best non-router current-matrix result; still a hand-assembled additive resource-basis learner. |
| CBP | n/t in interaction output dirs | n/t | `-` final 1.204, 0/30, d=-4.71 | `-` final 1.195, 2/30, d=-1.54 | `~` final 0.190, 19/30, d=+0.16 | Plasticity baseline, not a clear feature-construction win. |
| MLP vs UPGD on sklearn digits | n/a | n/a | n/a | n/a | n/a | External sanity check: fair MLP beats UPGD on final-window and held-out accuracy. |

## Interaction Benchmark Detail

| Output | Setup | Best supported result | Caveat |
|---|---|---|---|
| `step2_interaction_stationary_exact_8seed` | Stable pair oracle, dim=10, 8 seeds | `interaction_shadow_safe` final 0.0175 +/- 0.0104 vs MLP 0.4412; oracle-pair fraction 0.607 | Demonstrates exact product discovery only in stationary in-class setting. |
| `step2_interaction_sparse_exact_8seed` | Nonstationary sparse pair oracle, dim=10, 8 seeds | `interaction_shadow_safe` final 0.8253 vs MLP 0.8747; oracle-pair fraction 0.312 | Positive but small relative to seed variance. |
| `step2_interaction_sparse_slowutility_8seed` | Same sparse pair setup, slower utility | `interaction_shadow_safe` final 0.7468 vs MLP 0.8747; oracle-pair fraction 0.404 | Suggests recurrent/slow utility helps, still in-class only. |
| `step2_interaction_exhaustive_exact_16seed` | Nonstationary sparse pair oracle, exhaustive candidates, 16 seeds | Exhaustive candidates mean-loss diff +0.1116, 16/16 wins; oracle-pair fraction 0.671. Best final-window is recurrent shadow: 0.8446 vs MLP 0.9083, 11/16 wins | Mean-loss claim is stronger than final-window claim; exhaustive candidates enumerate the oracle class. |
| `step2_interaction_dim20_recurrent_8seed` | Larger dim=20 pair oracle, 8 seeds | Recurrent shadow final diff +0.0519, 7/8 wins; oracle-pair fraction 0.167 | Scaling weakens oracle overlap and margins. |
| Smoke dirs | 1-seed / <=120 steps | Only smoke coverage | Not evidence. |

## Supported Claims

- Exact interaction feature discovery works on streams whose hidden features are
  sparse pair products. The strongest committed evidence is stationary shadow
  candidates and 16-seed exhaustive/interim performance.
- The old "16/16 wins over MLP" style claim is metric-sensitive. In canonical
  Part A, interaction wins 15/16 on mean loss but only 8/16 on final-window MSE.
- With fairer MLP capacity on the same pair stream, interaction keeps a smaller
  mean-loss edge but final-window performance ties: 15/30 wins vs MLP(64) and
  15/30 vs MLP(64,64).
- UPGD was the first audited method that beat the best MLP on all three
  synthetic out-of-hypothesis-class streams: 30/30, 30/30, and 29/30 wins.
  The newer target-structure UPGD variant also closes the external one-hot
  digit matrix without a portfolio by using sum-style loss only for simplex
  targets and mean-style loss otherwise.
- D18 persistent trace is superseded but remains strong historical evidence on
  the broader controlled/synthetic/digits matrix. It is positive on every
  aggregate final-window MSE row over 14 regimes at 10 seeds (`138/2/0`
  seed-level wins/losses/ties) and remains positive by mean on the two hard
  digit risk rows at 30 seeds, including against a projected-MLP check.
- Context/output disentanglement supports a narrower claim about pair-product
  discovery: learned pair banks can recover oracle-active features, but low
  final loss in hidden recurring contexts also requires output memory or
  context-conditioned slopes.

## Context Disentanglement Detail

The two-pass probe in `step2_context_disentanglement.py` freezes learned
interaction features, then retrains online linear readouts from scratch. In the
default 3-seed run, final-window losses were:

| Representation/readout | Final-window loss | Interpretation |
|---|---:|---|
| oracle features + context-indexed heads | 0.0278 | Low-loss upper bound when both representation and context-specific slopes are available. |
| oracle features + context-gated slopes | 0.0286 | Matches indexed heads; explicit feature x context slopes are enough. |
| learned features + context-indexed heads | 0.1312 | Learned features help substantially when readout adaptation is not the bottleneck. |
| learned features + context-gated slopes | 0.1327 | Matches indexed heads with a single crossed-feature linear readout. |
| oracle features + one shared hidden-context head | 0.3534 | Good features alone do not solve hidden context-specific output slopes. |
| oracle features + one-hot context bias | 0.3588 | Context bias alone does not solve slope switching. |
| raw + one shared hidden-context head | 0.5169 | Baseline without constructed features. |

This does not make pair-product discovery universal. It clarifies which part of
the non-stationary pair benchmark is representation construction and which part
is output/context adaptation.

## Second-Wave Follow-Ups

These are not replacements for the 30-seed canonical suite, but they are useful
for ranking the next experiments.

| Follow-up | Result | Scientific effect |
|---|---|---|
| CBP in `step2_out_of_class.py` | 30-seed canonical run: CBP is MLP-like and never close to UPGD on the three synthetic streams. | Keep CBP as a plasticity baseline; do not treat it as the Step 2 solution. |
| Rare-task utility knobs | `utility_aggregation="max"` improves rare-task-masked aggregate loss in a 4-seed smoke, but not rarest-task final MSE and mildly worsens normal final loss. | Keep as opt-in; needs downstream utility evidence before changing defaults. |
| Compositional candidate min-age | Prevents newborn candidates from being recycled before earning utility; no clear UPGD/MLP performance gap reduction. | Improves validity of the negative compositional result. |
| External online digits | MLP test accuracy 0.9477 vs UPGD 0.9354 over 5 seeds; UPGD wins 0/5 on held-out accuracy. | Prevents any universal UPGD claim; demands broader external/non-synthetic benchmarks. |

## Unsupported Or Overstated Claims

- "Step 2 is solved as a theorem of universal feature construction" is
  unsupported. The current supervised promotion matrix is solved empirically by
  target-structure UPGD, a single nonlinear online learner with hidden-feature
  utility and bounded low-utility perturbation, not by a theorem that one
  recursive generator is universal.
- `CompositionalFeatureLearner` now has strong controlled-suite evidence in the
  pure recursive setting, but it remains caveated: the best tanh-heavy
  conservative single mechanism is positive by mean on all six controlled
  probes, while polynomial and nonlinear robustness still trail the promoted
  UPGD path.
- Generic random feature generate-and-test is not supported by committed JSON in
  the audited output set.
- The first non-synthetic benchmark is now present in canonical outputs, and
  early UPGD was negative there. Later target-structure UPGD closes the current
  supervised Step 2 matrix without a deployment portfolio/router. External
  evidence is stronger now, including compact true OpenML MNIST, true
  Fashion-MNIST delayed-context evidence, million-step SCR, and a completed
  one-seed 800-block full-source/full-task OPMNIST run. The full OPMNIST
  result is positive for online MSE, online accuracy, and final-window MSE, but
  negative for final-window accuracy and all-permutation held-out test metrics;
  the remaining replication boundary is multi-seed retained-view
  generalization, not task-count completion.
