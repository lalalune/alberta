# Step 2 Feature Discovery Research Plan

This note defines the concrete research program for Alberta Plan Step 2:
supervised feature finding under a fixed feature budget. The current codebase
now has two different kinds of evidence:

- a controlled, interpretable solution for sparse pair-product discovery; and
- a stronger practical baseline, `UPGDLearner`, that beats fair MLP baselines on
  three synthetic out-of-hypothesis-class supervised streams.

This is meaningful progress, but it is not a general solution to arbitrary
recursive representation learning. The pair-product result is intentionally
narrow, and UPGD is a soft plasticity/perturbation mechanism rather than a full
smart generator plus explicit candidate-testing resource manager.

See also `step2_directions_swot.md` for the ranked experimental directions and
SWOT analysis, and `step1_step2_step3_readiness.md` for the handoff audit from
Steps 1-2 into Step 3.

## Scientific Target

Step 2 is not merely "use an MLP."  The target is a learner that can:

- construct new nonlinear features from existing signals;
- test their utility online across vector-valued supervised tasks;
- preserve useful features;
- discard low-utility features;
- do all of this under a bounded feature budget and without a special training
  phase.

## Implemented Scaffold

- `NonlinearFeatureDiscoveryStream`: a vector-target, nonstationary supervised
  benchmark with hidden nonlinear latent features and recurring relevance
  contexts.
- `InteractionFeatureDiscoveryStream`: a sharper Step 2 benchmark where hidden
  oracle features are exact pairwise products `x_i * x_j`, with exactly
  `active_pairs_per_context` useful products per task/context.
- `FixedBudgetFeatureLearner`: a one-hidden-layer feature bank with explicit
  feature slots, per-feature utility, feature ages, generator provenance, and
  utility-based replacement.
- `FixedBudgetInteractionLearner`: an exact product-feature learner that
  proposes pairwise feature constructions, trains shadow candidates on residual
  error, promotes high-utility candidates, and discards low-utility active
  features.
- `CompositionalFeatureLearner`: a bounded feature DAG whose active and
  candidate features can compose existing constructed features through product,
  sum, tanh, and gated operations. It now has regression coverage for
  topological validity, cascade replacement, and candidate minimum-age refresh.
  Its current random search strategy still does not produce robust wins over
  fair MLP baselines on the canonical out-of-class suite.
- `UPGDLearner`: a utility-perturbed-gradient-descent learner over a shared MLP
  trunk. It tracks low-utility hidden weights and perturbs them instead of
  deleting whole slots. This is the strongest current Step 2 empirical result.
- Causal future utility: `one_step_output_loss_reduction` estimates the
  one-step counterfactual reduction in squared output error if a feature's
  output weight receives the current LMS residual update. This credits
  unweighted shadow candidates before outgoing weights have accumulated, while
  remaining causal and temporally uniform.
- `CBPMultiHeadMLPLearner`: continual-backprop-style hidden-unit replacement.
  This is implementation-sound after the multi-layer utility-gradient fix, but
  should be treated as plasticity preservation rather than full feature
  construction.
- Shadow candidates: optional candidate features train against residual error
  without contributing to prediction until promoted.
- Recurrent utility traces: slower utility decay can preserve features that were
  useful in previous recurring contexts.
- Exhaustive interaction candidates: when the candidate budget covers the
  finite pair-product space, candidates can continuously test every possible
  `x_i * x_j` feature while active prediction remains limited to a smaller
  feature budget.
- Generator priors: random features, parent mutation, and imprint generation.
- `feature_discovery_experiments.py`: experiment runner comparing linear/MLP
  baselines, static feature banks, random generate-and-test, shadow candidates,
  conservative/fast promotion, and generator priors.
- `step2_context_disentanglement.py`: a two-pass probe that freezes discovered
  pair-product features and retrains linear readouts to separate representation
  quality from output-weight/context adaptation.
- `step2_external_online.py`: an externally grounded online supervised
  benchmark using sklearn's bundled digits dataset. It is not an Alberta-style
  agent stream, but it is a useful non-synthetic sanity check for MLP vs UPGD.

## Experiment Matrix

1. Benchmark validation
   - Use `NonlinearFeatureDiscoveryStream`.
   - Vary feature dimension, number of hidden latents, context length, and
     active latent count.
   - Required outcome: linear baselines should fail when nonlinear latent
     structure dominates, while sufficiently large nonlinear learners improve.

2. Current MLP baseline
   - Use `MultiHeadMLPLearner` with `hidden_sizes=()` for linear and
     `hidden_sizes=(H,)` for MLP+ObGD.
   - Required outcome: quantify whether ordinary streaming MLP learning already
     matches or beats explicit feature lifecycle methods.

3. Utility measurement
   - Current utility combines outgoing-weight magnitude, activation, and online
     feature credit.
   - Next ablations should compare weight-only, gradient-only, ablation-loss,
     and task-balanced utility.

4. Random generate-and-test
   - Use `generate_test_random`.
   - Required outcome: demonstrate whether utility-based random replacement
     beats a static feature bank under equal feature budget.

5. Shadow candidate pool
   - Use `shadow_candidates_random`, `shadow_candidates_safe`, and
     `shadow_candidates_fast`.
   - Required outcome: measure the tradeoff between interim performance and
     speed of realized utility.

6. Generator comparison
   - Use `generator_priors`.
   - Next ablations should isolate random, parent mutation, and imprint
     generation rather than mixing them.

7. Multitask utility
   - Current targets are vector-valued and all heads contribute to utility.
   - `FixedBudgetInteractionLearner` now has opt-in `utility_aggregation="max"`
     and `utility_retention_decay` knobs. Smoke evidence shows mean utility can
     dilute single-head evidence, but max utility is not a clean default win.
   - Next work should add task-frequency correction and rare-task protection
     only with downstream loss evidence.

8. Utility-scaled perturbation
   - Implemented as `UPGDLearner`.
   - Next ablations should test whether the gain comes from utility-scaled
     perturbation specifically, from sparse initialization/layer normalization,
     or from broader MLP hyperparameters.

9. Structured compositional features
   - Implemented as `CompositionalFeatureLearner`.
   - Current evidence is mostly negative: the DAG can construct features of
     features and now has a tiny polynomial edge, but its random generator loses
     badly on frequency and compositional streams. The high-value next
     experiment is guided generation, not more claims about the existing random
     search.
   - Candidate refresh now respects `candidate_min_age`, which makes aggressive
     recycling schedules scientifically cleaner but did not close the UPGD/MLP
     performance gap in the 30-seed canonical rerun.

10. Meta-learned resource manager
   - Not implemented yet.
   - Treat generator choice, replacement rate, and promotion blend as online
     decisions, not fixed hyperparameters.

## Current Demonstrations

These are controlled demonstrations. They resolve a concrete Step 2 subproblem:
bounded construction, testing, retention, and deletion of sparse pairwise
features in continual supervised vector prediction. They do not by themselves
establish general feature discovery.

0. Canonical out-of-class suite
   - Source: `outputs/step2_canonical/out_of_class_results.json`.
   - Result: `UPGDLearner` beats the best fair MLP on all three streams:
     polynomial degree-3 (`30/30` wins), frequency mismatch (`30/30` wins), and
     two-layer compositional oracle (`29/30` wins).
   - Interpretation: this is the strongest current practical Step 2 result.
     It supports utility-based plasticity/perturbation as a serious supervised
     feature-finding direction, but it does not expose explicit constructed
     feature provenance or solve smart recursive feature generation.

1. Stationary sparse pair-product ablation
   - Command:
     ```bash
     PYTHONPATH=src python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" \
       --benchmark interaction --num-steps 2500 --seeds 8 \
       --feature-dim 10 --n-tasks 2 --n-contexts 1 --context-length 2500 \
       --active-pairs 2 --n-features 8 --candidate-count 64 \
       --replacement-interval 40 --min-feature-age 40 --candidate-min-age 20 \
       --step-size-output 0.04 --mlp-hidden 8 --mlp-step-size 0.03 \
       --noise-std 0.01 \
       --output-dir outputs/step2_interaction_stationary_exact_8seed
     ```
   - Result: `interaction_shadow_safe` reached mean final-window loss
     `0.0175 +/- 0.0104`, while same-budget `mlp_obgd` reached
     `0.4412 +/- 0.0885`.
   - Interpretation: shadow candidate testing can discover and retain the
     correct constructed product features when the hidden feature set is stable.
     Learned active features overlapped hidden oracle pairs by about `0.61` of
     unique active slots, versus about `0.09` for a static random product bank.

2. Non-stationary sparse pair-product benchmark, first pass
   - Command:
     ```bash
     PYTHONPATH=src python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" \
       --benchmark interaction --num-steps 2500 --seeds 8 \
       --feature-dim 10 --n-tasks 2 --n-contexts 4 --context-length 300 \
       --active-pairs 1 --n-features 8 --candidate-count 64 \
       --replacement-interval 40 --min-feature-age 40 --candidate-min-age 20 \
       --step-size-output 0.04 --mlp-hidden 8 --mlp-step-size 0.03 \
       --noise-std 0.01 \
       --output-dir outputs/step2_interaction_sparse_exact_8seed
     ```
   - Result: `interaction_shadow_safe` reached mean final-window loss
     `0.8253 +/- 0.2218`; same-budget `mlp_obgd` reached
     `0.8747 +/- 0.2248`; linear reached `0.9635 +/- 0.2351`.
   - Interpretation: this is a positive but weak Step 2 result.  The explicit
     feature-lifecycle learner is best on mean final-window loss and has much
     higher oracle-pair overlap than a static product bank, but the margin over
     MLP is small relative to seed variance.  This should be treated as a
     promising demonstration, not a decisive result.

3. Non-stationary sparse pair-product benchmark, recurrent/exhaustive candidate
   pass
   - Command:
     ```bash
     PYTHONPATH=src python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" \
       --benchmark interaction --num-steps 2500 --seeds 16 \
       --feature-dim 10 --n-tasks 2 --n-contexts 4 --context-length 300 \
       --active-pairs 1 --n-features 8 --candidate-count 64 \
       --replacement-interval 40 --min-feature-age 40 --candidate-min-age 20 \
       --step-size-output 0.04 --mlp-hidden 8 --mlp-step-size 0.03 \
       --noise-std 0.01 \
       --output-dir outputs/step2_interaction_exhaustive_exact_16seed
     ```
   - Mean-loss result, measuring interim performance across the whole stream:
     `interaction_exhaustive_candidates` reached mean loss `0.8903`; same-budget
     `mlp_obgd` reached `1.0019`.  Paired by seed, MLP-minus-method mean loss
     was `0.1116 +/- 0.0116`, with the exhaustive candidate method beating MLP
     on `16/16` seeds.
   - Final-window result: `interaction_shadow_recurrent` had the best
     final-window loss (`0.8446 +/- 0.1547`) versus `mlp_obgd`
     (`0.9083 +/- 0.1522`).  The paired final-window margin was positive but
     smaller: `0.0638 +/- 0.0289`, with wins on `11/16` seeds.
   - Feature-construction result: `interaction_exhaustive_candidates` had mean
     hidden-oracle pair overlap `0.671` (`4.31` oracle hits on average), versus
     `0.091` (`0.69` hits) for the static random product bank.
   - Interpretation: this is the strongest controlled product-feature result.
     Exhaustive shadow testing decisively improves interim performance and
     demonstrably finds useful constructed features. Final-window performance is
     still less decisive because the benchmark changes output weights by
     context without exposing a context variable; retaining the right features
     does not by itself solve recurring output-weight adaptation.

4. Context/output-adaptation disentanglement
   - Sources: `outputs/step2_context_disentanglement/results.json` and
     `outputs/step2_context_disentanglement_workerB_default/results.json`.
   - Result: learned pair banks recovered most oracle-active pairs in the small
     run, but low error required context-indexed heads or explicit
     context-gated feature slopes. Final-window loss:
     `oracle_augmented:context_indexed = 0.0278`,
     `oracle_augmented:context_gated_slopes = 0.0286`,
     `learned_augmented:context_indexed = 0.1312`,
     `learned_augmented:context_gated_slopes = 0.1327`,
     `oracle_augmented:observable_single = 0.3588`, and
     `raw:hidden_single = 0.5169`.
   - Interpretation: the final-window caveat is real and experimentally
     separated. Good feature construction improves the representation, but a
     single output head cannot remember context-specific slopes when the
     context is hidden. A one-hot context bias is not enough; the readout needs
     context-conditioned feature slopes. Step 2 should therefore evaluate
     representation quality separately from output-memory/context adaptation.

5. CBP as a Step 2 baseline
   - Source: `outputs/step2_canonical/out_of_class_results.json`.
   - Result: in the 30-seed canonical run, CBP behaves like an MLP baseline and
     does not challenge UPGD. Final MSE on polynomial/frequency/compositional:
     UPGD `0.5767/0.6335/0.1634`; CBP `1.2036/1.1951/0.1903`; best MLP
     `1.1458/1.1689/0.1908`.
   - Interpretation: CBP is useful as a plasticity baseline, but current
     evidence does not support it as the main Step 2 solution.

6. External online digits benchmark
   - Source: `outputs/step2_canonical/digits_online_results.json`.
   - Result: on sklearn digits, 5 seeds x 3000 online training steps, fair MLP
     beats UPGD: final-window accuracy `0.9668` vs `0.9496`; held-out test
     accuracy `0.9477` vs `0.9354`.
   - Interpretation: this is the strongest anti-universality result in the
   current workspace. UPGD is strong on synthetic nonstationary streams but
   is not a universal replacement for fair MLP training.

7. Future-utility estimator smoke audit
   - Source:
     `outputs/worker_s2_future_utility/{rare_task,interaction,nonlinear}`.
   - Implementation: `src/alberta_framework/core/future_utility.py` plus
     opt-in `future_utility_mix` in fixed-budget feature learners.
   - Rare-task result, 4 seeds x 1200 steps: pure counterfactual future utility
     improved rare final MSE versus mean utility by `0.5353` and raised rare
     pair survival from `0.25` to `0.50`, but common final MSE worsened by
     `0.2310` and active final MSE worsened on `0/4` seeds. It is therefore
     not a safe rare-task default.
   - Interaction-stream result, 3 seeds x 600 steps:
     `interaction_shadow_future` beat `mlp_obgd` on final-window loss in
     `3/3` seeds (`mlp - method = 0.1365 +/- 0.0514`), but remained behind
     existing `interaction_exhaustive_candidates` (`0.8046` vs `0.7296` mean
     final-window loss).
   - Nonlinear-stream result, 3 seeds x 600 steps:
     `shadow_candidates_future` lost to `mlp_obgd` on `0/3` seeds
     (`mlp - method = -0.1100 +/- 0.0409`) and was also slightly worse than
     `shadow_candidates_safe`.
   - Interpretation: future utility is implemented and scientifically useful
     as an opt-in diagnostic/candidate-credit signal, but it does not close the
     fair-MLP gap on the current nonlinear Step 2 stream and it introduces a
     common-task regression in the rare-task setting. The remaining gap is
     controlled weighting between immediate rare-feature preservation and
     current-task error, not merely the absence of a causal future signal.

## Critical Risks

- Utility is confounded by downstream weights.  A feature may look useless
  because the output head has not learned to use it yet.
- In the non-stationary interaction benchmark, the target mapping changes by
  hidden context while the observation distribution does not expose that
  context.  This intentionally stresses continual adaptation, but it means final
  error is partly an output-weight memory problem rather than purely a feature
  construction problem.
- Random replacement can erase rare but important features.  Minimum age and
  shadow candidates reduce but do not eliminate this.
- Candidate promotion can damage interim performance if candidate output weights
  are copied too aggressively.
- The core controlled benchmark suite is still synthetic.  It is necessary for
  controlled science, but success there does not imply real agent-state
  discovery; the external digits result is small and negative for UPGD alone,
  even though the later strict portfolio closes the current supervised Step 2
  benchmark matrix.
- A one-layer feature bank is easier to analyze than a full MLP, but it is not a
  complete answer to recursive construction of features from features.
- UPGD's utility is still backward-looking and per-weight. Its strong empirical
  performance does not yet answer how to estimate likely future utility, how to
  explain constructed features, or how to prioritize feature proposals from
  prior construction experience.
- The current compositional DAG is an important substrate, but the committed
  evidence says its generator/search policy is too weak. Treat it as a negative
  control until guided generation shows a reproducible advantage.
- The external digits benchmark shows UPGD can lose to a fair MLP on a small
  real dataset. Any Step 2 claim must now specify the stream family and
  nonstationarity regime where the method is expected to help.

## Smoke Command

```bash
PYTHONPATH=src python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" \
  --quick --output-dir outputs/step2_feature_discovery
```

## Main Metrics

- final-window mean squared error;
- final-over-first loss ratio;
- number of replacement or promotion events;
- final mean/min/max feature utility;
- constructed-feature oracle overlap on interaction benchmarks;
- learning-curve area under loss for interim-performance comparisons.
