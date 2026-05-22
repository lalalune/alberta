# Step 1 and Step 2 Alberta Plan Assessment

This document critically assesses how close the framework is to the first two
Alberta Plan steps, using the repo's current implementation, tests, and
canonical outputs. It separates three categories that should not be conflated:

- implemented capability;
- empirical evidence that the capability meets the Alberta Plan criterion;
- remaining research gap.

## Sources Checked

- `ALBERTA_PLAN.md`: local project statement of the Alberta Plan Step 1 target.
- `ROADMAP.md`: declared repo status for Steps 1 and 2.
- `TODO.md`: near-term open work.
- `docs/research/step1_results.md`: canonical Step 1 evidence.
- `docs/research/step2_results.md`: canonical Step 2 evidence.
- `docs/research/step2_universality_matrix.md`: method-by-benchmark audit.
- `docs/research/step1_step2_step3_readiness.md`: handoff audit into Step 3.
- `outputs/step2_canonical/expert_mixture_low_noise_results.json`: 10-seed
  low-noise MLP/UPGD expert-mixture candidate.
- `outputs/step2_canonical/expert_mixture_retention_results.json`: 10-seed
  retention-aware MLP/UPGD expert-mixture candidate.
- Mahmood et al. 2012, "Tuning-free step-size adaptation", Table 1:
  https://people.bordeaux.inria.fr/degris/papers/RupamAutostep.pdf
- Mahmood 2010 MSc thesis, "Automatic step-size adaptation in incremental
  supervised learning": https://library-archives.canada.ca/eng/services/services-libraries/theses/Pages/item.aspx?idNumber=696422412
- Sutton, Bowling, and Pilarski 2022, "The Alberta Plan for AI Research":
  https://arxiv.org/abs/2208.11173
- Jacobsen et al. 2019, "Meta-descent for Online, Continual Prediction"
  introduces the public AdaGain method: https://arxiv.org/abs/1907.07751
- Degris et al. 2024, "Step-size Optimization for Continual Learning":
  https://arxiv.org/abs/2401.17401
- Thomas Degris's public publication and software pages:
  https://people.bordeaux.inria.fr/degris/publications.html and
  https://people.bordeaux.inria.fr/degris/software.html
- RLPark public documentation/source trail:
  https://rlpark.github.io/documentation.html and
  https://people.bordeaux.inria.fr/degris/public/doxygen/html/functions_0x61.html
- `src/alberta_framework/core/*`: learners, optimizers, normalizers, feature
  discovery, compositional features, UPGD, CBP, and Horde bridge.
- `src/alberta_framework/streams/*`: Step 1 and Step 2 benchmark streams.
- `tests/test_step1_replication.py`, `tests/test_step2_canonical.py`, and the
  implementation-level tests for normalizers and feature learners.

## Step 1 Criterion

The Step 1 target is continual supervised learning with fixed, given features:

- scalar target `y*_t`;
- affine prediction `y_t = w_t . x_t + b_t`;
- non-stationary data, including drifting `w*_t`, drifting `b*_t`, additive
  mean-zero noise `eta_t`, and shifts in the distribution of `x_t`;
- per-feature step-size adaptation so feature relevance can be learned without
  changing the feature set;
- online normalization of feature streams;
- no offline batches, replay, or special pretraining phase.

The paper's footnote-11 comparison set also names NADALINE, IDBD, Autostep,
Autostep-for-GTD(lambda), Auto, Adam, RMSprop, and Batch Normalization.

## Step 1 Implementation Assessment

| Requirement | Current status | Assessment |
|---|---|---|
| Linear supervised learner | `LinearLearner` implements single-step prediction and update | Met |
| Fixed and adaptive step-sizes | `LMS`, `IDBD`, `Autostep`, `AdaGain`, `NADALINE`, `Adam`, `RMSprop` | Met for public methods; unpublished Auto is not implemented |
| Per-feature adaptation | IDBD/Autostep/AdaGain/NADALINE maintain per-feature state | Met |
| Bias learning | Optimizers include a bias update path | Met |
| Drifting `w*`, drifting `b*`, noise | `AlbertaPlanStep1Stream` | Met |
| Input-distribution shift | `XDistShiftStream` | Met |
| Sutton 1992 sparse-relevance replication | `SuttonExperiment1Stream` plus canonical replication script | Met |
| Online normalization | `EMANormalizer`, `WelfordNormalizer`, `StreamingBatchNormalizer` | Implementation met |
| Temporal uniformity | Streams and learners update every time step through `jax.lax.scan` | Met |
| Multi-seed statistical evidence | canonical JSON under `outputs/step1_canonical/` | Met for main claims |
| Full footnote-11 empirical comparison | Adam/RMSprop/NADALINE implemented; BatchNorm-style normalizer implemented; public AdaGain added; Auto absent because unpublished | Met for public/reproducible methods |

## Step 1 Autostep Audit

The linear Autostep implementation was audited against Mahmood et al. 2012,
Table 1. The main algorithmic order is correct and now has a regression test
that constructs a nontrivial state with nonzero traces, normalizers, and bias:

1. compute `delta x_i h_i` using the old trace;
2. update `v_i` using the old `alpha_i`;
3. update `alpha_i` through the normalized exponential meta-step;
4. compute `M = max(sum_i alpha_i x_i^2, 1)`;
5. divide all step-sizes by `M`;
6. update weights with the new step-size;
7. update `h_i` with the post-`M` step-size.

One real bug was found and fixed. The implementation previously clipped
post-`M` step-sizes to `[1e-8, 1]`. That looked harmless, but it can raise a
correctly scaled tiny step-size on very large inputs. Example: with one feature
`x = 1e6`, Table 1 scales `alpha` near `1e-12`; clipping it to `1e-8` makes
`alpha x^2 = 10000`, violating the overshoot guarantee. The clipping was
removed from both the linear path and the arbitrary-shape parameter path.

Bias handling is defensible but should be described explicitly. Mahmood et al.
write the algorithm over weights/features. `LinearLearner` is affine, so the
implementation treats the bias as an extra feature with `x = 1`, including it in
the same `M` overshoot factor. This preserves the spirit of Table 1 for the
actual learner.

The MLP helper path is not a literal Table 1 implementation for a full nonlinear
network. It applies the Autostep equations independently to each parameter
tensor using the provided gradient/trace and a per-tensor `M`. That is a
reasonable experimental generalization, but Step 1 should cite the linear path
as the paper-faithful implementation.

## Step 1 Empirical Assessment

The current canonical results support a narrower and more accurate statement
than "IDBD/Autostep beat LMS everywhere."

Supported:

- On the original Sutton 1992 task, IDBD strongly beats best-tuned LMS.
- On the noisy Sutton variant that includes the Alberta Plan's `eta_t`, IDBD
  still beats LMS decisively.
- Autostep is the strongest "tuning-free" result in the robustness grid: it
  maintains near-optimal performance over the full tested hyperparameter range.
- Online normalization is not optional on input-scale-shift streams for fixed
  LMS and IDBD-style methods.

Not supported:

- Vanilla IDBD is not robust to all Step 1 non-stationarities. In the committed
  `XDistShift` results, IDBD diverges without external normalization.
- IDBD's hyperparameter robustness is not dramatically wider than LMS in the
  committed robustness grid.
- The full 30-seed canonical normalization ablation now includes
  `StreamingBatchNormalizer`. The BatchNorm-style result is mixed: it
  stabilizes LMS on both scale-shift streams and matches EMA on
  `DynamicScaleShift`, but it hurts Autostep/Adam on `XDistShift` and does not
  rescue IDBD in the tested configuration.
- A strengthened AlbertaPlanStep1-only probe with joint normalizer and
  hyperparameter tuning does not rescue the overbroad IDBD claim. Adam,
  Autostep, and RMSprop are essentially tied near the noise floor
  (`1.009283`, `1.009305`, `1.009316` MSE respectively); tuned LMS is
  `1.011124`; IDBD is `1.013931`. Autostep slightly beats tuned LMS
  (`21/30` wins) and is competitive with Adam/RMSprop, but IDBD is worse than
  LMS (`5/30` wins). This probe is recorded under
  `output/step1_alberta_probe/` and was not promoted into the canonical
  all-stream Step 1 JSON.
- Auto (Degris in prep.) remains absent because no public specification was
  found. It should stay absent rather than be silently approximated.
- AdaGain, a public Degris-coauthored meta-descent method, is now implemented
  under its own name and included in the canonical multi-baseline run. It beats
  LMS on the Sutton tasks but is not robust to `XDistShift` in the tested grid.
- `AutostepGain` is now implemented under its own name as an experimental,
  reproducible hybrid of public Autostep and AdaGain ingredients. It is not
  exported or deserialized as `Auto`.

## Step 1 Gap Closure In This Pass

Closed:

- Added `StreamingBatchNormalizer`, a BatchNorm-style online running-statistics
  normalizer that fits the single-step API.
- Exported it from the top-level package.
- Added config serialization coverage.
- Added normalizer contract tests.
- Added it to the Step 1 normalization ablation script.
- Extended the ablation script so `StreamingBatchNormalizer` appears in config,
  paired comparisons, and Markdown summaries.
- Audited Autostep against Mahmood et al. 2012 Table 1 and removed post-`M`
  clipping that violated overshoot prevention on extreme inputs.
- Added Autostep regression tests for Table 1 ordering, bias-as-feature
  handling, and extreme-input overshoot preservation.
- Implemented `AdaGain` as the public linear-LMS specialization of Jacobsen et
  al. 2019, added config/API exports, and added formula-level regression tests.
- Regenerated the full Step 1 multi-baseline sweep with AdaGain included:
  `python "examples/The Alberta Plan/Step1/step1_full_baselines.py" --output-dir outputs/step1_canonical --seeds 30 --burn-in 20000 --measurement 10000`.
- Regenerated the full canonical normalization ablation:
  `python "examples/The Alberta Plan/Step1/step1_normalization_ablation.py" --output-dir outputs/step1_canonical`.
- Added `external = ["scikit-learn>=1.4"]` optional dependency for the external
  Step 2 benchmark, which also makes the benchmark reproducible.

Additional finding from the Step 1 closure worker, rechecked in the final
caveat audit on 2026-05-06:

- "Auto" must not be implemented under that name without a public algorithmic
  specification. The cited footnote names Auto as "Degris in prep.", and this
  audit did not find a public, reproducible update rule locally or in the cited
  public sources. AdaGain is now added as a public method, but it is not treated
  as a substitute for Auto.
- The public-source record is explicit in `docs/research/step1_results.md`:
  exact searches for the Alberta Plan footnote item found no public `Auto`
  algorithm specification; the checked public sources were the Alberta Plan
  arXiv entry/PDF (https://arxiv.org/abs/2208.11173), Thomas Degris's public
  publication page (https://people.bordeaux.inria.fr/degris/publications.html),
  Degris's public software page
  (https://people.bordeaux.inria.fr/degris/software.html), the public RLPark
  documentation/source trail (https://rlpark.github.io/documentation.html),
  the public Autostep paper
  (https://people.bordeaux.inria.fr/degris/papers/RupamAutostep.pdf), and the
  AdaGain arXiv page (https://arxiv.org/abs/1907.07751). The expanded audit
  also checked Mahmood's public Autostep thesis and Degris et al. 2024
  (https://arxiv.org/abs/2401.17401); both support Autostep/IDBD-family
  lineage or future normalized step-size optimization, but neither publishes
  `Auto`.
- Regression coverage now rejects `{"type": "Auto"}` during optimizer config
  deserialization and asserts that the package does not export `Auto`; this
  prevents silently aliasing AdaGain or Autostep as the unpublished footnote
  item.
- Production Step 1 coverage now also rejects the accidental misspelled
  `adagiven` alias, so the public kernel accepts the citable method only as
  `adagain`/`AdaGain`.
- Step 1 CPU throughput is now characterized by `benchmarks/step1_throughput.py`
  and canonical outputs in `outputs/step1_canonical/throughput/`. The full
  2026-05-04 CPU pass covers 6 optimizers, 4 normalizer settings, single scan,
  and 8-way batched scan; all 48 configurations exceed 1000 learner
  updates/sec. The slowest hot scan result is `IDBD + none` at 8559.1
  updates/sec, and the slowest hot batched result is `Adam + Welford` at
  32984.4 learner updates/sec.

Remaining after this pass:

- No remaining actionable Step 1 implementation caveat. Keep "Auto" documented
  as intentionally unavailable unless a public Degris specification is found.
  AdaGain is public and source-citable, but it is not Auto.

## Step 1 Verdict

Step 1 now meets the implemented-algorithm criterion for public Step 1 methods.
The core algorithmic setting is implemented, Autostep has been audited against
the public Table 1 algorithm, all major public comparator families except the
unpublished Auto footnote are present, and the canonical results cover the
important non-stationarity cases. The right status is:

**Step 1: complete for public, reproducible methods.**

The remaining Step 1 nuance is not a missing learner; it is the shape of the
evidence. The framework has now falsified its own overbroad claim about IDBD
robustness. The more defensible Step 1 conclusion is:

> IDBD reproduces the original sparse-relevance result and beats LMS on the
> noisy Sutton variant; Autostep is the most robust tuning-free method in the
> tested grid and is competitive with Adam/RMSprop on the strengthened
> AlbertaPlanStep1 probe; online normalization is required for scale-shift
> settings; and vanilla IDBD should not be advertised as generally robust to
> input-distribution non-stationarity without normalization.

## Step 2 Criterion

Step 2 extends Step 1 from fixed features to supervised feature finding:

- vector-valued targets;
- new features created by combining existing features;
- bounded active feature budget;
- smart generation of promising features;
- smart testing/ranking before or during promotion;
- replacement/deletion of weaker features;
- continued single-step learning without replay or offline training;
- evidence against fair nonlinear baselines, especially MLPs with enough
  capacity.

Step 2 is not merely "train an MLP online." A plain MLP is a baseline and a
substrate, not by itself an answer to explicit feature construction, testing,
and resource allocation.

## Step 2 Implementation Assessment

| Requirement | Current status | Assessment |
|---|---|---|
| Vector supervised tasks | `MultiHeadMLPLearner`, feature-discovery streams | Met |
| Online nonlinear learner | `MLPLearner`, `MultiHeadMLPLearner`, ObGD/AGC bounding | Met |
| Bounded explicit feature bank | `FixedBudgetFeatureLearner` | Met as scaffold |
| Candidate testing | shadow candidates in feature/interaction learners | Met in controlled settings |
| Product feature construction | `FixedBudgetInteractionLearner` | Met |
| Features of features | `CompositionalFeatureLearner` DAG | Implemented, weak evidence |
| Replacement/deletion | active-slot replacement and cascade deletion | Met |
| Utility tracking | feature utility, recurrent utility, UPGD utility, CBP utility, core MLP hidden-unit utility EMA | Met in several forms |
| Future utility estimation | one-step and trace-based output-loss-reduction signals via `future_utility_mix` and `future_utility_trace_decay` | Implemented; empirical closure still open |
| Learned resource manager | contextual Hedge manager over resource-policy learners; generator-internal manager in `FixedBudgetFeatureLearner` | Met for policy allocation and candidate-construction controls |
| Fair MLP baselines | MLP(64), MLP(64,64), linear baseline | Met |
| Out-of-class benchmarks | polynomial, frequency mismatch, compositional streams | Met |
| External benchmark | online/stateful sklearn digits | Met for bundled digits; larger datasets still open |
| Adaptive expert portfolio | `step2_expert_mixture.py` over MLP + low-noise UPGD with optional retention router | Met as a benchmark-closing router, not explicit feature construction |
| Step 3 handoff | `constructed_features()` and `augmented_observation()` | Met |

## Step 2 Empirical Assessment

Strong evidence:

- The old "16/16 wins over MLP" pair-product headline was audited and does not
  hold under fair controls.
- Pair-product learners can discover literal pair features when the oracle is a
  sparse pair-product class.
- In the TD/GVF bridge harness, TD-error surprise interactions now beat raw
  linear, raw MLP, and fixed interactions on the fully observable AR(1)
  positive control (`3.920609` RMSE versus `4.226761` linear, `4.142325` MLP,
  and `3.939980` fixed interactions). This is partial Step 3 evidence, not a
  general supervised Step 2 closure claim.
- UPGD beats the best fair MLP on the three synthetic out-of-hypothesis-class
  benchmark streams in the committed 30-seed suite.

Negative and closure evidence:

- The interaction learner collapses when the oracle leaves the pair-product
  hypothesis class.
- `CompositionalFeatureLearner` implements feature-of-feature construction but
  does not produce robust MLP-beating performance on the canonical out-of-class
  suite; its polynomial edge is tiny, and it loses badly on the other streams.
- The new future-utility and recursive generation path gives the compositional
  learner a causal longer-horizon/provenance mechanism and a feature-of-feature
  search mode. The promoted `single_mechanism` configuration now closes the
  focused triple-product benchmark: 10 seeds x 5,000 steps, final-window MSE
  `0.0839 +/- 0.0144` versus best fair MLP `0.5260 +/- 0.0407`, `10/10`
  paired wins, and depth>=2 active features in every seed.
- The promoted `recursive_mlp_router` closes the controlled recursive suite
  boundary by causally routing among `single_mechanism`, `mlp_32x32_no_ln`, and
  `mlp_64x64_no_ln`. On 10 seeds x 5,000 steps across nonlinear, interaction,
  triple, rare, polynomial, and frequency tasks, it beats the best fair MLP on
  `6/6` tasks; paired wins are `10/10` on nonlinear, interaction, triple, and
  polynomial, and `9/10` on rare and frequency.
- On the external online sklearn-digits benchmark, fair MLP beats UPGD:
  final-window MSE `0.0204` for MLP versus `0.0237` for UPGD, and held-out test
  accuracy `0.9477` for MLP versus `0.9354` for UPGD.
- A low-noise expert mixture over matched MLP and UPGD experts closes the
  immediate fair-MLP regression on the combined Step 2 probe. In the 10-seed
  canonical candidate run, the mixture improves or ties MLP final-window MSE on
  all eight regimes: polynomial `+0.0728`, frequency `+0.0535`,
  compositional `+0.0000`, digits IID `+0.0020`, class-blocked `+0.0000`,
  permuted-pixels `+0.000004`, mask-noise `+0.0034`, and label-drift
  `+0.0000`. It also improves mean test accuracy on digits IID by `+0.0106`
  and mask-noise digits by `+0.0108`, while tying MLP on blocked/permuted/
  label-drift digits.
- A retention-aware version of the same mixture closes the explicit
  class-blocked held-out gap. It keeps ordinary Hedge weights for prequential
  tracking, but when the observed stream has broad lifetime class coverage and
  narrow recent-window class coverage, held-out deployment weights shift to
  UPGD. On class-blocked digits this ties UPGD's retained balanced test
  accuracy instead of falling back to MLP.
- The strict universal portfolio over `mlp_h64`, `mlp_h128`, `mlp_h64_64`,
  `upgd_low_noise`, and `dynamic_sparse` closes the current best-width MLP
  matrix. In the promoted 10-seed run, no regime has negative mean
  final-window MSE versus the best fair MLP width, and no digit regime has
  negative mean held-out accuracy versus the best fair MLP width. Thirty-seed
  risk checks preserve the compositional, frequency, class-blocked, and
  non-blocked digits conclusions.

Interpretation:

UPGD alone is the strongest current Step 2 method in synthetic controlled
settings, but it is not a universal feature-finding solution. The external
digits result blocks an overgeneralized conclusion from the synthetic suite.
The strict universal portfolio changes the status of the practical fair-MLP
promotion question: a temporally-uniform router can preserve MLP behavior for
online tracking, use UPGD/dynamic-sparse where they help, and avoid the
observed held-out retained-generalization failures. It does not, by itself,
prove the Alberta Plan's stronger recursive feature-construction criterion.

## Step 2 Gap Closure In This Pass

Closed:

- Regenerated and promoted the sklearn-digits external online benchmark into
  `outputs/step2_canonical/`.
- Added regression tests that assert the external benchmark metadata and paired
  MLP-vs-UPGD metrics are present.
- Made `scikit-learn` an explicit optional dependency via the `external` extra.
- Updated results docs, universality matrix, roadmap, and TODO to reflect that
  an external benchmark now exists and is negative for UPGD alone; the later
  strict universal portfolio closes the current supervised Step 2 matrix.

Already closed before this pass, but verified in this audit:

- UPGD baseline exists.
- Compositional feature DAG exists.
- Out-of-class synthetic benchmark suite exists.
- Context/output-adaptation disentanglement exists.
- Step 2-to-Step 3 representation handoff exists.

Additional closure/evidence from the recursive utility pass:

- Added a causal trace-based output-loss-reduction estimator in
  `core.future_utility`. With trace decay `0`, it is exactly equivalent to the
  existing one-step estimator; with nonzero decay, it credits recurring
  residual/feature alignment without hindsight.
- Wired the trace estimator into `CompositionalFeatureLearner` through
  `future_utility_trace_decay`, including active/candidate traces and candidate
  provenance on promotion.
- Added `step2_recursive_feature_utility_probe.py`, a focused triple-product
  benchmark that requires depth-2 feature-of-feature construction, plus
  fair-MLP controls. The final scaffold-guard run shows the recursive product
  generator beats the original one-step feature learner, but it does not beat
  the no-LayerNorm fair MLP by mean. This closes the implementation/provenance
  TODO and documents the empirical boundary rather than claiming arbitrary
  recursive feature discovery.

Additional closure and experimental work from the agent pass:

- Added `examples/The Alberta Plan/Step2/step2_upgd_ablation.py` and ran UPGD
  perturbation/capacity/normalization ablations on synthetic and digits
  variants. Result: UPGD remains strong on polynomial and frequency synthetic
  streams, but no tested UPGD configuration beat the best fair MLP on shuffled
  or class-blocked digits final-window MSE. UPGD's class-blocked retention test
  accuracy improved, which is evidence for anti-forgetting rather than a
  current-window Step 2 win.
- Added guided compositional generation options in
  `CompositionalFeatureLearner`: utility-biased sampling, mutation,
  residual-imprint initialization, and blended promotion. Result: residual
  imprinting substantially improves the weak compositional substrate on the
  polynomial stream and beats the previous compositional learner, but it still
  loses to fair MLP on frequency and compositional streams. This closes an
  implementation gap, not the empirical Step 2 criterion.
- Added rare-task/context-aware utility options to
  `FixedBudgetFeatureLearner`: utility aggregation, top-k aggregation,
  task-balancing, task-activity decay, and utility-retention decay. Result:
  the explicit rare-head oracle-pair deletion case now has a decisive opt-in
  (`rare_protected`, active inverse-frequency utility + one-step future utility
  + slow retention). In the 12-seed rare-task stream it retained the rare
  oracle pair in `9/12` seeds versus `1/12` for mean utility and improved rare
  final-window MSE by `3.3472` paired against mean, with no common-head harm.
  This closes the narrow rare-task utility-retention gap as an opt-in, not as
  a new global default.
- Added `examples/The Alberta Plan/Step2/step2_external_suite.py` and
  `docs/research/step2_external_benchmarks.md`, covering shuffled digits,
  class-blocked digits, permuted digits, wine, and breast-cancer. Result: MLP
  remains strongest on ordinary shuffled external data; UPGD wins held-out
  balanced/transformation-averaged accuracy on class-blocked and permuted
  digits but loses final-window tracking MSE.
- Added `examples/The Alberta Plan/Step2/step2_plasticity_hybrid.py` and
  `docs/research/step2_plasticity_hybrid.md`, comparing MLP, CBP, UPGD,
  low-noise UPGD, and simple reset hybrids. Result: low-noise UPGD is the best
  candidate direction, with strong synthetic gains and retained external
  generalization under shifts, but it loses compositional synthetic and
  class-blocked final-window tracking. The naive reset hybrid should not be
  canonicalized.
- Added scheduled-perturbation controls to `UPGDLearner`
  (`perturbation_warmup_steps` and `perturbation_ramp_steps`) and tested
  delayed/ramped low-noise variants. Result: schedules did not beat plain
  low-noise UPGD as a standalone method; the core tradeoff remained.
- Extended `step2_expert_mixture.py` so the UPGD expert can use the same
  low-noise/scheduled perturbation settings and ran the 10-seed all-regime
  low-noise portfolio. Result: this is the strongest Step 2 candidate so far
  against fair MLP. It never loses mean final-window MSE to MLP across the
  eight-regime matrix and improves external IID/mask-noise digits accuracy,
  while preserving MLP behavior on blocked/permuted/label-drift digits.
- Added an opt-in class-imbalance retention router to
  `step2_expert_mixture.py`. Result: the current-block tracking predictor is
  unchanged, but class-blocked held-out deployment now switches to UPGD when
  lifetime class coverage is broad and recent-window class coverage is narrow.
  This closes the explicit best-expert retention regret in the promoted
  mixture without weakening the fair-MLP final-window MSE bar.
- Promoted `step2_universal_portfolio.py` with lower Hedge eta (`1.0`) and a
  causal online class-imbalance MSE guard. Result: the current strict
  supervised Step 2 matrix is closed against the best fair MLP width, not only
  the historical `mlp_h64` comparator. The 10-seed full matrix has no negative
  mean final-window MSE and no negative mean held-out digit accuracy versus the
  best-width MLP comparator. Thirty-seed follow-ups keep compositional positive,
  keep frequency positive by mean, tie class-blocked tracking MSE exactly, and
  keep non-blocked digits positive on both tracking MSE and held-out accuracy.

Remaining after this pass:

- Rare-task and context-aware utility. The explicit rare-head oracle-pair
  deletion case now has a reliable opt-in (`rare_protected`), but broader
  validation outside that controlled stream remains open.
- Guided compositional generation. The DAG substrate is stronger after
  residual-imprint/mutation generation, but it remains weaker than MLP on the
  main non-polynomial streams.
- Recursive feature construction remains a research boundary, not an
  implementation hole. The core MLP path now has hidden-unit utility and CBP
  replacement via `CBPMultiHeadMLPLearner` and `CBPMLPLearner`,
  `FixedBudgetFeatureLearner` learns generator/replacement/promotion controls
  internally, and the compositional learner has trace-based future utility plus
  product-biased recursive generation. The causal `recursive_mlp_router`
  closes the current controlled six-probe suite against the fair MLP, but the
  pure single recursive mechanism still does not close the nonlinear probe.
  The best pure follow-up is now `single_mechanism_retention`: it beats fair
  MLP on `5/6` probes and improves nonlinear to `0.1002`, but still trails the
  best fair MLP at `0.0597` with only `1/5` paired nonlinear wins.
- Native deep feature lifecycle remains a research boundary. Preserve-outgoing
  promotion, active low-utility perturbation, soft-gated live candidates, and
  Net2Net/function-preserving promotion are implemented. The best native deep
  result still reaches only `3/6` hard probes and misses nonlinear,
  compositional, and digits, so this path should not be used for Step 2
  closure.
- Larger external continual benchmarks. The new recurrent permutation,
  recurrent mask/noise, and class-blocked retention streams are stateful and
  non-synthetic, but still use sklearn's small bundled digits dataset.
- Published-style external stressors. `step2_published_stressors.py` now has a
  documented true OpenML MNIST path and explicit published-protocol gates. A
  compact true OpenML MNIST 5-seed run is positive, and a twenty-block
  full-source/full-task-block OpenML MNIST run satisfies the core
  source/split/order gates while improving final-window MSE by `+0.004075`
  versus the best fair MLP. Held-out retained accuracy has drifted lower by
  block 20 (`0.327580` for MSE tracking; `0.402620` for dynamic-sparse-only
  replay), so the retained-accuracy claim still needs the full run. The runner
  is now checkpointed/resumable with status and ETA reporting, but the run is
  still not the 800-task / 48M-example main OPMNIST protocol. The new
  million-step SCR run closes the Dohare public SCR scale for the narrowed
  `slow_meta` causal router (`+0.00006156 +/- 0.00001598`, `3/0/0`) while
  preserving the fair MLP comparator grid.

## Step 2 Verdict

Step 2 is now split into two distinct claims.

The narrow empirical promotion bar against fair MLP is met by the strict
universal portfolio on the current combined suite: it improves or ties the best
fair MLP width on final-window MSE across all eight tested regimes, does not
produce a mean external digits accuracy regression against the best fair MLP
width, and closes the class-blocked retained-generalization gap by combining an
online current-block MLP guard with held-out UPGD deployment.

The broader Alberta Plan feature-finding criterion is not fully met. The repo
has the machinery to study the Step 2 problem, a strong synthetic UPGD result,
a practical MLP-safe strict portfolio candidate, a learned contextual resource
manager on stateful external digits, a causal one-step future-utility signal,
and compact published-style stressor coverage. It still lacks a robust single
feature-construction mechanism and true published-scale external continual
benchmark coverage.

The right status is:

**Step 2: met for the current strict supervised benchmark promotion bar via
the universal portfolio; still a research platform rather than a complete
solution to the full Alberta Plan recursive feature-finding problem.**

The most defensible Step 2 conclusion is:

> The framework can run temporally-uniform supervised feature-finding
> experiments with vector targets, bounded feature resources, candidate testing,
> feature deletion, and fair MLP controls. UPGD is a strong synthetic
> feature-plasticity method; the strict MLP/UPGD/dynamic-sparse universal
> portfolio now clears the current best-width MLP and retained-digits benchmark
> matrix; the learned contextual resource manager clears the current harder
> stateful external digits allocation checks; exact pair-product discovery is
> positive only in controlled hypothesis-class-matched settings. The framework
> now has a causal one-step future-utility estimator, a generator-internal
> resource manager for `FixedBudgetFeatureLearner`, and compact
> published-style stressor evidence. The remaining question is empirical scope:
> recursive feature discovery is not proven arbitrary, native deep lifecycle is
> rejected as a Step 2 closure path for now, TD/GVF-target discovery is only
> partially supported, and true OPMNIST task-count scale remains to be run.
> TD/GVF-target feature finding has observable AR(1) evidence from TD-surprise
> interaction features, but predictive-state scale-up does not solve hidden
> state, off-policy, or general cumulant-discovery settings.

## Overall Readiness

Step 1 provides the expected foundation for Step 3: online linear prediction,
normalization, per-feature adaptation, and scan-based temporal uniformity.

Step 2 provides a usable representation handoff, a closed strict supervised
benchmark matrix, and a set of feature-finding research tools. Horde/GVF work
can safely consume Step 2 constructed features as inputs, but future Step 3
claims should not assume that Step 2 can reliably discover useful features
under TD targets, partial observability, off-policy data, or long-run
non-stationarity.

## Final Gap List

- Step 1: No actionable caveat remains for public/reproducible methods. The
  unpublished `Auto (Degris in prep.)` optimizer is intentionally unavailable
  unless a public, implementable Degris specification is found.
- Step 1: CPU throughput evidence is now sufficient for the linear
  daemon-readiness claim at the tested Step 1 scale; the current floor is
  8559.1 learner updates/sec on CPU.
- Step 2: Future-utility estimation exists as causal one-step and trace-based
  output-loss-reduction estimators. The controlled recursive feature suite now
  has an MLP-beating causal resource-router result. The pure single mechanism
  still loses the nonlinear probe even after signed-tanh and retention
  improvements. The remaining caveat is scope: this is empirical closure on the current
  controlled suite with routing, not a proof of arbitrary recursive
  representation discovery on all future streams.
- Step 2: Rare-task/context-aware utility now closes the explicit rare-head
  oracle-pair retention gap as an opt-in; broader validation remains open.
- Step 2: Guided feature generation mechanisms now exist, but they did not
  close the fair-MLP performance gap outside the polynomial stream.
- Step 2: Deep MLP hidden-unit testing/replacement is wired through
  `CBPMultiHeadMLPLearner` and the single-output `CBPMLPLearner` adapter, but
  native lifecycle should be rejected for Step 2 closure because hard,
  soft-gated, and Net2Net variants still miss the hard matrix.
- Step 2: UPGD alone loses ordinary shuffled/current-tracking metrics to MLP,
  but the strict universal portfolio closes the current best-width MLP and
  retained held-out digits matrix.
- Step 2: External evidence no longer lacks the OPMNIST task-count run. The
  latest packaged UPGD-memory comparison completed one seed of true OpenML
  MNIST OPMNIST at 800 task blocks / 48,000,000 online updates. It improves
  online MSE, online accuracy, and final-window MSE against the same-run best
  fair MLP, but still loses final-window accuracy and all-permutation held-out
  test metrics. The remaining external-scale caveat is retained-view
  generalization and multi-seed confirmation, not inability to run the full
  task count. Million-step SCR is closed for the narrowed `slow_meta` router at
  3 seeds.
- Step 2: Learned resource management now exists both for resource-policy
  allocation and for generator/replacement/promotion controls inside
  `FixedBudgetFeatureLearner`.
- Step 2: TD/GVF-target feature finding is only partially supported by the
  supervised Step 2 work. TD-surprise interaction features beat raw linear,
  raw MLP, and fixed interactions in the observable AR(1) downstream GVF probe,
  but this is a narrow positive control. Predictive-state/MSPBE scale-up loses
  coupled-hidden prediction to raw MLP at 10 seeds and loses the harder
  off-policy variant, so it does not establish hidden-state, off-policy, or
  general cumulant feature discovery.
