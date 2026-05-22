# Step 1/2 Review and Step 3 Readiness

This audit maps the Alberta Plan's first three roadmap steps to the current
framework.  The purpose is to make Step 3 depend only on capabilities that Step
1 and Step 2 have actually supplied.

## Step 1: Continual Supervised Learning With Given Features

### Plan Requirements

- Continual supervised prediction from feature vector `x_t` to scalar target
  `y*_t`.
- Non-stationary targets and feature distributions.
- Per-feature learning-rate adaptation so feature relevance affects learning
  speed.
- Online normalization of individual feature streams.
- Temporal uniformity: no offline training phases or special batch passes.

### Current Support

- `LinearLearner` implements scalar online prediction with `predict()` and
  `update()` single-step APIs.
- `LMS`, `IDBD`, and `Autostep` support fixed and meta-learned per-feature
  step-sizes.
- `EMANormalizer` and `WelfordNormalizer` support online normalization.
- Tracking utilities record step-size and normalization histories.
- `TDLinearLearner`, `TDIDBD`, and `AutoTDIDBD` provide the linear temporal
  prediction bridge needed by Step 3.

### Step 3 Readiness From Step 1

Step 3 can safely assume:

- given feature vectors can be consumed online;
- scalar and vector prediction heads can update every time step;
- feature-scale drift can be handled by online normalizers;
- per-feature adaptation exists for linear TD prediction.

Open issue:

- Per-parameter IDBD/Autostep for MLPs exists, but the most mature Step 3
  temporal traces are still linear/head-level rather than full nonlinear
  forward-view traces through a shared trunk.

## Step 2: Supervised Feature Finding

### Plan Requirements

- Vector-valued supervised targets, with each component treated as a task.
- Features constructed by combining existing features.
- A bounded resource budget for active represented features.
- Candidate feature testing before or during promotion.
- Utility assignment that considers current effects and likely future utility.
- Replacement/deletion of less promising features.
- No special offline training period.

### Current Support

- `NonlinearFeatureDiscoveryStream` provides hidden nonlinear latent features
  and vector targets.
- `InteractionFeatureDiscoveryStream` provides a sharper controlled benchmark:
  hidden oracle features are exact products `x_i * x_j`.
- `FixedBudgetFeatureLearner` manages bounded nonlinear tanh features, utility,
  candidate testing, generator provenance, and replacement.
- `FixedBudgetInteractionLearner` manages literal product features, including:
  active feature slots;
  shadow candidate slots;
  random, parent-mutation, imprint, and exhaustive candidate strategies;
  recurrent utility traces for recurring contexts;
  promotion and deletion decisions.
- `CompositionalFeatureLearner` implements a bounded feature DAG whose features
  can compose earlier constructed features, not just raw inputs. Its topology
  and cascade-replacement invariants are tested, but the current random
  generator does not yet robustly beat MLP baselines.
- `UPGDLearner` provides the strongest current practical Step 2 result:
  utility-scaled perturbation of low-utility hidden weights in a shared MLP
  trunk.
- `step2_expert_mixture.py` provides the current MLP-safe candidate:
  discounted Hedge routing over matched fair MLP and low-noise UPGD experts,
  plus an opt-in class-imbalance retention router for held-out deployment.
  Both experts are updated at every step.
- `CBPMultiHeadMLPLearner` provides a continual-backprop-style plasticity
  baseline with per-unit utility and replacement.
- `step2_external_online.py` provides an externally grounded, prequential
  supervised sanity check using sklearn digits.
- `constructed_features()` exposes the learned representation itself.
- `augmented_observation()` exposes `concat(raw_observation, constructed_features)`
  so Step 3 can consume Step 2 representations.

### Current Evidence

There are now two distinct evidence classes:

- Controlled sparse product-feature setting: shadow/exhaustive interaction
  testing finds real product features and improves interim mean loss when the
  oracle is inside the learner's pair-product hypothesis class.
- Canonical out-of-class suite:
  `outputs/step2_canonical/out_of_class_results.json` shows `UPGDLearner`
  beating the best fair MLP on all three tested streams: polynomial degree-3
  (`30/30` wins), frequency mismatch (`30/30` wins), and two-layer
  compositional oracle (`29/30` wins).
- Rigged-vs-fair controls show the old pair-product headline collapses to
  chance-level final-window wins once the MLP is not capacity-starved.
- Context/output disentanglement shows that even oracle constructed features
  do not solve hidden context-specific output slopes with one shared output
  head; context-indexed readouts or context-gated feature slopes unlock the
  low-loss upper bound, while one-hot context bias alone does not.
- A 5-seed online digits benchmark is negative for UPGD alone: fair MLP beats
  UPGD on final-window and held-out accuracy. This is not a continuing agent
  stream, but it is a useful guardrail against universal single-method claims.
- A 10-seed low-noise expert-mixture run closed the historical fair-MLP-h64
  matrix: the mixture improved or tied fair MLP final-window MSE across the
  eight synthetic/digits regimes and preserved mean external digits accuracy.
- The retention-aware promoted mixture then closed the class-blocked held-out
  retention gap by deploying UPGD when observed lifetime class coverage was
  broad but the recent final window was class-narrow.
- The stricter universal portfolio is the current Step 2 closure candidate: it
  compares against the best fair MLP width per seed, improves or ties mean
  final-window MSE on all eight regimes, and improves or ties mean held-out
  digits accuracy, with 30-seed scale checks on the previously fragile rows.

Interpretation:

- The framework has a concrete, temporally uniform solution to bounded
  supervised feature finding for sparse pairwise products, and a strong
  practical utility-based MLP plasticity baseline in UPGD. The strongest
  practical Step 2 result is now the strict MLP/UPGD/dynamic-sparse universal
  portfolio rather than UPGD alone.
- The controlled recursive feature-construction suite now has a causal
  resource-router result that beats the best fair MLP on all six suite tasks.
  This does not prove arbitrary recursive representation discovery on every
  future stream, nor does it solve representation learning in partially
  observable stateful environments or nonlinear feature finding embedded
  directly inside a Horde/GVF TD update.

## Step 3: What It Needs From Steps 1 and 2

Step 3 says to repeat the above two representation steps for sequential
real-time GVF prediction.  That implies two phases.

### Phase A: GVF Prediction With Given Features

Needed:

- online feature normalization and step-size adaptation from Step 1;
- vector-headed prediction from Step 2;
- temporal-difference targets and eligibility traces.

Current status:

- `HordeLearner` wraps `MultiHeadMLPLearner` with `HordeSpec` metadata.
- `GVFSpec` and `HordeSpec` represent cumulants, gamma, lambda, and demon type.
- Gamma-0 Hordes are tested to match `MultiHeadMLPLearner`.
- Temporal GVF targets `cumulant + gamma * V(s')` are implemented.
- Per-head trace decay is supported; shared nonlinear trunk traces are guarded
  because naive error-weighted trunk traces are not forward-view correct.
- `src/alberta_framework/steps/step3.py` now exposes a narrow given-feature
  production surface: `Step3HordeConfig`, `make_step3_horde()`, and
  `run_step3_smoke()`.

Verdict:

- Step 3 Phase A has enough from Step 1 and Step 2 to proceed for given feature
  vectors and multi-head GVF prediction.
- The packaged surface is a smoke/integration kernel only. It deliberately does
  not promote a general TD feature-discovery method.

### Phase B: GVF Prediction With Feature Finding

Needed:

- a representation handoff from Step 2 to Step 3;
- constructed features usable as the state/features for Horde;
- eventual temporal feature construction from old signals and traces.

Current status:

- `constructed_features()` and `augmented_observation()` provide the handoff.
- `build_step2_to_step3_arrays()` packages the causal array convention:
  row `t` is `concat(raw_t, constructed_t)`, and the Horde bootstrap input is
  the shifted augmented row `t+1`.
- `tests/test_horde.py::TestStep2ToStep3Bridge` verifies that a Horde can run
  on Step 2 augmented observations.
- `tests/test_step3_production.py` verifies the production helper's handoff
  shapes, shifted next-observation convention, finite Horde updates, config
  validation, and smoke result serialization.
- `CumulantDiscovery` now uses the GVF/nexting transition convention
  `c_{t+1} = projection @ next_observation` instead of leaking the current
  observation into the cumulant.
- `step3_feature_discovery_eval.py` evaluates downstream target-GVF RMSE after
  a Step 2 discovery warmup. The i.i.d. observable+squares probe is negative
  against MLP (`2.0350` discovered auxiliary cumulants vs `2.0284` raw MLP
  RMSE), which is expected because `obs_t` has little causal information about
  nonlinear `target_{t+1}`. The AR(1) probe gives the feature family a fair
  Markov prediction problem. The latest TD-surprise follow-up is a narrow
  positive control: TD-error-scored interaction features beat raw linear
  (`3.920609` vs `4.226761`), raw MLP (`3.920609` vs `4.142325`), and the
  fixed all-interactions control (`3.920609` vs `3.939980`) over 3 seeds on
  the fully observable AR(1) stressor. The same mechanism does not solve the
  coupled-hidden or off-policy probes. Predictive-state/MSPBE follow-ups add a
  useful shared-rollout evaluation path, but scale-up rejects robust closure:
  the default 10-seed coupled-hidden run loses to raw MLP (`2/8/0` paired
  wins), and the harder off-policy variant loses to raw clipped-IS TD (`1/4/0`).

Verdict:

- Step 3 has the mechanical interface it needs to consume Step 2 features.
- The remaining research problem is not wiring. It is making the TD/GVF result
  robust beyond the observable, hypothesis-class-matched AR(1) probe, and adding
  features based on older signals, traces, and state construction rather than
  only current supervised observations.

## Critical Gaps Before Claiming Full Step 3 Readiness

- Full nonlinear eligibility traces through a shared trunk remain unresolved.
- Feature finding under bootstrapped TD targets has partial downstream GVF RMSE
  evidence, not closure: TD-surprise interactions beat fair raw baselines in an
  observable AR(1) positive control, but predictive-state scale-up still leaves
  coupled-hidden AR(1) favoring raw MLP and harder off-policy behavior-mismatch
  favoring raw clipped-IS TD.
- Trace/history features exist as fixed features, but they are not yet part of
  learned Step 2 construction.
- Off-policy GVF learning is not yet covered by the Step 2 feature-discovery
  machinery.
- The Step 3 production surface is now present for given-feature Horde smoke
  and Step 2 array handoff, but no production helper should be treated as a
  default answer for TD/GVF feature discovery.
- Throughput evidence is local-core only: it measures in-process JAX scan loops,
  not daemon end-to-end rlsecd/security-gym transport, serialization,
  checkpointing, monitoring, or environment-step overhead.
- Step 2 evidence remains dominated by controlled synthetic streams; Step 3
  should add conditional/partially observable sequential testbeds before
  claiming general perception learning.
- External supervised evidence is mixed: UPGD alone loses ordinary digits
  tracking to fair MLP, while the retention-aware low-noise portfolio is
  MLP-safe and fixes the observed class-blocked held-out retention gap. Compact
  published-style stressors add permuted-digits and true-OpenML wins, the
  twenty-block OPMNIST core-protocol run is positive on final-window MSE and
  resumable toward full scale, and million-step SCR is now closed for a
  narrowed causal router.
  Step 3 should still avoid inheriting any Step 2 method as a default answer
  without task-family-specific validation because OPMNIST task-count scale,
  native deep lifecycle, and TD/GVF discovery remain open.
