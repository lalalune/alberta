# Alberta Framework Roadmap

Building the foundations of Continual AI, one step at a time.

The Alberta Framework follows the 12-step "retreat and return" strategy from the Alberta Plan for AI Research (Sutton et al., 2022). Each step builds on the previous, starting from the simplest possible setting and incrementally adding complexity.

## Step 1: Meta-Learned Step-Sizes — Complete (v0.1.0–v0.4.0)

**Goal**: Demonstrate that IDBD and Autostep with meta-learned step-sizes can match or beat hand-tuned LMS on non-stationary supervised learning problems.

**Delivered**:
- LMS, IDBD (Sutton 1992), Autostep (Mahmood et al. 2012) optimizers
- Linear learners with pluggable optimizers
- Online feature normalization (EMA and Welford)
- JIT-compiled scan-based learning loops with `jax.lax.scan`
- Batched multi-seed experiments via `jax.vmap`
- Step-size and normalizer tracking for meta-adaptation analysis
- TD-IDBD and AutoTDIDBD for temporal-difference learning (Kearney et al. 2019)
- Publication-quality experiment infrastructure (statistics, visualization, export)
- Factorial studies with multiple non-stationarity types and scale ranges

## Step 2: Nonlinear Function Approximation — Supervised Matrix Accepted (v0.5.0–v0.17.x)

**Goal**: Extend from linear to nonlinear function approximation while maintaining streaming, single-step updates. Demonstrate that ObGD's overshooting prevention enables stable MLP learning in the continual setting.

**Delivered**:
- ObGD bounding (Elsayed et al. 2024) with dynamic step-size bounding, decoupled as `Bounder` ABC
- AGC bounding (Brock et al. 2021) for per-unit adaptive gradient clipping
- `MLPLearner` with parameterless LayerNorm, LeakyReLU, sparse initialization
- Composable architecture: any Optimizer + optional Bounder + optional Normalizer
- `MultiHeadMLPLearner` for multi-task continual learning (shared trunk, NaN masking)
- bsuite benchmark integration with Q-learning agents (Autostep DQN, LMS DQN, Adam DQN)
- `ContinuingWrapper` for episodic-to-continuing conversion (Alberta Plan Step 6 preview)
- Agent lifecycle tracking (`step_count`, `birth_timestamp`, `uptime_s`)
- Representation utility logging for bsuite experiments
- Hybrid optimizer (`head_optimizer`) for trunk/head optimizer split on MLPLearner and MultiHeadMLPLearner
- Checkpoint utilities (`save_checkpoint`/`load_checkpoint`) for persisting learner state
- Learner config serialization (`to_config`/`from_config`) for all learners, optimizers, bounders, and normalizers
- Built-in JIT compilation on `MLPLearner` and `MultiHeadMLPLearner` `predict()`/`update()` methods
- Daemon usage guide (`docs/guide/daemon-usage.md`) covering single-step API, JIT warmup, checkpoints, config serialization, and feature diagnostics
- `MultiHeadMLPLearner` linear baseline support (`hidden_sizes=()`)
- Feature relevance diagnostics (`compute_feature_relevance`, `compute_feature_sensitivity`, `relevance_to_dict`) for periodic daemon reporting
- IDBD-MLP optimizer (Meyer): per-parameter adaptive step-sizes for MLPs via `IDBDParamState`, with `h_decay_mode` (`prediction_grads`/`loss_grads`)
- Orbax checkpointing: replaced hand-rolled npz+json with `orbax-checkpoint` for versioned pytree serialization; added `load_checkpoint_metadata` and `checkpoint_exists` utilities
- Production target-structure UPGD default for the supervised vector-target Step 2 matrix
- Strict digit/readout UPGD preset for the one-branch online-MSE digit conflict
- UPGD-memory, associative-memory, and temporal-context Step 2 production helpers
- Step 2 pipeline modes for identity, temporal-context, UPGD, and associative features

**Remaining research boundary**:
- TD/GVF-target feature discovery belongs to Step 3 and remains open
- No theorem of universal recursive representation learning is claimed
- Full published-scale OPMNIST now has a completed one-seed 800-task OpenML
  MNIST run for the latest UPGD-memory/MLP comparison; the remaining external
  scale boundary is multi-seed confirmation and all-metric closure. A follow-up
  single-UPGD H128 run wins held-out all-permutation accuracy, but fair MLP
  baselines still win final-window accuracy and held-out test MSE
- Comparison studies across diverse non-stationarity types
- AdaptiveObGD (Appendix B of Elsayed et al. 2024) with RMSProp-style second-moment normalization

## Step 3: GVF Prediction & Horde — Given-Feature Gate Complete (v0.15.0+)

**Goal**: Move from supervised prediction to General Value Function (GVF) predictions using the Horde architecture. Formalize rlsecd's existing multi-head predictions as GVF demons, extend to temporal predictions (γ > 0) with eligibility traces, and build the foundation that Step 4 control will use.

This follows the Alberta Plan's Step 3 ("Prediction I: Continual GVF prediction learning" — Sutton et al. 2022, p.8): *"Repeat the above two steps for sequential, real-time settings where the data is not i.i.d., but rather is from a process with state and the task is generalized value function (GVF) prediction."* The reference architecture is Horde (Sutton et al. 2011), where knowledge is represented as a large number of approximate value functions learned in parallel, each with its own policy, pseudo-reward, pseudo-termination, and pseudo-terminal-reward.

**Key insight**: rlsecd is already a Horde — its 5 prediction heads are implicit GVF demons with γ=0 (single-step prediction) and π=behavior (passive observation). Step 3 formalizes this, then extends to temporal predictions (γ > 0) and off-policy learning, which are prerequisites for control demons in Step 4.

### Phase 1: GVF Types & Demon Specification

A GVF demon is defined by four "question functions" (Sutton et al. 2011, §3):
- **π** (policy) — what behavior is this knowledge about?
- **γ** (pseudo-termination) — when does the prediction horizon end?
- **r** (pseudo-reward / cumulant) — what signal are we predicting?
- **z** (pseudo-terminal-reward) — what value at termination?

A demon with a fixed target policy π is a **prediction demon** (knowledge). A demon whose target policy is greedy w.r.t. its own GVF (π = greedy(q̂)) is a **control demon** (goals). Conventional value functions and SARSA Q-functions are special cases.

**Deliverables**:
- `GVFSpec` dataclass: `(cumulant_fn, gamma_fn, policy, terminal_reward_fn)` — the four question functions
- `DemonType` enum: `PREDICTION` vs `CONTROL`
- `HordeSpec`: collection of `GVFSpec` entries, one per head in `MultiHeadMLPLearner`
- Formalize rlsecd's 5 heads as `GVFSpec` instances (all γ=0, π=behavior — validates the types against an existing system)

### Phase 2: TD(λ) for MLP — Eligibility Traces

Eligibility traces are essential for temporal GVF predictions (γ > 0) and for efficient credit assignment in control. We have traces for linear TD (TD-IDBD/AutoTDIDBD) but not for MLP.

**Delivered local scope**:
- Per-parameter trace arrays on `MultiHeadMLPLearner`
- Head TD(λ) trace updates integrated with Optimizer/Bounder composition
- Per-demon trace decay via `per_head_gamma_lamda`
- Accumulating vs replacing trace modes
- ObGD-compatible trace scaling after bounded updates

**Research boundary**:
- Nonlinear shared-trunk traces with γλ > 0 remain guarded because the current
  VJP path folds per-head errors into the trunk cotangent before trace
  accumulation. Use `IndependentDemonHorde` when temporal traces need
  independent nonlinear trunks.

### Phase 3: Horde Learning Loop

**Delivered**:
- `HordeLearner` or extend `MultiHeadMLPLearner` to accept `HordeSpec`
- Per-demon TD targets computed from each demon's question functions
- Per-demon γ handling (some heads γ=0 single-step, others γ>0 temporal)
- Scan-based Horde learning loop for JIT compilation
- Prediction testbed: predict-next-observation on security-gym streams, random walk streams

### Phase 4: Off-Policy Prediction (Stretch)

The Horde paper uses GQ(λ) (Maei & Sutton 2010) for off-policy learning — each demon can learn about a target policy π different from the behavior policy b. This requires importance sampling ratios π(s,a)/b(s,a).

**Delivered local scope**:
- Linear off-policy TD with Retrace-style clipping
- ETD linear learner plumbing and tests
- Off-policy linear prediction tests

**Research boundary**:
- Nonlinear Horde/GQ/GTD with per-demon importance sampling remains open.
- The security-gym counterfactual action question requires the external
  rlsecd/active-defense rollout path and logs.

### Downstream: rlsecd as a Formal Horde

rlsecd's current 5 heads become proper GVF demons:

| Head | Cumulant (r) | γ | π | Type |
|------|-------------|---|---|------|
| is_malicious | binary label | 0 | behavior | prediction |
| attack_type | categorical | 0 | behavior | prediction |
| severity | severity score | 0 | behavior | prediction |
| session_risk | EMA risk | 0 | behavior | prediction |
| next_event_type | next event | 0 | behavior | prediction |

Step 3 extends these to temporal predictions (γ > 0): "will this session become malicious in the next N events?" Then Step 4 adds a control demon that takes defensive actions.

### Decision Point: Discounted vs Average Reward

When implementing temporal GVF predictions (γ > 0), we need to decide whether to use discounted reward, average reward, or both. The Alberta group argues that discounted reward is "a hack" — the discount factor γ conflates two distinct roles (prediction horizon and value weighting), and average-reward formulations are more natural for continuing, non-episodic agents. The Alberta Plan explicitly calls for average-reward methods in Steps 5–6, so building on discounted reward first may create technical debt. On the other hand, discounted reward is simpler to implement initially and has more established tooling (e.g., standard TD(λ), SARSA). Decide before committing to the Phase 2/3 TD target computation.

## Step 4: Control — Step 4a Complete (v0.16.0)

**Goal**: Introduce action selection. Move from prediction-only agents to control using the GVF/Horde infrastructure from Step 3. A control demon is a GVF where π = greedy(q̂) — prediction and control are the same mechanism (Sutton et al. 2011, §4).

The Alberta Plan's Step 4 ("Control I: Continual actor-critic control" — Sutton et al. 2022, p.8) says: *"The critic would presumably be that resulting from Steps 1-3."* The GVF prediction machinery from Step 3 IS the critic.

### Step 4a: SARSA — On-Policy TD Control — Complete (v0.16.0)

**Goal**: Add the first control demon to the Horde. SARSA is on-policy (behavior policy = target policy), so no importance sampling is needed — the simplest possible control demon.

**Delivered**:
- `SARSAAgent` wrapping `HordeLearner` with epsilon-greedy action selection and SARSA target computation
- `SARSAConfig`: n_actions, gamma, epsilon schedule (linear decay)
- Control demons use gamma=0 internally; real discount in `SARSAConfig.gamma` (SARSA target computed externally)
- Gumbel trick tie-breaking for uniform action selection among equal Q-values
- NaN-masking: only taken action's head receives target per step
- Mixed Horde: optional prediction demons coexist with control demons
- Three learning loops: `run_sarsa_episode` (episodic), `run_sarsa_continuing` (daemon-style), `run_sarsa_from_arrays` (JIT-compiled scan)
- Integration with all composable components (Optimizer, Bounder, Normalizer)
- Config serialization roundtrip via `to_config()` / `from_config()`
- Trunk trace guard: `MultiHeadMLPLearner` validates trunk `gamma * lamda = 0` when hidden layers present
- 30 tests, example (`sarsa_cartpole.py`), documentation (`sarsa-control.md`)

**Downstream: rlsecd as active defender**:
- Framework-side contracts map the 6 security-gym actions
  (pass/alert/throttle/block_source/unblock/isolate) to action heads.
- rlsecd still needs the external `--gym-control` daemon loop: existing
  prediction demons + one SARSA/Horde-AC control path.
- Prediction demons continue learning knowledge; control demon learns to act.
- The active-defense rollout should generate `(state, action, reward,
  outcome)` experience for the autoresearch LLM oracle pipeline once the
  rlsecd/chronos-sec repos and logs are available.

### Step 4b: Actor-Critic — Implemented, Provisional

**Delivered**:
- Linear/discrete and continuous actor-critic cores
- Horde-backed actor-critic adapter using a Step 3 Horde critic
- Pipeline `control_mode="horde_ac"` with one synchronized critic state
- bsuite comparison reports showing current actor-critic evidence is not yet
  strong enough to replace SARSA as the canonical Step 4 learner

**Remaining research boundary**:
- Promote actor-critic only after predefined evidence beats Q/SARSA on seeded
  continuing-control benchmarks.

## Steps 5–6: Continuing Control — Primitive Implemented

**Goal**: Transition from episodic to continuing (average-reward) formulations, which are more natural for long-lived agents.

**Delivered primitive surface**:
- `DifferentialTDLearner` for average-reward prediction
- `DifferentialSARSAAgent` for continuing average-reward control
- Production facades `steps.step5` and `steps.step6`
- Scan-compatible tests and smoke probes

**Remaining research boundary**:
- Nonlinear average-reward Horde/GVF integration
- Average-reward actor-critic with shared nonlinear features
- Continuing-control benchmark evidence beyond primitive probes

## Step 7: Planning — Primitive Implemented

**Goal**: Add bounded background planning from a learned transition model while
preserving foreground, continuing real-time control.

**Delivered primitive surface**:
- `Step7DynaConfig` combining Step 6 differential SARSA and Step 8 one-step
  world-model configs
- Real transition update first, then a fixed `planning_steps` budget of
  model-generated one-step backups
- Fixed-size transition memory for memory-backed planning anchors
- Search-control strategies for random, reward-prioritized,
  transition-surprise-prioritized, and predecessor-prioritized planning
- Warmup-gated planning so the world model must receive real transitions before
  generated backups are accepted
- Real action context restoration after planning, so background backups do not
  corrupt the live environment interaction state
- Production facade `steps.step7` with scan-compatible smoke tests
- Seeded one-state continuing-control sample-efficiency benchmark:
  Step 7 reward-prioritized Dyna improves final-window reward from `0.92` to
  `1.00` and Q-gap by `+4.79` over Step 6 real-only differential SARSA across
  5 seeds (`outputs/step7_dyna/results.json`)

**Remaining research boundary**:
- Full prioritized-sweeping queues with recursive priority propagation and
  learned search-control
- Off-policy accounting for imagined behavior
- Short-rollout and option/subtask discovery variants
- Seeded benchmark evidence that planning improves continuing control without
  model-bias regressions

## Step 8: One-Step World Model — Primitive Implemented

**Goal**: Learn a compact one-step environment predictor that supports Dyna-style background planning.

The world model predicts reward and next observation from the current observation and action.
It is the model-learning component used inside Step 7 (Dyna planning) and is also the building
block for the error-gated dreaming in Step 9.

**Delivered primitive surface**:
- `OneStepWorldModel` predicting scalar reward and per-channel next observation (no discount head)
- Action-conditioned input `concat(observation, one_hot(action))` with optional observation-by-action product features
- Discrete (`n_actions` one-hot) and continuous/vector (`n_actions=None`) action support
- Observation bounds tracking (`observation_min`, `observation_max`) for imagination clipping
- Step count and production facade `steps.step8` with config serialization and smoke tests

**Remaining research boundary**:
- Multi-step world models and latent dynamics prediction
- Ensemble disagreement as an uncertainty signal for dream gating
- Seeded benchmark evidence of world-model prediction error curves

## Step 9: Guarded Dreaming — Primitive Implemented

**Goal**: Extend Dyna's one-step planning to error-gated, real-state-anchored dreaming while
preserving the continuing average-reward control formulation from Step 6.

Step 9 introduces two improvements over the simple warmup-gated Dyna in Step 7:

1. **Error gating**: Dreams are accepted only when the world model's running prediction-error
   EMA is below a configurable threshold (`dreaming_max_model_error`). This protects the
   control policy from a poorly calibrated model during early learning.
2. **Buffer-anchored imagination**: A ring buffer of recent real observations supplies dream
   anchors instead of always using the current state. This improves state-space coverage of
   imagined experience without additional environment interaction.

The world model is upgraded to `ActionConditionedWorldModel`, which adds a learned discount/
termination head needed for principled multi-step rollouts. The control learner remains the
linear differential SARSA agent from Step 6 (`DifferentialSARSAAgent`).

**Delivered primitive surface**:
- `Step9DreamingConfig`: observation dim, n_actions, world-model and dreaming hyperparameters,
  planning budget, buffer capacity
- `Step9DreamingState`: combined control + world-model + observation-buffer state
- `step9_update`: real transition update → world model update → buffer add → guarded dream scan
- `run_step9_scan`: JIT-compiled scan over real continuing transition arrays
- `run_step9_smoke`: deterministic integration probe
- 22 tests covering config validation, warmup gating, error gating, zero-budget, scan shapes,
  buffer growth, and long-horizon fineness

**Remaining research boundary**:
- Multi-step rollout dreaming (horizon > 1) with behavior model policy
- Prioritized dream selection (surprise × utility scoring)
- Ensemble or latent-space uncertainty for the dream acceptance gate
- Seeded benchmark evidence that guarded dreaming improves continuing control over
  Step 7 one-step Dyna on continuing gymnasium tasks

## Step 10: STOMP Progression — Primitive Implemented

**Goal**: Introduce temporal abstraction via the STOMP progression (SubTasks, Options, Models,
Planning).  The agent can now execute temporally extended actions (options) defined by
feature-reaching subtasks, learn multi-step outcome models for each option, and act over an
extended action space of both primitive and option actions.

This corresponds to Alberta Plan Step 10 ("Intelligence I: Temporal abstraction and options" —
Sutton et al. 2022).  The reference architecture is the options framework (Sutton, Precup &
Singh 1999) applied to the continuing average-reward control setting from Step 6.

The STOMP components:
* **SubTasks** — Feature-reaching sub-problems.  Each subtask is a `SubtaskSpec` specifying a
  feature index, threshold, pseudo-reward scale, and maximum option duration.
* **Options** — Temporally extended actions.  Each option has an intra-option differential
  Q-policy trained with subtask pseudo-rewards (same average-reward formulation as Step 6).
* **Models** — Per-option outcome models.  At option termination the model observes cumulative
  pseudo-reward, accumulated discount, and the start→end state delta, updating EMA statistics
  and a linear next-state predictor.
* **Planning** — The base agent acts over the extended action set {primitives} ∪ {options}.
  When an option is selected its intra-option policy drives primitive environment actions until
  termination.

**Delivered primitive surface**:
- `SubtaskSpec` / `STOMPSpecArrays`: subtask definitions and their JAX-array representations
- `IntraOptionPoliciesState`, `OptionModelsState`, `STOMPState`: batched-over-options state
- `STOMPAgent`: `init()`, `start()`, `update()`, `scan()` with `jax.lax.cond` option branching
- `STOMPConfig` / `Step10STOMPConfig`: fully serializable configuration with `to_config()` /
  `from_config()`
- `make_step10_stomp_agent`, `init_step10_state`, `step10_update`, `run_step10_scan`,
  `run_step10_smoke`: standard production facade
- `core/options.py`: JAX-compatible STOMP core with functional, scan-friendly implementations
- 36 tests covering config validation, config roundtrip, factory, init, option termination,
  option max-step cap, base Q update, option model update, scan shapes, two-subtask runs,
  smoke probe, and 200-step fineness check

**Remaining research boundary**:
- Option and subtask discovery (rather than hand-specified subtasks)
- Semi-MDP planning: using option models for multi-step backups at the base level
- Off-policy intra-option learning (importance-sampling corrections)
- Seeded benchmark evidence that options improve continuing control over flat Step 6 on
  continuing gymnasium tasks with reachable sub-goals

## Step 11: OaK (FC-STOMP) — Primitive Implemented

**Goal**: Add Feature Construction to STOMP, producing the OaK (Options and Knowledge)
architecture.  OaK extends Step 10 with three mechanisms that keep the option set useful over
the lifetime of a continual agent.

This corresponds to Alberta Plan Step 11 ("Intelligence II: Feature construction and options" —
Sutton et al. 2022).  The three OaK additions over STOMP are:

1. **Utility tracking** — An exponential moving average (EMA) of pseudo-reward accumulates a
   per-option utility score at every time step.
2. **Curation** — `curate()` compares the lowest-utility option against a configurable
   threshold.  Below-threshold options are replaced: Q-weights, eligibility traces, option
   models, and utility statistics are reset and a new `SubtaskSpec` is drawn.
3. **Option keyboard** — A real-valued chord vector blends option Q-functions into a composite
   Q-vector: `Q_w(s,a) = Σ_i w_i Q_i(s,a)` (Barreto et al. 2019), enabling exponentially
   many composite behaviors from a finite option set.

**Delivered primitive surface**:
- `OaKConfig`, `OaKState`, `OaKAgent` with utility EMA, curation, and keyboard
- Scan-compatible utility update via `jnp.where`; curation is Python-level (outside JIT)
- `keyboard_q_values` / `keyboard_action`: L1-normalised blended Q-values and epsilon-greedy
- `Step11OaKConfig` / `Step11SmokeResult` production facade
- `core/oak.py` building on `core/options.py`
- 32 tests covering config roundtrip, factory, init, utility EMA, scan shapes, curation,
  keyboard, smoke, and 200-step fineness

**Remaining research boundary**:
- Learned feature construction (auto-generated subtask features)
- Gradient-based or information-theoretic curation selection signals
- Keyboard chord vector learning (meta-gradient or bandit-style)
- Seeded benchmark evidence that curation maintains option quality over long horizons

## Step 12: Prototype-IA — Primitive Implemented

**Goal**: Demonstrate Intelligence Amplification (IA) — an IA agent that increases the
decision-making capacity of a *partner* agent.  The IA agent is not standalone; it amplifies
another agent's intelligence via two augmentation streams.

This corresponds to Alberta Plan Step 12 ("Intelligence III: Prototype-IA" — Sutton et al. 2022).
Reference: Mathewson et al. (2023, "Communicative Capital").

1. **Exo-cerebellum** — Multi-output linear predictor anticipating future observation features.
   Prediction vector becomes an augmented feature channel:
   `augmented_obs = concat(partner_obs, predictions)`.
2. **Exo-cortex** — OaK-based (Step 11) agent learning from partner experience and broadcasting
   greedy action recommendations.  The partner can accept or ignore the recommendation.

**Delivered primitive surface**:
- `ExoCerebellumConfig` / `ExoCerebellumState` / `ExoCerebellumAgent`: vectorised multi-output
  online predictor with cyclic cumulant targets
- `ExoCortexAgent`: `OaKAgent` wrapper adding `recommend(state, obs)`
- `IAConfig` / `IAState` / `IAAgent`: paired cerebellum + cortex with `update()` and `scan()`
- `IAUpdateResult` / `IAArrayResult`: per-step `predictions`, `cerebellum_errors`,
  `recommendation`, `augmented_obs`, `cortex_td_error`
- `Step12IAConfig` / `Step12SmokeResult` production facade
- `core/intelligence_amplification.py` building on `core/oak.py`
- 30+ tests covering config validation, obs-dim mismatch guard, config roundtrip, factory,
  init shapes, update shapes/dtypes, augmented-obs concat verification, scan shapes,
  cerebellum weight update, smoke, and 200-step fineness

**Remaining research boundary**:
- Communication protocol for recommendation acceptance / rejection
- Multi-partner IA coordination
- Learned augmentation channel selection
- Exo-cortex with nonlinear function approximation
- Seeded benchmark evidence that IA augmentation improves partner decision-making

## References

- Sutton, R.S. (1992). "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"
- Sutton, R.S., Modayil, J., Delp, M., Degris, T., Pilarski, P.M., White, A., & Precup, D. (2011). "Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction." *Proc. 10th AAMAS*, pp. 761–768.
- Mahmood, A.R., Sutton, R.S., Degris, T., & Pilarski, P.M. (2012). "Tuning-free Step-size Adaptation"
- Kearney, A., Veeriah, V., Travnik, J., Pilarski, P.M., & Sutton, R.S. (2019). "Learning Feature Relevance Through Step Size Adaptation in Temporal-Difference Learning"
- Sutton, R.S., et al. (2022). "The Alberta Plan for AI Research"
- Sutton, R.S. (2025). "The OaK Architecture: A Vision of SuperIntelligence." *RLC 2025*.
- Elsayed, M., Lan, Q., Lyle, C., & Mahmood, A.R. (2024). "Streaming Deep Reinforcement Learning Finally Works"
- Brock, A., De, S., Smith, S.L., & Simonyan, K. (2021). "High-Performance Large-Scale Image Recognition Without Normalization"
- Maei, H.R. & Sutton, R.S. (2010). "GQ(λ): A general gradient algorithm for temporal-difference prediction learning with eligibility traces." *Proc. 3rd Conf. on AGI*.
- Barreto, A., Borsa, D., Hou, S., Cabi, S., Aytar, Y., Sherfield, Z., Hessel, M., Silver, D., & Munos, R. (2019). "The Option Keyboard: Combining Skills in Reinforcement Learning." *NeurIPS*.
- Mathewson, K., Pilarski, P.M., & Sutton, R.S. (2023). "Communicative Capital." *Neural Computing and Applications*.
- Meyer, E. (2025). "IDBD for MLPs" — https://github.com/ejmejm/phd_research/blob/main/phd/jax_core/optimizers/idbd.py
