# TODO

Immediate next steps and near-term work items for the Alberta Framework.

## Step 2 — Remaining Work

- [ ] Full OPMNIST solution gate: at least 3 completed published-scale seeds
      (800 tasks, 48,000,000 updates/seed, all held-out permutation views) with
      the promoted Step 2 learner beating the best fair MLP comparator on
      online MSE, online accuracy, final-window MSE, final-window accuracy,
      held-out test MSE, and held-out test accuracy. This is the condition for
      `opmnist_solution_status(...)[solved_opmnist_step2] == true`; fresh
      `step2_upgd_memory_opmnist.py` outputs write this as `solution_status`.
      Promotion command: `python benchmarks/step2_opmnist_solution_gate.py
      <result_json>` must exit 0 without `--allow-unsolved`.
      Run-plan command: `python benchmarks/step2_opmnist_full_run_plan.py
      --write-plan outputs/step2_opmnist_solution_full/plan.json`.
      The default plan targets `step2_hybrid_memory_trace` and
      `step2_hybrid_memory_trace_adaptive_sharp` against h64/h128 fair MLP and
      sharpened-MLP comparators.
      Parallel seed outputs must be merged with
      `benchmarks/step2_opmnist_merge_seed_results.py` before promotion.
      Preferred coordinator: `python
      benchmarks/step2_opmnist_solution_pipeline.py --run-next --no-dry-run`
      to resume the first missing seed, then `--merge-ready --audit
      --no-dry-run` once all split seed results exist.
      Use `--run-next --run-next-chunks N --no-dry-run` for bounded scheduler
      windows; it checkpoints/statuses partial progress without writing final
      result JSONs.
      Fresh seed results must retain their `manifest` blocks with argv, git,
      environment, method-list, and source-hash provenance for paper writing.
      The merged artifact must preserve these manifests and SHA-256 hashes in
      `manifest.split_results`; `solved_opmnist_step2` stays false unless
      `artifact_provenance.provenance_complete` is true.
- [x] Neuron utility tracking (per-hidden-unit EMA of gradient magnitude) — `MLPLearner(track_neuron_utility=True)`, `dormant_neuron_fraction`, `reset_dormant_neurons`
- [ ] Feature generation and testing ("generate and test" mechanisms)
- [ ] Nonlinear feature discovery for streaming problems
- [ ] Comparison studies: MLPLearner across diverse non-stationarity types (drift, abrupt, periodic)
- [x] AdaptiveObGD (Appendix B of Elsayed et al. 2024) — RMSProp-style second-moment normalization
- [ ] More bsuite sweep experiments and analysis (beyond catch/cartpole)

## Step 3 — GVF Prediction & Horde

Formalize rlsecd's multi-head predictions as GVF demons (Sutton et al. 2011, "Horde"), extend to temporal predictions with eligibility traces, and build the prediction infrastructure that Step 4 control will use. Per the Alberta Plan (Sutton et al. 2022, p.8): the critic in Step 4 "would presumably be that resulting from Steps 1-3."

### Phase 1: GVF Types & Demon Specification — Complete (v0.15.0)
- [x] `GVFSpec` dataclass in `core/types.py` — question functions: cumulant_index, gamma, lamda, terminal_reward
- [x] `DemonType` enum: `PREDICTION` (fixed target policy) vs `CONTROL` (π = greedy(q̂))
- [x] `HordeSpec` — collection of `GVFSpec` entries with pre-computed gamma/lambda arrays
- [x] Validate types by expressing rlsecd's 5 heads as `GVFSpec` instances (all γ=0, π=behavior)
- [x] Config serialization: `GVFSpec.to_config()` / `from_config()` consistent with existing patterns
- [x] `HordeLearner` wrapping `MultiHeadMLPLearner` with per-demon gamma/lambda and TD targets
- [x] Per-head trace decay via `per_head_gamma_lamda` on `MultiHeadMLPLearner`
- [x] Scan-based Horde learning loop + batched variant
- [x] Test: Horde with all γ=0 demons matches `MultiHeadMLPLearner` behavior exactly
- [x] Trunk trace guard: `MultiHeadMLPLearner` validates trunk `gamma * lamda = 0` when hidden layers present

### Phase 2: TD(λ) Eligibility Traces for MLP
- [x] Per-parameter eligibility trace arrays on `MultiHeadMLPLearner` (matching weight shapes in each layer)
- [x] Head TD(λ) update rule: `e_t = γ_t * λ * e_{t-1} + ∇_θ q̂(s,a)` integrated with Optimizer/Bounder composition
- [x] Trace decay λ configurable per demon (Horde §4: λ is an "answer function") via `per_head_gamma_lamda`
- [x] Accumulating vs replacing traces option (`TraceMode` enum, replacing traces in `MultiHeadMLPLearner`)
- [x] Integration with ObGD bounding — traces scaled by bounding factor after trunk and head bounding
- [x] Test: TD(λ=0) / γλ=0 resets traces on the MLP path and supports arbitrary γ with λ=0
- [x] Test: linear MLP (`hidden_sizes=()`) permits temporal traces and roundtrips per-head γλ
- [ ] Research boundary: nonlinear shared-trunk temporal traces with γλ>0 remain guarded; use `IndependentDemonHorde` for per-demon nonlinear trunks

### Phase 3: Off-Policy Prediction (Stretch)
- [x] Linear off-policy TD with Retrace-style clipping and ETD plumbing
- [ ] Nonlinear Horde importance sampling ratios π(s,a)/b(s,a) per demon
- [ ] GQ(λ) or GTD(λ) for stable nonlinear off-policy learning with function approximation (Maei & Sutton 2010)
- [x] Linear off-policy prediction demon test: learn about a policy different from behavior
- [ ] Test on security-gym: "what would happen if we blocked this IP?" (prediction about untaken action)

## Step 4a — SARSA (On-Policy TD Control) — Complete (v0.16.0)

- [x] `SARSAAgent` wrapping `HordeLearner` — control demons + optional prediction demons
- [x] Control demons: gamma=0 internally, real discount in `SARSAConfig.gamma`, SARSA target computed externally
- [x] ε-greedy action selection with Gumbel trick tie-breaking, linear epsilon decay
- [x] NaN-masking: only taken action's head updated per step
- [x] `run_sarsa_episode` — single episode gymnasium loop
- [x] `run_sarsa_continuing` — continuing loop for streaming environments (daemon-style)
- [x] `run_sarsa_from_arrays` — JIT-compiled scan for pre-collected data (security-gym pattern)
- [x] Config serialization roundtrip via `to_config()` / `from_config()`
- [x] 30 tests: init, action selection, update logic, epsilon decay, bounding, serialization, scan
- [x] Example: `sarsa_cartpole.py` (episodic CartPole)
- [x] Example: `sarsa_trading.py` (continuing FOREX/STOCKS via gym-anytrading)
- [x] Documentation: `docs/guide/sarsa-control.md`

### Remaining Step 4a Work
- [x] bsuite catch/cartpole comparison: SARSA agent alongside existing DQN agents
- [x] Multi-seed statistical comparison of SARSA vs Q-learning (AlbertaAgent)
- [x] Broader primary bsuite SARSA-vs-Q report

Artifacts:

- `outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md`
- `outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md`

### Downstream Integration (rlsecd)
- [ ] External: rlsecd `--gym-control` mode: existing 5 prediction demons + SARSA control demon
- [x] Framework-side contract maps 6 security-gym actions (pass/alert/throttle/block_source/unblock/isolate) to action heads
- [x] Local SARSA scan throughput sustains >1000 steps/sec on CPU
- [ ] External: rlsecd end-to-end throughput must include parsing, feature extraction, learner update, checkpoint/reporting, and action dispatch
- [ ] External: generate `(state, action, reward, outcome)` experience for autoresearch LLM oracle pipeline from rlsecd/security-gym rollouts

## Step 4b — Actor-Critic (Implemented, Provisional)

- [x] Stream AC(lambda): Actor-critic with eligibility traces
- [x] Policy gradient with ObGD-style overshooting prevention hook
- [x] Continuous and discrete action spaces
- [x] Horde-backed actor-critic critic path using Step 3 machinery
- [x] bsuite runner/reporting path for Q/SARSA/AC/Horde-AC comparisons
- [x] `NonlinearHordeActorCriticAgent`: MLP actor with `jax.grad` policy gradient,
  eligibility traces through all layers, sparse init, 28 tests, bsuite adapter (nlhac)
- [x] 10-seed catch/0 bsuite evidence: nlhac closes 20 regret units vs linear AC
  (458 vs 478); SARSA (374) remains the dominant baseline at this horizon
- [x] Autostep-for-actor: per-weight step-size adaptation for the actor MLP —
  complete for `NonlinearHordeActorCriticAgent` (fea98e1) and
  `AverageRewardHordeActorCriticAgent`; re-gate promotion pending seeded evidence

## rlsecd Integration

- [x] AF-1: Checkpoint utilities — `save_checkpoint`/`load_checkpoint` + `to_config()`/`from_config()` (rlsecd needs to consume)
- [x] AF-1: Orbax checkpointing migration — replaced npz+json with `orbax-checkpoint`, added `load_checkpoint_metadata`/`checkpoint_exists`
- [x] AF-3: Document single-step learner API for daemon use (`docs/guide/daemon-usage.md`)
- [x] AF-4: JIT-compile `predict()`/`update()` on MLPLearner and MultiHeadMLPLearner (upstream)
- [x] AF-2: Get permission from Edan Meyer to publish IDBD-MLP
- [x] AF-2: Merge IDBD-MLP into main (Meyer adaptation with IDBDParamState, 18 tests)
- [ ] External: AF-2 IDBD-MLP 100k-event replay test in rlsecd
- [ ] External: AF-2 IDBD-MLP full 1.6M log stability test
- [ ] External: simplify rlsecd SecurityAgent to use Orbax checkpoint utilities (format v2)
- [ ] External: simplify rlsecd SecurityAgent to use framework config serialization
- [ ] External: integrate `compute_feature_relevance` into rlsecd periodic reporting (60s interval)

## Step 8 — One-Step World Model (Primitive Implemented)

- [x] `OneStepWorldModel`: reward + next-observation prediction from `concat(obs, action_one_hot)`
- [x] Observation bounds tracking for imagination clipping
- [x] Production facade `steps.step8` with config serialization and smoke tests
- [x] Bounded multi-step rollout consumer for action-conditioned world models — `DreamRolloutConfig`, `ActionConditionedDreamWorld`, and `dream_rollout`
- [x] Ensemble uncertainty signal — `step8_ensemble_predict` reports reward, next-observation, and total disagreement
- [x] Seeded benchmark: `benchmarks/step8_world_model_prediction.py` proves online reward/next-observation prediction and ensemble-disagreement collapse over 10 seeds
- [x] Step 8 solution gate — `benchmarks/step8_solution_gate.py`

## Step 9 — Guarded Dreaming (Completion Gate Accepted)

- [x] `Step9DreamingConfig`: unified config for control + world model + dreaming knobs
- [x] `Step9DreamingState`: combined control + world-model + behavior-model + observation-buffer state
- [x] `step9_update`: real update → world model update → behavior-model update → buffer add → prioritized guarded dream scan
- [x] Error gate: dreams accepted only when `model_error_ema <= dreaming_max_model_error`
- [x] `RecentObservationBuffer`: ring buffer for real-state dream anchors
- [x] Warmup gate: dreams blocked until `step_count >= dreaming_warmup_steps`
- [x] Zero-budget path (scan over empty arange) is JIT-compatible
- [x] `run_step9_scan` / `run_step9_smoke` with config roundtrip and 27 tests
- [x] Production facade `steps.step9`; exported from `steps/__init__.py`
- [x] Multi-step rollout dreaming (horizon > 1) with a learned behavior model
- [x] Prioritized dream selection (surprise × utility from `score_dream_candidates`)
- [x] Seeded benchmark evidence: guarded dreaming vs Step 7 one-step Dyna on continuing tasks — `benchmarks/step9_guarded_dreaming.py` proven: 9/10 seeds win over naive Dyna, +0.0117 Phase 2 improvement on 1-state switching bandit
- [x] Step 9 solution gate — `benchmarks/step9_solution_gate.py`

## Step 10 — STOMP Progression (Completion Gate Accepted)

- [x] `SubtaskSpec` / `STOMPSpecArrays`: feature-reaching subtask definitions and JAX arrays
- [x] `IntraOptionPoliciesState`, `OptionModelsState`, `STOMPState`: batched state for N options
- [x] `STOMPAgent`: `init()`, `start()`, `update()`, `scan()` with `jax.lax.cond` option branching
- [x] Intra-option differential Q-learning with pseudo-rewards per subtask
- [x] Option outcome models: EMA cumulative reward, EMA discount, linear next-state delta predictor
- [x] Option termination: feature threshold OR max-step cap, both as JAX boolean ops
- [x] Extended action space: base agent acts over {primitives} ∪ {options}
- [x] `STOMPConfig` / `Step10STOMPConfig`: `to_config()` / `from_config()` roundtrip
- [x] Production facade `steps.step10`: `make_step10_stomp_agent`, `init_step10_state`,
      `step10_update`, `run_step10_scan`, `run_step10_smoke`
- [x] 45 tests: config validation, roundtrip, factory, init, termination, max-step, scan shapes,
      two-subtask runs, smoke, 200-step fineness
- [x] Option discovery: `feature_to_subtask_specs()` and `subtasks_from_feature_scores()` auto-discover subtasks from Q-weight importance — `benchmarks/step10_feature_autodiscovery.py` proven: 10/10 seeds discover correct features [5,4] from Q-weights
- [x] Semi-MDP planning with option models for multi-step base Q backups — `_differential_semidp_q_update` in `core/options.py`; verified in 42 tests and step10_stomp_options benchmark
- [x] Off-policy intra-option learning with clipped importance-sampling corrections
- [x] Seeded benchmark evidence: options vs flat Step 6 on continuing tasks with sub-goals — `benchmarks/step10_stomp_options.py` proves STOMP accelerates control on 6-state chain vs flat DifferentialSARSA
- [x] Step 10 solution gate — `benchmarks/step10_solution_gate.py`

## Step 11 — OaK (FC-STOMP) (Completion Gate Accepted)

- [x] `OaKConfig`: wraps `STOMPConfig`, adds `utility_ema_decay` and `curation_threshold`
- [x] `OaKState`: extends `STOMPState` with `utility_ema`, `execution_counts`, `cumulative_pseudo_rewards`
- [x] `OaKAgent`: `init()`, `start()`, `update()`, `scan()`, `curate()`, `to_config()`
- [x] Scan-compatible utility EMA via `jnp.where` (no Python branching on JAX values)
- [x] `keyboard_q_values` / `keyboard_action`: L1-normalised blended Q-values
- [x] `Step11OaKConfig` / `Step11SmokeResult` production facade
- [x] `core/oak.py` building on `core/options.py`
- [x] 41 tests: config roundtrip, factory, init, utility EMA, scan shapes, curation, keyboard, learned feature construction, keyboard chord learning, smoke, 200-step fineness
- [x] Learned feature construction (auto-generated subtask features)
- [x] Keyboard chord vector learning (bandit-style)
- [x] Seeded benchmark evidence that curation maintains option quality over long horizons — `benchmarks/step11_oak_curation.py`; post-curation recovery 0.935 on 8/10 seeds
- [x] Step 11 solution gate — `benchmarks/step11_solution_gate.py`

## Step 12 — Prototype-IA (Primitive Implemented)

- [x] `ExoCerebellumConfig` / `ExoCerebellumState` / `ExoCerebellumAgent`: vectorised multi-output predictor
- [x] `ExoCortexAgent`: OaK wrapper adding `recommend(state, obs)` (argmax base Q)
- [x] `IAConfig` / `IAState` / `IAAgent`: paired cerebellum + cortex with `update()` and `scan()`
- [x] `IAUpdateResult` / `IAArrayResult`: per-step predictions, cerebellum_errors, recommendation, augmented_obs, cortex_td_error
- [x] `Step12IAConfig` / `Step12SmokeResult` production facade
- [x] `core/intelligence_amplification.py` building on `core/oak.py`
- [x] 30+ tests: config validation, obs-dim mismatch guard, config roundtrip, factory, init shapes, update shapes/dtypes, augmented-obs concat, scan shapes, weight update, smoke, 200-step fineness
- [ ] Communication protocol for recommendation acceptance / rejection
- [x] Exo-cortex with nonlinear function approximation — via `OaKConfig.stomp.base_hidden_sizes`
- [x] Seeded benchmark evidence that IA augmentation improves partner decision-making — `benchmarks/step12_ia_augmentation.py`; cerebellum MSE ≈ 0 vs 0.167 zero-baseline; cortex 60% accuracy

## PrototypeAgent — All 12 Steps Integrated (v0.21.0)

- [x] `PrototypeAgentConfig`: unified config for OaK + world model + dreaming + Horde + IA
- [x] `PrototypeAgentState`: combined state (oak, world_model, buffer, horde, ia, step_count)
- [x] `PrototypeAgent`: single agent integrating all 12 Alberta Plan steps
  - Step 1/2: IDBD / nonlinear FA (internal to OaK's differential Q-learner)
  - Step 3: Parallel GVF prediction demons via HordeLearner (optional)
  - Step 4: SARSA on-policy control (internal to OaK/STOMP)
  - Steps 5/6: Average-reward / differential SARSA (internal to OaK/STOMP)
  - Step 7: Dyna planning budget (n_dreams_per_step)
  - Step 8: ActionConditionedWorldModel dynamics (optional)
  - Step 9: GuardedDreamer + RecentObservationBuffer anchored dreaming (optional)
  - Steps 10/11: STOMPAgent wrapped by OaKAgent — temporal abstraction + curation
  - Step 12: IAAgent companion cerebellum + cortex (optional)
- [x] `feature_to_subtask_specs`: extract SubtaskSpecs from OaK Q-weight feature importance
- [x] `PrototypeUpdateResult` / `PrototypeArrayResult`: per-step diagnostics
- [x] `run_prototype_scan` / `run_prototype_smoke`: JIT-compiled loop and quick validity check
- [x] 50 tests: config validation, roundtrip, init, act, update (minimal+full), scan, curation,
      auto-subtask specs, feature_to_subtask_specs, serialization, dreaming mechanics, 200-step
      fineness
- [x] Exported from `alberta_framework.core` public API
- [x] End-to-end benchmark on continuous gym task showing value from each added component
- [ ] Real robot / sim-to-real demonstration

## Infrastructure

- [ ] Update CHANGELOG.md with each release (moved from CLAUDE.md)
- [ ] Keep bsuite running on Python 3.13 via PYTHONPATH workaround
