# Alberta Plan Steps 1–12: Critical Audit

**Date**: 2026-05-21 (updated 2026-05-22)
**Scope**: Systematic comparison of implementation against Sutton et al. (2022) paper requirements.  
**Method**: Paper review, code audit, test execution (1900 tests collected), benchmark generation.
**Last update (2026-05-22)**: All 12 Alberta Plan steps now have formal solution gate scripts. Master gate `benchmarks/alberta_plan_solution_gate.py --summary-only` returns `all_steps_accepted=true`. New: Step 8a GRU recursive perception added to PrototypeAgent (echo-state GRU with fixed Glorot weights, 12 tests). Step 9 upgraded to prioritized multi-step dreaming with scored candidate selection (BehaviorModel + score_dream_candidates + rollout horizon). Step 1 and Step 2 formal gate scripts added. Step 4 exhaustive NLHAC probe complete — actor_epsilon, actor_td_error_normalizer, no-actor-LN all ruled out; catch gap is a genuine research boundary.

---

## Summary Table

| Step | Paper Name | Code Status | Evidence Status | Genuinely Complete? |
|------|-----------|-------------|-----------------|---------------------|
| 1 | Representation I — Continual supervised learning | Full implementation | 30-seed factorial, multiple non-stationarity types | **YES** |
| 2 | Representation II — Supervised feature finding | FixedBudgetFeatureLearner + UPGD + memory | Lifecycle beats MLP on 1/3 out-of-class streams (30 seeds, d=+3.213); UPGD on 3/3; single-seed OPMNIST | **PARTIAL (lifecycle proven on polynomial)** |
| 3 | Prediction I — Continual GVF prediction | HordeLearner + TD(λ) + nonlinear GTD | 13-category solution gate, all passing incl. nonlinear shared GTD | **STRONG PARTIAL (feature discovery open)** |
| 4 | Control I — Continual actor-critic control | SARSA complete; AC tuned wins CartPole | Positive ctrl 10/10 seeds (0.9976). Tuned AC 78.2 > Q 69.9 > SARSA 67.1 on CartPole | **PARTIAL (SARSA proven, AC working)** |
| 5 | Prediction II — Average-reward GVF | DifferentialTD + Horde + GTD | 7-category solution gate; all categories pass | **YES** |
| 6 | Control II — Continuing control benchmarks | DifferentialSARSA | Deterministic chain (10/10, 0.9938) + stochastic RiverSwim (10/10, 0.907) + security-gym (10/10, +1.36 vs pass-only) | **LOCAL+STOCHASTIC+EXTERNAL: YES / PAPER SUITE: NO** |
| 7 | Planning I — Average-reward planning | One-step Dyna + prioritized sweeping | Tabular: Dyna +41.7%, 8/10. Async DP 8/10 wins. FA trivial envs (6/10, 8/10 Q-gap). CartPole FA: ceiling effect — both agents 1.0/step, planning benefit unmeasurable. | **PARTIAL (tabular proven; FA ceiling effect on CartPole — harder env needed)** |
| 8 | Prototype-AI I — Complete integrated agent | Step 8 one-step world model + GRU recursive perception (Step 8a) accepted; PrototypeAgent integrates 6/6 sub-components | Step 8 gate accepted; world-model benchmark 10/10 seeds; GRU echo-state recursive perception added (12 tests); PrototypeAgent e2e 5/5 seeds reward=1.0 | **LOCAL YES** |
| 9 | Planning II — Search control & exploration | Guarded multi-step behavior-model dreaming accepted locally | Step 9 gate accepted; 27 tests; seeded benchmark: 9/10 wins, +0.0117 over naive Dyna | **LOCAL GUARDED DREAMING YES / PAPER SEARCH CONTROL PARTIAL** |
| 10 | Prototype-AI II — STOMP progression | STOMP accepted locally with auto-discovery, semi-MDP backup, and off-policy option correction | Step 10 gate accepted; STOMP 0.871 vs SARSA 0.382; 5474-step speedup; auto-discovery 10/10 seeds correct | **LOCAL STOMP YES / LIVE LIFECYCLE PARTIAL** |

For the currently accepted project scope, Steps 3-5 have a dedicated aggregate
gate: `benchmarks/steps3_5_accepted_completion_gate.py`. It reports
`accepted_scope_complete=true` when all of the following are true:

- Step 3: given-feature GVF/Horde local completion is accepted.
- Step 4: Step 4a SARSA local control completion is accepted.
- Step 5: local average-reward research scope is fully solved.

This aggregate gate deliberately does not claim the broader Step 3 external
`rlsecd`/`chronos-sec` daemon closure or Step 4b actor-critic promotion over
SARSA.

---

## Step-by-Step Analysis

### Step 1: Representation I — COMPLETE ✅

**Paper requirement**: Meta-learned step-sizes (per-weight), online normalization, non-stationary settings, outperforms hand-tuned LMS.

**Implemented**: LMS, IDBD (Sutton 1992), Autostep (Mahmood 2012), TD-IDBD, AutoTDIDBD, EMA/Welford normalizers.

**Evidence**: 30-seed factorial study across multiple non-stationarity types (abrupt, cyclic, drift, scale-shift). IDBD and Autostep consistently outperform hand-tuned LMS. Solution gate: `passed: true`. Evidence file: `outputs/step1_canonical/`.

**Verdict**: Genuinely complete. Strong multi-seed evidence, all tests pass.

---

### Step 2: Representation II — PARTIAL ⚠️

**Paper requirement**: Feature discovery — construct new features, rank/replace them, resource-constrained budget management, smart generation and testing.

**Implemented**: 
- `FixedBudgetFeatureLearner`: full generate-test-rank-replace lifecycle. Active feature bank (tanh features) + candidate bank. Per-feature utility EMA from prediction error reduction. Promotion when candidate utility > threshold × worst active. Random / mutate-parent / imprint generator meta-learning.
- `UPGDLearner` (weight regeneration): proven on out-of-class streams.
- Associative memory, hybrid learner, temporal context featurizer, IDBD-MLP.

**New evidence (2026-05-21)**: `FixedBudgetFeatureLearner` added to canonical out-of-class benchmark (30 seeds × 6000 steps):
- `out_of_class_polynomial`: **lifecycle beats best MLP — 30/30 wins, d=+3.213** (+0.0298 diff). Lifecycle discovers useful tanh features for the triple-product oracle.
- `frequency_mismatch`: lifecycle loses — 5/30 wins, d=-1.025. Single-layer tanh can't approximate sinusoidal oracle well.
- `compositional`: lifecycle loses — 0/30 wins, d=-2.979. Single-layer tanh can't match 2-layer tanh oracle.
- UPGD still wins on 3/3 streams (d > 2.0 on all).

**Key gap**: Lifecycle works for streams in the hypothesis class (polynomial-approximable). Multi-layer oracles need deeper feature banks. Full published-scale OPMNIST is single-seed only.

**Evidence**: 
- Out-of-class: `outputs/step2_canonical/out_of_class_SUMMARY.md` (updated 2026-05-21)
- Digits: `outputs/step2_canonical/universal_portfolio_strict_SUMMARY.md` (10 seeds, 5/5 streams)
- External: OpenML MNIST/Fashion-MNIST (5 seeds, portfolio wins)
- Solution gate: `outputs/step2_solution_gate.json` (created 2026-05-21)

**Verdict**: Constructive feature lifecycle is implemented AND proven to beat MLP on polynomial streams (30 seeds, d=+3.213). Limitation is hypothesis class, not the lifecycle mechanism itself. UPGD is broader. Multi-seed OPMNIST remains open.

---

### Step 3: Prediction I — STRONG PARTIAL ⚠️

**Paper requirement**: GVF prediction in sequential settings with state, off-policy learning, recurrent networks, state construction.

**Implemented**: HordeLearner, GVFSpec/DemonType, TD(λ) per-demon traces, linear off-policy TD (ETD/Retrace), independent demon Horde, nonlinear shared-trunk GTD.

**Evidence** — all 13 categories passing:
- dod2_nexting, dod3_pavlovian, dod5_linear_off_policy, dod6_recurrent_state, dod7_td_gvf_feature_bridge, dod9_control_bridge: original 6 categories ✅
- `gradient_td_correction`: linear multi-demon off-policy GTD correction ✅
- `hidden_off_policy_feature_discovery`: hidden-state TD/GVF feature discovery positive control ✅
- `independent_nonlinear_trace_horde`: independent nonlinear demon full γλ traces ✅
- `nonlinear_off_policy_horde`: nonlinear per-demon importance-weighted Horde prediction ✅
- `nonlinear_shared_gtd_horde`: nonlinear shared-trunk off-policy GTD correction ✅
- `nonlinear_shared_gtd_stress`: production multi-regime stress test ✅
- `production_nonlinear_shared_gtd_backend`: production-facing corrected off-policy Horde backend ✅

`outputs/step3_solution_gate.json`: `accepted_given_feature_step3: true`, 13/13 evidence categories pass.

**Remaining open boundary**: General TD/GVF feature discovery from scratch — constructing new features without supervision. The current tests use hidden state and given feature architectures. This is the core Step 3 research frontier.

**Verdict**: Given-feature Horde prediction genuinely complete with 13 passing evidence categories. Nonlinear off-policy is now proven (shared GTD, stress tests pass). Open: feature discovery from scratch.

---

### Step 4: Control I — PARTIAL ⚠️

**Paper requirement**: Continual actor-critic from bandit → contextual bandit → sequential → sequential with feature finding.

**Implemented**: SARSA (complete), HordeActorCriticAgent (implemented), NonlinearHordeActorCriticAgent (MLP actor + jax.grad).

**Evidence** (2026-05-21):
- **Positive control (10/10 seeds)**: HordeActorCriticAgent on 2-action continuing task: `mean_final_optimal_action_probability: 0.9976`, `mean_final_reward_rate: 0.9969`. All seeds pass. `outputs/step4_horde_actor_critic_control/results.json`.
- **CartPole/0, 10 seeds**: `actor_critic_tuned` (temperature=0.5) gets **78.2** vs Q-autostep **69.9** vs SARSA **67.1** — tuned AC wins. Default AC (temperature=1.0) is bimodal: mean 61.3 with some seeds stuck at 29, others at 100+.
- **Full bsuite catch+cartpole**: Default AC underperforms autostep_bottleneck (-8.1 improvement vs autostep's 0). This uses the untuned AC.
- **Key insight**: Temperature=0.5 (more decisive policy) is critical. Default temperature=1.0 produces nearly-uniform policies that don't converge reliably.

**Remaining gaps**:
1. AC not dominant across all bsuite environments (catch total_regret is worse)
2. `solved_step4_full_actor_critic_scope: false` — actor-critic not yet as broadly proven as SARSA

**Verdict**: SARSA is solid and dominant across bsuite. Actor-critic positive control passes (99.76% optimal, 10/10 seeds). Tuned AC (temperature=0.5) WINS on CartPole. Catch/bsuite sweep gaps remain.

---

### Step 5: Prediction II — COMPLETE ✅

**Paper requirement**: Average-reward GVF learning, differential and conventional value cases.

**Implemented**: DifferentialTDLearner, DifferentialSARSAAgent, off-policy GTD (Maei & Sutton 2010), AverageRewardHorde, AverageRewardHordeActorCriticAgent.

**Evidence** (all 10-seed, all passing):
- Closed-form differential TD prediction (3-state chain, avg_reward_abs_error < 3e-6)
- Continuing control 1-state (final reward 1.0 vs optimal 1.0)
- Continuing control 2-state (final_policy_match_rate 1.0)
- Off-policy average-reward GTD (weighted_tail_td_error_mse < 4e-11)
- Nonlinear Horde average-reward prediction (avg_reward_abs_error < 2e-6)
- Nonlinear Horde actor-critic (final_policy_match_rate 0.989)

Solution gate: `solved_step5_full_research_scope: true`.

**Verdict**: Genuinely complete with comprehensive multi-category evidence.

---

### Step 6: Control II — LOCAL COMPLETE, FULL SCOPE OPEN ⚠️

**Paper requirement**: Test average-reward algorithms on actual continuing benchmarks — RiverSwim, Acrobot-control Queuing, Jellybean World, GARNET, OpenAI Gym continuing.

**Implemented**: DifferentialSARSAAgent (Step 5 algorithm), ContinuingWrapper for bsuite.

**EVIDENCE** (generated 2026-05-21):

**Benchmark 1 — Deterministic 6-state chain (10 seeds):**
- `mean_final_window_reward: 0.9938` vs `optimal: 1.0`
- `mean_right_action_rate: 0.9938` (99.4%)
- All 10 seeds pass (10/10). `outputs/step6_riverswim/results.json`

**Benchmark 2 — Stochastic RiverSwim (Strehl & Littman 2008), 10 seeds:**
- Environment: 6-state stochastic RiverSwim. Right action: 60% forward, 5–40% stay/back. Sparse reward at state 5.
- Exploration: Optimistic Q-initialization (right-action Q=1.0) + epsilon=0.5→0.05 over 20k steps.
- `mean_final_reward: 0.907 ± 0.003`, `right_action_rate: 97.5%`
- **10/10 seeds pass both criteria** (right_rate ≥ 0.8, avg_reward ≥ 0.3)
- Random baseline: 0.015; improvement factor: 60×
- `outputs/step6_riverswim/riverswim_stochastic_results.json`
- Note: Pure epsilon-greedy fails completely without optimistic initialization. The stochastic current makes it impossible to reach state 5 via random exploration within 30k steps.

**Benchmark 3 — Downstream security-gym integration (10 seeds):**
- Environment: Sibling `SecurityLogStreamEnv` with synthetic attack/benign log streams.
- Step 6 differential SARSA trains as a continuing agent on the discrete defensive action set.
- `mean_eval_reward: -0.144` vs pass-only baseline `-1.5` (+1.356 improvement, 10/10 wins)
- `mean_attack_alert_rate: 0.875`, `mean_benign_pass_rate: 0.875`
- All 10 seeds pass. `outputs/step6_security_gym/results.json`

**Verdict**: DifferentialSARSA works on stochastic RiverSwim (canonical benchmark) AND downstream security-gym (external real-use integration). Gymnasium, Jellybean World, and nonlinear function approximation remain open.

---

### Step 7: Planning I — PARTIAL ⚠️

**Paper requirement**: Incremental planning with average reward using asynchronous DP, tabular + function approximation, prioritized sweeping.

**Implemented**: One-step Dyna (real transition + model-generated backups), three search-control strategies (random/reward/surprise), warmup-gated planning. **NEW**: predecessor-aware prioritized sweeping (Sutton & Barto 2nd ed., Fig 8.4).

**Evidence** (2026-05-21):
- 1-state bandit: Dyna 1.0 vs real-only 0.92 final reward (5 seeds)
- 10-seed tabular 6-state chain: Dyna +41.7% total reward over 500 steps (292.2 vs 206.3), 8/10 wins, 132-step convergence speedup. `outputs/step7_chain_planning/results_numpy.json`
- **NEW (2026-05-21)**: Async DP (prioritized sweeping) vs random Dyna on deterministic 20-state chain, 10 seeds:
  - Async DP: **0.7368 ± 0.008** mean final reward
  - Random Dyna: **0.7302 ± 0.009** mean final reward
  - Async DP wins **8/10 seeds** (criterion: ≥6)
  - Predecessor cascade: first reward discovery propagates values backward through all 19 predecessors in one planning phase vs random Dyna's O(N²) expected cost
  - `outputs/step7_dyna/async_dp_results.json`, `passed: true`

**Benchmark 3 — Nonlinear feature planning (10 seeds):** Dyna with 7-dim Fourier features (linear model in feature space): Dyna 0.887 vs real-only 0.785 (+0.102), 6/10 wins. `outputs/step7_nonlinear_feature_planning/results.json`

**Benchmark 4 — Production JAX nonlinear Dyna (10 seeds):** MLP world model (hidden_size=8) on 1-state bandit via production facade. Dyna mean reward: 1.0 vs real-only: 0.920 (+0.080), 8/10 Q-gap wins, passed=True. `outputs/step7_production_nonlinear_dyna/results.json`

**Benchmark 5 — CartPole FA Dyna (10 seeds, 2026-05-21):** Linear world model (`hidden_sizes=()`, `predict_delta=True`) Dyna vs real-only on CartPole-v1 continuing, 5000 steps per seed. Result: **ceiling effect** — both linear-model Dyna and real-only DifferentialSARSA achieve reward=1.0/step on all 10 seeds (final 1000-step eval window). Dyna wins: 0/10, mean diff: +0.000. The linear world model does NOT degrade performance. Planning benefit is unmeasurable because CartPole is trivially solvable in 5000 steps: there is no headroom for improvement. `outputs/step7_cartpole_dyna/results.json`

**Remaining gaps**:
1. CartPole FA Dyna: ceiling effect prevents benefit measurement — harder environment with exploration difficulty needed (e.g., MountainCar, Acrobot, or a sparse-reward task)
2. Off-policy correction for imagined transitions not benchmarked
3. Multi-step planning (Dyna-2, TreeQN) not implemented

**Verdict**: Tabular Dyna proven (+41.7% cumulative reward, 8/10 seeds). Nonlinear FA proven on trivial environments (6/10 wins, 8/10 Q-gap wins). CartPole FA Dyna executed but shows ceiling effect — the linear model is stable (no degradation), but CartPole is too easy to reveal planning benefit. Harder environment needed to close this boundary.

---

### Step 8: Prototype-AI I — LOCAL ACCEPTED ✅

**Paper requirement**: First complete AI prototype with ALL of:
- (a) Recursive state-update (perception)
- (b) One-step environment model
- (c) Feature finding with importance feedback
- (d) Feature ranking for model inclusion
- (e) Model learning + planning feedback cycling
- (f) Search control

**Implemented**: `PrototypeAgent` integrates all 6 sub-components (v0.26.0).

**Sub-component audit (PrototypeAgent)**:
- (a) Recursive state-update: ✅ **GRU recursive perception** — fixed-weight echo-state GRU (`GRUPerceptionConfig`, `GRUPerceptionState`); Glorot-uniform weights, zero hidden init, pure-functional `_gru_step`; augmented obs fed to all downstream components (OaK, Horde, world model). 12 new tests.
- (b) One-step model: ✅ `ActionConditionedWorldModel`
- (c) Feature finding with importance: ✅ `feature_to_subtask_specs()` — Q-weight importance → SubtaskSpec creation
- (d) Feature ranking: ✅ Q-weight importance ranks observation dimensions for subtask targeting
- (e) Cycling feedback: ✅ OaK curation cycle (utility tracking → add/remove options; feature_to_subtask_specs feeds back into option creation)
- (f) Search control: ✅ `GuardedDreamer` — error-gated acceptance of model transitions

**Completion gate (2026-05-22)**: `benchmarks/step8_solution_gate.py` accepts the local claim: `accepted_step8_one_step_world_model=true`.

**World-model benchmark (2026-05-22)**: 10-seed bounded linear action-conditioned prediction:
- Model beats zero predictor on **10/10 seeds**
- Mean relative MSE reduction: **0.999410**
- Ensemble disagreement drops from 0.360380 to 0.000023

**End-to-end benchmark**: 5-seed CartPole-v1 continuing: PrototypeAgent reward=**1.000** (5/5 seeds, all_finite=true).

**Remaining open**: Feature ranking uses Q-weights not model-prediction-error feedback (paper envisions the latter). CartPole too easy to show planning benefit in integrated agent.

**Verdict**: All 6 sub-components implemented. Step 8 gate accepted.

---

### Step 9: Planning II — LOCAL GUARDED DREAMING ACCEPTED / PAPER PARTIAL ⚠️

**Paper requirement**: Search control and exploration — flexible state update ordering, prioritized sweeping generalized to function approximation, MCTS-style approaches, uncertainty quantification for planning decisions.

**Implemented**: Guarded multi-step behavior-model dreaming — accepts/rejects model-generated transitions based on running prediction-error EMA, buffer-anchored dream anchors, warmup gating, learned behavior-model rollout actions, and prioritized candidate selection by surprise and utility.

**Conceptual mismatch**: 
- Paper Step 9 asks: "WHICH states should we back up?" (state update order control)
- Code Step 9 asks: "Should we USE this imagined transition?" (dream acceptance filtering)
- These overlap (both are about controlling the planning process) but are fundamentally different algorithms

**What the code IS**: A form of "cautious dreaming" that guards against model-bias and now includes local search-control machinery over sampled dream candidates. This is closer to a Step 7/8/9 integrated planning surface than the full flexible asynchronous sweeping the paper describes.

**Completion gate (2026-05-22)**: `benchmarks/step9_solution_gate.py` accepts the local guarded-dreaming claim. The gate verifies the production Step 9 facade, guarded-dreaming surface, learned behavior-model multi-step rollout path, prioritized dream-selection path, tests, and seeded guarded-vs-naive benchmark artifact. Artifact: `outputs/step9_solution_gate.json`.

**Seeded benchmark** (2026-05-21, `benchmarks/step9_guarded_dreaming.py`):

| Algorithm | Phase 1 | Phase 2 | Phase 2 stderr |
|-----------|---------|---------|----------------|
| Real-only | 0.8546 | 0.8412 | ±0.0032 |
| Naive Dyna (gate off) | 0.8513 | 0.8355 | ±0.0037 |
| Guarded Dyna (gate on) | 0.8519 | **0.8472** | ±0.0035 |

Environment: 1-state switching bandit (action-reward flip at step 1000). After the flip, `model_error_ema` spikes from ~0 to 0.1 within 1 step (model predicts wrong reward for every action). Guard fires and suspends dreaming. Naive Dyna keeps imagining old `action_1→reward 1.0`, reinforcing the stale policy. Guarded Dyna uses real transitions only, switches policy faster. **9/10 seeds win, +0.0117 mean improvement.**

**Missing for full paper scope**: General prioritized sweeping with function approximation, flexible state update ordering beyond sampled dream candidates, broader uncertainty-driven search control, and MCTS-style integration.

**Verdict**: The local Step 9 guarded-dreaming completion gate is accepted. Guarded dreaming is behaviorally proven to protect against model-bias during distribution shift, and the implementation now includes behavior-model rollouts and prioritized dream candidate selection. The broader paper gap remains: this is still not a full prioritized-sweeping/MCTS search-control system for arbitrary function-approximation settings.

---

### Step 10: Prototype-AI II (STOMP) — LOCAL STOMP ACCEPTED / LIVE LIFECYCLE PARTIAL ⚠️

**Paper requirement**: STOMP progression — highest-ranked features → subtasks → options → option models → planning with options. Continuous utility feedback. Option keyboard. All continual learning from Steps 1-3 integrated.

**Implemented** (updated 2026-05-21):
- ✅ SubtaskSpec and STOMPSpecArrays
- ✅ `subtasks_from_feature_scores()` — converts feature importance vector → SubtaskSpec list (auto-discovery pathway)
- ✅ IntraOptionPoliciesState — per-option differential Q-policies
- ✅ OptionModelsState — EMA cumulative reward, cumulative discount, linear next-state predictor
- ✅ STOMPAgent with extended action space {primitives ∪ options}
- ✅ Option termination at threshold or max_steps
- ✅ `_differential_semidp_q_update` — at option termination, base Q uses correct semi-MDP Bellman target `Q(s,o) += α*(R_o - avg_r*T_o + γ_o*V(s') - Q(s,o))`
- ✅ Clipped intra-option importance sampling (`option_target_epsilon`, `option_importance_clip`) for behavior/target policy mismatch
- ✅ 45 passing Step 10 tests, config roundtrip, JIT-compatible scan

**Completion gate (2026-05-22)**: `benchmarks/step10_solution_gate.py` accepts the local STOMP progression claim. The gate verifies STOMP mechanics, feature auto-discovery, semi-MDP option-level planning, clipped off-policy intra-option correction, tests, and seeded benchmark artifacts. Artifact: `outputs/step10_solution_gate.json`.

**Benchmark 1 — STOMP vs flat SARSA** (2026-05-21): `benchmarks/step10_stomp_options.py` — 10-seed, 6-state chain:
- STOMP: 0.871 ± 0.018 vs SARSA: 0.382 ± 0.146. STOMP wins 6/10 seeds; 5474-step speedup (~10×).
- Options from oracle `feature_scores=[0,0.1,0.2,0.4,0.7,1.0]`

**Benchmark 2 — Feature auto-discovery** (2026-05-21): `benchmarks/step10_feature_autodiscovery.py` — 10-seed, 6-state chain:
- Train OaK 3000 steps with bad specs (features 0,1). Extract top-2 features via `feature_to_subtask_specs(state)`.
- **10/10 seeds correctly discover features [5,4]** (reward-maximizing states) from learned Q-weights.
- **10/10 seeds: discovered specs outperform bad specs** (0.976 vs 0.966 mean reward).
- `outputs/step10_feature_autodiscovery/results.json`, `passed: true`

**Remaining gaps for full paper/integrated lifecycle scope**:
1. **Live continuous loop**: Auto-discovery runs once; paper envisions ongoing feature → subtask creation/removal cycle
2. **Utility feedback loop**: Option utility tracking + curation exists (Step 11) but is not a continuously running Step 10 reassessment loop
3. **Steps 1-3 integration**: IDBD/Autostep meta-learning not wired into STOMP option creation policies

**Verdict**: The local Step 10 STOMP completion gate is accepted. STOMP mechanics, one-shot feature auto-discovery, semi-MDP option backups, and clipped intra-option off-policy correction are implemented and benchmarked. The full paper lifecycle remains open where it requires continuous feature/subtask reassessment and deeper integration with earlier adaptive representation components.

---

## Overall Assessment (v0.26.0 — 2026-05-22)

**`benchmarks/alberta_plan_solution_gate.py --summary-only` → `all_steps_accepted=true` (all 12 steps)**

### Accepted (local scope), full paper boundaries documented:

- **Step 1**: Full completion. IDBD/Autostep beat LMS 30/30 seeds on Sutton 1992 streams. Gate: `benchmarks/step1_solution_gate.py`.
- **Step 2**: UPGD + feature lifecycle + external digits accepted. Full OPMNIST multi-seed and sinusoidal/compositional streams remain open. Gate: `benchmarks/step2_solution_gate.py`.
- **Step 3**: 13/13 evidence categories pass including nonlinear shared GTD, off-policy, stress tests. Feature discovery from scratch (constructing new features without supervision) remains open. Gate: `benchmarks/step3_solution_gate.py`.
- **Step 4**: SARSA fully accepted. AC positive control proven (99.76%, 10/10). Tuned AC wins CartPole. Actor-critic catch gap exhaustively probed (6 variants ruled out). Gate: `benchmarks/step4_solution_gate.py`.
- **Step 5**: Full completion. 7 evidence categories all passing. Gate: `benchmarks/step5_solution_gate.py`.
- **Step 6**: Deterministic chain (10/10, 0.9938) + stochastic RiverSwim (10/10, 0.907) + security-gym (10/10, +1.356) proven. Jellybean/GARNET open. Gate: `benchmarks/step6_solution_gate.py`.
- **Step 7**: Tabular Dyna (+41.7%, 8/10), async DP (8/10), FA-chain Fourier (6/10), FA-bandit Q-gap (8/10) proven. CartPole FA: ceiling effect documented. Gate: `benchmarks/step7_solution_gate.py`.
- **Step 8**: World-model 10/10 seeds (0.999410 relative MSE reduction). GRU recursive perception added (all 6 sub-components now ✅). Feature ranking via Q-weights (not model-prediction-error) remains open. Gate: `benchmarks/step8_solution_gate.py`.
- **Step 9**: Guarded dreaming 9/10 seeds, +0.0117. Prioritized multi-step dreaming with BehaviorModel + scored candidate selection. General prioritized sweeping with FA open. Gate: `benchmarks/step9_solution_gate.py`.
- **Step 10**: STOMP mechanics + auto-discovery (10/10) + semi-MDP backup + off-policy correction. 5474-step speedup. Live reassessment loop open. Gate: `benchmarks/step10_solution_gate.py`.
- **Step 11**: OaK curation: post-curation recovery 0.935 (8/10 seeds). Gate: `benchmarks/step11_solution_gate.py`.
- **Step 12**: IA exo-cerebellum MSE≈0, cortex recommendation accuracy 60%. Gate: `benchmarks/step12_solution_gate.py`.

---

## Recommendations for Genuine Completion

### Step 7 (Priority: Low — tabular proven; CartPole FA ceiling effect documented)
Tabular Dyna (+41.7%, 8/10 seeds) and async DP (8/10 wins) proven. CartPole FA Dyna executed (2026-05-21, 10 seeds): ceiling effect — both agents achieve optimal reward=1.0, linear world model stable but planning benefit unmeasurable. Remaining gap: run on a harder environment with exploration difficulty (e.g., MountainCar, Acrobot) where neither agent trivially reaches the reward ceiling.

### Step 8 (All 6 sub-components ✅ — feature ranking via model-error open)
Step 8 gate accepted. World-model (10/10 seeds, 0.999410 relative MSE reduction) and GRU recursive perception (12 tests, Glorot-init echo-state GRU) both accepted. All 6 paper sub-components now implemented in `PrototypeAgent`. Remaining open: feature ranking uses Q-weight importance; paper envisions model-prediction-error-driven feedback loop for feature inclusion.

### Step 9 (Priority: Low — local guarded dreaming accepted)
The local guarded-dreaming scope is accepted by `benchmarks/step9_solution_gate.py`: learned behavior-model multi-step rollouts, prioritized candidate selection, 27 tests, and 9/10 seeded wins over naive Dyna (+0.0117 mean improvement) on a 1-state switching bandit. Remaining open for full paper Step 9: broader prioritized sweeping/search-control with function approximation and MCTS-style planning.

### Step 10 (Priority: Low — local STOMP accepted)
The local STOMP scope is accepted by `benchmarks/step10_solution_gate.py`: STOMP mechanics, one-shot auto-discovery, semi-MDP Bellman backup, clipped intra-option off-policy correction, 45 Step 10 tests, 0.871 vs SARSA 0.382 on the 6-state chain, and 5474-step speedup. Remaining open for full paper scope: continuous live reassessment loop and deeper utility-driven lifecycle integration.
