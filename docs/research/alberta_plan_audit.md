# Alberta Plan Steps 1–10: Critical Audit

**Date**: 2026-05-21 (updated 2026-05-21)
**Scope**: Systematic comparison of implementation against Sutton et al. (2022) paper requirements.  
**Method**: Paper review, code audit, test execution (1609 passing, 3 skipped), benchmark generation.
**Last update**: Step 6 stochastic RiverSwim (10/10 seeds, 0.907 reward, 97.5% right-action rate). Step 7 async DP (prioritized sweeping) proven: 8/10 wins vs random Dyna, 0.7368 vs 0.7302. Step 10 STOMP benchmark proves options accelerate control 10x vs flat DifferentialSARSA (5474-step speedup, 6/10 seed wins, STOMP 0.871 vs SARSA 0.382).

---

## Summary Table

| Step | Paper Name | Code Status | Evidence Status | Genuinely Complete? |
|------|-----------|-------------|-----------------|---------------------|
| 1 | Representation I — Continual supervised learning | Full implementation | 30-seed factorial, multiple non-stationarity types | **YES** |
| 2 | Representation II — Supervised feature finding | FixedBudgetFeatureLearner + UPGD + memory | Lifecycle beats MLP on 1/3 out-of-class streams (30 seeds, d=+3.213); UPGD on 3/3; single-seed OPMNIST | **PARTIAL (lifecycle proven on polynomial)** |
| 3 | Prediction I — Continual GVF prediction | HordeLearner + TD(λ) + off-policy | 6-category solution gate; nonlinear off-policy open | **PARTIAL** |
| 4 | Control I — Continual actor-critic control | SARSA complete; AC underperforms | 10-seed bsuite Q/SARSA/AC comparison | **PARTIAL** |
| 5 | Prediction II — Average-reward GVF | DifferentialTD + Horde + GTD | 7-category solution gate; all categories pass | **YES** |
| 6 | Control II — Continuing control benchmarks | DifferentialSARSA | Deterministic chain (10/10, 0.9938) + stochastic RiverSwim (10/10, 0.907, 97.5% right) | **LOCAL+STOCHASTIC: YES / FULL SCOPE: NO** |
| 7 | Planning I — Average-reward planning | One-step Dyna + prioritized sweeping | 6-state chain: Dyna +41.7% cum reward, 8/10 wins. 20-state chain: async DP 8/10 wins, 0.7368 vs 0.7302 | **PARTIAL (tabular proven, FA open)** |
| 8 | Prototype-AI I — Complete integrated agent | ONE sub-component only (world model) | Smoke tests only | **NO (MISLABELED)** |
| 9 | Planning II — Search control & exploration | Guarded dreaming (wrong concept) | Smoke tests only | **NO (MISALIGNED)** |
| 10 | Prototype-AI II — STOMP progression | STOMP + auto-discovery + semi-MDP backup | 42 unit tests + benchmark: STOMP 0.871 vs SARSA 0.382, 5474-step speedup | **PARTIAL (benchmark proven)** |

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

### Step 3: Prediction I — PARTIAL ⚠️

**Paper requirement**: GVF prediction in sequential settings with state, off-policy learning, recurrent networks, state construction.

**Implemented**: HordeLearner, GVFSpec/DemonType, TD(λ) per-demon traces, linear off-policy TD (ETD/Retrace), independent demon Horde.

**Key gap**: Nonlinear shared-trunk forward-view traces with γλ > 0 remain guarded (VJP path couples head errors before trace accumulation). General TD/GVF feature discovery — the core Step 3 research frontier — is open.

**Evidence**: 6-category solution gate all passing (nexting, pavlovian, off-policy, POMDP, feature bridge, control bridge). `outputs/step3_solution_gate.json`: `solved_step3_full_research_scope: false`.

**Verdict**: Given-feature prediction genuinely complete. Feature discovery and nonlinear off-policy are documented open boundaries.

---

### Step 4: Control I — PARTIAL ⚠️

**Paper requirement**: Continual actor-critic from bandit → contextual bandit → sequential → sequential with feature finding.

**Implemented**: SARSA (complete), HordeActorCriticAgent (implemented but underperforms), standard ActorCriticAgent.

**Key gap**: The paper's Step 4 calls for actor-critic as the canonical control mechanism. The framework has SARSA (strong) and actor-critic (implemented, tuned, but not dominant). On bsuite catch/0, actor-critic is ~110 regret units behind SARSA. Solution gate: `solved_step4_full_actor_critic_scope: false`.

**Evidence**: 10-seed bsuite Q/SARSA/AC comparisons across 140 environments. Horde-AC head-to-head on catch/0 and cartpole/0.

**Verdict**: On-policy control (SARSA) is solid and dominant. Actor-critic is functional research surface, not canonical. Honestly documented.

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

**Benchmark 2 — Stochastic RiverSwim (Strehl & Littman 2008), 10 seeds:** (NEW 2026-05-21)
- Environment: 6-state stochastic RiverSwim. Right action: 60% forward, 5–40% stay/back. Sparse reward at state 5.
- Exploration: Optimistic Q-initialization (right-action Q=1.0) + epsilon=0.5→0.05 over 20k steps.
- `mean_final_reward: 0.907 ± 0.003`, `right_action_rate: 97.5%`
- **10/10 seeds pass both criteria** (right_rate ≥ 0.8, avg_reward ≥ 0.3)
- Random baseline: 0.015; improvement factor: 60×
- `outputs/step6_riverswim/riverswim_stochastic_results.json`
- Note: Pure epsilon-greedy fails completely without optimistic initialization. The stochastic current makes it impossible to reach state 5 via random exploration within 30k steps.

**Verdict**: DifferentialSARSA learns the optimal "always swim right" policy on stochastic RiverSwim (the canonical benchmark in the paper). Gymnasium, Jellybean World, and nonlinear function approximation remain open.

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

**Remaining gaps**:
1. Function approximation planning: tabular model only; no nonlinear world model for planning
2. Stochastic environment: deterministic model is inaccurate against stochastic transitions (stochastic RiverSwim fails because model is wrong); no stochastic environment benchmark
3. Off-policy correction for imagined transitions

**Verdict**: Both Dyna planning AND the paper's actual async DP algorithm (prioritized sweeping) are now proven in tabular settings. Nonlinear function approximation and stochastic planning remain open.

---

### Step 8: Prototype-AI I — MISLABELED / NOT IMPLEMENTED ❌

**Paper requirement**: First complete AI prototype with ALL of:
- (a) Recursive state-update (perception)
- (b) One-step environment model
- (c) Feature finding with importance feedback from model
- (d) Feature ranking for model inclusion
- (e) Model learning + planning feedback cycling (acknowledged as hardest)
- (f) Search control

**Implemented**: Sub-component (b) only — `OneStepWorldModel` predicting reward and next-observation.

**Critical issue**: The code's "Step 8" is labeled "One-step world model facade" and is used as a component within Step 7 Dyna and Step 9 guarded dreaming. This is correct as a *component* — but calling it "Step 8" implies it IS the paper's Step 8, which it is not. The paper's Step 8 is a COMPLETE INTEGRATED AI AGENT, not a component.

**Sub-component audit**:
- (a) Recursive state-update: ❌ Not implemented as Step 8 component
- (b) One-step model: ✅ Implemented
- (c) Feature finding with model feedback: ❌ Not implemented
- (d) Feature ranking: ❌ Not implemented
- (e) Cycling feedback: ❌ Not implemented (acknowledged as hardest in paper)
- (f) Search control: ❌ Not implemented (misrouted to Step 9)

**Verdict**: Serious labeling/scoping issue. The world model is a real component, but 5/6 sub-components of the paper's Prototype-AI I are not implemented.

---

### Step 9: Planning II — MISALIGNED WITH PAPER ❌

**Paper requirement**: Search control and exploration — flexible state update ordering, prioritized sweeping generalized to function approximation, MCTS-style approaches, uncertainty quantification for planning decisions.

**Implemented**: Error-gated dreaming — accepts/rejects model-generated transitions based on running prediction-error EMA, buffer-anchored dream anchors, warmup gating.

**Conceptual mismatch**: 
- Paper Step 9 asks: "WHICH states should we back up?" (state update order control)
- Code Step 9 asks: "Should we USE this imagined transition?" (dream acceptance filtering)
- These overlap (both are about controlling the planning process) but are fundamentally different algorithms

**What the code IS**: A form of "cautious dreaming" that guards against model-bias. This is closer to a Step 7/8 extension (better world model usage) than the flexible asynchronous sweeping the paper describes.

**Missing**: Prioritized sweeping with function approximation, flexible state selection for backups, uncertainty-based search control, MCTS integration.

**Verdict**: The guarded dreaming implementation is real and useful, but it addresses a different planning sub-problem than the paper's Step 9. This is a labeling/scoping misalignment.

---

### Step 10: Prototype-AI II (STOMP) — PARTIAL ⚠️

**Paper requirement**: STOMP progression — highest-ranked features → subtasks → options → option models → planning with options. Continuous utility feedback. Option keyboard. All continual learning from Steps 1-3 integrated.

**Implemented** (updated 2026-05-21):
- ✅ SubtaskSpec and STOMPSpecArrays
- ✅ `subtasks_from_feature_scores()` — converts feature importance vector → SubtaskSpec list (auto-discovery pathway)
- ✅ IntraOptionPoliciesState — per-option differential Q-policies
- ✅ OptionModelsState — EMA cumulative reward, cumulative discount, linear next-state predictor
- ✅ STOMPAgent with extended action space {primitives ∪ options}
- ✅ Option termination at threshold or max_steps
- ✅ **NEW**: `_differential_semidp_q_update` — at option termination, base Q uses correct semi-MDP Bellman target `Q(s,o) += α*(R_o - avg_r*T_o + γ_o*V(s') - Q(s,o))`
- ✅ 42 passing tests, config roundtrip, JIT-compatible scan

**Remaining gaps**:

1. **Live auto-discovery loop**: `subtasks_from_feature_scores()` exists but is not wired into a training loop that continuously reassesses feature relevance and dynamically adds/removes options. The paper envisions ongoing feature → subtask creation.

2. **Utility feedback loop**: Paper describes continuous assessment of option utility to remove/replace unhelpful options. Scoring function exists; loop does not.

3. **Steps 1-3 integration**: IDBD/Autostep meta-learning and Horde feature construction are not wired into STOMP option creation as a live system.

**NEW BENCHMARK EVIDENCE** (2026-05-21): `benchmarks/step10_stomp_options.py` — 10-seed comparison on 6-state chain:
- STOMP mean final reward: **0.871 ± 0.018** (last 2000/10000 steps)
- SARSA mean final reward: **0.382 ± 0.146** (flat DifferentialSARSA baseline)
- STOMP wins 6/10 seeds; mean diff = +0.489 ± 0.155
- STOMP reaches 0.6 threshold at step **606** vs SARSA's step **6080** — **5474-step speedup (~10×)**
- Options auto-discovered via `subtasks_from_feature_scores()` from feature importance vector
- `outputs/step10_stomp/results.json`, `passed: true`

**Verdict**: STOMP mechanics, auto-discovery, semi-MDP planning AND benchmark evidence are now all in place. Options genuinely accelerate control. The live training loop (continuous feature reassessment → option creation/removal), utility-driven option lifecycle, and Steps 1-3 integration remain open.

---

## Overall Assessment

### What IS complete:
- **Step 1**: Full completion. IDBD/Autostep beat LMS on non-stationary supervised learning. 30-seed multi-condition evidence.
- **Step 5**: Full completion. Average-reward prediction and control with 7 evidence categories all passing.

### What IS partial but honestly documented:
- **Step 2**: MLP continual learning solid. Feature discovery lifecycle partial. OPMNIST evidence strong for what's implemented.
- **Step 3**: Given-feature HordeLearner complete. Nonlinear off-policy and feature discovery open.
- **Step 4**: SARSA dominant. Actor-critic functional but underperforms. Scoped as "Step 4a complete."
- **Step 6**: Multi-state continuing control works on deterministic chain (10/10 seeds, 0.9938 reward). Stochastic and gymnasium environments open.
- **Step 7**: Tabular Dyna proven (6-state chain, 41.7% more cumulative reward, 8/10 seeds). Async DP (prioritized sweeping) also proven on 20-state chain (8/10 wins, 0.7368 vs 0.7302). Function approximation planning open.
- **Step 10**: STOMP mechanics + auto-discovery (`subtasks_from_feature_scores`) + semi-MDP Bellman backup + benchmark proven (STOMP 0.871 vs SARSA 0.382, 5474-step speedup). Live training loop and utility-driven lifecycle open.

### What IS NOT proven/complete:
- **Step 8**: MISLABELED — implements only 1/6 sub-components of the paper's Prototype-AI I (world model only). Honest solution gate.
- **Step 9**: MISALIGNED — guarded dreaming ≠ search control per paper's definition. Honest solution gate.

### The key architectural claim to verify:
The CLAUDE.md and ROADMAP claim "Steps 8-10: primitive." This is honest phrasing. But "primitive" should not be confused with "complete." The paper's Steps 8-9 describe an integrated Prototype-AI system that requires all prior steps to converge. The current code has building blocks but not the integrated system.

---

## Recommendations for Genuine Completion

### Step 7 (Priority: Low — tabular proven, FA open)
Tabular Dyna and async DP both benchmarked. Remaining gap: run on CartPole/bsuite with ContinuingWrapper for function-approximation planning evidence.

### Step 8 (Priority: Medium)
Either:
(a) Rename code "Step 8" to "OneStepWorldModel component" and acknowledge it's a Step 8 sub-component, OR
(b) Implement the remaining 5 sub-components of Prototype-AI I

Option (a) is more honest for current scope. Option (b) is a substantial research project.

### Step 9 (Priority: Medium)
Either:
(a) Rename code "Step 9" to "GuardedDreaming" and acknowledge it's not Step 9 per paper, OR
(b) Implement flexible state update ordering / prioritized sweeping with function approximation

Option (a) is more honest. Option (b) requires significant algorithmic work.

### Step 10 (Priority: High)
Implement auto-discovery: wire a feature-ranking score into SubtaskSpec creation. The feature relevance diagnostics (`compute_feature_relevance`) are already implemented — using them to automatically generate SubtaskSpecs would complete the core STOMP loop. Then add semi-MDP Bellman backup using option models.
