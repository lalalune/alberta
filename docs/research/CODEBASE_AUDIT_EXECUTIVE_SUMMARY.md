# Alberta Framework Codebase Audit: Executive Summary

**Date:** 2026-05-21  
**Auditor:** Claude Code  
**Subject:** Alignment between current implementation and Alberta Plan paper requirements (arXiv:2208.11173)

---

## Quick Assessment

The alberta-framework is a **world-class research platform for Steps 1-4** of the Alberta Plan but a **scaffolded prototype for Steps 5-12**. Implementation quality is high where it exists; gaps are in integration scope, not algorithmic sophistication.

| Aspect | Verdict |
|--------|---------|
| **Step 1 (Linear continual learning)** | ✓ Complete; audited against paper |
| **Step 2 (Supervised feature discovery)** | ◐ Partial; strong synthetic, external benchmarks harder |
| **Step 3 (GVF prediction)** | ◐ Partial; prediction works, off-policy learning incomplete |
| **Step 4 (Actor-critic control)** | ◐ Partial; SARSA on-policy only, off-policy not implemented |
| **Steps 5-9 (Average-reward, planning)** | ✗ Minimal/Primitive; planning interface exists, not integrated |
| **Steps 10-12 (STOMP/Oak/IA)** | ✗ Primitive; frameworks sketched, lifecycle management not continuous |

**Interpretation:** The framework is **sufficient for Steps 1-4 research**. Moving to Steps 5-12 requires significant integration work, not just adding new components.

---

## Key Findings: What Paper Demands vs. What's Built

### Temporal Uniformity
**Paper Demands:**
> "All learning, planning, and meta-learning happen on every time step in a continuing, non-stationary setting. No special training phases, offline batches, or episodic boundaries."

**Implementation Status:** ✓ Achieved for Steps 1-4
- Single-step API enforced (`predict()`, `update()`)
- No episodic loops in core learners
- Foreground operations lock to real-time cadence

**Implementation Status:** ✗ Partial for Steps 5-12
- Planning interface exists but not fully async DP
- Feature/option lifecycle not continuously updated
- Steps 10-12 have skeleton; continuous operation not proven

---

### Representation Learning (Steps 1-2)

**Paper Demands:**

Step 1:
- Per-feature adaptive step-sizes (meta-learned, not hand-tuned)
- Online normalization (input scaling/offsetting)
- Handle non-stationarity: drifting weights, drifting bias, input-shift
- Tuning-free algorithms outperform LMS on sparse-relevance tasks

Step 2:
- Feature generation (products, compositional)
- Candidate testing and ranking
- Resource budgeting (bounded feature bank)
- Utility tracking across multi-task outputs
- Feature deletion/replacement without catastrophic forgetting
- Beat fair MLPs on out-of-hypothesis-class tasks (polynomial, frequency, compositional)

**Current Status:**

Step 1: ✓ **COMPLETE**
- IDBD, Autostep, Autostep-for-GTD(λ), Adam, RMSprop all implemented
- Audited against Mahmood et al. 2012 Table 1 (bug fix applied)
- Online normalization: EMANormalizer, WelfordNormalizer, StreamingBatchNormalizer
- Canonical results: 30-seed evidence on Sutton 1992 sparse-relevance + extensions
- CPU throughput: >8,500 updates/sec (daemon-ready)

Step 2: ◐ **PARTIAL**
- ✓ Product features (FixedBudgetInteractionLearner)
- ✓ Compositional features (CompositionalFeatureLearner with recursive DAG)
- ✓ Candidate testing and utility tracking
- ✓ Feature deletion/replacement mechanism
- ◐ Synthetic results strong: UPGD beats MLP on polynomial, frequency, compositional
- ✗ External benchmark (sklearn digits) shows MLP beats UPGD (0.0204 vs 0.0237 MSE)
- ◐ Workaround: Universal portfolio (strict expert mixture) closes current benchmark but doesn't prove arbitrary feature discovery

**Gap:** Step 2 is a strong research platform but not yet a complete representation-learning solution. UPGD works on synthetic out-of-class tasks; external validation still open.

---

### Prediction & Control (Steps 3-4)

**Paper Demands:**

Step 3:
- GVF prediction on sequential data (not i.i.d.)
- Per-demon γ/λ (independent trace decay per prediction target)
- TD learning with eligibility traces
- Feature discovery under TD targets (harder than supervised)
- Off-policy learning (ideally)

Step 4:
- Actor-critic: critic from Steps 1-3, actor learned in parallel
- Progression: k-arm bandit → contextual bandit → sequential with features
- Feature sharing between actor and critic without divergence
- Continual learning under policy changes

**Current Status:**

Step 3: ◐ **PARTIAL**
- ✓ HordeLearner wraps MultiHeadMLPLearner for GVF demons
- ✓ Per-demon γ/λ traces implemented
- ✓ TD learning with SAR(S)A targets
- ✓ Multi-head feature bank (inherits from Step 2)
- ◐ Off-policy: infrastructure exists (eligibility traces, Q-learning), but convergence guarantees not proven on suite
- ✓ Example: TD-surprise interaction features beat raw MLP on observable AR(1) control task

Step 4: ◐ **PARTIAL**
- ✓ SARSAAgent wraps HordeLearner for on-policy control
- ✓ Bandit → contextual → sequential progression exists
- ✓ Episodic/continuing/scan loops
- ✗ Off-policy actor-critic not implemented (SARSA is on-policy; GTD/TDC for critic not integrated with actor)
- ✓ Feature ranking for control (inherited from Step 2/3)

**Gap:** Steps 3-4 are functionally complete for on-policy learning. Off-policy (critical for Step 10 options) needs GTD/TDC integration + convergence proofs.

---

### Average-Reward & Planning (Steps 5-9)

**Paper Demands:**

Step 5: Extend GVF to average-reward criterion (differential value V^ρ)
Step 6: Continuing benchmarks (River Swim, Access-Control, Jellybean World, GARNET)
Step 7: Asynchronous DP with function approximation for average-reward
Step 8: Integrated agent (Prototype-AI I): perception + one-step model + feature-guided planning
Step 9: Search control (state prioritization in async DP)

**Current Status:**

Step 5: ✗ **MINIMAL**
- Average-reward DP exists in tabular form
- Not integrated with Steps 1-4 feature learning or GVF infrastructure
- Differential value learning not connected to step-size meta-learning

Step 6: ✗ **MINIMAL**
- Continuing environments mentioned but not a primary focus
- Step 1-4 work uses synthetic episodic streams
- River Swim, GARNET benchmarks not in examples

Step 7: ✗ **PRIMITIVE**
- Planning interface exists (`Plan` ABC)
- Asynchronous DP skeleton; not full Bertsekas algorithm
- Search control not implemented (random/breadth-first only)

Step 8: ✗ **PRIMITIVE**
- One-step Dyna-style planning exists
- Feature-guided planning not integrated (planning doesn't feed utility back to features)
- Model learning + feature finding + planning cycle not closed

Step 9: ✗ **PRIMITIVE**
- Planning interface allows custom update orders
- Prioritized sweeping not implemented for function approximation
- Uncertainty-aware prioritization absent

**Gap:** Steps 5-9 are the **critical integration bottleneck**. The pieces exist but not wired together. A team needs to:
1. Connect average-reward DP to GVF prediction learning
2. Implement asynchronous DP + small backups
3. Wire model feedback into feature ranking
4. Build prioritization based on TD error / value uncertainty

---

### Temporal Abstraction & Lifelong Learning (Steps 10-12)

**Paper Demands:**

Step 10: STOMP progression
- Features → SubTasks (highest-ranked become goals) → Options (policies for subtasks) → Models (option transition models) → Planning
- Off-policy option learning
- Long-horizon planning via option models

Step 11: Oak architecture
- Utility assessment for all elements (features, subtasks, options, models)
- Continuous deletion/replacement cycle
- Option Keyboard: index options by vectors; combine options via chords

Step 12: Intelligence Amplification
- Dual-agent system: augment partner (human or AI) via GVF predictions (exo-cerebellum) and action recommendations (exo-cortex)
- Learn partner's preferences online
- Both human-agent and agent-agent settings

**Current Status:**

Step 10: ✗ **PRIMITIVE**
- ✓ Options framework exists (option definition, termination)
- ✗ Feature-driven subtask discovery not implemented (subtasks are manual)
- ✗ Off-policy option learning not tested at scale
- ✗ Option models not integrated with planning
- ✗ STOMP cycle (features → subtasks → options → models → planning) not closed

Step 11: ✗ **PRIMITIVE**
- ✓ Feature/option deletion logic exists
- ✗ Utility assessment loop not continuous (manual intervention required)
- ✗ Reranking doesn't drive subtask changes
- ✗ Option Keyboard mentioned but not integrated
- ✗ Lifecycle management not proven on continuing tasks

Step 12: ✗ **PRIMITIVE**
- ✓ GVF prediction layer (foundation for exo-cerebellum)
- ✗ Dual-agent coordination not implemented
- ✗ Partner preference learning not integrated
- ✗ Bandwidth management for signals/recommendations absent

**Gap:** Steps 10-12 are **research questions, not engineering tasks**. The framework provides GVF and option infrastructure, but:
- Feature-driven abstraction discovery is unproven
- Option lifecycle management needs theoretical grounding
- Dual-agent IA requires novel learning algorithms (not just synthesis of existing methods)

---

## Implementation Quality Where Present

### Excellent
- **Step 1 algorithms**: IDBD, Autostep — thorough, well-tested, audited against papers
- **Step 2 synthetic results**: UPGD competitive on polynomial/compositional benchmarks
- **GVF infrastructure**: Horde/MultiHeadMLPLearner solid
- **Continuous testing**: Regression tests for major algorithms
- **Code organization**: Clear separation of concerns (learners, optimizers, normalizers, streams)

### Good
- **Documentation**: CLAUDE.md, detailed research audits, runnable examples
- **JAX integration**: Immutable state, JIT compilation, vmap for batch runs
- **Config serialization**: Most components have to_config/from_config
- **Benchmarking**: Throughput measured, canonical results stored

### Needs Work
- **Off-policy convergence proofs**: GTD/TDC integration incomplete
- **Average-reward integration**: Separate from Steps 1-4 pipeline
- **End-to-end Prototype-AI**: No single agent class that demonstrates Steps 1-8 together
- **Lifelong learning proofs**: Step 11 Oak lifecycle not validated on long-run tasks

---

## Codebase Audit Documents

This audit has generated three reference documents (in `/docs/research/`):

1. **ALBERTA_PLAN_12STEPS_AUDIT.md** (46 KB)
   - Complete step-by-step breakdown with exact paper quotes
   - Algorithmic requirements, success criteria, deployment specs
   - Use for: Deep understanding, gap analysis, implementation planning

2. **ALBERTA_PLAN_12STEPS_QUICK_REF.txt** (13 KB)
   - Concise one-page-per-step summary
   - Current framework status for each step
   - Time budgets, architectural differences
   - Use for: Quick lookup, team briefing, roadmap planning

3. **PAPER_AUDIT_INDEX.md** (4 KB)
   - Navigation guide to audit documents
   - Summary table of implementation status
   - Cross-references to related docs

---

## Recommendations

### For Research Continuation (Next 1-2 Years)

**Short Term (1-3 months):**
1. **Off-policy GTD/TDC integration** (Step 3-4 blocker)
   - Add GTD/TDC critic learning
   - Prove convergence with function approximation
   - Integrate with SARSA actor

2. **Average-reward DP bridge** (Step 5-7 foundation)
   - Connect Sutton et al. 2008 async DP to Steps 1-4 GVF infrastructure
   - Test on continuing benchmarks (River Swim, GARNET)
   - Implement small backups, prioritized sweeping with FA

**Medium Term (3-6 months):**
3. **Prototype-AI I integration** (Step 8)
   - Build single agent class combining foreground (perception/policy/learning) + background (model/planning)
   - Wire feature ranking feedback from planning
   - Demonstrate planning improves over model-free on controlled domain

4. **Search control** (Step 9)
   - Implement uncertainty-aware prioritization
   - Test state-ordering heuristics against random
   - Integrate with async DP

**Long Term (6-12 months):**
5. **Feature-driven STOMP** (Step 10)
   - Rank features by utility; top-k become subtasks
   - Learn off-policy option policies for subtasks
   - Learn option models; validate against environment

6. **Oak lifecycle validation** (Step 11)
   - Continuous utility assessment loop (every N steps)
   - Automated deletion/promotion on long-run tasks
   - Measure hierarchy improvement over time

### For Deployment Readiness

- **Real-time API**: Steps 1-4 are daemon-ready (<100 ms per update loop)
- **Steps 5-9**: Achievable with async planning background; prototype recommended
- **Steps 10-12**: Requires research; not deployment-ready until proven on multi-month runs

### For Codebase Documentation

Update ROADMAP.md to reflect:
- Step 1: ✓ Complete
- Step 2: ◐ Partial (synthetic ok; external harder)
- Steps 3-4: ◐ Partial (on-policy ok; off-policy needed)
- Steps 5-9: Plan 6-month integration sprint
- Steps 10-12: Mark as "Research Direction" (not yet designed)

---

## Conclusion

The Alberta framework is a **high-fidelity implementation of Alberta Plan Steps 1-4**, with strong empirical results on synthetic continual learning tasks. The code quality, testing, and documentation are excellent.

Advancing to Steps 5-12 requires:
1. Closing the off-policy learning gap (GTD/TDC)
2. Integrating average-reward DP across learning and planning
3. Building Prototype-AI as a single integrated agent
4. Researching feature-driven abstraction discovery (STOMP)
5. Validating lifelong learning at scale (Oak)

This is **challenging but achievable work** that aligns with the paper's vision. The framework provides a solid foundation; the next phase is integration and validation at increasing scale and complexity.

---

**Generated:** 2026-05-21 by Alberta Plan Audit  
**Source:** Sutton, Bowling, Pilarski (2023) arXiv:2208.11173  
**Reference:** /docs/research/PAPER_AUDIT_INDEX.md for full documents
