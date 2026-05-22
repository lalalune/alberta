# Alberta Plan Paper Audit Index

This directory contains a complete structured extraction of the Alberta Plan paper (Sutton, Bowling, Pilarski 2023, arXiv:2208.11173).

## Documents Generated

### 1. ALBERTA_PLAN_12STEPS_AUDIT.md
**Full, detailed analysis** (45 KB)
- Complete 12-step descriptions with exact quotes from paper
- Key algorithmic requirements for each step
- Success criteria ("What done looks like")
- Robot/real-world deployment requirements
- Referenced papers for each step
- Architectural vision and system design
- Gaps and provisional notes

**Use when:** You need comprehensive understanding of a specific step, exact algorithm requirements, or comparison with another method.

### 2. ALBERTA_PLAN_12STEPS_QUICK_REF.txt
**Concise reference table** (5 KB)
- One-paragraph summary per step
- Problem statement, key innovation, success criteria
- Implementation status in current framework
- Temporal uniformity & foreground/background separation
- Architectural differences from standard RL
- Robot/real-world deployment time budget

**Use when:** You need quick lookup, status overview, or to brief team members.

## Summary: Current Implementation Status

| Phase | Steps | Status | Notes |
|-------|-------|--------|-------|
| **Foundations** | 1-2 | COMPLETE / PARTIAL | Step 1 done; Step 2 beats synthetic but struggles external |
| **Core Prediction/Control** | 3-6 | PARTIAL / MINIMAL | GVF works; off-policy needs; average-reward scaffolded |
| **Planning & Integration** | 7-9 | PRIMITIVE | Planning interface exists; search control minimal |
| **Hierarchical Abstraction** | 10-11 | PRIMITIVE | STOMP/Oak frameworks sketched; not continuous |
| **Intelligence Amplification** | 12 | PRIMITIVE | GVF prediction base; no dual-agent coordination |

**Key Gap:** The framework is a **strong research platform** for Steps 1-4, but Steps 5-12 require significant integration work to realize the paper's vision of a unified, continually learning hierarchical agent.

## How to Use This Audit

1. **For code review:** Read the Quick Ref first, then deep-dive into the audit for specific steps.
2. **For gap analysis:** Compare current implementation against "Success Criteria" and "Key Algorithmic Requirements" sections.
3. **For robot deployment:** See "Robot/Real-World Requirements" in each step + Quick Ref time budget section.
4. **For researcher onboarding:** Start with Quick Ref, then read Executive Summary in full audit.
5. **For roadmap planning:** Use the Framework Status sections to identify which steps need work.

## Key Findings

### What the Paper Demands
- **Temporal uniformity**: Every algorithm runs on every time step (no offline phases)
- **Continuing operation**: No episodic resets; systems must handle indefinite, non-stationary environments
- **Integrated hierarchy**: Automatic discovery of features → subtasks → options → models, all from experience
- **Real-time**: Daemon-compatible single-step API with <100 ms per loop budget

### What's Implemented
- Linear Step 1 (IDBD, Autostep) with excellent foundational work
- Nonlinear Step 2 (MLP, feature discovery) with strong synthetic results
- GVF Step 3 prediction layer with Horde
- SARSA Step 4 control on-policy

### What's Needed for Full Vision
- Off-policy learning for Steps 3-4 (critical for Step 10 options)
- Average-reward integration (Steps 5-7) across learning, planning, and control
- Full asynchronous DP with search control (Steps 7-9)
- STOMP end-to-end (Step 10): features → subtasks → options → models → planning cycle
- Oak lifecycle management (Step 11): continuous utility assessment and element replacement
- Dual-agent IA coordination (Step 12): partner adaptation and bandwidth management

## Cross-References

- **CLAUDE.md**: Project overview and local conventions
- **ROADMAP.md**: High-level project status
- **TODO.md**: Near-term work items
- **step1_step2_alberta_plan_assessment.md**: Deep analysis of Steps 1-2 implementation
- **Alberta-Plan paper**: https://arxiv.org/abs/2208.11173
