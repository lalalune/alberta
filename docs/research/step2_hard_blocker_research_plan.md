# Step 2 Hard-Blocker Research Plan

Date: 2026-05-05

## Goal

Close the remaining Step 2 blockers without weakening the acceptance bar:

- full published-scale OPMNIST, not a compact analogue;
- a self-contained feature-construction mechanism that does not need an MLP
  fallback to cover nonlinear streams;
- native deep feature lifecycle that improves deep MLPs rather than acting as a
  diagnostic wrapper;
- TD/GVF feature discovery that works beyond an observable interaction positive
  control.

The working standard is not "find one more positive probe." A candidate should
win against the best fair MLP comparator on the relevant stress suite, preserve
temporal uniformity, avoid task labels and offline stream-wide selection, and
produce reproducible outputs with tests.

## Current Gaps And Limitations

### OPMNIST 800-Task Scale

The implementation gap is mostly closed: the runner has true OpenML MNIST,
canonical 60k/10k split, 60,000-example task blocks, deterministic pixel
permutations, no task id, prediction-before-update, all experts updated every
step, and resumable checkpoints. The task-count evidence gap is now closed for
one seed: the latest-best UPGD-memory/MLP comparison completed all 800 task
blocks (`48,000,000/48,000,000` examples).

Post-completion update: the full run remains scientifically mixed. UPGD-memory
beats the best same-run fair MLP on online MSE, online accuracy, and
final-window MSE, but fair MLP baselines still win final-window accuracy and
all-permutation held-out test metrics. The remaining blocker is therefore not
compute-scale completion; it is retained-view generalization and multi-seed
confirmation.

The strongest immediate direction is to improve the retained-view objective
without damaging the online-tracking win.

### Single Recursive Mechanism

The current pure mechanism is strong when the right algebraic primitives enter
the candidate bank. It loses nonlinear online learning even though the signed
tanh scaffolds show representational capacity: heldout nonlinear loss improves,
but final-window online loss still trails fair MLP. That suggests the blocker is
credit allocation and replacement timing, not raw expressivity.

Plausible fixes should change how candidates are scored and retained, not just
add more basis functions. The sweep tested geometric dictionaries,
energy/novelty scoring, and redundancy-tolerant retention. The first two were
negative. Retention is the best pure mechanism so far: it wins 5/6 probes and
improves nonlinear to `0.1002`, but fair MLP is still `0.0597`.

### Native Deep Lifecycle

The native lifecycle wrapper trains shadow candidates, then swaps units into an
already-adapted network. The main failure mode is that promotion is disruptive
or too weak: candidates are not live in the downstream computation while being
tested, so their utility estimate is only an approximation to true downstream
usefulness. Preserve-outgoing promotion and active perturbation help diagnose
the problem but do not solve it.

The sweep tested soft-gated live candidates and a function-preserving Net2Net
pivot. Both were negative for promotion. Net2Net improved the native deep
ceiling to 3/6 by paired mean, but still lost nonlinear, compositional, and
digits. Native deep lifecycle should be deprioritized for Step 2 closure.

### TD/GVF Feature Discovery

TD-surprise interactions now close one narrow positive control. The remaining
failures are expected: hidden state requires state construction, and off-policy
TD requires utility estimates aligned with MSPBE/emphatic weighting rather than
supervised one-step residuals. A Step 2 feature signal based only on current
supervised observations is the wrong object for these TD/GVF regimes.

The sweep tested predictive-state construction and an MSPBE-like off-policy
proxy. The 3-seed result was positive, but 10-seed and harder-hidden follow-ups
rejected robust closure: predictive-state features lost to raw MLP on
coupled-hidden AR(1), and off-policy gains did not survive the harder variant.
The shared off-policy rollout plumbing should be kept; the closure claim should
not.

## Candidate Ideas And Critical Assessment

### 1. Finish Full OPMNIST

Assessment: mandatory. This is the only way to remove the published-scale claim
boundary. It is not intellectually novel, but it is the highest-leverage
blocker because the implementation is already strict and positive through ten
blocks.

Risk: the full run may reverse after many permutations. That is an empirical
fact we need, not a failure to avoid.

Success bar: 800/800 blocks complete, published-scale flag true, portfolio
nonnegative vs best fair MLP on final-window MSE and retained held-out accuracy.

### 2. Tune OPMNIST Deployment Objective

Assessment: viable but secondary. Dynamic sparse replay is better than the
canonical MSE-tracking deployment at 10 blocks. A guarded deployment rule could
improve retained accuracy without changing online training.

Risk: overfitting to early permutations. It should be evaluated as a replay
variant from checkpoints, not substituted into the main run without evidence.

Success bar: deployment rule improves held-out accuracy on partial and later
checkpoints without degrading final-window MSE.

### 3. Add More Nonlinear Basis Functions

Assessment: useful but insufficient alone. Signed tanh already shows that
representability is not the whole problem. RBF/spline/hinge features could
cover the nonlinear miss, but without better scoring they may repeat the same
online allocation failure.

Risk: bloated candidate banks masquerading as discovery.

Success bar: pure mechanism beats fair MLP online on nonlinear while preserving
interaction, triple, polynomial, rare, and frequency wins.

### 4. Budgeted Geometric Dictionary

Assessment: rejected. A novelty-gated RBF/Nystrom dictionary is a
geometric answer to nonlinear streams: cover the visited input manifold with
local basis functions and score centers by residual reduction per feature
energy. It is simple, causal, and mathematically grounded.

Result: the implemented geometric dictionary lost all six probes, including
nonlinear (`0.2978` versus best fair MLP `0.0847`) and all algebraic probes
against `single_mechanism`. Local residual patches churned and did not produce
reusable features.

Success bar: closes nonlinear without sacrificing algebraic probes; ideally
also helps OPMNIST/digits.

### 5. Orthogonal Residual Matching

Assessment: rejected in its tested form. Normalize candidate utility by energy and
novelty against active features, approximating online matching pursuit. This
directly attacks the current scoring failure.

Result: `candidate_scoring_mode="energy_novelty"` improved triple and rare but
destroyed nonlinear (`0.4029` versus best fair MLP `0.0301`) and sacrificed
interaction/polynomial/frequency. The novelty term suppressed correlated but
useful scaffolds.

Success bar: signed-tanh/RBF candidates are selected when useful, while product
features remain protected on algebraic tasks.

### 6. Soft-Gated Deep Lifecycle

Assessment: rejected. Hard replacement estimates utility
off-path. Soft gates let candidates receive real downstream gradients before
promotion, then convert useful candidates to active units.

Result: soft-gated variants reached only 2/6 probes; Net2Net/function-preserving
promotion reached only 3/6 and still lost nonlinear, compositional, and digits.

Success bar: one native deep variant beats fair MLP on at least 5/6 hard probes
without relying on an external UPGD feature bank.

### 7. Jacobian/NTK Coverage

Assessment: mathematically attractive but more complex. Score candidate units
by the novelty of their prediction-gradient direction, not only output loss.
This could improve deep lifecycle and recursive generation by avoiding
redundant features.

Risk: JAX/Jacobian cost and implementation complexity. This should be a
secondary add-on after energy/novelty matching.

Success bar: improves candidate selection in compact probes without large
runtime blowup.

### 8. Predictive-State / Koopman Features For TD/GVF

Assessment: partial then rejected for robust closure. Hidden-state TD/GVF problems need features that
summarize history and predict future observations/cumulants. Multi-timescale
traces, auxiliary short-horizon predictions, and GVF-feedback features are the
right geometry: approximate a predictive state.

Result: compact 3-seed coupled-hidden/off-policy means were positive, but the
10-seed default run lost coupled-hidden to raw MLP (`2/8/0` paired wins), and
the harder hidden run lost `0/5/0`.

Success bar: beat raw MLP on coupled-hidden AR(1) and improve over raw clipped
IS on off-policy probes.

### 9. Emphatic / MSPBE Feature Utility

Assessment: promising but not closed. Squared TD error is not the right
objective under behavior-target mismatch. Candidate scoring should weight by
clipped IS/emphasis and measure projected Bellman-error reduction.

Result: the MSPBE proxy beat raw clipped-IS TD in the default 10-seed run
(`7/3/0`) and some sensitivities, but lost in the harder off-policy variant
(`1/4/0`). A real GTD/emphatic criterion is the next defensible version.

Success bar: feature-augmented off-policy TD beats raw clipped-IS TD on both
observable and coupled-hidden probes.

### 10. Universal Portfolio Integration

Assessment: partial, not promoted. A single mechanism may
remain elusive, but the strict Step 2 bar already allows temporally-uniform
routing. A universal portfolio can include MLP, dynamic sparse/UPGD, geometric
dictionary, recursive algebraic features, and native deep lifecycle, with
guarded deployment for external retention.

Result: the all-fronts artifact summary returns `decision=partial`. Strict
supervised, recursive controlled, million-step SCR, and one-seed full-task
OPMNIST online-tracking/MSE evidence are closed. OPMNIST retained-view metrics
and TD/GVF bridge evidence are not Step 2 closure routes.

Risk: overclaiming. This closes an engineering benchmark only if every route is
online, causal, and compared against best fair MLP, and if the docs clearly say
the winner is a portfolio.

Success bar: one canonical command beats/ties best fair MLP on the strict
synthetic, recursive, stateful external, OPMNIST, SCR, and TD/GVF bridge
matrix.

## Implementation Plan

1. Run/monitor full OPMNIST from the 20-block checkpoint and evaluate
   deployment variants at intermediate checkpoints.
2. Build a budgeted geometric dictionary candidate with residual-energy and
   novelty scoring; compare against current recursive mechanisms on the
   six-probe suite.
3. Upgrade recursive candidate scoring with energy/novelty normalization and
   short-trace residual matching, using signed tanh/RBF candidates as opt-in
   generators.
4. Replace hard-only native deep testing with a soft-gated candidate path, then
   harden useful candidates; run the six-probe deep lifecycle matrix.
5. Add predictive-state and emphatic/MSPBE scoring variants to the TD/GVF
   harness; require wins on coupled-hidden and off-policy probes.
6. Integrate any positive candidates into a canonical universal Step 2
   portfolio only after they beat local fair MLP controls in their own
   ownership scopes.

## Sweep Outcome

The sweep did not produce a universal Step 2 closure. It did sharpen the
research boundary:

- Continue OPMNIST to 800/800 blocks. The algorithmic path is ready; the
  blocker is compute and retained-accuracy drift.
- Keep recursive retention as the best pure self-contained mechanism so far,
  but do not promote it until nonlinear closes.
- Stop treating native deep lifecycle as a Step 2 closure path. It remains
  useful instrumentation and a future plasticity research path.
- Keep the shared off-policy rollout plumbing and MSPBE proxy as useful Step 3
  research tools, but reject robust TD/GVF feature-discovery closure.
- Use portfolio closure only for the strict supervised matrix and recursive
  controlled suite, not as an unqualified Alberta Plan Step 2 solution.

## Promotion Rules

A candidate can be promoted only if:

- it has focused tests for new state/config/scoring behavior;
- `ruff check .`, `mypy src/`, and `pytest tests/` pass;
- the experiment command and output directory are recorded;
- the result is compared against the best fair MLP width, not a weaker legacy
  baseline;
- the claim is scoped to the exact benchmark it actually closes.
