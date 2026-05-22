# Step 2 Universal Representation-Learning Acceptance

Date: 2026-05-06.

This note defines what this repository may defensibly claim for Alberta Plan
Step 2. It separates the current supervised empirical acceptance matrix from
external-scale evidence, theorem-level guarantees, and Step 3 TD/GVF claims.

## Executive Boundary

The defensible current claim is:

> Target-structure UPGD is the promoted single-learner Step 2 result. It closes
> the current supervised empirical acceptance matrix against fair MLP baselines
> without a portfolio, router, replay buffer, task id, or MLP fallback. This is
> evidence for online utility-guided representation adaptation under bounded
> supervised updates. It is not a global theorem of universal representation
> learning.

The current state is:

- **Supervised empirical matrix**: closed by target-structure UPGD.
- **External-scale evidence**: supportive but incomplete at the performance
  level. Full 800-task Online Permuted MNIST is protocol-scale confirmed for
  one seed, but the metric result is mixed and multi-seed closure is still open.
- **CIFAR-scale/image-scale evidence**: exploratory. A real CIFAR-10 smoke run
  now validates the runner/data path, but canonical multi-seed evidence is not
  recorded.
- **Theorem-level guarantee**: no global theorem yet. The current theorem is a
  conditional update-scale and low-utility exploration guarantee, not a
  universal approximation or dominance theorem.
- **TD/GVF feature discovery**: explicitly Step 3 and out of scope for Step 2
  closure.

## Claim Levels

Use these levels consistently.

| Level | Status | Required evidence | Permitted language |
|---|---|---|---|
| L0: smoke/path validation | Complete for many runners | Import/tests/single seed/tiny run | "runner works", "path validated" |
| L1: local positive result | Common | Same-run comparator, small seed count, focused task | "positive probe", "candidate" |
| L2: supervised Step 2 empirical acceptance | Current promoted level | Full supervised matrix, fair MLP comparators, paired multi-seed stats, no task id/replay/router | "closes the current supervised Step 2 empirical matrix" |
| L3: external-scale replication | Not closed for OPMNIST | Published-style protocol, full task count or explicit scale gate, fair baselines, primary metrics nonnegative | "external-scale replicated" only after gates pass |
| L4: theorem-level guarantee | Not available | Formal assumptions, proof of approximation/adaptation/regret/convergence under finite resources | "theorem", "guarantee", "universal under assumptions" |

Do not collapse L2 into L3 or L4.

## Empirical Acceptance Matrix

The supervised Step 2 empirical matrix is a bounded benchmark claim, not an
unbounded no-free-lunch claim. A method satisfies it only if all conditions
below hold.

### Method Constraints

- Single continually updated learner with one active prediction path.
- No deployment portfolio, output router, expert selector, replay buffer, task
  id, future labels, held-out test feedback, or dataset-specific branch.
- Prediction-before-update on each online example.
- All trainable components update temporally uniformly, or any inactive update
  path is justified as a causal per-step mechanism such as intervaled
  low-utility perturbation.
- Shared nonlinear representation is used for vector targets, not one separate
  hand-tuned learner per regime.

### Comparator Constraints

- Same-run fair MLP baselines are required.
- MLP widths must be plausible rather than intentionally capacity-starved.
- Baselines must receive comparable preprocessing, ObGD/bounding choices where
  applicable, online update counts, and target encodings.
- The comparison should use paired seed/regime statistics.

### Task Families

The current supervised matrix includes:

| Family | Purpose | Acceptance standard |
|---|---|---|
| Dense synthetic out-of-hypothesis-class streams: polynomial, frequency, compositional | Tests nonlinear feature adaptation outside pair-product feature banks | Positive final-window MSE against same-run best fair MLP |
| Dense-zero regression stress | Tests exact-zero target coordinates without treating them as sparse classification | Nonnegative versus fair MLP and better target-normalization behavior than target-density |
| Sparse multilabel stress | Tests sparse targets that are not one-hot/simplex classification | Positive versus fair MLP and no target-density-style overboost |
| Sklearn-digits online regimes: IID, class-blocked, permuted pixels, mask noise, label drift | Tests external supervised one-hot targets under nonstationarity | Positive aggregate final-window MSE and held-out/test accuracy against fair MLP, with hard-row caveats reported |
| Controlled recursive feature-construction probes | Tests whether feature-of-feature mechanisms work on designed recursive targets | Supportive evidence only unless the same method also passes the full promoted matrix |

### Current Accepted Result

Target-structure UPGD is accepted for L2 because the recorded evidence says it:

- uses a causal target-structure loss rule rather than a dataset router;
- preserves one-hot/simplex update pressure and dense-regression mean scaling;
- wins the dense synthetic out-of-class streams against fair MLP;
- passes dense-zero and sparse-multilabel stressors;
- closes the current digits matrix with density-equivalent one-hot behavior;
- remains a single UPGD learner with explicit online hidden-feature utility and
  bounded low-utility perturbation.

The compute-efficient `UPGDLearner.step2_default(n_heads)` branch strengthens
the L2 practical claim by showing a width-32 lean Rademacher target-structure
variant can beat the MLP64 comparator while running faster in the recorded JAX
scan throughput benchmark. It does not change the theorem boundary.

## External-Scale Evidence

External-scale evidence is a separate replication tier. It is not required to
say the current supervised Step 2 matrix is closed, but it is required before
claiming published-scale or broad real-world continual-learning superiority.

### Sufficient External-Scale Evidence

For a published-style external claim, a run must record protocol gates and pass
them. For Online Permuted MNIST, this means:

- true MNIST source, not sklearn digits or a small fallback;
- canonical 60,000/10,000 split;
- 60,000-example task blocks;
- no task id to the learner;
- sequential single-pass examples within each task;
- random pixel permutations for all tasks;
- full requested task count when claiming published scale;
- fair MLP comparators trained online in the same protocol;
- primary online and held-out metrics nonnegative versus best fair MLP.

### Current OPMNIST State

The current OPMNIST evidence has advanced from partial checkpoints to a
completed one-seed protocol-scale artifact:

- Current promoted artifact:
  `outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json`.
- Follow-up promoted artifact:
  `outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_results.json`.
- Completed progress: 800 full 60,000-example blocks out of 800; 48,000,000
  online examples.
- The artifact uses true OpenML MNIST, the canonical 60,000/10,000 split, 800
  random pixel permutations, no task id, prediction-before-update, and all 800
  held-out permutation views.
- It is mixed on performance. The latest-best UPGD-memory artifact wins online
  MSE, online accuracy, and final-window MSE against the same-run best fair
  MLP. The single-UPGD softmax-H128 artifact additionally wins
  all-permutation held-out test accuracy against `mlp_h128`. Fair MLP
  comparators still win final-window accuracy in both artifacts and held-out
  all-permutation test MSE in the single-UPGD artifact.

Therefore the only permissible wording is that OPMNIST is protocol-scale
confirmed for one seed and still open as a multi-seed/all-metric performance
boundary. Do not say the full OPMNIST performance result is closed unless the
result file passes the explicit published-scale and metric gates.

### CIFAR And Other Image-Scale Evidence

CIFAR or CIFAR-like results are exploratory unless they meet the same rules:
online one-pass protocol, no task id, no replay unless explicitly declared as a
different setting, fair MLP/continual baselines, enough seeds, predeclared
metrics, and committed result artifacts. A single exploratory run may motivate
work; it cannot upgrade the Step 2 claim.

## Theorem-Level Guarantees

The current defensible theorem-level content is conditional and narrow. It must
be split by claim type.

| Claim type | Current status | Acceptable wording |
|---|---|---|
| Expressivity | Finite-width MLP expressivity only; universal approximation needs growing width or a dense dictionary. | "expressive over the configured finite resource class" |
| Online adaptation | Causal SGD plus target-structure scaling, ObGD, and low-utility perturbation. No convergence theorem. | "supports online adaptation under bounded updates" |
| Regret/competitiveness | No regret theorem for the exact UPGD recursion. Finite expert-selection regret is available only for an idealized selector. | "a possible reduction for future sieve-style variants" |
| Empirical universality | Closed for the declared supervised Step 2 matrix. | "empirically universal over the current supervised matrix" |

The implemented learner supports these theorem-level invariants:

- Target-structure loss gives update-scale consistency for one-hot/simplex
  targets, dense regression including exact-zero coordinates, and sparse
  multilabel targets.
- ObGD-style bounding constrains per-step displacement under the configured
  envelope.
- Nonzero low-utility perturbation gives low-utility weights an exploration
  channel on perturbation steps.

A future L4 theorem can only claim universal representation learning after it
adds the missing assumptions and lemmas: bounded inputs/targets, a rich
generated-feature dictionary or restricted target class, a finite resource
schedule, recurrence/excitation, a declared stationary or bounded-variation
nonstationary model, and a precise baseline comparator class.

This does not prove:

- global convergence;
- regret against the best representation in hindsight;
- discovery of all useful features under finite budget;
- universal recursive feature construction;
- dominance over all MLPs, all streams, or all target processes;
- TD/GVF feature discovery.

It must state whether it proves approximation, adaptation rate, regret,
stability, or convergence; these are different claims.

## Out-Of-Scope For Step 2

The following must not be used to claim Step 2 universal representation
learning:

- TD, TD(lambda), true-online TD, off-policy TD, GVF, Horde, nexting, or
  control results.
- Surprise-driven cumulant discovery or predictive-state GVF feedback.
- Step 4 SARSA/control performance.
- Bsuite control results.

These may demonstrate that Step 2 representations can be handed to Step 3, or
that Step 3 has its own feature-discovery progress. They do not close Step 2.

## What Evidence Is Not Sufficient

Insufficient evidence includes:

- wins over an under-width or otherwise weak MLP baseline;
- a pair-product learner winning only when the hidden oracle is pair products;
- a portfolio/router closing a matrix while the claim says "single learner";
- held-out deployment routing that uses test feedback or task ids;
- smoke runs, one-seed runs, or capped external subsets presented as scale
  closure;
- final-window wins that hide large held-out retention regressions;
- held-out retention wins that hide current-window tracking collapse;
- CIFAR or OPMNIST partial runs described without protocol gates;
- theoretical language based only on empirical dominance.

## Permissible Language

Use:

- "Target-structure UPGD closes the current supervised Step 2 empirical
  acceptance matrix."
- "This is an empirical bounded-matrix claim."
- "The current evidence supports utility-guided online representation
  adaptation under supervised one-pass nonstationarity."
- "OPMNIST is protocol-scale confirmed for one 800-task seed, with a
  single-UPGD H128 follow-up that wins held-out test accuracy but still has
  mixed metric results and no multi-seed performance closure yet."
- "No global theorem of universal representation learning is currently
  established."
- "TD/GVF feature discovery remains Step 3 research."

Do not use:

- "Step 2 is solved" without the words "current supervised empirical matrix."
- "Universal representation learning is proved."
- "Published-scale OPMNIST is closed."
- "CIFAR validates the claim" for exploratory runs.
- "TD/GVF feature discovery is solved by Step 2."

## Remaining Blockers

The remaining blockers for stronger claims are:

1. Complete the full 800-task OPMNIST run, or keep it explicitly outside the
   Step 2 acceptance blocker.
2. Add canonical external-scale image/tabular/regression streams with enough
   seeds and protocol gates before making broad external claims.
3. Resolve the class-blocked tracking-versus-retention tradeoff more cleanly,
   or continue to report it as a limitation.
4. Prove a formal L4 theorem under explicit stream and budget assumptions, if
   theorem language is desired.
5. Keep TD/GVF feature discovery on the Step 3 evidence track.

## Source Notes

This rubric is consistent with the current evidence notes:

- `docs/research/step2_current_best.md`
- `docs/research/step2_compute_efficient_upgd.md`
- `docs/research/step2_target_structure_upgd_stress.md`
- `docs/research/step2_target_density_upgd_closure.md`
- `docs/research/step2_universality_matrix.md`
- `docs/research/step2_opmnist_800_task_closure.md`
- `docs/research/step2_published_mnist_closure.md`
- `docs/research/step2_external_benchmarks.md`
- `docs/research/step2_worker_p_tdgfv_feature_discovery.md`
