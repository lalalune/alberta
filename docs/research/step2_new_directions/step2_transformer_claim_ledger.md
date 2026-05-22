# Step 2 Transformer Claim Ledger

Date: 2026-05-07

Purpose: keep Step 2 claim language tied to recorded evidence. This file is a
claim-to-evidence ledger, not a result summary. Use it before writing abstracts,
slides, README language, or production-readiness notes.

## Claim Levels

| Level | Meaning | Allowed scope |
|---|---|---|
| L0 | Path or smoke validation | Imports, tiny runs, single-seed or runner wiring only |
| L1 | Local positive candidate | Focused benchmark with fair comparator, but limited scale or post hoc selection |
| L2 | Current supervised Step 2 empirical matrix | Prespecified supervised matrix, paired multi-seed fair MLP comparators, single learner |
| L3 | External-scale replication | Published-style protocol, full requested scale, fair baselines, primary metrics pass |
| L4 | Theorem-level guarantee | Formal assumptions and proof of approximation, adaptation, regret, or convergence |

Do not collapse L2 into L3 or L4.

## Claim-To-Evidence Matrix

| Claim area | Current evidence | Current level | Allowed phrases | Banned phrases | Gate to upgrade |
|---|---|---:|---|---|---|
| Universal learner / universal representation | `docs/research/step2_universal_representation_learning_acceptance.md` says target-structure UPGD closes the current supervised empirical matrix, but not a global theorem. `docs/research/step2_current_best.md` explicitly rejects a universal "UPGD beats every fair MLP" claim after external reruns. | L2 | "Target-structure UPGD closes the current supervised Step 2 empirical acceptance matrix." "Empirical bounded-matrix claim." "Promoted single learner for online vector-target continual learning." | "Universal representation learning is proved." "Step 2 is solved" without "current supervised empirical matrix." "UPGD dominates all MLPs, streams, or target processes." "Universal learner" as an unconditional noun. | L4 proof with stated assumptions, comparator class, finite resource schedule, recurrence/excitation conditions, and a precise theorem target. |
| Tiny Shakespeare transformer | `advantage_memory_transformer_followup.md` records exploratory 10-seed Tiny Shakespeare gains for replay-capped post-FFN memory at 3000, 5000, and 10000 steps, with small margins. `advantage_memory_transformer_paperworthy_protocol.md` says the result is not paperworthy yet. `advantage_memory_transformer_confirmatory_lockbox_protocol.md` freezes a 30-seed validation/lockbox protocol but does not report it as passed. | L1 | "In exploratory Tiny Shakespeare runs, replay-gated slow memory under a tight resource cap produced small paired improvements over the tuned FFN baseline." "Tiny Shakespeare is transformer-facing integration evidence." | "Replay-capped memory transformers beat tuned fast transformers." "Transformer FFN replacement is proven." "Production language-model win." "Paperworthy Tiny Shakespeare advantage" before the frozen protocol passes. | 30 paired seeds, frozen byte-indexed train/validation/lockbox splits, corpus hashes, 4096 held-out contexts per seed, 3000/5000/10000 horizons, Holm correction across primary metrics, smallest-effect gate, parameter- and compute-matched FFN controls, complete manifests. |
| External sequence memory | `external_sequence_memory_benchmarks.md` tests four bounded non-Shakespeare or algorithmic probes. The replay-capped memory path is neutral to slightly negative on held-out NLL and does not improve accuracy. The local text probe is generated because no cached public non-Shakespeare corpus was present. | L1 negative / open | "The current replay-capped post-FFN memory candidate did not generalize as a positive external sequence result." "External sequence validation remains open." | "External sequence memory passed." "The Tiny Shakespeare memory mechanism generalizes beyond Shakespeare." "The prototype path behaves as addressable KV memory." | Real public non-Shakespeare corpus with checksum/provenance; query-only sparse-recall metrics; true addressable KV and oracle baselines; pre-FFN/KV placement retest; longer horizon after gate/cap failure is fixed. |
| OPMNIST / retained image-view memory | `step2_opmnist_800_task_closure.md` now records a 47/800-block checkpoint and evaluated 47-block snapshot: final-window MSE and final-window accuracy favor the portfolio, held-out test MSE favors it, but held-out test accuracy trails the best fair MLP. `published_scale_external_claim_supported=false`. `step2_current_best.md` records compact 3-seed evidence for JAX prototype memory and the UPGD-memory hybrid as retained-view components. | L1 to partial L3, not closed | "OPMNIST is a positive partial external snapshot and an open published-scale boundary." "The latest local OPMNIST status is 47/800 blocks, 2,820,000/48,000,000 examples, with published-scale support false." "JAX prototype memory is the current retained-view memory component." | "Published-scale OPMNIST is closed." "800-task OPMNIST replicated." "Class-view retention is solved." "The 47-block snapshot proves broad external-scale superiority." | Complete 800 blocks / 48,000,000 examples, full aggregation, all primary metrics nonnegative against the best same-run fair MLP, protocol gates true, no task-id or held-out-feedback leakage, and reproducible checkpoint/status artifacts. |
| Production JAX Step 2 | `step2_paper_and_deployment_quality_plan.md` supports a narrow deployment-candidate claim for width-32 `UPGDLearner.step2_default` versus MLP64 in recorded JAX scan benchmarks. `throughput_transformer_memory_2026-05-07.md` shows transformer replay memory is materially slower than FFN. `src/alberta_framework/steps/step2.py` exposes Step 2 UPGD, prototype memory, and UPGD-memory factories; `src/alberta_framework/pipeline.py` does not expose memory modes. | L2 for UPGD scan benchmark; standalone memory API partial; transformer memory not production-efficient | "Recorded JAX scan benchmarks show width-32 target-structure UPGD faster than MLP64 for the measured workloads." "PrototypeMemory and UPGDMemory are standalone JAX PyTree/scan-compatible Step 2 learners." "The end-to-end pipeline currently exposes `temporal_context`, `upgd`, and `identity`, not memory modes." | "Same-width UPGD is cheaper." "Transformer memory is production-efficient." "The pipeline supports Step 2 retained-view memory." "Production-ready outside online continual supervised workloads." | Exact deployment candidate frozen; config and checkpoint roundtrips; single-step and scan tests; domain shadow evaluation; monitoring/rollback plan; docs matching the code surface; transformer memory needs fused production kernels and competitive throughput. |

## Phrase Ledger

Allowed:

- "current supervised Step 2 empirical matrix"
- "single non-router target-structure UPGD learner"
- "bounded online supervised workloads"
- "exploratory Tiny Shakespeare transformer-facing evidence"
- "external sequence memory result is currently negative/open"
- "OPMNIST is partial; published-scale flag remains false"
- "standalone Step 2 memory learners are scan-compatible"

Banned:

- "universal learner" without an explicit empirical-matrix qualifier
- "universal representation learning is proved"
- "Step 2 is solved" without "current supervised empirical matrix"
- "Tiny Shakespeare proves transformer replacement"
- "external sequence validation passed"
- "published-scale OPMNIST is closed"
- "pipeline supports Step 2 memory modes"
- "production ready" without checkpoint/config/single-step/scan/docs gates and a deployment shadow evaluation

## Ledger Notes

- The newest local OPMNIST status file says 47/800 blocks. Older Step 2 claim
  notes still say 46/800. Use the 47/800 status for new language, and have the
  OPMNIST owner reconcile stale summaries.
- The transformer memory throughput note is about a replay-capped research
  runner. It must not be used as evidence of production efficiency.
- The external sequence note is negative evidence and should be shown when
  discussing transformer-memory generalization.
