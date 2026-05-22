# Step 2 Completion Criteria

Date: 2026-05-08.

This note turns "100000% complete" into separate evidence gates. A Step 2
claim is complete only for the gate and scope named by its evidence. A stronger
sentence cannot inherit weaker evidence.

## Current Status

The current repository is not complete at every possible Step 2 scope. It has a
scoped theorem result, an implementation-complete Step 2 substrate, smoke
confirmation, non-published external evidence, and a single-seed
published-scale OPMNIST protocol artifact. It does not have arbitrary
distribution-free universality or multi-seed published-scale OPMNIST
performance closure.

```json
{
  "schema": "alberta.step2.completion_criteria.v1",
  "date": "2026-05-08",
  "overall_100000_percent_complete": false,
  "overall_reason": "Published-scale OPMNIST protocol is confirmed for one seed, but arbitrary distribution-free universality is rejected rather than proved, and multi-seed published-scale performance closure is not established.",
  "evidence_gates": {
    "theorem_complete": {
      "passes": true,
      "status": "scoped_pass",
      "scope": "conditional finite-resource causal and finite/generated-class theorem",
      "passing_claims": [
        "finite_resource_causal",
        "finite_generated_class"
      ],
      "rejected_claims": [
        "arbitrary_recursive_universality",
        "distribution_free_universality"
      ],
      "required_evidence": [
        "bounded observable targets",
        "declared finite or growing resource schedule",
        "finite/generated candidate class",
        "causal temporal-uniform update rule",
        "explicit approximation, regret, retention, and drift terms"
      ],
      "replacement_theorem": "docs/research/step2_upgd_recursive_feature_discovery_theory.md",
      "impossibility_references": [
        "docs/research/step2_distribution_free_limits.md",
        "docs/research/step2_associative_memory_theory.md",
        "docs/research/step2_compositional_no_regret.md"
      ]
    },
    "implementation_complete": {
      "passes": true,
      "status": "pass",
      "scope": "repo implementation surface for Step 2 learners and associative substrate",
      "required_evidence": [
        "causal prediction-before-update learner APIs",
        "bounded finite-resource feature or memory budget",
        "Step 2 facade or pipeline integration",
        "config serialization and unit coverage for the promoted path"
      ],
      "evidence": [
        "src/alberta_framework/core/associative_memory.py",
        "src/alberta_framework/steps/step2.py",
        "tests/test_associative_memory.py",
        "tests/test_pipeline.py"
      ]
    },
    "smoke_confirmed": {
      "passes": true,
      "status": "pass",
      "scope": "small local protocol and wiring confirmation only",
      "required_evidence": [
        "deterministic smoke command",
        "finite metrics",
        "manifest written",
        "no published-scale claim made by the smoke artifact"
      ],
      "evidence": [
        "benchmarks/step2_associative_opmnist_confirmation.py",
        "tests/test_step2_associative_opmnist_confirmation.py"
      ]
    },
    "external_confirmed": {
      "passes": true,
      "status": "non_published_external_pass",
      "scope": "external or externally grounded data below published OPMNIST scale",
      "required_evidence": [
        "real external dataset metadata or explicitly named external-style protocol",
        "no synthetic fallback counted as external",
        "scale, seeds, and protocol limitations disclosed",
        "published-scale flag remains separate"
      ],
      "evidence": [
        "docs/research/step2_associative_evidence_gate.md",
        "docs/research/step2_associative_opmnist_confirmation.md",
        "tests/test_step2_opmnist_protocol.py"
      ]
    },
    "published_scale_confirmed": {
      "passes": true,
      "status": "single_seed_published_scale_protocol_pass",
      "scope": "Dohare-style full OPMNIST scale for one configured seed",
      "required_evidence": [
        "promoted result artifact with full protocol metadata",
        "mnist_published_scale true",
        "true MNIST 60000/10000 split",
        "800 random pixel permutations",
        "60000 online updates per task block",
        "48000000 completed online updates per configured seed",
        "prediction before update every step",
        "no task id provided to learner",
        "held-out evaluation covers all permutation views",
        "single-seed limitation disclosed"
      ],
      "evidence": [
        "outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json",
        "outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_SUMMARY.md",
        "outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_results.json",
        "outputs/step2_canonical/upgd_memory_opmnist_single_upgd_h128_800block_1seed_SUMMARY.md"
      ],
      "limitations": [
        "one configured seed",
        "the latest-best UPGD-memory artifact wins online MSE, online accuracy, and final-window MSE against the best MLP comparator",
        "the single-UPGD softmax-H128 artifact wins online MSE, online accuracy, final-window MSE, and all-permutation held-out test accuracy against MLP-H128",
        "MLP comparators still win final-window accuracy in both promoted artifacts and all-permutation held-out test MSE in the single-UPGD artifact",
        "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py writes solution_status.solved_opmnist_step2=false until a multi-seed full-scale artifact wins all core metrics",
        "benchmarks.step2_associative_opmnist_confirmation.canonical_opmnist_artifact_status independently reports solved_opmnist_step2=false for current canonical single-seed artifacts"
      ]
    }
  }
}
```

## Gate Semantics

`theorem_complete` is allowed to pass only as a scoped theorem. The current
passing scope is finite-resource, causal, and finite/generated-class. Claims of
arbitrary recursion or distribution-free universality are rejected. They can be
closed only as an impossibility/replacement result that names why the stronger
demand is not a valid theorem target.

`implementation_complete` says the code surface exists and has tests. It does
not imply external data performance or a universal theorem.

`smoke_confirmed` says small runs exercise the protocol and write the expected
artifacts. It never implies external or published-scale confirmation.

`external_confirmed` says there is non-synthetic or externally grounded evidence
with its limitations disclosed. It does not imply the Dohare-style OPMNIST task
count has been run.

`published_scale_confirmed` now means the promoted artifact satisfies the
full 800-task, 48,000,000-update OPMNIST protocol for one configured seed.
It does not mean a multi-seed result exists, that UPGD-memory wins every metric,
or that Step 2 has a distribution-free universal representation theorem.
Partial runs, dry runs, synthetic fallbacks, sklearn-digits smoke runs, and
forged status booleans do not satisfy this gate.

The stricter `canonical_opmnist_artifact_status` audit in
`benchmarks/step2_associative_opmnist_confirmation.py` reserves
`solved_opmnist_step2=true` for a full-scale artifact with at least three
completed seeds and positive candidate-vs-best-MLP deltas on online MSE,
online accuracy, final-window MSE, final-window accuracy, held-out test MSE,
and held-out test accuracy. The current promoted artifact intentionally fails
that stronger solution gate.

The promoted UPGD-memory OPMNIST runner now writes the same gate into every
fresh result as `solution_status` via `opmnist_solution_status` in
`examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py`. This is the
authoritative pass/fail field for future full-scale multi-seed artifacts.
Fresh runner outputs also include `manifest` with argv, git metadata, runtime
environment, method list, and source SHA-256 hashes. A promoted multi-seed
artifact must preserve these per-seed manifests through the split-result JSONs
and the merged artifact's `manifest.split_results` rows, including SHA-256
hashes for every consumed seed result, so the paper can be reproduced from
exact code and command metadata.
`opmnist_solution_status(...).artifact_provenance.provenance_complete` must be
true; otherwise `solved_opmnist_step2` remains false even if the protocol and
performance metrics pass.

The command-line promotion check is:

```bash
python benchmarks/step2_opmnist_solution_gate.py \
  outputs/step2_canonical/upgd_memory_opmnist_latest_best_800block_1seed_results.json
```

It exits nonzero unless `solved_opmnist_step2` is true. Use
`--allow-unsolved` only for diagnostic reports that must print the current
status without claiming closure.

Generate the canonical 3-seed full-run command and follow-up audit command with:

```bash
python benchmarks/step2_opmnist_full_run_plan.py \
  --write-plan outputs/step2_opmnist_solution_full/plan.json
```

The generated plan is not evidence by itself; it is the preregistered command
surface for producing the artifact that the promotion gate will audit.
By default, it targets the current promoted OPMNIST candidate family:
`step2_hybrid_memory_trace` and `step2_hybrid_memory_trace_adaptive_sharp`,
with `mlp_h64`, `mlp_h128`, `mlp_h64_sharp`, and `mlp_h128_sharp` as fair
same-run comparators. Override `--only-methods` only when deliberately running
a new preregistered candidate set.
It also includes per-seed split commands and a merge command:
`python benchmarks/step2_opmnist_merge_seed_results.py ... --output <merged.json>`.
The merged JSON is the artifact passed to `step2_opmnist_solution_gate.py`.

For the actual long run, prefer the resumable coordinator:

```bash
python benchmarks/step2_opmnist_solution_pipeline.py \
  --write-plan outputs/step2_opmnist_solution_full/plan.json \
  --write-status outputs/step2_opmnist_solution_full/pipeline_status.json
```

This prints the split-seed readiness state without starting a run. To advance
one missing seed at a time, rerun it with `--run-next --no-dry-run`. For
bounded scheduler windows, use `--run-next --run-next-chunks N --no-dry-run`;
this writes checkpoints and status after `N` chunks without writing a final
result JSON unless the seed has actually completed all configured steps. When
all three split result JSON files exist, rerun with `--merge-ready --audit
--no-dry-run`. The coordinator still does not create evidence by itself: the
promotion artifact remains the merged JSON plus a zero-exit solution-gate audit.

## Required Language

Acceptable:

- "Step 2 is implementation-complete for the current finite-resource learner
  surface."
- "The theorem status is complete for the conditional finite/generated-class
  statement."
- "The promoted UPGD-memory OPMNIST artifact is published-scale confirmed for
  one seed, but it is not a multi-seed or unqualified performance win."

Not acceptable:

- "Step 2 is 100000% complete."
- "The repo proves arbitrary recursive feature discovery."
- "Distribution-free universality is solved."
- "UPGD-memory wins full OPMNIST" without naming the metric and seed scope.
