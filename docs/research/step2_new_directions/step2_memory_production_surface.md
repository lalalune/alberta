# Step 2 Memory Production Surface

Date: 2026-05-07

Purpose: audit what Step 2 memory currently exposes, what the end-to-end
pipeline does not expose, and the exact gates required before production-ready
claim language is allowed.

## Audit Verdict

Step 2 memory is exposed as standalone production-facing Step 2 factories, but
not as an end-to-end `AlbertaPipeline` Step 2 mode.

Do not claim that retained-view memory is available through the Step 1-4
pipeline until a pipeline owner adds a mode, config, state field, update path,
scan path, and tests.

## Current Standalone Surface

| Surface | Evidence | Status |
|---|---|---|
| Fixed-budget prototype memory | `src/alberta_framework/steps/step2.py:69` defines `Step2MemoryConfig`; `src/alberta_framework/steps/step2.py:189` defines `make_step2_memory_learner`; `src/alberta_framework/core/prototype_memory.py:113` defines `PrototypeMemoryLearner`. | Exposed outside pipeline |
| UPGD plus prototype memory hybrid | `src/alberta_framework/steps/step2.py:85` defines `Step2HybridConfig`; `src/alberta_framework/steps/step2.py:206` defines `make_step2_hybrid_learner`; `src/alberta_framework/core/upgd_memory.py:265` defines `UPGDMemoryLearner`. | Exposed outside pipeline |
| Package facade exports | `src/alberta_framework/steps/__init__.py:19` imports Step 2 memory configs/factories and `src/alberta_framework/steps/__init__.py:64` exports them. | Exposed through `alberta_framework.steps` |
| Single-step API | `PrototypeMemoryLearner.predict/update` and `UPGDMemoryLearner.predict/update` are JIT-wrapped single-step methods. Existing tests cover finite predictions, step counts, fixed-budget allocation, target-trace behavior, and both-component updates. | Partially accepted |
| Scan API | `run_prototype_memory_arrays` and `run_upgd_memory_arrays` use `jax.lax.scan`; existing tests include scan/JIT compatibility. | Partially accepted |

The standalone memory surface is suitable for focused Step 2 probes and
downstream callers that instantiate the learner directly. It is not yet enough
for an end-to-end pipeline claim.

## Pipeline Exposure Audit

| Pipeline point | Evidence | Memory exposure |
|---|---|---|
| Mode enum | `src/alberta_framework/pipeline.py:64` defines `Step2Mode = Literal["temporal_context", "upgd", "identity"]`. | No memory mode |
| Config fields | `src/alberta_framework/pipeline.py:268` has `features`, `upgd`, `horde`, `control`, `horde_ac`, `step2`, and `control_mode`. | No memory config |
| Validation | `src/alberta_framework/pipeline.py:278` only accepts `temporal_context`, `upgd`, and `identity`. | Memory rejected as unknown |
| State | `src/alberta_framework/pipeline.py:350` stores `feature_state`, `upgd_state`, `horde_state`, `control_state`, `last_features`, and `step_count`. | No memory state |
| Construction | `src/alberta_framework/pipeline.py:459` builds temporal context or UPGD only. | No memory learner |
| Feature path | `src/alberta_framework/pipeline.py:549` returns temporal-context features, UPGD trunk features, or raw identity observations. | No memory prediction/features |
| Update path | `src/alberta_framework/pipeline.py:655` accepts `upgd_targets` only for Step 2 learning. | No memory target/update |
| Array scan | `src/alberta_framework/pipeline.py:785` scans observations, rewards, termination flags, cumulants, and optional UPGD targets. | No memory targets/state |
| Pipeline tests | `tests/test_pipeline.py:219` covers UPGD Step 2; there is no pipeline memory test. | Not exposed |

Result: the pipeline currently supports learned UPGD hidden features, not retained
prototype memory or UPGD-memory blended predictions.

## Acceptance Tests

These are the exact gates before claim language can say "production-ready Step 2
memory" or "pipeline supports Step 2 memory."

| Gate | Required acceptance tests | Current status |
|---|---|---|
| Checkpoint | Add `test_step2_memory_checkpoint_roundtrip`: instantiate `make_step2_memory_learner`, run at least one update, `save_checkpoint` with metadata containing `config.to_dict()`, `load_checkpoint` into a fresh template, assert state trees and predictions match. Add `test_step2_hybrid_checkpoint_roundtrip` with the same pattern for `make_step2_hybrid_learner`. If pipeline memory is added, add `test_pipeline_step2_memory_checkpoint_roundtrip` for `AlbertaPipelineState`. | Blocked. Generic checkpoint utilities exist, but memory-specific state roundtrips are not tested. |
| Config | Add `from_dict` or `from_config` roundtrips for `Step2MemoryConfig` and `Step2HybridConfig`, then test JSON serialization with `json.dumps/json.loads`. If pipeline memory is added, extend `AlbertaPipelineConfig.to_dict/from_dict` and test memory-mode roundtrip plus invalid-combination failures. | Blocked for Step 2 facade configs. Existing tests only check `to_dict` snapshots; underlying core `PrototypeMemoryConfig` and `UPGDMemoryConfig` have roundtrip tests. |
| Single-step | Keep existing `tests/test_prototype_memory.py` and `tests/test_upgd_memory.py` coverage. Add facade-level assertions that `make_step2_memory_learner` and `make_step2_hybrid_learner` produce finite `predict`, one-step `update`, monotonic `step_count`, bounded prototype count, and both-component updates for the hybrid. If pipeline memory is added, add `test_pipeline_with_step2_prototype_memory_single_step` and `test_pipeline_with_step2_upgd_memory_single_step`. | Partial. Standalone core and facade smoke tests exist; pipeline memory single-step does not exist. |
| Scan | Keep `test_run_prototype_memory_arrays_is_scan_compatible` and `test_upgd_memory_scan_runner_is_jit_compatible`. Add facade-level scan tests through Step 2 factories. If pipeline memory is added, add `test_pipeline_step2_memory_scan_is_finite` checking shapes, finite values, memory state advancement, Horde/control advancement, and no shape drift under `run_arrays`. | Partial. Core scan tests exist; pipeline memory scan does not exist. |
| Docs | Update `docs/guide/step1-step2-production.md` to show both raw prototype memory and hybrid usage, including checkpoint/config examples after those gates exist. Update `docs/guide/end-to-end-pipeline.md` to match the real pipeline modes. If memory is not integrated, it must explicitly say memory is standalone only. If memory is integrated, it must document memory target shape, update semantics, scan arrays, and checkpoint template construction. | Blocked. The pipeline guide is stale relative to code: it says no UPGD extractor or Horde actor-critic wrapper, while code now has both. It also has no memory-mode contract. |

Minimum command set after code changes:

```bash
source .venv/bin/activate
pytest tests/test_prototype_memory.py tests/test_upgd_memory.py tests/test_production_steps.py tests/test_pipeline.py tests/test_checkpoints.py -q
ruff check .
mypy
```

For documentation-only edits, run at least:

```bash
git diff --check
```

## Production Readiness Gates

Standalone Step 2 memory can be called production-facing only after:

- memory and hybrid checkpoint roundtrips pass;
- facade configs have reversible JSON serialization;
- single-step `predict/update` tests cover finite metrics and fixed-budget state;
- scan/JIT tests cover the factory-created learners, not only core classes;
- docs include direct usage, checkpoint restore, and claim boundaries.

Pipeline Step 2 memory can be claimed only after:

- `Step2Mode` includes explicit memory mode names or a separate memory policy;
- `AlbertaPipelineConfig` carries memory/hybrid config and roundtrips it;
- `AlbertaPipelineState` stores memory or hybrid state;
- `init`, `predict`, `update`, and `run_arrays` advance memory state causally;
- target semantics are explicit for one-hot/simplex targets, NaN inactive heads,
  and non-classification streams;
- Step 3 and Step 4 feature handoff is defined. In particular, decide whether
  memory predictions are outputs, auxiliary features, or a separate supervised
  head outside the Horde/control feature path;
- tests cover checkpoint/config/single-step/scan/docs.

## Blockers

- Pipeline memory integration is not a small patch. It needs an API decision:
  feature augmentation, supervised prediction head, or both.
- Step 2 facade configs lack reversible `from_dict` methods, so config
  serialization is not production-complete at the facade level.
- Memory-specific checkpoint tests are missing.
- End-to-end pipeline docs are stale relative to current code and still do not
  declare the memory boundary.
- Claim notes disagree on OPMNIST progress: newer status says 47/800 blocks,
  older claim notes still say 46/800.

## Recommended Owners

| Area | Recommended owner |
|---|---|
| Pipeline memory modes and `AlbertaPipelineState` changes | Pipeline/API owner |
| Step 2 facade config roundtrips and memory checkpoint tests | Step 2 production-surface owner |
| Prototype/UPGD-memory scan and single-step regression tests | Core memory owner |
| `docs/guide/end-to-end-pipeline.md` and `docs/guide/step1-step2-production.md` updates | Docs owner after API owner decides memory scope |
| OPMNIST status reconciliation and published-scale language | OPMNIST experiment owner |
| Tiny Shakespeare and external sequence claim status | Transformer-memory research owner |
