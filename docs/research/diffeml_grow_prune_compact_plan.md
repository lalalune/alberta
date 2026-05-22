# DiffEML Grow/Prune/Compact Plan

Worker 3 scope: hard-circuit pruning and compaction after training. The goal is
not to add another soft model. The goal is to take a hardened DiffEML circuit,
remove dead Boolean structure, and produce metadata for a smaller EML-derived
Boolean DAG.

## What This Pass Does

The new pruning utilities live in
`src/alberta_framework/core/diffeml_pruning.py`.

They operate on hard two-input gates:

- source indices for left and right inputs
- a four-bit Boolean truth-table mask
- deployed readout metadata

The pass detects:

- constant gates: masks or simplifications that always produce `0` or `1`
- identity aliases: gates that collapse exactly to an input or prior source
- duplicate gates: exact structural duplicates after commutative canonicalization
- unused readout features: final features not consumed by the deployed head
- unused unique gates: gates that are valid but not reachable from the readout

The output is a `CircuitCompactionResult` with compacted global-DAG metadata,
not a patched runner. That is intentional: a true compactor should be allowed
to rewrite a final-layer duplicate to an input, constant, or earlier retained
gate. The current image runner uses a narrower previous-layer source namespace,
so direct execution needs a small packed-DAG integration step.

## Larp Audit

The clean claim is:

> A learned relaxed DiffEML circuit can be hardened into an EML-derived Boolean
> circuit, and the resulting hard circuit can be evaluated, pruned, compacted,
> and compared under circuit resource accounting.

Claims that are not clean yet:

- A float linear head can make the circuit look better than the Boolean circuit
  deserves. Treat linear-head results as an ablation unless the paper claim is
  explicitly about Boolean features plus a conventional classifier.
- Random width growth is not a principled scaling strategy. It may increase
  chance coverage, but it does not by itself demonstrate that EML structure is
  doing the work.
- Compression after training is only meaningful if hardened accuracy is
  preserved. Storage compression without hard accuracy is bookkeeping.
- Universality should be framed at the circuit level: EML-threshold templates
  can realize the complete two-input Boolean basis; circuits over a complete
  Boolean basis can represent Boolean functions; continuous approximation needs
  a separate bit-encoding or continuous-DiffEML argument.

## Grow/Prune/Compact/Fine-Tune Loop

1. Train an oversized relaxed DiffEML circuit with a pure or explicitly
   quantified readout.
2. Harden gate selectors to truth-table masks and harden readout metadata.
3. Run `compact_hard_circuit`.
4. Measure hard accuracy before and after compaction. This should be exact when
   the runner supports the compact global-DAG source model.
5. Fine-tune the remaining relaxed circuit initialized from the compacted hard
   circuit, optionally regrowing a small budget of new gates.
6. Repeat until validation accuracy no longer improves under a fixed gate or
   byte budget.

## Integration Points

The pruning module deliberately avoids editing shared DiffEML code. Full
integration should touch these places:

- `diffeml_image.selected_gate_mask_arrays`: export hard masks as
  `HardGateLayer` objects or an equivalent adapter.
- `diffeml_image.CircuitWiring`: expose left/right source arrays with the input
  dimension used by the source namespace.
- `diffeml_image.packed_hard_features` and `eval_packed_circuit_chunk`: add a
  packed global-DAG evaluator that resolves `SourceRef` dependencies instead of
  only `input + const_true + previous_layer`.
- readout export in `packed_hard_logits`: pass linear weights, group-sum class
  counts, or class-vote ids into `compact_hard_circuit`.
- result metadata: add `CircuitCompactionResult.to_config()` to model artifacts
  beside compiled storage accounting.

## Scaling Hypothesis

The credible scaling route is structured overproduction followed by exact hard
pruning:

- grow more local or class-conditional gates than needed
- keep hard selectors honest with entropy/temperature schedules
- prune constants, aliases, duplicates, and readout-unreachable gates
- compact to a global Boolean DAG
- fine-tune or regrow under a fixed compute budget

This is more EML-like than adding a bigger float head because the retained
object is still a discrete circuit whose gates are EML-realizable Boolean
functions.

## Acceptance Criteria

A production-quality version of this path should show:

- exact packed logits before and after compaction
- hard accuracy preserved under compaction
- artifact records with original gates, compacted gates, duplicate counts,
  constant counts, unused readout features, and byte estimates
- ablations for no-prune, constants-only, duplicates-only, readout-prune-only,
  and full compact
- comparison at equal hard-gate budgets rather than only equal train-time width
