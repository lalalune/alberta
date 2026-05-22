# Step 2 Associative Evidence Gate

Date: 2026-05-07.

## Purpose

This note records the evidence boundary for the new associative Step 2 core
promotion. The promotion is an implementation closure: the repository now has a
fixed-budget associative memory core, Step 2 facade wiring, end-to-end pipeline
mode, smoke tests, and positive sequence-memory artifacts. It is not a formal
universality result and it does not close published-scale OPMNIST.

## Guard Contract

- `associative_core_promotion`: `implementation_closure`
- `formal_universality_supported`: `false`
- `published_scale_opmnist_supported`: `false`
- `associative_positive_artifact`: `outputs/step2_new_directions/associative_external_hybrid_900_10seed/results.json`
- `associative_sparse_kv_focus_artifact`: `outputs/step2_new_directions/sparse_kv_associative_probe_900_30seed_focus/results.json`
- `opmnist_status_artifact`: `outputs/step2_canonical/opmnist_true_mnist_40block_mse_results.json`
- `formal_universality_required_artifact`: `docs/research/step2_associative_formal_universality.md`
- `opmnist_published_scale_required_artifact`: `outputs/step2_canonical/opmnist_true_mnist_800task_results.json`

## What Is Closed

The associative work closes the implementation path for a causal associative
Step 2 substrate:

- prediction occurs before the current write;
- context windows produce discrete associative features;
- online rows are updated by a bounded table budget and learned utility;
- optional scope controllers learn soft feature-family, suffix-window, and
  effective-budget gates from causal loss advantage and replacement pressure;
- the Step 2 and pipeline facades can expose associative probability features;
- smoke tests cover config round trip, prediction-before-write behavior,
  repeated binding, fixed-budget replacement, and facade smoke behavior.

The strongest current sequence artifact is the 10-seed hybrid external smoke at
`outputs/step2_new_directions/associative_external_hybrid_900_10seed/results.json`.
The `hybrid_token_suffix_norm4` row beats the FFN baseline by held-out eval NLL
on every seed for `block_shift_markov`, `delayed_copy`, `sparse_kv_recall`, and
`local_text_motif`. The 30-seed focused sparse-KV artifact also shows the
suffix-pair associative mechanism beating the FFN baseline on every seed.

This is enough to promote the associative core as a real implementation surface
and a credible mechanism lead for fast key/value binding.

## What Is Not Closed

The associative promotion does not establish formal universality. The feature
families, suffix window, and effective budget can now be softened by optional
learned gates, but the operation set, maximum context width, maximum table
budget, calibration scale, and protocol scope are still declared choices. A
formal claim needs its own artifact with explicit stream assumptions,
finite-resource schedule, approximation/adaptation statement, and proof
obligations.

The promotion also does not close Online Permuted MNIST at the published scale.
The broader Step 2 evidence set now has a separate one-seed UPGD-memory
artifact with all 800 60,000-example task blocks complete, but that is not an
associative-memory result and it is not a multi-seed or all-metric performance
closure.

## Gate Hook

Do not edit the public gates for this small guard. The obvious future hook is
to add optional associative artifact arguments and pure JSON checks to
`benchmarks/step2_upgd_evidence_gate.py`, following its existing `GateCheck`
and `write_summary` pattern. The lightweight console presence gate in
`src/alberta_framework/cli.py:evidence_gate_main` has a second hook if the
associative artifacts become required canonical files.
