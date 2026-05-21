# DiffEML Hard-Synthesis Plan

This note defines the harder DiffEML research direction: the deployed object is
a hard Boolean artifact, not a relaxed neural model. The experiment harness is:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_hard_synthesis_suite.py" \
  --run matrix \
  --scale smoke \
  --output outputs/diffeml_hard_synthesis/matrix_smoke.json
```

Cheap backend-free validation uses:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_hard_synthesis_suite.py" \
  --run all \
  --scale smoke \
  --dry-run
```

## Claim Boundary

Promoted results must satisfy all four rules:

- No float head in the deployed classifier.
- Hard packed or deployed metrics are primary; soft metrics are diagnostics.
- Every deployed gate mask has an executable EML witness.
- Image-bit tasks include same-feature baselines.

A run that violates any rule can still be useful as an ablation, but it cannot
support a hard-synthesis claim.

## Mathematical Rationale

DiffEML hard synthesis treats a trained model as a finite Boolean function. Once
continuous inputs are thresholded, each task has inputs in `{0,1}^d` and outputs
in either `{0,1}` or `{0,...,C-1}`. This makes the deployed hypothesis auditable:
it is a composition of gate masks, wiring, and a discrete readout.

Packed bitset synthesis represents each Boolean feature over a batch as machine
words. A candidate gate mask implements a truth table on two bitsets with bitwise
operations, so `packed_hard_accuracy` is the real deployment metric. This is the
fastest path for XOR, thresholded halfspaces, checkerboards, and binary digit-bit
tasks.

ECOC readouts encode each class `c` as a binary codeword `m(c)`. The model learns
hard bit heads and predicts by nearest Hamming distance:
`argmin_c H(h(x), m(c))`. This preserves a pure discrete readout for multiclass
tasks, provided the code heads are hard circuits and the codebook is counted as
metadata rather than a float head.

ANF sparse Boolean polynomials use algebraic normal form over `GF(2)`:
`f(x) = xor_{S in T} prod_{i in S} x_i`. Sparse ANF is the natural description
for parity-like structure such as XOR and checkerboards. Success means the
selected terms compile to AND/XOR gates with EML witnesses for the corresponding
gate masks.

Decision trees and BDDs compile thresholded predicates into branching programs.
A tree can memorize partitions, but an ordered BDD exposes whether the rule has
shared substructure under a variable order. Success requires reporting the
compiled node count, gate-mask witnesses, and packed agreement, not only tree
training accuracy.

## Matrix

Smoke scale enumerates:

| Task | Purpose | Included directions |
| --- | --- | --- |
| `xor` | Minimal nonlinearity and parity witness | packed bitset, ANF |
| `diagonal_halfspace` | Thresholded continuous linear boundary | packed bitset, tree/BDD |
| `checkerboard` | Alternating finite partition | packed bitset, ANF, tree/BDD |
| `small_digits_even_odd_bits` | Binary image-bit task with same-feature baselines | packed bitset, tree/BDD |
| `small_digits_mod3_bits` | Small multiclass image-bit ECOC task | ECOC |
| `multiclass_ecoc_toy` | Controlled multiclass code-decoding task | ECOC |

Full scale keeps the same tasks but expands sample counts and compatible
direction coverage. The harness is backend-free for matrix construction and
lazy-loads future modules only for non-dry executions.

## Success

This direction is successful if the first paper-quality artifact shows:

- XOR recovered exactly with hard masks and EML witnesses across seeds.
- Diagonal halfspace and checkerboard solved by packed/deployed circuits with a
  small soft-hard gap.
- ECOC toy and small digit-bit ECOC runs decode using Hamming distance without a
  float classifier head.
- ANF and BDD rows provide smaller or more interpretable hard circuits than the
  raw packed-gate search on at least one task.
- Image-bit rows include majority, same-feature logistic, and same-feature MLP
  baselines before any external comparison.

## Failure

The direction should be considered failed, or at least not yet publishable, if:

- Soft accuracy is high but packed or deployed hard accuracy is low.
- The best rows need a linear head or uncounted real-valued readout metadata.
- Gate masks cannot be rendered as executable EML witnesses.
- ECOC succeeds only by learning code heads that are not hard circuits.
- ANF or BDD compression destroys accuracy or merely memorizes the smoke split.
- Image-bit improvements disappear against same-feature baselines.

The intended outcome is not just better accuracy. The intended outcome is a
small, inspectable, packed-deployable Boolean artifact whose math and byte count
are both clear.

## Implemented Backends

The current package includes executable hard-synthesis backends in
`alberta_framework.core.diffeml_hard_synthesis`:

- `run_packed_bitset_gate_synthesis`: greedy two-input Boolean mask search with
  packed `uint64` evaluation.
- `run_anf_sparse_polynomial`: sparse ANF over `GF(2)` compiled to AND/XOR EML
  gate masks.
- `run_tree_bdd_compilation`: Boolean decision tree compiled to NOT/AND/OR EML
  gate masks.
- `run_ecoc_readout`: one hard binary circuit per code bit, then packed Hamming
  ECOC decode.

Smoke runs cap effective gate budgets for high-dimensional rows so the suite is
usable during development. Result records include both `requested_gate_budget`
and the effective smoke budget. Full-scale runs keep the requested budgets and
should be treated as deliberate experiments.

## Current Smoke Artifact

Executed command:

```bash
python "examples/The Alberta Plan/Step2/step2_diffeml_hard_synthesis_suite.py" \
  --run all \
  --scale smoke \
  --seeds 0 \
  --output outputs/diffeml_hard_synthesis/smoke_results.json
```

All 11 rows completed. Every row reported `deploy_uses_float_head=false` and
`eml_witness_coverage=true`.

| Run | Hard accuracy | Gates | Bytes | Majority | Same-feature float baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| `packed_xor_seed0` | 1.000 | 1 | 14 | 0.500 | n/a |
| `anf_xor_seed0` | 1.000 | 1 | 14 | 0.500 | n/a |
| `packed_diagonal_halfspace_seed0` | 0.891 | 16 | 149 | 0.547 | n/a |
| `tree_bdd_diagonal_halfspace_seed0` | 0.875 | 9 | 86 | 0.547 | n/a |
| `packed_checkerboard_seed0` | 0.641 | 12 | 113 | 0.516 | n/a |
| `anf_checkerboard_seed0` | 1.000 | 5 | 50 | 0.516 | n/a |
| `tree_bdd_checkerboard_seed0` | 0.969 | 46 | 419 | 0.516 | n/a |
| `packed_small_digits_even_odd_bits_seed0` | 0.828 | 8 | 77 | 0.562 | logistic 0.867, MLP 0.805 |
| `tree_bdd_small_digits_even_odd_bits_seed0` | 0.914 | 26 | 239 | 0.562 | logistic 0.867, MLP 0.805 |
| `ecoc_small_digits_mod3_bits_seed0` | 0.758 | 25 | 233 | 0.375 | logistic 0.867, MLP 0.797 |
| `ecoc_multiclass_ecoc_toy_seed0` | 1.000 | 5 | 54 | 0.250 | n/a |

This is evidence that EML-witnessed hard circuits can learn useful functions
under the suite constraints. It is not yet evidence that the method beats
DiffLogic at published scale. The checkerboard row is solved by an explicitly
topology-aware ANF construction, `structured_grid_boundary_parity`, which uses
the finite partition boundaries and compiles the rule as six single-variable
parity terms. That row should be reported as structured synthesis, not as a
generic greedy-search result. The tree row also improves when smoke mode uses a
task-aware effective depth of 6 instead of the matrix's generic depth-4 cap.
