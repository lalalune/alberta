# External Sequence Memory Benchmarks

Date: 2026-05-07

## Question

The replay-capped post-FFN memory transformer is the current Step 2 sequence
candidate from the Tiny Shakespeare follow-up. This pass asks whether that
candidate still helps outside the original Tiny Shakespeare stream, and whether
it survives a stronger stateful association probe.

## Benchmarks

The runner now implements four bounded external probes:

- `block_shift_markov`: a contiguous character-like stream with three recurring
  second-order Markov grammars and abrupt block shifts;
- `delayed_copy`: an algorithmic next-token probe where the queried lag changes
  by block and the context includes payload-token distractors;
- `sparse_kv_recall`: random key-pair to value bindings are rebound across
  episodes, distractors intervene, and held-out evaluation queries target the
  final one-shot binding table;
- `local_text_motif`: a deterministic non-Shakespeare prose-like character
  stream.

No cached non-Shakespeare text corpus was present locally. The only cached
`.txt` files found were Tiny Shakespeare copies:

- `output/data/tinyshakespeare/input.txt`
- `output/subagents/transformer_ffn/data/tinyshakespeare.txt`

To avoid making this external check depend on a flaky network fetch, the
non-Shakespeare text benchmark uses a deterministic local text-like generator.
Its result should be treated as a corpus-diversity smoke, not as a public-corpus
claim.

All probes compare the FFN transformer against the replay-capped post-FFN scalar
advantage memory candidate with `gate_objective=replay`, `replay_size=128`,
`gate_max=0.15`, `train_loss_mode=memory`, and `reset_mode=meta_ema`.

## Commands

Smoke:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" \
  --benchmarks all \
  --steps 80 \
  --seeds 1 \
  --eval-steps 32 \
  --final-window 32 \
  --block-size 32 \
  --d-model 16 \
  --mlp-hidden 32 \
  --proto-count 16 \
  --replay-size 16 \
  --output-dir outputs/step2_new_directions/external_sequence_memory_v2_smoke
```

Bounded 3-seed comparison:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_sequence_external_memory_benchmarks.py" \
  --benchmarks all \
  --steps 900 \
  --seeds 3 \
  --eval-steps 256 \
  --final-window 256 \
  --output-dir outputs/step2_new_directions/external_sequence_memory_v2_3seed
```

## Results

Positive diffs favor replay-capped memory. The full artifact is
`outputs/step2_new_directions/external_sequence_memory_v2_3seed/SUMMARY.md`.

| Benchmark | Metric | FFN | Replay memory | Diff |
| --- | --- | ---: | ---: | ---: |
| `block_shift_markov` | final-window NLL | 3.5153 +/- 0.0090 | 3.5155 +/- 0.0090 | -0.0002 +/- 0.0000 |
| `block_shift_markov` | held-out NLL | 3.5227 +/- 0.0345 | 3.5229 +/- 0.0346 | -0.0002 +/- 0.0001 |
| `block_shift_markov` | fast-only held-out NLL | 3.5227 +/- 0.0345 | 3.5224 +/- 0.0342 | +0.0003 +/- 0.0003 |
| `delayed_copy` | final-window NLL | 3.1853 +/- 0.0268 | 3.1853 +/- 0.0268 | -0.0001 +/- 0.0000 |
| `delayed_copy` | held-out NLL | 3.0912 +/- 0.0677 | 3.0913 +/- 0.0678 | -0.0001 +/- 0.0001 |
| `delayed_copy` | fast-only held-out NLL | 3.0912 +/- 0.0677 | 3.0915 +/- 0.0673 | -0.0003 +/- 0.0006 |
| `sparse_kv_recall` | final-window NLL | 3.0062 +/- 0.0242 | 3.0063 +/- 0.0242 | -0.0001 +/- 0.0000 |
| `sparse_kv_recall` | held-out NLL | 2.8197 +/- 0.0334 | 2.8197 +/- 0.0335 | -0.0000 +/- 0.0000 |
| `sparse_kv_recall` | fast-only held-out NLL | 2.8197 +/- 0.0334 | 2.8199 +/- 0.0332 | -0.0003 +/- 0.0003 |
| `local_text_motif` | final-window NLL | 2.5938 +/- 0.0230 | 2.5939 +/- 0.0230 | -0.0002 +/- 0.0000 |
| `local_text_motif` | held-out NLL | 2.4771 +/- 0.0383 | 2.4774 +/- 0.0383 | -0.0003 +/- 0.0000 |
| `local_text_motif` | fast-only held-out NLL | 2.4771 +/- 0.0383 | 2.4758 +/- 0.0385 | +0.0013 +/- 0.0004 |

Accuracy did not improve. On `local_text_motif`, fast-only evaluation improved
slightly after memory-regularized training, but enabling the memory residual
made held-out NLL worse.

Diagnostics:

| Benchmark | Final-window advantage | Gate-update advantage | Gate | Active prototypes | Allocation rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `block_shift_markov` | -0.000511 +/- 0.000075 | -0.000259 +/- 0.000103 | 0.149955 +/- 0.000014 | 64.0 | 1.0 |
| `delayed_copy` | -0.001257 +/- 0.000085 | +0.000342 +/- 0.000100 | 0.149400 +/- 0.000037 | 64.0 | 1.0 |
| `sparse_kv_recall` | -0.000347 +/- 0.000030 | +0.000025 +/- 0.000027 | 0.149969 +/- 0.000011 | 64.0 | 1.0 |
| `local_text_motif` | -0.000813 +/- 0.000112 | -0.000167 +/- 0.000286 | 0.149475 +/- 0.000014 | 64.0 | 1.0 |

## Assessment

The current replay-capped post-FFN memory candidate does not generalize as a
positive external sequence result. Adding the stronger sparse associative recall
probe did not expose a useful memory advantage; it produced a displayed-zero
held-out NLL difference and a slightly negative fast-only result. The
non-Shakespeare text-like stream also stayed negative when the memory residual
was enabled.

The dominant failure is the cap/gate mechanism. The gate sits at the configured
cap on every benchmark despite negative measured final-window advantage. Replay
advantage is noisy enough to keep the gate open on `delayed_copy` and
`sparse_kv_recall`, while the memory-enabled path remains slightly worse.

Prototype budget is not the first explanation in this run. The sparse recall
probe uses 64 active keys and the default 64 prototypes, but active prototypes
are saturated and allocation rate remains 1.0. That points to replacement and
novelty dynamics, not merely too few slots. The mechanism is also mismatched to
the benchmark: a residual prototype basis is not yet behaving like an
addressable key/value memory, even on an intentionally small binding table.

Placement remains unresolved. The small `local_text_motif` fast-only gain with a
memory-enabled training path suggests the slow branch can regularize or perturb
the fast branch, but the post-FFN residual placement still damages direct
memory-enabled inference.

## Remaining Gaps

This is still bounded validation, not a final external protocol:

- run a real cached public non-Shakespeare corpus with checksum/provenance
  instead of the deterministic local text generator;
- report query-only metrics for sparse recall separately from interference
  examples;
- compare against a true addressable KV-memory baseline and an oracle binding
  baseline on `sparse_kv_recall`;
- sweep `gate_max`, gate objective, and per-prototype gates externally;
- retest pre-FFN KV placement on the same external probes;
- sweep prototype count, novelty threshold, and bandwidth to separate budget
  limits from replacement dynamics;
- run longer horizons after the gate no longer opens at cap under negative
  advantage.
