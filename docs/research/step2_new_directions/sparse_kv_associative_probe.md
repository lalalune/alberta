# Sparse KV Associative Probe

Date: 2026-05-07.

Purpose: isolate the remaining external benchmark failure. The residual
prototype-memory transformer can close its prediction gate, but it does not act
like an addressable key/value memory on `sparse_kv_recall`. This probe asks
whether a simple online associative learner with learned feature utility can
solve that same stream.

## Runner

New runner:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python benchmarks/step2_sparse_kv_associative_probe.py \
  --steps 900 \
  --seeds 10 \
  --eval-steps 256 \
  --final-window 256 \
  --output-dir outputs/step2_new_directions/sparse_kv_associative_probe_900_10seed_norm
```

Focused full-pair selector ablation:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python benchmarks/step2_sparse_kv_associative_probe.py \
  --steps 900 \
  --seeds 3 \
  --eval-steps 256 \
  --final-window 256 \
  --no-ffn \
  --output-dir outputs/step2_new_directions/sparse_kv_associative_probe_full_selective_3seed
```

Artifacts:

- `outputs/step2_new_directions/sparse_kv_associative_probe_900_10seed_norm/results.json`
- `outputs/step2_new_directions/sparse_kv_associative_probe_900_10seed_norm/SUMMARY.md`
- `outputs/step2_new_directions/sparse_kv_associative_probe_900_30seed_focus/results.json`
- `outputs/step2_new_directions/sparse_kv_associative_probe_900_30seed_focus/SUMMARY.md`
- `outputs/step2_new_directions/associative_external_all_900_3seed/results.json`
- `outputs/step2_new_directions/associative_external_all_900_3seed/SUMMARY.md`
- `outputs/step2_new_directions/associative_external_hybrid_900_10seed/results.json`
- `outputs/step2_new_directions/associative_external_hybrid_900_10seed/SUMMARY.md`
- `outputs/step2_new_directions/sparse_kv_associative_probe_full_selective_3seed/results.json`
- `outputs/step2_new_directions/sparse_kv_associative_probe_full_selective_3seed/SUMMARY.md`

## Mechanism

The associative probe is deliberately small and causal:

- predict before writing the current example;
- maintain sparse feature-to-label rows;
- update rows with recency-biased exponential writes;
- learn per-feature utility online from feature-level loss advantage;
- compare token-only, recent suffix-pair, and full position-pair feature
  families;
- normalize feature logits as an explicit calibration ablation.

The probe is not a promoted Step 2 learner. It is a mechanism test for the
unresolved external sparse key/value recall failure.

## 30-Seed Focused Result

The focused confirmatory probe ran only FFN, token-only memory, and the two
normalized suffix-pair memories:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python benchmarks/step2_sparse_kv_associative_probe.py \
  --steps 900 \
  --seeds 30 \
  --eval-steps 256 \
  --final-window 256 \
  --variants token_utility suffix_pair_utility_norm4 suffix_pair_utility_norm8 \
  --output-dir outputs/step2_new_directions/sparse_kv_associative_probe_900_30seed_focus
```

Lower NLL is better.

| Method | Eval NLL | Eval accuracy | Diff vs FFN eval NLL | Paired wins | Steps/s | Feature touches/step |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_ffn_transformer` | `2.7600 +/- 0.0101` | `0.1094 +/- 0.0044` | n/a | n/a | `1629.3` | n/a |
| `token_utility` | `2.7131 +/- 0.0134` | `0.2272 +/- 0.0065` | `+0.0469 +/- 0.0108` | `24/30` | `3272.0` | `32.0` |
| `suffix_pair_utility_norm4` | `1.4830 +/- 0.0329` | `0.6400 +/- 0.0132` | `+1.2770 +/- 0.0307` | `30/30` | `3685.6` | `28.0` |
| `suffix_pair_utility_norm8` | `1.2213 +/- 0.0380` | `0.5966 +/- 0.0123` | `+1.5387 +/- 0.0362` | `30/30` | `3461.4` | `28.0` |

This is now a confirmed mechanism lead, not a one-off smoke result. The
suffix-pair associative learner is both substantially better and faster than
the FFN baseline on held-out sparse key/value recall.

## External Breadth Smoke

The most important follow-up is the single hybrid feature-family learner. It
uses one associative substrate with both positional-token and local suffix-pair
features under the same learned utility weighting:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python benchmarks/step2_sparse_kv_associative_probe.py \
  --benchmarks all \
  --steps 900 \
  --seeds 10 \
  --eval-steps 256 \
  --final-window 256 \
  --variants hybrid_token_suffix_norm4 hybrid_token_suffix_norm8 \
  --output-dir outputs/step2_new_directions/associative_external_hybrid_900_10seed
```

Paired eval-NLL differences below are baseline FFN minus row method, so positive
values favor the hybrid associative learner.

| Benchmark | Hybrid | Eval-NLL diff vs FFN | Paired wins | Eval accuracy diff vs FFN |
|---|---|---:|---:|---:|
| `block_shift_markov` | `hybrid_token_suffix_norm4` | `+0.4230 +/- 0.0821` | `10/10` | `+0.2023` |
| `delayed_copy` | `hybrid_token_suffix_norm4` | `+0.1396 +/- 0.0174` | `10/10` | `+0.1168` |
| `sparse_kv_recall` | `hybrid_token_suffix_norm4` | `+1.0237 +/- 0.0223` | `10/10` | `+0.5980` |
| `local_text_motif` | `hybrid_token_suffix_norm4` | `+1.5334 +/- 0.0503` | `10/10` | `+0.4215` |

The `norm8` hybrid also wins every delayed-copy, sparse-KV, and local-text seed,
but loses one block-shift seed. The safer leading hybrid is therefore
`hybrid_token_suffix_norm4`.

This is the first external sequence result in this pass that looks like a
single simple learner rather than post hoc feature-family selection. It is
still not a full Step 2 solution, but it is a credible next candidate:
fast/slow memory, learned feature utility, local conjunctions, and calibrated
evidence in one online mechanism.

For context, the fixed-family breadth smoke was:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python benchmarks/step2_sparse_kv_associative_probe.py \
  --benchmarks all \
  --steps 900 \
  --seeds 3 \
  --eval-steps 256 \
  --final-window 256 \
  --variants token_utility suffix_pair_utility_norm4 suffix_pair_utility_norm8 \
  --output-dir outputs/step2_new_directions/associative_external_all_900_3seed
```

Paired eval-NLL differences below are baseline FFN minus row method, so positive
values favor the associative learner.

| Benchmark | Best associative variant | Eval-NLL diff vs FFN | Paired wins | Readout implication |
|---|---|---:|---:|---|
| `block_shift_markov` | `suffix_pair_utility_norm4` | `+0.6246 +/- 0.2605` | `3/3` | local pair features help recurring grammar shifts |
| `delayed_copy` | `token_utility` | `+0.5656 +/- 0.0275` | `3/3` | pair features hurt; simple positional token memory is enough |
| `sparse_kv_recall` | `suffix_pair_utility_norm8` | `+1.4143 +/- 0.2425` | `3/3` | key/value binding needs local conjunctions |
| `local_text_motif` | `suffix_pair_utility_norm4` | `+1.4103 +/- 0.0561` | `3/3` | local pair features capture repeated motif structure |

The fixed-family smoke changed the design target. Fixed suffix-pair memory is
too static because it loses delayed-copy. Fixed token memory is too weak because
it barely moves sparse-KV. The hybrid run above shows that one associative
substrate can carry both feature families without losing the delayed-copy
benefit.

## 10-Seed Broad Ablation

Lower NLL is better.

| Method | Eval NLL | Eval accuracy | Steps/s | Feature touches/step | Final rows |
|---|---:|---:|---:|---:|---:|
| `baseline_ffn_transformer` | `2.7780 +/- 0.0177` | `0.1039 +/- 0.0079` | `1569.8` | n/a | n/a |
| `token_utility` | `2.7394 +/- 0.0229` | `0.2137 +/- 0.0125` | `3012.9` | `32.0` | `533.3` |
| `suffix_pair_utility_norm4` | `1.5014 +/- 0.0540` | `0.6164 +/- 0.0293` | `3125.4` | `28.0` | `1933.4` |
| `suffix_pair_utility_norm8` | `1.2399 +/- 0.0828` | `0.5863 +/- 0.0273` | `3241.0` | `28.0` | `1933.4` |
| `full_pair_utility_norm4` | `3.4138 +/- 0.0076` | `0.2004 +/- 0.0108` | `289.5` | `496.0` | `104889.6` |

The prior residual prototype-memory rescue runs were effectively neutral on
`sparse_kv_recall`: FFN held-out NLL around `2.819`, static replay around
`2.819`, and rescue around `2.820`. The suffix-pair associative probe is
therefore not a marginal improvement over that failure mode; it is a different
mechanism class that actually reads the final one-shot binding table.

## Calibration Ablation

The unnormalized suffix-pair variant had good accuracy but terrible NLL:

| Variant | Eval NLL | Eval accuracy |
|---|---:|---:|
| `suffix_pair_utility_sum` | `13.1000 +/- 1.7945` | `0.6223 +/- 0.0297` |
| `suffix_pair_utility_norm4` | `1.5014 +/- 0.0540` | `0.6164 +/- 0.0293` |
| `suffix_pair_utility_norm8` | `1.2399 +/- 0.0828` | `0.5863 +/- 0.0273` |

This matters for the universal-learner direction: the readout needs an explicit
normalization/calibration contract. Raw additive evidence can be accurate but
unusable under log-loss.

## Full-Pair Ablation

The full all-position-pair family is the obvious attempt to remove the static
recent-suffix assumption. It is not competitive yet:

| Variant | 3-seed eval NLL | 3-seed eval accuracy | Steps/s | Feature touches/step |
|---|---:|---:|---:|---:|
| `full_pair_utility_norm4` | about `3.43` | about `0.20` | about `500` | `496` |
| `full_pair_selective_norm16` | about `2.64` | about `0.24` | about `490` | `496` |

More aggressive learned utility helps, but it is still slower and worse than
the compact suffix-pair block. The immediate candidate should not enumerate all
pairs. It should learn a compact read/write scope around recent token structure
and only expand when utility warrants the extra computation.

## Interpretation

What this proves:

- The sparse-KV external failure is not unsolvable and is not just a data issue.
- Explicit fast associative binding can beat the FFN baseline strongly on the
  held-out final-binding query distribution.
- Token-only memory is insufficient; useful recursive features need at least
  local conjunctions.
- Calibration is as important as recall accuracy under NLL.
- Full brute-force feature conjunctions are the wrong computational path.

What this does not prove:

- It does not rescue the current transformer-memory candidate.
- It does not meet the full Step 2 bar by itself, because the suffix window and
  feature families are still configured rather than fully learned.
- It does not address Tiny Shakespeare or OPMNIST yet.
- The final-window online NLL is not expected to be low on this generator,
  because the final binding for each active key is first presented once and the
  learner must predict before writing it. The relevant success criterion here is
  held-out query recall after the one-shot write.

## Next Candidate

The next Step 2 learner should combine the earlier fast/slow thesis with this
new mechanism:

- fast transformer or UPGD feature state for general sequence modeling;
- slow associative key/value rows for explicit binding and retention;
- learned read/write scope that starts with positional-token and recent local
  conjunction features, then expands under utility pressure;
- normalized evidence accumulation, not raw additive memory logits;
- separate write gate, read gate, and calibration gate;
- recency overwrite for rebinding tasks;
- resource manager that pays for higher-order features only when their
  feature-level advantage is positive.

This is the clearest productive pivot from the failed residual-prototype
candidate: keep the fast/slow architecture, but replace residual prototype
values with a calibrated associative binding substrate.
