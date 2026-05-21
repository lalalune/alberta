# Transformer Fast/Slow Grind Conclusion

Date: 2026-05-06

## Scope

This pass tested the requested transformer directions with independent workers
and an integrated local runner:

- learned residual gates;
- gate inputs from novelty, uncertainty, and recent token loss;
- decoupled fast/slow learning rates;
- zero/no-reset/meta-EMA prototype value reset policies;
- prototype memory as a KV-like attention adapter; and
- longer token budgets up to 5000 online steps.

The main target was whether a slow+fast transformer memory block can beat a
tuned plain FFN transformer on Tiny Shakespeare.

## Implemented Experiments

New scripts:

- `step2_tiny_shakespeare_gated_memory_transformer.py`: integrated local
  runner with per-channel learned gate, novelty/entropy/loss diagnostics,
  decoupled fast/slow rates, meta-EMA reset, and post-FFN plus pre-FFN KV
  placements.
- `step2_tiny_shakespeare_gated_proto_transformer.py`: worker learned gate
  with scalar/channel/group modes and diagnostic gate inputs.
- `step2_tiny_shakespeare_proto_kv_attention.py`: worker prototype-KV adapter
  inside attention.
- `step2_tiny_shakespeare_fast_slow_transformer.py`: worker simpler fast FFN
  plus slow prototype path with learned gate.

Reports:

- `gated_proto_transformer_worker.md`
- `proto_kv_attention_worker.md`
- `fast_slow_transformer_worker.md`
- `long_budget_transformer_worker.md`
- `gate_resource_manager_critique.md`

One resumed meta-reset worker did not return before timeout and was closed. Its
scope was covered locally by the integrated runner, which ran `meta_ema`,
`zero`, and `none` reset modes.

## Best Positive Results

### Required 800-Step Horizon

Several slow+fast variants beat the tuned FFN on one metric at 800 steps.

| Candidate | Final-window NLL vs FFN | Eval perplexity vs FFN | Assessment |
| --- | ---: | ---: | --- |
| Simple fast/slow worker | 3.502008 vs 3.503801 | 34.835220 vs 34.894415 | Beats mean on both, small and not per-seed robust on eval PPL |
| Gated prototype worker | 3.504763 vs 3.503801 | 33.564642 vs 34.894415 | Eval win, final NLL loss |
| Prototype-KV hybrid | 3.507629 vs 3.503801 | 33.888149 vs 34.894415 | Eval win, final NLL loss |
| Local gated memory, bias -2 | 3.296752 vs 3.256544 | 29.853861 vs 30.161136 | Eval win, final NLL loss |

The 800-step signal is real but fragile. The slow path behaves like a useful
short-horizon regularizer/adaptation path, not yet like a superior general
learner.

### Longer Budgets

Longer budgets falsified the strongest 800-step claims.

| Candidate | Horizon | Final-window NLL | Eval perplexity | Assessment |
| --- | ---: | ---: | ---: | --- |
| Simple fast/slow worker | 2000 | 3.330712 vs FFN 3.323867 | 26.928391 vs FFN 26.681372 | loses both |
| Gated prototype worker | 3000 | 3.297891 vs FFN 3.274380 | 25.412567 vs FFN 25.057534 | loses both |
| Prototype-KV hybrid | 3000 | 3.278263 vs FFN 3.274380 | 24.978309 vs FFN 25.057534 | eval win only |
| Prototype basis | 5000 | 2.7778 vs FFN 2.7866 | 19.9615 vs FFN 19.3734 | final NLL win only |
| Local gated memory, bias -2 | 3000 | 2.945358 vs FFN 2.952114 | 22.215795 vs FFN 21.042216 | final NLL win only |

The repeated pattern is clear: slow memory can improve online/final-window
prediction, but held-out eval perplexity still favors the tuned FFN.

## Mechanistic Findings

What works:

- Fast + slow is better than pure slow memory.
- Learned gates are necessary; ungated hybrids usually overfit or lag.
- KV-style placement inside attention is a credible interface and gives small
  held-out eval improvements.
- Decoupled slow learning rates matter; slow LR around `0.15-0.20` is often
  needed for online NLL gains.
- Tight novelty thresholds are needed in transformer hidden space; loose
  thresholds collapse to one prototype.

What fails:

- No variant beats the best tuned FFN on both final-window NLL and held-out
  eval perplexity at longer budgets.
- Higher slow-path learning rates improve online NLL but hurt held-out
  perplexity.
- Meta-EMA, zero reset, and no reset did not change the conclusion in the local
  3000-step run.
- UPGD-FFN perturbations were effectively neutral and slower.
- Compute is worse than the baseline for all memory variants.

## Conclusion

This is not a solved transformer replacement.

The strongest defensible conclusion is:

> Slow+fast memory blocks are useful online adaptation mechanisms and can beat a
> tuned FFN on short-horizon or online/final-window metrics, but the current
> implementations do not robustly beat a tuned FFN transformer on held-out
> next-token prediction.

The right next research direction is not more manual prototype tuning. The gap
is objective mismatch: the slow path optimizes immediate online fit and keeps
hurting held-out perplexity. A serious next candidate needs an explicit
generalization-aware resource objective, for example a learned gate trained
against delayed validation/replay loss or predictive compression rather than
only the current token loss.

## Follow-Up: Advantage-Gated Memory

Follow-up experiments in
`advantage_memory_transformer_followup.md` tested this objective-mismatch
hypothesis with a measured fast-vs-slow advantage gate.

The best confirmed result is narrow but real: scalar advantage-gated slow
memory improves the fast-only deployed transformer over 10 seeds at both 3000
and 5000 online steps. The memory residual itself still hurts held-out
perplexity, so this is best understood as a training-time auxiliary slow path,
not a solved inference-time memory block.

Per-prototype utility gates produced the first small memory-enabled wins over
the FFN on both final-window and held-out metrics, but the winning placement
changed between 3000 and 5000 steps. That makes it promising, not canonical.

### Replay-Capped Update

The next pass implemented the hard target directly: delayed replay advantage
plus an explicit resource cap. The current leading config is:

- `gate_objective=replay`;
- `replay_size=128`;
- `gate_lr=0.5`;
- `gate_l2=0.1`;
- `gate_max=0.15`;
- post-FFN memory placement.

Over 10 seeds, this single config beat the tuned FFN on both final-window NLL
and held-out perplexity at 3000, 5000, and 10000 online steps. The wins are
small at 3000/5000 and clearer by 10000:

| Horizon | FFN final NLL | Post replay final NLL | FFN eval PPL | Post replay eval PPL |
| ---: | ---: | ---: | ---: | ---: |
| 3000 | 2.9025 | 2.9023 | 22.4428 | 22.4360 |
| 5000 | 2.8686 | 2.8683 | 20.6829 | 20.6484 |
| 10000 | 2.7227 | 2.7217 | 21.9730 | 21.5362 |

This changes the conclusion: replay-capped slow memory is now a credible
transformer Step 2 candidate. It is still not a complete universal learner
claim because the budget cap is static, throughput is roughly half the FFN
baseline in the research runner, and evidence is still limited to Tiny
Shakespeare.

## Validation

Passed:

- `python -m py_compile` on all new transformer scripts.
- focused `mypy --follow-imports=skip` on all new transformer scripts.
- `pytest tests/test_prototype_basis.py -q`: 6 passed.

Repo-wide note:

- `ruff check .` currently fails on unrelated dirty-worktree files
  `src/alberta_framework/__init__.py` and `src/alberta_framework/core/upgd_memory.py`.
  Targeted lint for the new scripts passed.
