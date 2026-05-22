# Prototype-KV Attention Worker

Date: 2026-05-06

## Scope

Implemented an isolated Tiny Shakespeare prototype-KV attention experiment in
`examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_kv_attention.py`.
No shared core files were edited.

The adapter uses final attention queries as prototype memory keys. Prototype
centers update online every token by nearest-center EMA or allocation. Memory
value rows and a scalar retrieval gate are regular trainable parameters and get
cross-entropy gradients through the next-token loss.

Compared methods:

- `baseline_ffn_transformer`: existing tuned attention plus FFN baseline.
- `proto_kv_attention`: attention plus prototype-KV retrieval before readout.
- `hybrid_ffn_proto_kv`: attention plus prototype-KV retrieval before FFN.

## Runs

Smoke:

```bash
source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_kv_attention.py' --steps 50 --seeds 1 --eval-steps 64 --final-window 32 --output-dir 'outputs/step2_new_directions/proto_kv_attention_worker/smoke'
```

Required 2-seed run:

```bash
source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_kv_attention.py' --steps 800 --seeds 2 --eval-steps 256 --final-window 256 --output-dir 'outputs/step2_new_directions/proto_kv_attention_worker'
```

Multi-slot 800-step probe:

```bash
source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_kv_attention.py' --steps 800 --seeds 2 --eval-steps 256 --final-window 256 --memory-novelty-threshold 0.001 --memory-temperature 0.01 --gate-init 0.2 --output-dir 'outputs/step2_new_directions/proto_kv_attention_worker/threshold001_temp001_800'
```

Longer multi-slot probe:

```bash
source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_kv_attention.py' --steps 3000 --seeds 2 --eval-steps 512 --final-window 512 --memory-novelty-threshold 0.001 --memory-temperature 0.01 --gate-init 0.2 --output-dir 'outputs/step2_new_directions/proto_kv_attention_worker/threshold001_temp001_3000'
```

## Results

### Required 800-step run

| Method | Final-window NLL | Eval NLL | Eval perplexity |
|---|---:|---:|---:|
| Baseline FFN | 3.503801 | 3.548977 | 34.894415 |
| Proto-KV | 3.548880 | 3.581169 | 36.004332 |
| FFN + Proto-KV | 3.507629 | 3.519596 | 33.888149 |

At 800 steps, pure Proto-KV did not beat baseline. The hybrid beat baseline on
mean eval perplexity by 1.006266 points, but did not beat baseline
final-window NLL.

### Multi-slot 800-step probe

| Method | Final-window NLL | Eval NLL | Eval perplexity | Active memory slots |
|---|---:|---:|---:|---:|
| Baseline FFN | 3.503801 | 3.548977 | 34.894415 | n/a |
| Proto-KV | 3.550024 | 3.582137 | 36.037493 | 25.277344 |
| FFN + Proto-KV | 3.507505 | 3.519690 | 33.890888 | 11.691406 |

Lowering the novelty threshold made the memory genuinely multi-slot, but it did
not change the conclusion.

### Longer 3000-step multi-slot probe

| Method | Final-window NLL | Eval NLL | Eval perplexity | Active memory slots |
|---|---:|---:|---:|---:|
| Baseline FFN | 3.274380 | 3.210800 | 25.057534 | n/a |
| Proto-KV | 3.275581 | 3.224170 | 25.400635 | 58.939453 |
| FFN + Proto-KV | 3.278263 | 3.209853 | 24.978309 | 24.522461 |

At 3000 steps, FFN + Proto-KV again slightly beat baseline mean eval perplexity
by 0.079225 points and eval NLL by 0.000947, but lost mean final-window NLL by
0.003883. The effect is too small and seed-mixed to treat as a Step 2 win.

## Assessment

Prototype-KV attention is implemented and functional, including gradient-trained
memory values and online query-key center updates. The hybrid has a repeated
held-out eval-perplexity hint, but no robust final-window NLL improvement over
the tuned FFN baseline. Pure Proto-KV is weaker than the baseline in these
settings.

