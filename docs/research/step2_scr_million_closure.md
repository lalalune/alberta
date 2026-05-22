# Step 2 SCR Million-Step Closure

Status: closed.

The Slowly-Changing Regression hard blocker is closed for the Step 2
prediction-space portfolio under the Dohare public SCR parameter shape and the
required published-scale stream length. The final run used 3 seeds, 1,000,000
online examples per seed, the Dohare public SCR config (`m=20`, `f=15`,
`T=10000`, target hidden `100`, `beta=0.7`), no task-id leakage, and a final
window of 100,000 examples.

## Scope

The full router/knob portfolio was tried at smoke and 100k scale. The
published-scale run was intentionally narrowed to the best 100k causal router,
`slow_meta`, to avoid spending million-step time on every router variant. The
fair MLP comparator was not weakened: the final run still compares against the
same exact fair MLP grid, `mlp_h64`, `mlp_h128`, and `mlp_h64_64`.

The `slow_meta` router overrides were:

```json
{
  "router_policy": "meta",
  "hedge_eta": 1.0,
  "hedge_discount": 0.999,
  "router_decay": 0.005,
  "guard_tolerance": 0.0
}
```

The final run also used `dynamic_rewire_interval=2000`. This only affects the
dynamic sparse expert; the fair MLP comparator grid is unchanged.

## Commands

Focused tests and lint:

```bash
source .venv/bin/activate && pytest tests/test_step2_published_stressors.py tests/test_step2_scr_router_search.py -v
source .venv/bin/activate && ruff check "examples/The Alberta Plan/Step2/step2_published_stressors.py" "examples/The Alberta Plan/Step2/step2_scr_router_search.py" tests/test_step2_published_stressors.py tests/test_step2_scr_router_search.py
```

Smoke over all router variants:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" --smoke --router-variants all --output-dir outputs/step2_scr_million_smoke --result-prefix scr_million_smoke
```

100k one-seed Dohare-public-config probe over all router variants:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" --steps 100000 --n-seeds 1 --final-window 10000 --scr-preset dohare_paper --router-variants all --output-dir outputs/step2_scr_million_100k --result-prefix scr_million_100k
```

One-seed million-step confirmation:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" --million-scr --router-variants slow_meta --output-dir outputs/step2_scr_million_slow_meta --result-prefix scr_million_slow_meta --note-path docs/research/step2_scr_million_closure.md
```

Three-seed million-step closure run:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" --steps 1000000 --n-seeds 3 --final-window 100000 --scr-preset dohare_paper --dynamic-rewire-interval 2000 --router-variants slow_meta --output-dir outputs/step2_scr_million_slow_meta_3seed --result-prefix scr_million_slow_meta_3seed --note-path docs/research/step2_scr_million_closure.md
```

## Results

| Run | Seeds | Steps | Best variant | Router final-window MSE | Best fair MLP final-window MSE | Diff vs best fair MLP | Wins/Losses/Ties | Published-scale closed | Runtime |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| Smoke | 1 | 120 | `convex_reference` | 0.06569078 | 0.07182548 | +0.00613470 | 1/0/0 | false | 51.27 s |
| 100k probe | 1 | 100,000 | `slow_meta` | 0.00096248 | 0.00114132 | +0.00017884 | 1/0/0 | false | 184.77 s |
| 1e6 confirmation | 1 | 1,000,000 | `slow_meta` | 0.00070416 | 0.00075250 | +0.00004834 | 1/0/0 | true | 350.91 s |
| 1e6 closure | 3 | 1,000,000 | `slow_meta` | 0.00079090 +/- 0.00005410 | 0.00085247 +/- 0.00006853 | +0.00006156 +/- 0.00001598 | 3/0/0 | true | 484.55 s |

Final 3-seed per-seed diffs, positive favors router:

| Seed | Router final-window MSE | Best fair MLP final-window MSE | Diff |
|---:|---:|---:|---:|
| 0 | 0.00070416 | 0.00075250 | +0.00004834 |
| 1 | 0.00089030 | 0.00098366 | +0.00009337 |
| 2 | 0.00077826 | 0.00082123 | +0.00004298 |

The best fair MLP in all three final seeds was `mlp_h64_64`.

## Protocol Gates

Final 3-seed run status flags:

```json
{
  "uses_dohare_public_scr_config": true,
  "scr_steps": 1000000,
  "scr_min_published_steps": 1000000,
  "scr_meets_published_step_count": true,
  "scr_task_id_provided_to_learner": false,
  "scr_uses_online_stream_only": true,
  "matches_dohare_public_scr_protocol": true,
  "published_scale_scr_closed": true
}
```

The shared status gate now prevents published-scale SCR status from becoming
true unless the SCR config matches the public Dohare parameter shape, the run
uses at least 1,000,000 online examples, the learner receives no task id, and
the stream is online-only with a fixed target network.

## Conclusion

Closed. The 3-seed million-step run clears the best fair MLP comparator on every
seed with the published-scale SCR protocol gate true. This should be reported
as a narrowed published-scale SCR closure for the `slow_meta` causal router, not
as evidence that every router variant was run at paper scale.
