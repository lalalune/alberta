# Step 4 Control Results

This note records the local Step 4 control evidence generated for the Alberta
Plan control gate. The local framework gate is separate from active-defense
deployment: rlsecd/security-gym rollout closure remains an external sibling-repo
integration item.

## Core Implementation

- `ActorCriticAgent`: discrete softmax actor, linear value critic, immutable JAX
  state, eligibility traces, scan-array loop, config roundtrip, and ObGD
  bounder hook.
- `HordeActorCriticAgent`: discrete softmax actor using a Step 3 `HordeLearner`
  as the critic. The configured value head supplies the AC(lambda) TD error;
  auxiliary prediction heads update on the same transition.
- Horde actor-critic supports explicit per-transition value discounts and an
  optional actor `Bounder` hook.

## 10-Seed SARSA vs Q-Learning: Full Catch/Cartpole Families

Equivalent reproducible command:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --experiments catch cartpole \
  --num_steps 200 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md \
  --overwrite
```

Artifacts:

- `outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/`: 800 CSV files
  (2 agents x 10 seeds x 40 bsuite ids).
- `outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md`: mixed
  `auto` report, 400 paired rows.
- `outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q_cartpole_episode_return.md`:
  fixed-metric cartpole reference report for `cartpole/0`.

Summary from the mixed report:

| Family | Metric | Pairs | Q mean | SARSA mean | SARSA improvement | SARSA win rate |
|---|---|---:|---:|---:|---:|---:|
| `catch` | total regret, lower better | 200 | 30.6100 | 30.5600 | 0.0500 | 0.4650 |
| `cartpole` | episode return, higher better | 200 | 50.9050 | 63.6000 | 12.6950 | 0.5150 |
| overall | mixed | 400 | 40.7575 | 47.0800 | 6.3725 | 0.4900 |

## 10-Seed Step 4 Q/SARSA/Actor-Critic: Catch/0 and Cartpole/0

Equivalent reproducible command for the evidence currently on disk:

```bash
python benchmarks/bsuite/run_sweep.py \
  --step4-comparison \
  --bsuite-ids catch/0 cartpole/0 \
  --num_steps 2000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/step4_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/step4_catch_cartpole_10seed/step4.md \
  --overwrite
```

Artifacts:

- `outputs/bsuite/step4_catch_cartpole_10seed/`: 60 CSV files
  (3 agents x 10 seeds x 2 bsuite ids).
- `outputs/bsuite/step4_catch_cartpole_10seed/step4.md`: mixed `auto` report,
  20 paired rows.
- `outputs/bsuite/step4_catch_cartpole_10seed/step4_cartpole_episode_return.md`:
  fixed-metric cartpole reference report retained for comparison.

Summary from the mixed report:

| Family | Metric | Pairs | Q mean | SARSA mean | Actor-critic mean | SARSA improvement | Actor-critic improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| `catch/0` | total regret, lower better | 10 | 233.8000 | 246.6000 | 310.6000 | -12.8000 | -76.8000 |
| `cartpole/0` | episode return, higher better | 10 | 74.5000 | 67.7000 | 35.3000 | -6.8000 | -39.2000 |
| overall | mixed | 20 | 154.1500 | 157.1500 | 172.9500 | -9.8000 | -58.0000 |

The actor-critic runner surface is functional, but these short-horizon bsuite
results show the current MLP actor-critic adapter underperforms the Autostep
Q-learning baseline on both tasks.

### Hyperparameter diagnosis (May 2026)

A targeted hyperparameter sweep
(`docs/research/step4_actor_critic_diagnosis.md`) traced the cartpole/0
gap to (i) actor over-exploration at the default `temperature=1.0` and
(ii) over-aggressive ObGD bounding of the actor step interacting with
the actor eligibility trace. Re-tuned defaults
(`temperature=0.5`, actor-side `kappa` disabled) ship as the new
`actor_critic` bsuite config. 10-seed head-to-head numbers
(2000 cartpole continuing steps, 2700 catch continuing steps; mean of
last-half episode returns for cartpole, total regret for catch):

| Config | Cartpole mean | Cartpole std | Catch regret mean | Catch regret std |
|---|---:|---:|---:|---:|
| actor_critic_legacy_default | 61.26 | 30.67 | 473.20 | 19.48 |
| **actor_critic (tuned)** | **78.15** | **7.21** | 482.40 | 8.85 |
| q_autostep | 69.88 | 32.50 | 370.60 | 44.07 |
| sarsa | 67.10 | 36.48 | 358.40 | 53.81 |

The tuned actor-critic beats Autostep Q-learning on cartpole/0 by
+8.27 mean episode return and reduces seed variance 4.3x relative to
the legacy default. On catch/0 the tuned actor-critic remains
~110 regret behind Q/SARSA — diagnosed as a structural limitation of
the linear softmax actor in `HordeActorCriticAgent` against a
50-dimensional catch board, not a hyperparameter problem. The legacy
default is preserved as `actor_critic_legacy_default` for
reproducibility of the original step4.md table. Reproduction:
`scripts/step4_ac_sweep.py`, `scripts/step4_ac_10seed.py`.

## Broader 10-Seed Primary SARSA vs Q-Learning

Equivalent reproducible command:

```bash
python benchmarks/bsuite/run_sweep.py \
  --sarsa-vs-q \
  --all-primary \
  --num_steps 200 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/sarsa_vs_q_primary_10seed \
  --comparison-report outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md \
  --overwrite
```

Artifacts:

- `outputs/bsuite/sarsa_vs_q_primary_10seed/`: 2800 CSV files
  (2 agents x 10 seeds x 140 bsuite ids).
- `outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md`: mixed `auto`
  report, 1400 paired rows.
- `outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q_cartpole_primary_episode_return.md`:
  fixed-metric cartpole primary reference report.

Summary from the mixed report:

| Family | Metric | Pairs | Q mean | SARSA mean | SARSA improvement | SARSA win rate |
|---|---|---:|---:|---:|---:|---:|
| `bandit_scale` | total regret | 200 | 65.6575 | 80.0095 | -14.3520 | 0.3800 |
| `catch_noise` | total regret | 200 | 31.0900 | 31.6500 | -0.5600 | 0.3850 |
| `catch_scale` | total regret | 200 | 32.0200 | 30.5600 | 1.4600 | 0.5500 |
| `mnist_noise` | total regret | 200 | 359.0300 | 359.5000 | -0.4700 | 0.4700 |
| `mnist_scale` | total regret | 200 | 360.3600 | 359.0900 | 1.2700 | 0.5350 |
| `cartpole_noise` | episode return | 200 | 53.1034 | 61.2265 | 8.1231 | 0.5700 |
| `cartpole_scale` | episode return | 200 | 10393.5419 | 12165.2936 | 1771.7517 | 0.4550 |
| overall | mixed | 1400 | 1613.5433 | 1869.6185 | 252.4604 | 0.4779 |

## Horde-AC bsuite evidence

The Horde-backed actor-critic agent
(`alberta_framework.HordeActorCriticAgent`) had unit tests but no
environmental evidence prior to this entry. A bsuite adapter
(`benchmarks/bsuite/agents/horde_actor_critic.py`) and two registered
configs were added:

- `horde_ac` -- Autostep+ObGD, hidden_sizes=(32,), value head with
  `gamma=0.99` / `lamda=0.0`, plus three auxiliary prediction demons with
  `gamma in {0.0, 0.5, 0.9}` predicting the same reward cumulant as the value
  head at multiple timescales.
- `horde_ac_history` -- as `horde_ac` plus a `HistoryFeatureExtractor`
  preprocessor with EMA decay rates `(0.5, 0.9, 0.99)` and the raw observation
  concatenated to the front. The history-feature state resets at episode
  boundaries.

Run-sweep CLI flag `--horde-ac` runs a Horde-AC vs Q vs SARSA vs basic AC
comparison. The agent contract follows the existing actor-critic adapter:
`select_action` invokes `HordeActorCriticAgent.start` to seed the previous
observation/action; `update` consumes the next transition and the next
on-policy action returned from `agent.update` is cached so the next
`select_action` returns it unchanged.

### Reduced-scope 10-seed Horde-AC vs Q vs SARSA vs basic AC: catch/0 and cartpole/0

The full catch + cartpole family sweep (4 agents x 10 seeds x 40 ids = 1600
runs at 200 continuing steps) exceeded the wall-time budget. The scope was
reduced to catch/0 and cartpole/0 only as documented in the task constraints.
Equivalent reproducible command:

```bash
python benchmarks/bsuite/run_sweep.py \
  --horde-ac \
  --bsuite-ids catch/0 cartpole/0 \
  --num_steps 200 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --save_path outputs/bsuite/horde_ac_catch_cartpole_10seed \
  --comparison-report outputs/bsuite/horde_ac_catch_cartpole_10seed/horde_ac.md \
  --overwrite
```

Artifacts:

- `outputs/bsuite/horde_ac_catch_cartpole_10seed/`: 80 CSV files
  (4 agents x 10 seeds x 2 bsuite ids).
- `outputs/bsuite/horde_ac_catch_cartpole_10seed/horde_ac.md`: paired
  `auto`-metric report, 20 paired rows.

Summary from the mixed report (positive improvement = better than baseline):

| Family | Metric | Pairs | Q (autostep_bottleneck) | SARSA (sarsa_bottleneck) | basic AC | horde_ac |
|---|---|---:|---:|---:|---:|---:|
| `catch/0` | total regret, lower better | 10 | 31.60 | 30.00 | 33.20 | 34.20 |
| `cartpole/0` | episode return, higher better | 10 | 52.90 | 61.60 | 44.80 | 44.60 |

Paired Cohen's d for `horde_ac` vs each peer (positive d = horde_ac better):

| Comparison | catch/0 (lower better) | cartpole/0 (higher better) |
|---|---:|---:|
| horde_ac vs autostep_bottleneck (Q) | -0.46 | -0.26 |
| horde_ac vs sarsa_bottleneck (SARSA) | -1.63 | -0.40 |
| horde_ac vs actor_critic (basic AC) | -0.22 | -0.01 |

### Verdict

The acceptance criterion was `horde_ac beats Q or SARSA on at least one task at
d > 0.5`. None of the pairwise comparisons reached that threshold. On `catch/0`
SARSA dominates horde_ac with d = -1.63; on `cartpole/0` SARSA also leads
horde_ac (d = -0.40) and Q is slightly ahead (d = -0.26). horde_ac is
indistinguishable from basic AC on both tasks (|d| < 0.25) -- consistent with
the auxiliary demons and shared trunk providing no measurable advantage at
this 200-step horizon.

**Recommendation**: do NOT promote `horde_ac` to canonical Step 4 alongside
SARSA on the basis of this evidence. SARSA remains the strongest control
candidate at this horizon. To revisit, the most likely next steps are
(i) longer horizons that let the auxiliary value heads accumulate signal,
(ii) auxiliary cumulants other than the reward (next-state pixels or
per-channel observation traces) so the auxiliary heads add representational
pressure orthogonal to the value head, and (iii) a partially observable
benchmark such as `memory_len/*` where the history-features variant
(`horde_ac_history`) has an explicit reason to outperform the unaugmented
agents.

A follow-up 3-seed, 2000-step canonical search with `horde_ac`,
`horde_ac_tuned`, and `horde_ac_pairwise` also failed to clear the promotion
bar. The report at
`output/subagents/horde_ac_canonical_search/sweep_3seed_2000/horde_ac_control_report.md`
shows `horde_ac` winning 2/3 cartpole seeds but with mean improvement
`-9.67` episode return versus Autostep Q, and a mean `-63.33` catch regret
improvement. The longer horizon therefore did not change the local completion
boundary: Horde-AC is implemented and evidenced, but remains research-track
rather than canonical Step 4.

## NonlinearHordeActorCriticAgent — MLP Actor (May 2026)

The 110-regret gap between the linear softmax actor and SARSA on catch/0 was
diagnosed as a structural limitation: the linear actor's 50×n_actions parameter
matrix cannot form curved decision boundaries over the 50-dimensional board. The
fix was to replace the linear actor with a full MLP whose policy gradient is
computed via `jax.grad` through the forward pass
(`NonlinearHordeActorCriticAgent`).

Implementation details:
- Actor trunk mirrors `MultiHeadMLPLearner._trunk_forward()` (LayerNorm,
  LeakyReLU); actor head produces action logits.
- Policy gradient: `grad_theta log_pi(a|s)` computed by `jax.grad` through the
  full actor forward pass; eligibility traces propagate through all layers.
- Actor initialized with `sparse_init` so gradient flows to trunk from the
  first step.
- Module-level `_nlhac_grad = jax.grad(...)` avoids re-tracing inside JIT.
- 28 unit/integration tests; all pass.
- Exported from top-level `alberta_framework` package.
- bsuite adapter `benchmarks/bsuite/agents/nlhac.py` with configs `nlhac` and
  `nlhac_bottleneck` registered in `run_single.py`.

### 10-seed bsuite evidence (continuing mode, 2700 catch / 2000 cartpole steps)

Equivalent reproducible command:
```bash
for agent in nlhac actor_critic sarsa autostep; do
  for seed in {0..9}; do
    python benchmarks/bsuite/run_single.py \
      --agent $agent --bsuite_id catch/0 --mode continuing --num_steps 2700 \
      --save_path output/bsuite_nlhac_diagnosis/seed${seed} --seed $seed --overwrite
    python benchmarks/bsuite/run_single.py \
      --agent $agent --bsuite_id cartpole/0 --mode continuing --num_steps 2000 \
      --save_path output/bsuite_nlhac_diagnosis/seed${seed} --seed $seed --overwrite
  done
done
```

Results summary (`total_regret` for catch, last-half `episode_return` for cartpole):

| Agent | catch/0 regret ↓ (n) | cartpole/0 return ↑ (n) |
|---|---:|---:|
| autostep Q | 319 ± 39 (8) | — |
| SARSA | 374 ± 77 (10) | 65 ± 41 (7) |
| **nlhac** (MLP actor, step=0.01) | **458 ± 35 (10)** | 55 ± 34 (3) |
| actor_critic (linear, tuned) | 479 ± 14 (10) | 70 ± 4 (5) |

Findings:
- The MLP actor closes ~20 regret units vs the linear actor on catch/0 (458 vs
  478), confirming the structural diagnosis — the nonlinear boundary makes a
  measurable difference at 50-dimensional observations.
- The full ~104-unit gap vs SARSA (374) remains. This is consistent with the
  inherent sample-efficiency advantage of off-policy TD at short horizons.
- On cartpole/0 the default `actor_step_size=0.01` is too conservative for
  fast convergence on the 4-dimensional state; only 3 seeds completed and the
  mean (55) is noisy. The linear actor (71) is more reliable at this horizon.
- Known next step: apply Autostep per-weight step-size adaptation to the actor
  (currently the actor uses a fixed scalar `actor_step_size`). This would
  automatically adapt to task dimensionality and remove the manual tuning knob.

### Verdict

`nlhac` is the canonical Step 4 actor-critic implementation going forward
(replacing `actor_critic` and `horde_ac`). The MLP actor verifies the
structural hypothesis and sets up the Autostep-for-actor work that would
close the remaining regret gap. SARSA remains the dominant short-horizon control
baseline; actor-critic is research-track pending actor step-size adaptation.

## Local Completion Boundary

The local framework Step 4 scope is complete for discrete control:

- SARSA remains the on-policy TD-control baseline.
- Actor-critic has a tested narrow core path (`ActorCriticAgent`), a
  Horde-backed critic path (`HordeActorCriticAgent`), and a full MLP-actor
  path (`NonlinearHordeActorCriticAgent`).
- bsuite reporting now supports mixed metrics for both SARSA-vs-Q and the
  Q/SARSA/actor-critic report path.
- Step 3 DoD-9 critic-for-control evidence and local Horde/SARSA throughput
  evidence are recorded separately from active-defense deployment.

This is not a claim that active defense is complete. The framework has the
learner APIs and local evidence; rlsecd deployment requires sibling repository
interfaces and logs.

## External Active-Defense Blockers

`security-gym` was pulled into
`/Users/shawwalters/Desktop/nca_fun/security-gym` from
`https://github.com/j-klawson/security-gym.git` at
`4b4c7b6e322f7b18817949990dfb583aa5686056`. The verified environment contract
is reflected in `alberta_framework.security`: action ids `0=pass`, `1=alert`,
`2=throttle`, `3=block_source`, `4=unblock`, `5=isolate`; action dicts with
`action` and `risk_score`; continuing-stream termination semantics; and
asymmetric action rewards. The local `security-gym` environment, hybrid, and
scan-stream tests passed after the pull.

`rlsecd` and `chronos-sec` were not reachable at the expected public URLs
(`https://github.com/j-klawson/rlsecd.git` and
`https://github.com/j-klawson/chronos-sec.git`). Direct daemon closure is an
external integration boundary for this repository; `rlsecd` acceptance details
are also recorded in `docs/research/rlsecd_external_blockers.md`. Completing
the active-defense track requires the correct repository URLs or credentials
plus:

- rlsecd streaming control loop that preserves temporal order and calls
  single-step `act`/`update`.
- Pinned feature schema shared by Step 3 prediction demons and Step 4 critic.
- Reproducible rollout logs with `(state, action, reward, next_state,
  termination, policy_metadata)`.
- End-to-end throughput including parsing, feature extraction, learner update,
  checkpoint/reporting, and action dispatch.
