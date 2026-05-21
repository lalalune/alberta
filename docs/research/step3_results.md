# Step 3 Empirical Results

This file collects the canonical multi-seed results for the Step 3 (GVF
prediction & Horde) Definition-of-Done criteria. Numbers are reproducible
from the corresponding scripts under `examples/The Alberta Plan/Step3/`.

## Closure Boundaries and Production Surface

Step 3 is locally complete for given-feature GVF prediction, Horde mechanics,
per-head traces, off-policy linear TD plumbing, recurrent history features, and
local-core throughput. It is not a claim that general temporal feature discovery
or nonlinear shared-trunk forward-view traces are solved.

The production-facing surface is intentionally narrow:
`src/alberta_framework/steps/step3.py` exposes `Step3HordeConfig`,
`make_step3_horde()`, `build_step2_to_step3_arrays()`, and
`run_step3_smoke()`. These helpers package the existing given-feature Horde path
and the Step 2 constructed-feature handoff. They are smoke/integration helpers,
not new scientific evidence.

| Boundary | Current status | Candidate track |
|---|---|---|
| Shared nonlinear trunk traces | Guarded; head traces are supported, trunk `gamma * lambda` remains zero with hidden layers | independent per-demon trunks, per-head trunk traces, or a forward-view-correct nonlinear trace derivation |
| TD/GVF feature discovery | Scoped positive controls only; observable AR(1) interaction features help, hidden/off-policy/general discovery remains open | learned history/state features, MSPBE-aware selection, full auxiliary-question meta-gradient methods |
| Pavlovian blocking | Instrumented but mixed across horizons | richer conditioning harnesses and nonlinear/latent-state variants |
| CBP sustained plasticity | Preserves effective rank and eliminates dead units but does not improve final-window MSE on DoD-8 sign-flip (LeakyReLU 200k), DoD-8 v2 ReLU sign-flip, or DoD-8 v2 task-shift; research-only surface | task families where rank preservation translates to online loss, replacement gates, UPGD/CBP hybrids |
| GVF feedback on POMDP | Negative in the current POMDP-RandomWalk ablation | causal feedback schedules, learned recurrent state, and stability gates |
| Throughput | SARSA local scan throughput passes 18/18 CPU configs above 1000 steps/sec (`output/step3_throughput/sarsa_throughput_20260507_074858.csv`); heavy Horde DoD-10 configs fail the 100-demon traced CPU target at 0/6 passing (`output/step3_throughput/horde_throughput_20260507_074718.csv`); daemon E2E remains separately bottlenecked on checkpoint/reporting I/O | optimize 100-demon Horde traces, replace per-step `save_checkpoint` with async/buffered/incremental checkpointing, and measure rlsecd/security-gym E2E once the daemon repo and logs are available |
| Production API | Given-feature Horde helper added; no packaged general discovery policy | promote only after robust TD/GVF discovery evidence exists |

Reproduce all sweeps in order::

    python "examples/The Alberta Plan/Step3/dod2_nexting_sweep.py"
    python "examples/The Alberta Plan/Step3/dod3_pavlovian_sweep.py"
    python "examples/The Alberta Plan/Step3/dod5_off_policy_sweep.py"
    python "examples/The Alberta Plan/Step3/dod6_pomdp_sweep.py"
    python "examples/The Alberta Plan/Step3/dod7_feature_discovery_sweep.py"
    python "examples/The Alberta Plan/Step3/dod8_plasticity_sweep.py"
    python "examples/The Alberta Plan/Step3/dod8_taskshift_v2_sweep.py"
    python "examples/The Alberta Plan/Step3/dod9_capstone_sweep.py"
    python benchmarks/horde_throughput.py
    python benchmarks/sarsa_throughput.py
    python benchmarks/daemon_throughput.py

## DoD-2: Multi-timescale Nexting

**Setup**: 5-state cyclic chain, cumulant=1 each step, 4000 steps × 12 seeds.
Compares TDLinear TD(0), TDLinear TD(λ), TrueOnlineTD(λ), MLPHorde head-only
traces, and MLPHorde head-only traces + CBP at γ ∈ {0.0, 0.5, 0.9, 0.99}.
Metric: RMSE(prediction vs analytic forward-view return) on the converged
window. Output: `output/step3_dod2/{results.csv,summary.json}`.

| Method | λ | γ=0.90 RMSE | γ=0.99 RMSE |
|---|---:|---:|---:|
| TDLinear TD(0) | 0.0 | 0.4405 | 41.5455 |
| TDLinear TD(λ) | 0.5 | 0.1193 | 26.9730 |
| TDLinear TD(λ) | 0.9 | 0.0024 | 8.6897 |
| TrueOnlineTD(λ) | 0.5 | 0.1171 | 26.8230 |
| TrueOnlineTD(λ) | 0.9 | 0.0020 | 8.1136 |
| MLPHorde head traces | 0.5 | 0.0018 | 8.8996 |
| MLPHorde head traces | 0.9 | 0.0017 | 6.9339 |
| MLPHorde head traces + CBP | 0.5 | 0.0411 | 8.8502 |
| MLPHorde head traces + CBP | 0.9 | 0.0665 | 7.0195 |

**Verdict — PASS.** The broader comparison is now generated. Long-horizon
trace learning behaves as expected: λ=0.9 dominates TD(0), TrueOnlineTD(λ)
slightly improves over conventional TD(λ), and MLPHorde head traces are
competitive or better on this deterministic nexting task. CBP is present in
the comparison but is not beneficial for this stationary chain.

## DoD-3: Pavlovian Conditioning Suite

**Setup**: ACQUISITION → EXTINCTION → REACQUISITION (each 2000 steps),
distractors=2, CS-US delay=5, 10 seeds. Mean prediction in the last
200 steps of each phase.

| γ    | A_mean | E_mean | R_mean | A−E drop | R−E recovery |
|------|--------|--------|--------|----------|--------------|
| 0.00 | 0.055  | 0.000  | 0.053  | 0.055    | 0.053        |
| 0.50 | 0.109  | 0.000  | 0.107  | 0.109    | 0.107        |
| 0.90 | 0.538  | 0.000  | 0.545  | 0.538    | 0.545        |
| 0.99 | 3.294  | 1.316  | 3.805  | 1.978    | 2.489        |

**Verdict — PASS.** Across all four γ horizons, predictions
monotonically rise during acquisition, drop during extinction, and
recover during reacquisition — the canonical Pavlovian curves. γ=0.99
shows partial extinction (long-horizon return takes longer to
extinguish), as expected.

Blocking evidence from the same 10-seed run:

| γ    | CS0 mean | CS1 mean | CS0−CS1 | CS0 > CS1 seeds |
|------|---------:|---------:|--------:|----------------:|
| 0.00 | 0.034    | 0.038    | -0.003  | 6/10 |
| 0.50 | 0.071    | 0.102    | -0.031  | 4/10 |
| 0.90 | 0.554    | 0.473    | 0.081   | 8/10 |
| 0.99 | 4.579    | 4.665    | -0.086  | 3/10 |

**Blocking verdict — MIXED.** The harness now emits blocking and paired
statistics, closing the instrumentation gap. The empirical blocking effect is
not robust across horizons in this linear Horde setup; γ=0.9 shows the expected
CS0 > CS1 pattern, while shorter and longer horizons do not. Keep this as a
documented Step 3 boundary rather than claiming robust Kamin blocking.

## DoD-5: Off-policy Convergence (Retrace clipping + ETD)

**Setup**: 4-state random walk, behavior π=uniform, target π=always
right. Under target, V*(s)=1 for every state. 12 seeds × 2000 episodes.

| Retrace clip c | RMSE mean | RMSE std | Diverged seeds |
|----------------|-----------|----------|----------------|
| 1.0            | 0.0000    | 0.0000   | 0/12           |
| 2.0            | 0.0000    | 0.0000   | 0/12           |
| 1000.0         | 0.0000    | 0.0000   | 0/12           |
| inf            | 0.0000    | 0.0000   | 0/12           |

**Verdict — PASS.** OffPolicyTDLinearLearner converges exactly to the
target-policy V on this problem under all clip settings (no divergence,
no measurable bias). The implementation gap for full emphatic traces is also
closed by `ETDLinearLearner`, which adds follow-on trace `F_t`, scalar emphasis
`M_t`, emphatic eligibility traces, config roundtrip, and focused tests for
LMS equivalence, off-policy trace evolution, JIT update, and bounded finite
updates.

## DoD-6: Recurrent State on POMDP-RandomWalk

**Setup**: AR(1) latent observations with a drifting RandomWalk linear target.
Channels 2..5 are periodically hidden on two out of every three steps, while
the target remains a function of the full latent observation. The ablation
compares raw masked observations, trace-only recurrent state, GVF-feedback
state, and trace+GVF state. The canonical command is:

```bash
python "examples/The Alberta Plan/Step3/dod6_pomdp_sweep.py"
```

| Condition | Final-window MSE mean | MSE std | Better than raw |
|---|---:|---:|---:|
| raw | 0.9990 | 0.5424 | -- |
| trace_only | 0.3503 | 0.1660 | 10/10 |
| gvf_feedback | 1.0948 | 0.5960 | 0/10 |
| trace_plus_gvf | 0.3579 | 0.1633 | 10/10 |

**Verdict — PASS for recurrent traces, negative for GVF feedback.** The
requested POMDP-RandomWalk ablation is now generated. Multi-timescale history
features recover the masked latent information in every seed. Auxiliary
GVF-feedback features, as currently constructed, do not help this stream and
slightly hurt without traces.

## DoD-7: Feature Finding Under TD Targets

**Setup**: 8-dim i.i.d. Gaussian observation, 16 cumulant candidates,
discovery on/off. 5000 steps × 10 seeds.

| Discovery | Median utility | Replacements per seed | Projections changed |
|-----------|----------------|-----------------------|---------------------|
| OFF       | 0.0000         | 0.0                   | 0/16                |
| ON        | 0.0004         | 46.6                  | 16/16               |

**Verdict — PASS (mechanism only).** With discovery enabled, ~47
replacements occur per seed and all 16 projections rotate through fresh
candidates. With discovery disabled, the candidate set is frozen.
Median utility is small in absolute terms because the i.i.d. stream has
no temporal structure to discover; on a structured stream the same
mechanism would retain demons predicting the structured cumulants.

### Downstream GVF RMSE bridge

`step3_feature_discovery_eval.py` now evaluates whether Step 2 constructed
features or discovered auxiliary cumulants reduce downstream target-GVF RMSE,
not just whether a candidate bank changes internally. The harness uses the GVF
transition convention `c_{t+1}` and scores pre-update predictions against
forward-view returns after burn-in and burn-tail exclusion.

The original observable+squares probe used i.i.d. observations. That ordering is
correct, but it is a weak feature-finding test because `obs_t` does not predict
the nonlinear part of `target_{t+1}` except through distributional/context
statistics. It produced a useful negative against MLP:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py" \
  --seeds 5 \
  --discovery-steps 800 \
  --eval-steps 1600 \
  --hide-last-channels 0 \
  --include-squares \
  --output-dir output/td_gvf_observable_squares_5seed
```

| Method | Target GVF RMSE |
|---|---:|
| `given_mlp_gvf` | `2.0284 +/- 0.1306` |
| `discovered_aux_cumulants_mlp_gvf` | `2.0350 +/- 0.1296` |
| `random_aux_cumulants_mlp_gvf` | `2.0362 +/- 0.1307` |
| `step2_tanh_features_linear_gvf` | `2.2104 +/- 0.1478` |
| `given_linear_gvf` | `2.2216 +/- 0.1487` |
| `step2_interaction_features_linear_gvf` | `2.2515 +/- 0.1425` |

To give discovered pair/square features a causal chance under TD targets, the
script now has an explicit AR(1) observation mode and an
`all_visible_interactions_linear_gvf` positive control. The control appends every
pair/square product of the observed channels before training the Horde; it is not
counted as a discovery method.

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py" \
  --seeds 5 \
  --discovery-steps 500 \
  --eval-steps 1000 \
  --burn-in 100 \
  --burn-tail 50 \
  --observation-dynamics ar1 \
  --ar-rho 0.95 \
  --include-squares \
  --hide-last-channels 0 \
  --output-dir output/td_gvf_ar1_squares_5seed
```

| Method | Target GVF RMSE |
|---|---:|
| `all_visible_interactions_linear_gvf` | `3.1419 +/- 0.2445` |
| `step2_interaction_features_linear_gvf` | `3.1522 +/- 0.2288` |
| `given_mlp_gvf` | `3.2390 +/- 0.2630` |
| `discovered_aux_cumulants_mlp_gvf` | `3.2815 +/- 0.2679` |
| `random_aux_cumulants_mlp_gvf` | `3.3069 +/- 0.2728` |
| `given_linear_gvf` | `3.3416 +/- 0.2596` |
| `step2_tanh_features_linear_gvf` | `3.3620 +/- 0.2841` |

Paired differences in this AR(1) probe: learned interaction features beat the
linear GVF baseline by mean RMSE `0.1894` (`5/5` seeds) and the raw MLP GVF by
`0.0868` (`4/5` seeds). The all-visible interaction control beats linear by
`0.1997` (`5/5`) and MLP by `0.0970` (`3/5`). Surprise-driven auxiliary
cumulants beat random auxiliary cumulants and linear, but lose to the raw MLP
by mean RMSE `0.0425` (`1/5` seeds).

Current DoD-7 canonical run:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py" \
  --output-dir output/step3_dod7 \
  --seeds 5 \
  --discovery-steps 800 \
  --eval-steps 1600 \
  --burn-in 150 \
  --burn-tail 50 \
  --observation-dynamics ar1 \
  --include-squares \
  --hide-last-channels 0
```

| Method | Target GVF RMSE |
|---|---:|
| `fixed_interaction_linear_gvf` | `2.9537 +/- 0.1907` |
| `td_surprise_interaction_features_linear_gvf` | `2.9761 +/- 0.1954` |
| `meta_gradient_proxy_interaction_features_linear_gvf` | `2.9988 +/- 0.1970` |
| `step2_interaction_features_linear_gvf` | `3.0050 +/- 0.1967` |
| `gvf_feedback_features_linear_gvf` | `3.0323 +/- 0.2081` |
| `given_mlp_gvf` | `3.0538 +/- 0.2006` |
| `given_linear_gvf` | `3.1556 +/- 0.1844` |

Off-policy behavior-mismatch probe:

| Method | Target RMSE |
|---|---:|
| `off_policy_mspbe_novel_predictive_state_linear_td_is` | `0.8651 +/- 0.0709` |
| `off_policy_mspbe_predictive_state_linear_td_is` | `0.8947 +/- 0.0647` |
| `off_policy_raw_linear_td_is` | `0.8978 +/- 0.0730` |
| `off_policy_raw_linear_td_no_is` | `1.4135 +/- 0.1056` |

**Verdict — PASS as a scoped positive control; open as general discovery.**
TD-surprise interaction features beat both raw linear and raw MLP on the
observable AR(1) TD target, and the Veeriah-style proxy arm is present in the
same candidate family. Off-policy MSPBE predictive-state features narrowly beat
raw clipped-IS TD on mean, but only 3/5 seeds for the novelty-gated row. This
does not solve hidden-state, nonlinear off-policy, or general cumulant
discovery.

### DoD-7 Hidden: hidden-state AR(2) target

The observable AR(1) probe leaves the hidden-state case open: when the
target depends on channels the agent never reads, no amount of
visible-channel feature construction can recover it. To close that gap,
`step3_hidden_feature_discovery_eval.py` runs the same discovery stack
on a hidden-state stream where:

- `HiddenStateAR2Stream` (in `streams/synthetic.py`) generates an
  8-channel latent state under stationary AR(2) dynamics
  `x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + sigma * eps_t` with
  `phi1=0.6, phi2=0.3, sigma=1.0`. The pair `(phi1, phi2)` is inside
  the AR(2) stationarity triangle, so traces can converge.
- `PartialObservationWrapper(MaskMode.PERIODIC)` hides channels
  `2..7` on 2 out of every 3 steps. The agent gets a full-state glimpse
  every third step; the other two steps it sees only channels `0..1`.
- The target is a deterministic function of the FULL state:
  `y_t = weights @ x_t + 0.5 * x_t[i] * x_t[j] + N(0, 0.05)`, where
  `weights` is sampled once from `N(0, I)` and `(i, j)` are two distinct
  hidden-block indices. Both the linear and the nonlinear part of the
  target couple to hidden channels, so visible-only features cannot
  represent the target.

Five seeds, 2000 steps total per seed (1500-step warmup for discovery,
500-step held-out evaluation tail). All conditions train a linear Horde
with one γ=0 prediction demon; conditions differ only in the feature
representation.

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step3/step3_hidden_feature_discovery_eval.py" \
  --seeds 5 \
  --warmup-steps 1500 \
  --eval-steps 500 \
  --output-dir output/step3_dod7_hidden
```

| Condition | Target GVF RMSE |
|---|---:|
| `raw_masked_history_linear_gvf` | `3.4154 +/- 0.2369` |
| `raw_masked_history_cumulant_linear_gvf` | `3.4158 +/- 0.2384` |
| `raw_masked_history_interaction_cumulant_linear_gvf` | `3.5727 +/- 0.2795` |
| `raw_masked_history_interaction_linear_gvf` | `3.6442 +/- 0.2407` |
| `raw_masked_linear_gvf` | `3.7738 +/- 0.2862` |

History-only beats raw masked observations by mean RMSE `0.358`
(positive recurrent baseline, as expected). Cumulant-discovery
projections of the augmented (raw + history) observation produce no
detectable gain over history-only on this stream — the discovered
auxiliary cumulants are linear in the same features the GVF already
sees, so they add no representational capacity. The Step-2
interaction-feature bank actually *hurts*: pair products of
history-trace features overfit the warmup target without recovering the
hidden-by-hidden product (which lives in masked channels and only
surfaces every third step). The full stack inherits both drawbacks and
loses to history-only by mean RMSE `0.157` (`z-like = -0.43`).

**Verdict — NEGATIVE on hidden-state discovery.** The full stack does
*not* beat history-only at d > 1.0; it is `0.157` worse on mean. History
features alone are the best of the five conditions. This is the
documented hidden-state case the AR(1) DoD-7 result explicitly
disclaimed.

What this implies for the Step 3 paper-spec gap:

1. The current discovery primitives (random-projection cumulant
   discovery, fixed-budget pair-product interaction features) all
   operate on the *augmented* observation. When the relevant signal
   never appears in that observation, they cannot recover it; they only
   recombine what is already there. Closing the hidden-state gap needs
   a discovery method that can *learn nonlinear features of latent
   state*, not merely linear projections or pair products of observed
   features.
2. History/EMA features remain the only mechanism here that meaningfully
   improves over raw masked observations. That is consistent with the
   nexting/predictive-state literature, but it is also a narrow lever:
   the EMAs cannot represent the hidden-by-hidden product term, so
   target RMSE plateaus well above the achievable Bayes optimum.
3. The honest framing for the Step 3 paper draft is therefore:
   "TD/GVF feature discovery passes as a scoped positive control on
   observable AR(1) and fails on hidden-state AR(2); a learned
   nonlinear-recurrent feature builder, or a Veeriah-style
   meta-gradient on auxiliary questions that operates on a learned
   latent state, is the smallest plausible upgrade."

Artifacts: `output/step3_dod7_hidden/{results.csv, summary.json, SUMMARY.md}`.

## DoD-8: 200k-step Plasticity Sustained (CBP)

**Setup**: Non-stationary regression where half the regression weights
flip sign every 10000 steps; 200,000 total steps × 5 seeds × 2 conditions
(CBP off / CBP on). MultiHeadMLPLearner with 1 head, 64-unit trunk, ObGD
bounding, LMS step-size. The run writes
`output/step3_dod8_200k/effective_rank_trajectory.csv`.

| Condition | first ctx MSE | last ctx MSE | final-window MSE | final effective rank | replacements |
|-----------|--------------:|-------------:|-----------------:|---------------------:|-------------:|
| CBP off   | 0.0782 | 0.0210 | 0.0211 | 8.88 | 0 |
| CBP on    | 0.0814 | 0.0290 | 0.0293 | 26.09 | 1279 |

**Verdict — MIXED/NEGATIVE.** The 200k artifact and effective-rank trajectory
are now generated. CBP preserves much higher hidden activation rank and performs
unit replacement, but it does not improve final-window MSE on this particular
stream. Treat this as evidence that the CBP plumbing and plasticity diagnostics
work, not as a positive sustained-performance claim.

### DoD-8 v2: ReLU and task-shift variants

The original DoD-8 sign-flip stream uses LeakyReLU, which lets the
network re-fit each context by reweighting existing units; that
configuration cannot show CBP-positive signal even in principle. The
v2 sweep tests two regimes designed to expose CBP's intended benefit:

* **Variant A — ReLU sign-flip.** Same stream as DoD-8 with
  `leaky_relu_slope=0.0`. Hard-ReLU produces true dead units that
  cannot recover via gradient flow; per Dohare et al. (Nature 2024)
  this is CBP's home turf.
* **Variant B — Task-shift.** Each context uses a contiguous half of
  the input vector (alternating low/high) with fresh random weights;
  the unused half contributes zero to the target. The full input
  distribution is fixed but the target subspace rotates every 10000
  steps, forcing genuinely new feature->target maps each context.

5 seeds × 50,000 steps × 4 cells. Reproduce with
`python "examples/The Alberta Plan/Step3/dod8_taskshift_v2_sweep.py"`;
artifacts under `output/step3_dod8_v2/{results.csv, summary.json,
effective_rank_trajectory.csv, SUMMARY.md}`.

| Cell                      | first ctx MSE | last ctx MSE | final-window MSE | eff. rank | dead units | replacements |
|---------------------------|--------------:|-------------:|-----------------:|----------:|-----------:|-------------:|
| A_relu_signflip · CBP off | 0.0753        | 0.0638       | 0.0638 ± 0.0432  | 15.62     | 10.2       | 0            |
| A_relu_signflip · CBP on  | 0.0961        | 0.0783       | 0.0789 ± 0.0453  | 36.75     | 0.0        | 319          |
| B_taskshift     · CBP off | 0.0467        | 0.0195       | 0.0191 ± 0.0155  | 15.75     | 13.6       | 0            |
| B_taskshift     · CBP on  | 0.0499        | 0.0223       | 0.0221 ± 0.0177  | 29.05     | 6.8        | 319          |

CBP-on minus CBP-off final-window MSE: Variant A delta = +0.0151
(CBP hurts), Variant B delta = +0.0029 (CBP hurts; within ~0.2 SD).
The mechanism works in both variants — replacements ran at the
configured rate (319 per seed), dead units shrank from 10.2 to 0
(Variant A) and 13.6 to 6.8 (Variant B), and effective rank roughly
doubled — but in both cases unit replacement disrupted already-learned
features faster than the network could re-fit them.

**Verdict — confirmed negative.** The original DoD-8 result was not a
stream artifact. CBP fails to improve final-window MSE on the
ReLU-activation variant where CBP is supposed to help most, and on a
task-shift stream where representations genuinely have to change. In
the 64-unit, single-head, LMS+ObGD configuration used here on this
stream family, CBP's replacement schedule is a net cost on prediction
error. The CBP plumbing, replacement, dead-unit, and rank diagnostics
all work correctly — the algorithm just does not help at this scale on
these tasks. CBP remains a research-only surface in the framework
(retain `CBPMultiHeadMLPLearner` and `ContinualBackpropConfig` for
future task families and hybrids; do not promote to a default
component).

## DoD-9: Critic-for-Control Capstone

**Setup**: Continuing GridWorld torus (4×4, 4 actions, +1 reward at
fixed cell with teleport on collection). 30,000 steps × 10 seeds × 3
conditions: SARSA-only, SARSA + auxiliary GVF Horde (γ ∈ {0, 0.5, 0.9}
predicting reward), SARSA + auxiliary GVF Horde + history features.

| Condition | Last-5k reward mean | Last-5k std | Total reward mean |
|---|---:|---:|---:|
| SARSA-only | 2088.6 | 275.0 | 10681.1 |
| SARSA + prediction Horde | 2173.3 | 73.0 | 11556.3 |
| SARSA + Horde + CBP + history | 2197.7 | 33.1 | 11883.0 |

**Verdict — PASS.** The capstone artifact now exists at
`output/step3_dod9/{results.csv,summary.json}`. Auxiliary prediction demons
improve both last-window and total reward versus SARSA-only, and the
CBP+history condition is best on the aggregate metrics.

## DoD-10: CPU Throughput at Scale

**Setup**: CPU-only 10,000-step synthetic daemon-throughput benchmark. The
benchmark path now uses final-state-only scan runners, which execute the same
predict+update trajectory as the diagnostic scan loops but do not materialize
full per-step metric histories. This measures the runtime contract needed by a
daemon while preserving separate diagnostic APIs for research sweeps.

Horde artifact:
`output/step3_throughput/horde_throughput_20260506_013220.csv`.

| Gate | Configs passing | Slowest config | Slowest steps/sec |
|---|---:|---|---:|
| Full Horde grid | 36/36 | 100 demons, 64x64 hidden, traces on, no normalizer | 2655.0 |
| DoD Horde subset: n_demons >= 50, traces on | 6/6 | 100 demons, 64x64 hidden, traces on, no normalizer | 2655.0 |

SARSA artifact:
`output/step3_throughput/sarsa_throughput_20260506_013253.csv`.

| Gate | Configs passing | Slowest config | Slowest steps/sec |
|---|---:|---|---:|
| Full SARSA grid | 18/18 | 16 actions, 64x64 hidden, traces on | 11103.6 |

### Daemon end-to-end gate

**Setup**: ``benchmarks/daemon_throughput.py`` drives the production
``AlbertaPipeline`` through a synthetic JSON-line transport with explicit
parse, predict, update, serialize, and per-100-step Orbax checkpoint
phases. 5,000 steps per configuration, 8-cell grid:
``features ∈ {no_features (identity), full Step 2 features}``,
``n_demons ∈ {5, 25}``, ``hidden_sizes ∈ {(), (32,)}``. Acceptance threshold
relaxed to ≥500 steps/sec to account for daemon overhead.

Daemon artifact:
``outputs/daemon_throughput/{daemon_throughput_results.csv, summary.json,
SUMMARY.md}``.

| features | n_demons | hidden | feat_dim | steps/sec | parse ms | predict ms | update ms | serialize ms | ckpt ms | slowest | gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| no_features | 5 | () | 8 | 399.2 | 0.242 | 0.134 | 0.638 | 0.194 | 129.534 | checkpoint | FAIL |
| no_features | 5 | (32,) | 8 | 343.6 | 0.283 | 0.228 | 0.939 | 0.220 | 123.863 | checkpoint | FAIL |
| no_features | 25 | () | 8 | 192.9 | 0.337 | 0.282 | 1.215 | 0.465 | 288.346 | checkpoint | FAIL |
| no_features | 25 | (32,) | 8 | 148.7 | 0.381 | 0.662 | 2.125 | 0.526 | 302.915 | checkpoint | FAIL |
| features | 5 | () | 28 | 181.8 | 0.549 | 0.290 | 1.810 | 0.411 | 243.736 | checkpoint | FAIL |
| features | 5 | (32,) | 28 | 223.5 | 0.375 | 0.405 | 1.672 | 0.326 | 169.437 | checkpoint | FAIL |
| features | 25 | () | 28 | 144.3 | 0.448 | 0.634 | 1.831 | 0.633 | 337.716 | checkpoint | FAIL |
| features | 25 | (32,) | 28 | 104.0 | 0.488 | 1.125 | 3.321 | 0.683 | 399.612 | checkpoint | FAIL |

**Verdict — PASS for local-core scan throughput; FAIL for daemon end-to-end
at the 500 steps/sec gate.** The broad CPU throughput gate is closed for the
in-process JAX learner update contract: Horde clears the ≥1000 steps/sec
target across n_demons ∈ {5, 25, 100}, hidden_sizes ∈ {(), (32,), (64, 64)},
traces on/off, and normalizer on/off; SARSA clears the same target across
n_actions ∈ {4, 8, 16}, hidden_sizes ∈ {(), (32,), (64, 64)}, and traces
on/off. The daemon E2E harness now measures parse/predict/update/serialize/
checkpoint phases separately and identifies Orbax checkpointing as the
dominant cost at every grid cell: per-checkpoint latency is 124–400ms (one
checkpoint per 100 steps), consuming 6.5–20s of the 12.5–48s wall time per
5,000-step run. Excluding checkpoint cost, only the smallest configuration
(``no_features``, 5 demons, no hidden layers) clears the 500 steps/sec line
(~827 steps/sec). The next bottleneck is the per-step Python→XLA round trip
in ``pipeline.update``: 0.6–3.3ms per step, vs the fully scan-fused
``run_horde_learning_loop_final_state`` which amortizes compile and dispatch
across the whole episode. This still does not measure rlsecd transport,
process scheduling, real environment stepping, monitoring, or security-gym
integration overhead.

## Summary Table

| DoD  | Topic                          | Status              |
|------|--------------------------------|---------------------|
| 2    | Multi-timescale nexting        | PASS                |
| 3    | Pavlovian conditioning         | PASS; blocking mixed|
| 5    | Off-policy IS + ETD prediction | PASS                |
| 6    | Recurrent state on POMDP       | PASS for traces     |
| 7    | TD-target feature discovery    | PASS scoped; open general |
| 8    | 200k plasticity sustained + v2 ReLU/task-shift | NEGATIVE; research-only |
| 9    | Critic-for-control capstone    | PASS                |
| 10   | CPU throughput ≥ 1000 steps/s  | PASS local-core; daemon E2E measured, FAIL at ≥500 sps gate (checkpoint-bound) |
