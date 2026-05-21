# Step 2 Simple Non-Router Closure Assessment

## Question

Can Step 2 be closed by a single learner rather than by a deployment portfolio,
router, or hand-selected expert set?

Working standard:

- one prediction path at deployment time;
- one online update rule used at every timestep;
- no output router, no expert selector, no portfolio over MLP/UPGD/sparse
  learners;
- internal communication and resource allocation are allowed only when they are
  part of the learner's own uniform update rule;
- comparison is against the same-run best fair MLP width, not a fixed weak MLP.

## Current Best Candidate

The best candidate is D18 persistent trace:

```bash
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --configs step2_persistent_trace \
  --output-dir outputs/step2_main_d18_persistent_trace_p6_all_10seed \
  --note-path docs/research/step2_main_d18_persistent_trace_p6_all_10seed.md
```

The canonical artifact was generated with the equivalent explicit command using
`--configs step2_gain_l2_0p1` plus the target-trace flags. The shorter named
config changes only the result method label.

D18 persistent trace is one additive predictor. It has a resource-managed RKHS
core, tanh/Fourier basis block, small polynomial/unified residual blocks, learned
block gains, online-discovered simplex projection for one-hot targets, and a
causal target trace gated by observed one-step target persistence. Every block
updates from the same residual. It does not route among output learners.

## Evidence

Canonical all-suite source:

- `outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_results.json`
- `outputs/step2_canonical/simple_d18_persistent_trace_all_10seed_SUMMARY.md`

Result: all 14 aggregate final-window MSE rows are positive against the same-run
best fair MLP. Seed-level wins/losses/ties are `138/2/0`.

| Group | Seed-level final-window MSE wins/losses/ties |
|---|---:|
| Controlled six-probe suite | `60/0/0` |
| Synthetic polynomial/frequency/compositional | `28/2/0` |
| Digits IID/class-blocked/permuted/mask-noise/label-drift | `50/0/0` |

Weakest aggregate rows:

| Row | Final-window MSE diff |
|---|---:|
| Digits class-blocked | `+0.001409875` |
| Digits mask-noise | `+0.009147615` |
| Synthetic compositional | `+0.042200536` |
| Synthetic polynomial | `+0.094076989` |

Because D18 projects one-hot digit predictions after detecting simplex target
geometry, the fair audit also projects the MLP comparator by converting its
final-window accuracy to one-hot MSE: `0.2 * (1 - accuracy)` for 10 classes.
The 30-seed risk source is:

- `outputs/step2_canonical/simple_d18_persistent_trace_risk_digits_30seed_results.json`
- `outputs/step2_canonical/simple_d18_persistent_trace_risk_digits_30seed_SUMMARY.md`

| Risk row | Raw MSE diff | Raw wins/losses/ties | Projected-MLP diff | Projected wins/losses/ties |
|---|---:|---:|---:|---:|
| Digits class-blocked | `+0.001473318` | `30/0/0` | `+0.000244444` | `11/2/17` |
| Digits mask-noise | `+0.009994925` | `29/1/0` | `+0.001466667` | `18/11/1` |

## Failed Simpler Alternatives

The four parallel simple-learner probes were informative and negative:

- Pure residual-birth learner: clean generic feature birth, but only `2/6`
  blocker rows beat fair MLP. It solved class-blocked and controlled-rare but
  failed mask-noise, polynomial, frequency, and compositional blockers.
- Pure kernel adaptive filtering: mechanically single additive learner, but no
  fixed kernel candidate cleared the six-blocker matrix.
- Wide random features with one NLMS readout: positive on controlled-rare,
  mask-noise, and compositional, but failed class-blocked mean MSE and lost
  synthetic polynomial/frequency on all paired seeds.
- Calibration/simplex-only mechanisms: strong on mask-noise for a softmax
  update variant, but failed class-blocked against the fair projected-MLP bar.

These failures matter. They show that D18's win is not just a projection trick,
not just a local residual memory, and not just a generic high-dimensional random
feature expansion.

## Promising Follow-Up Not Promoted

After the persistent-trace result was validated, the newer D18
`step2_canonical` context/prototype branch finished a 10-seed all-suite run with
an even cleaner raw final-window MSE table: `140/0/0` seed-level wins/losses/ties
against the same-run best fair MLP. Its 30-seed hard digit risk run also stayed
positive by mean: class-blocked raw `30/0/0`, mask-noise raw `29/1/0`.

It is not the promoted simple-candidate artifact here because its strict
projected-MLP class-blocked margin is much thinner: `+0.000022` by mean at 30
seeds with `10/10/10` projected wins/losses/ties. It also adds retained class
prototypes, making the mechanism less clean. It should be treated as a
promising follow-up to simplify and stress-test, not as a replacement for the
more conservative persistent-trace evidence.

A later all-suite `step2_canonical` rerun again reached raw `140/0/0`, but the
fair projected-MLP digit audit was negative on IID, label-drift, mask-noise,
and permuted-pixels rows. That kept the persistent-trace artifact as the best
D18 result. It is now superseded for the global simple non-router Step 2 claim
by target-structure UPGD.

## Claim Boundary

The current evidence supports:

> D18 persistent trace is a strong superseded single non-router Step 2 learner:
> it beats the same-run best fair MLP on every aggregate row of the
> 14-regime controlled/synthetic/digits promotion matrix, and its two hardest
> digit rows remain positive by mean at 30 seeds, including under a fair
> projected-MLP audit. The current promoted simple learner is target-structure
> UPGD.

The evidence does not yet support:

> Step 2 is theoretically solved by a clean universal feature-construction
> principle.

Reasons:

- D18 is still a hand-assembled additive resource-basis learner, not the current
  promoted target-structure UPGD learner and not a single recursive
  feature-growth rule.
- The successful mechanism uses fixed families with complementary biases:
  local RKHS memory, tanh/Fourier features, small polynomial/unified residuals,
  learned block gains, and causal target persistence.
- Published-scale OPMNIST remains outside this specific non-router result. The
  broader Step 2 evidence set now has a one-seed 800-task / 48,000,000-update
  UPGD-memory artifact, but that artifact is mixed: UPGD-memory wins online MSE,
  online accuracy, and final-window MSE while fair MLP comparators still win
  final-window accuracy and all-permutation held-out test MSE/accuracy.
- TD/GVF target feature discovery remains Step 3 research.
- Native deep MLP feature lifecycle remains weaker than D18 and the portfolio
  baselines on the hard blocker matrix.

## Bottom Line

For the user's stricter standard, this is a real advance: the current supervised
promotion matrix no longer requires a deployment portfolio/router to beat fair
MLP. The remaining work is not to patch another blocker row; it is to simplify
the D18 mechanism into a cleaner mathematical principle, or to continue
multi-seed external-scale replication and metric-level closure on the 800-task
OPMNIST run.
