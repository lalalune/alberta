# DiffEML Structured Topology Plan

The current credible scale path is not wider random wiring. Random wiring is a
reasonable control, but it is weak evidence for an EML-specific result because
the learned object can look like random Boolean features plus a readout. The
structured path should make the deployed circuit the main object: binary inputs,
two-input EML gates, deterministic source schedules, and a Boolean readout when
possible.

## Claim Boundary

The defensible mathematical claim is over binary feature vectors. A finite
Boolean circuit is universal for functions on `{0, 1}^d`; EML templates can
select elementary Boolean operations at each two-input node; therefore a large
enough hardened DiffEML circuit can represent any Boolean classifier on the
chosen discretization. Continuous approximation is inherited only through the
input discretization: thresholds or detectors partition the domain, then the
Boolean circuit maps cells to outputs. Any result must report how much accuracy
comes from the feature construction versus the EML circuit.

## Larp Risks To Remove

- Dense linear heads can do too much of the work. Keep them as baselines, but
  promote `group_sum`, `class_vote`, or `signed_class_vote` when making EML
  claims.
- Detector features can do too much of the work. Report matched linear and
  non-EML baselines on the exact binary features.
- Random wiring is not mathematically wrong, but it has high metadata cost and
  weak structure. Treat it as the DiffLogic-style control, not the final story.
- Structured local trees that underperform random wiring are not promoted just
  because they look image-like.
- Any topology without a packed hard evaluator is a planning candidate until a
  runner exists. The accounting must say `future_*` in `runner_wiring_mode`.

## Implemented Planning Specs

### Affine Expander

`affine_expander_topology_spec` replaces sampled source indices with a
deterministic degree-2 expander schedule. Each layer stores four modular-affine
coefficients:

```text
left(i)  = a_l * i + b_l mod m
right(i) = a_r * i + b_r mod m
```

The first layer uses the input feature modulus; deeper layers use the previous
layer width. Multipliers are chosen coprime to the modulus, so each source rule
is a permutation before pairing. This gives global sparse mixing with
descriptor-sized wiring storage while keeping every node a two-input EML gate.

Default readout: `class_vote`. That keeps deployment Boolean: final features
vote for classes rather than passing through dense float weights.

### Butterfly Class Bank

`butterfly_class_bank_topology_spec` uses a deterministic global butterfly mixer
followed by class-aligned butterfly banks. The final readout defaults to
`group_sum`, so the classifier is just a vote count over class-owned Boolean
features.

This directly tests the cleanest version of the hypothesis: can the circuit
construct class evidence internally, with no learned dense head? It also removes
almost all wiring metadata because each layer can be regenerated from a stride
and the known width/class-bank partition.

## Why These Are Less Larp

Both plans keep the hard model as:

```text
binary features -> two-input EML gates -> Boolean/count readout
```

There is no dense hidden layer, no continuous activation at deployment, and no
float readout in the default structured class-bank plan. Training can still use
relaxed selectors, but the deployable object is an explicit Boolean circuit.

## Scaling Path

1. Use random sparse as the matched gate-budget control.
2. Add packed hard runners for affine expander and butterfly class-bank.
3. Run exact matched budgets: same binary features, same gate count, same head
   mode, same train/test split.
4. Train oversize structured circuits, prune constant/duplicate/low-utility
   nodes, compact the remaining topology, then fine-tune from the compacted
   circuit.
5. Report deployed bytes as `wiring + gate masks + readout metadata`, not just
   train-time parameter count.

The strongest publishable outcome would be a structured Boolean DiffEML circuit
that beats random wiring and DiffLogic-style controls at matched hard-circuit
budget while retaining descriptor-sized wiring storage.
