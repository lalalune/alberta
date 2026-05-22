# Step 2 Wide Simple Search Results

This pass tested ten deliberately simple feature-finding mechanisms inspired by
self-replication, physics, cells, economics, and statistical mechanics.  Each
direction had three revisions.  The standard here was paired online supervised
Step 2 performance against fair MLP baselines; UPGD was kept as an incumbent
plasticity baseline.

## Scaled positives

### 1. Market auction feature economy

Best current candidate.  Tasks bid for candidate features, features pay rent,
and bankrupt features are replaced.  Rev C adds rent/bankruptcy.

Scaled run: `5 seeds x 2500 steps`, final window `750`.

| Stream | Best revision | Final MSE | Best MLP | Best MLP - method | Wins |
|---|---:|---:|---:|---:|---:|
| polynomial | Rev C rent/bankruptcy | 0.548283 +/- 0.053776 | 0.919805 | +0.371522 +/- 0.030697 | 5/5 |
| frequency | Rev C rent/bankruptcy | 1.112627 +/- 0.154545 | 1.281332 | +0.168705 +/- 0.115312 | 3/5 |
| compositional | Rev B task-balanced | 0.497447 +/- 0.126013 | 0.278150 | -0.219298 +/- 0.050783 | 0/5 |

Assessment: strong for polynomial and promising but noisy for frequency.  It
does not solve compositional two-layer tanh streams.  This is not universal yet,
but it is the best non-UPGD simple mechanism from this pass.

### 2. Replicator-mutator feature population

A population of feature recipes is selected by online utility; Rev B mutates
low-fitness features from high-fitness parents; Rev C adds elite memory.

Scaled run: `5 seeds x 2500 steps`, final window `750`.

| Stream | Best revision | Final MSE | Best MLP | Best MLP - method | Wins |
|---|---:|---:|---:|---:|---:|
| polynomial | Rev B mutation | 0.608634 +/- 0.125904 | 1.154352 | +0.545718 +/- 0.196827 | 5/5 |
| frequency | Rev B mutation | 0.883424 +/- 0.185745 | 1.499412 | +0.615988 +/- 0.150480 | 5/5 |

Assessment: surprisingly strong on both supported streams.  Caveat: the
initial recipe grammar includes triple products and sine features, so this is
evidence for useful budgeted search/selection over a broad grammar, not evidence
that the method can invent arbitrary representations from scratch.

### 3. Fast/slow dual-memory MLP

Fast weights adapt online while slow weights consolidate; Rev C gates fast/slow
predictions by recent loss.

Scaled run: `8 seeds x 4000 steps`, final window `1000`.

| Stream | Best revision | Final MSE | Best MLP | Best MLP - method | Wins |
|---|---:|---:|---:|---:|---:|
| polynomial | Rev C loss gate | 0.950511 +/- 0.126369 | 0.976911 | +0.026400 +/- 0.003370 | 8/8 |
| frequency | Rev B elastic anchor | 1.377051 +/- 0.319764 | 1.248730 | -0.128321 +/- 0.039328 | 0/8 |

Assessment: real but narrow.  It reliably beats MLP on polynomial, but loses
badly on frequency.  UPGD is far stronger on both streams in this scaled run.

## Rejected directions

| Direction | Best result | Verdict |
|---|---|---|
| Energy thermostat plasticity | Polynomial Rev B final MSE 0.648888 vs best MLP 0.612481 | Failed; worse final and interim performance. |
| Homeostatic feature setpoints | Failed on polynomial, frequency, compositional; Rev B was least bad | Failed; setpoint maintenance hurt performance. |
| Catalytic coactivation | Polynomial Rev A/B essentially tied but did not beat best MLP mean | Not worth scaling as implemented. |
| Immune clonal memory | Diverged badly: Rev A polynomial final MSE 769.5 vs best MLP 1.044 | Failed; numerically unstable. |
| Residual atoms | Polynomial Rev B final MSE 1.432493 vs best MLP 1.336800 | Failed; normalization helped but not enough. |
| Criticality controller | Polynomial Rev C final MSE 1.153685 vs best MLP 1.114137 | Failed; replacements were harmful. |
| Metropolis/Boltzmann acceptor | Longer 3-seed run: Rev A 1.472751 vs best MLP 1.041936 | Failed; too slow and negative. |

## Scientific conclusion

The best simple pattern is not generic reset/homeostasis/temperature control.
Those mostly add churn without task-relevant credit.  The successful pattern is
closer to a budgeted economy/ecology over candidate feature recipes:

1. Maintain many cheap candidate nonlinear features.
2. Assign survival credit from actual supervised error reduction.
3. Replace low-utility features with variants of useful features.
4. Preserve a small amount of historically useful structure.
5. Keep the readout online and non-disruptive.

This advances Step 2 for sparse feature discovery, especially when the grammar
contains the right atoms.  It does not yet solve universality.  The open problem
is compositional representation construction: when the useful feature is not in
the first-order recipe catalogue, market/replicator methods fail the same way
MLPs and UPGD can fail or over-adapt.

The next high-value experiments are:

1. Make the market/replicator grammar recursively compositional, but enforce a
   strict depth/resource cost so it cannot explode.
2. Add a compositional stream to the replicator script and test whether mutation
   can discover useful tanh-of-combination features rather than merely select
   pre-existing sine/triple atoms.
3. Hybridize UPGD with market allocation: UPGD supplies plastic hidden units,
   while auction credit decides which units are protected, cloned, or replaced.
4. Add out-of-distribution external controls, especially digits/permuted pixels,
   to reject methods that only win synthetic grammar-aligned streams.
5. Report compute-normalized performance, because Metropolis-like methods are
   not acceptable if they win only by excessive proposal evaluation.

## Follow-up Ablations

After the initial scale-up, the promising methods were ablated to understand
whether the wins came from the learning mechanism or merely from having the
right hand-coded feature grammar.

### Market Auction

The auction remains strong under severe resource cuts.

| Ablation | Stream | Best auction result | Best MLP | Diff | Wins |
|---|---|---:|---:|---:|---:|
| 32 slots | polynomial | 0.5653 | 1.2097 | +0.6444 | 3/3 |
| 32 slots | frequency | 0.8677 | 1.2993 | +0.4316 | 2/3 |
| 32 candidates | polynomial | 0.7970 | 1.2097 | +0.4127 | 3/3 |
| 32 candidates | frequency | 0.9576 | 1.2993 | +0.3417 | 2/3 |
| 16 slots | polynomial | 0.4632 | 1.2097 | +0.7465 | 3/3 |
| 16 slots | frequency | 0.6191 | 1.2993 | +0.6803 | 3/3 |
| 8 slots | polynomial | 1.2321 | 1.2097 | -0.0224 | 0/3 |
| 8 slots | frequency | 0.8726 | 1.2993 | +0.4267 | 3/3 |

Interpretation: the auction's success is not just from a large slot budget or
candidate pool.  A smaller budget often improves performance by raising
selection pressure.  Eight slots are too few for polynomial because the learner
must retain raw inputs and multiple triple products.  Frequency still works with
eight slots because a few sine/cosine atoms are enough.

Feature diagnostics support this interpretation.  In the scaled run, polynomial
auction Rev C ended with about 30.6 triple features, while frequency Rev C ended
with about 23.2 sine and 20.6 cosine features.  The method is selecting
structure appropriate to the stream.

Optimization direction: make slot pressure adaptive.  Start with a small budget,
increase only when marginal accepted-feature revenue remains high, and charge a
complexity rent by feature kind/depth.

### Replicator-Mutator

Replicator is more grammar-sensitive.

| Ablation | Stream | Best replicator result | Best MLP | Diff | Wins |
|---|---|---:|---:|---:|---:|
| baseline 96 population | polynomial | 0.6086 | 1.1544 | +0.5457 | 5/5 |
| baseline 96 population | frequency | 0.8834 | 1.4994 | +0.6160 | 5/5 |
| no seeded triples | polynomial | 1.2208 | 0.9925 | -0.2284 | 0/3 |
| no seeded triples | frequency | 0.6931 | 1.3316 | +0.6385 | 3/3 |
| 48 population | polynomial | 1.0938 | 0.9925 | -0.1014 | 0/3 |
| 48 population | frequency | 0.9102 | 1.3316 | +0.4215 | 3/3 |

Interpretation: polynomial success depends on having enough triple-product
coverage early.  Mutation did not rediscover enough triples quickly when they
were not seeded.  Frequency success is robust because the top selected features
are mostly sine atoms.

Optimization direction: use market-style candidate scoring before admitting
mutations.  The current replicator mutates from elite parents, but it does not
evaluate enough alternative children before spending a slot.  A cheap local
auction over mutation proposals should preserve the replicator mechanism while
reducing grammar brittleness.

### Market/Replicator Hybrid Attempts

The immediate hybrid tried here was to keep the market auction mechanism but
expand its candidate grammar with compositional `tanh` atoms.  This was meant
to test whether the compositional-stream failure was merely a missing-feature
problem.

| Variant | Compositional best auction | Best MLP | Diff | Verdict |
|---|---:|---:|---:|---|
| 16-slot auction, no tanh | 0.1989 | 0.1595 | -0.0394 | closer, still loses |
| unsigned tanh combos | 0.1784 | 0.1595 | -0.0189 | improves, still loses |
| signed tanh combos | 0.3553 | 0.1595 | -0.1958 | over-selects bad tanh atoms |
| signed tanh + cost 0.01 | 0.3467 | 0.1595 | -0.1873 | cost not enough |
| signed tanh + cost 0.05 | 0.3056 | 0.1595 | -0.1461 | cost helps but still loses |

Diagnostics from the signed-tanh run show why it failed: the auction filled
roughly 54-58 of 64 slots with `tanh` features, but those fixed signed sums
were still worse than the MLP's learned hidden units.  The problem is therefore
not just "include tanh"; it is adaptive construction of internal directions.

Interpretation: market selection works when the useful atom is already in the
candidate economy.  For compositional streams, a fixed catalogue of shallow
atoms is insufficient.  The next real improvement needs candidate features whose
parameters are themselves adapted briefly before being admitted, for example:

1. sample a small pool of candidate hidden units,
2. train each candidate for a few online residual-gradient steps on the recent
   buffer,
3. charge the candidate for that compute and complexity,
4. admit it only if its net residual-reduction revenue exceeds slot rent.

This is still a market/cost mechanism, but now the auction buys *trained
prototypes* rather than static atoms.

### Residual-Trained Prototype Attempt

The next implementation added residual-trained `rtanh` candidates:

1. sample random tanh units,
2. train each candidate for a few residual-gradient steps on the recent buffer,
3. score it by residual-reduction revenue minus optional cost,
4. admit it through the same auction,
5. optionally keep adapting admitted `rtanh` units online.

Compositional-only probe, `3 seeds x 2000 steps`, final window `600`:

| Variant | Best auction | Best MLP | Diff | Verdict |
|---|---:|---:|---:|---|
| trained `rtanh`, 32 candidates, 4 steps | 0.2007 | 0.1595 | -0.0412 | loses |
| trained `rtanh`, 64 candidates, 8 steps | 0.1927 | 0.1595 | -0.0332 | close to MLP(64,64), loses |
| trained `rtanh`, 16 slots | 0.2129 | 0.1595 | -0.0534 | loses |
| adaptive admitted `rtanh`, lr 0.003 | 0.1937 | 0.1595 | -0.0343 | loses |
| adaptive admitted `rtanh`, lr 0.01 | 0.1911 | 0.1595 | -0.0316 | best hybrid, still loses |

This improved over bad signed-static tanh features and roughly matched the
two-layer MLP baseline, but it still did not beat the best one-layer MLP.  The
current residual-training objective is too weak: it trains a single candidate
unit as a residual regressor, then asks a linear readout to use it in a
non-stationary multi-task stream.  For the compositional stream, the MLP's
jointly-trained hidden layer is still more effective.

Revised next step: candidate features should be trained as a small *set* with
competition/diversity, not independently.  The auction should admit a bundle
only if the bundle's net residual-reduction exceeds the sum of its slot and
compute costs.  Single-unit residual prototypes are not enough.

### Bundle Prototype Attempt

The bundle version was implemented next.  Each auction can train several small
bundles of residual-tanh units jointly on the recent buffer, solve a shared
linear readout for the bundle, subtract diversity/complexity cost, and admit the
whole bundle only if its per-slot revenue beats incumbent slot revenue.

Compositional-only probe, `3 seeds x 2000 steps`, final window `600`:

| Variant | Best auction | Best MLP | Diff | Bundle admissions | Verdict |
|---|---:|---:|---:|---:|---|
| bundle size 4 | 0.2966 | 0.1595 | -0.1371 | Rev C avg 1.7 | worse |
| bundle size 8 | 0.2683 | 0.1595 | -0.1088 | Rev C avg 1.0 | worse |
| bundle size 4, strict cost | 0.2795 | 0.1595 | -0.1200 | Rev C avg 1.0 | worse |

This branch failed.  Bundles were admitted, but they disrupted online
performance instead of improving it.  The best runs were the ones that admitted
few bundles.  This suggests the current buffer-trained residual objective is
misaligned with future online utility under context changes.

Updated mechanism diagnosis: the successful market is a good selector of
already-stable atoms.  It is not yet a good *constructor* of trainable
compositional state.  Strengthening Step 2 likely requires meta-learning the
candidate-construction process itself, not just scoring a bigger candidate pool.

### Meta-Constructor Attempt

The market auction was extended with a learned constructor prior.  Candidate
sampling is no longer purely uniform over the catalogue: feature families earn
an EMA score from live bid credit, and a separate delayed-survival EMA is
updated at later auctions from features that survived replacement pressure.
Evicted features feed back their realized slot utility before removal.  A
second version made this prior arm-level rather than kind-level, separating
arms such as `sin:f1`, `sin:f2`, `cos:f1`, `triple`, `pair`, `rtanh`, and
static `tanh` arity/scale.

Small probe, `3 seeds x 2000 steps`, final window `600`, candidate count `64`,
cost scale `0.001`:

| Variant | Stream | Best auction | Best MLP | Diff | Wins | Verdict |
|---|---|---:|---:|---:|---:|---|
| no meta-constructor | polynomial | 0.6721 | 1.2097 | +0.5376 | 3/3 | strong |
| kind live-only | polynomial | 0.6636 | 1.2097 | +0.5461 | 3/3 | small gain |
| kind delayed-survival | polynomial | 0.6658 | 1.2097 | +0.5438 | 3/3 | small gain |
| kind aggressive survival | polynomial | 0.6544 | 1.2097 | +0.5553 | 3/3 | best static meta |
| arm aggressive survival | polynomial | 0.6545 | 1.2097 | +0.5552 | 3/3 | ties best |
| no meta-constructor | frequency | 0.9553 | 1.2993 | +0.3440 | 2/3 | strong/noisy |
| kind/arm meta variants | frequency | 0.9553 | 1.2993 | +0.3440 | 2/3 | unchanged |
| no meta-constructor | compositional | 0.2978 | 0.1595 | -0.1383 | 0/3 | fails |
| kind aggressive survival | compositional | 0.2955 | 0.1595 | -0.1360 | 0/3 | tiny gain, fails |
| arm aggressive survival | compositional | 0.2998 | 0.1595 | -0.1403 | 0/3 | fails |
| residual `rtanh`, no meta | compositional | 0.1927 | 0.1595 | -0.0332 | 0/3 | close, fails |
| residual `rtanh`, kind survival | compositional | 0.1889 | 0.1595 | -0.0294 | 0/3 | best constructor, fails |
| residual `rtanh`, arm survival | compositional | 0.1967 | 0.1595 | -0.0372 | 0/3 | worse |
| residual `rtanh` + static tanh, kind survival | compositional | 0.2113 | 0.1595 | -0.0518 | 0/3 | worse |

Diagnostics were scientifically coherent.  Polynomial arms assigned highest
live credit to `triple`; frequency arms assigned highest credit to sine/cosine
arms; residual-tanh runs assigned non-zero survival credit to `rtanh`.  But
this did not translate into a compositional win.  The limiting factor is
therefore not just *which proposal family is sampled*; it is that the current
proposal family does not train and retain a hidden representation whose utility
survives context changes better than the MLP's online hidden layer.

Conclusion: meta-learning the constructor prior is a real improvement to the
market mechanism and should remain in the Step 2 implementation.  It strengthens
the successful sparse-feature story and makes the system more scientifically
legible, but it does not solve universality.  The next serious mechanism must
meta-learn *parameterized construction* and *retention*, not merely proposal
family frequencies.  The most defensible next experiment is a tiny
UPGD/CBP-style hidden-unit nursery feeding the auction: briefly train candidate
units online, admit only those with delayed survival credit, and protect/reseed
units whose descendants repeatedly survive future auctions.

### Persistent Hidden-Unit Nursery Attempt

The next attempt implemented that nursery.  A persistent population of `rtanh`
candidate units is trained online against the current residual on every step.
Mature units with high residual-reduction EMA are offered to the auction, and
weak mature units are periodically reset.  This differs from the earlier
residual-trained prototype, which trained candidates only on the latest buffer
at auction time.

Compositional-only probe, `3 seeds x 2000 steps`, final window `600`:

| Variant | Best auction | Best MLP | Diff | Wins | Verdict |
|---|---:|---:|---:|---:|---|
| residual `rtanh`, no nursery | 0.1927 | 0.1595 | -0.0332 | 0/3 | prior best |
| nursery 64, conservative | 0.1884 | 0.1595 | -0.0289 | 0/3 | improves, fails |
| nursery 128, aggressive reset | 0.2094 | 0.1595 | -0.0500 | 0/3 | worse |
| nursery 64, auction step 0.1 | 0.2774 | 0.1595 | -0.1180 | 0/3 | worse |
| nursery 64, auction step 0.05 | 0.3427 | 0.1595 | -0.1832 | 0/3 | worse |
| nursery 128 with 128 slots | 0.2334 | 0.1595 | -0.0739 | 0/3 | worse |
| nursery 64 + buffer-trained `rtanh` | 0.1856 | 0.1595 | -0.0261 | 1/3 | current best, still loses |

This narrows the compositional gap and beats the two-layer MLP baseline in this
small probe, but it still does not beat the best one-layer MLP.  The useful
mechanism appears to be persistent construction plus auction selection; the
harmful mechanisms are excessive slot budget, slower readout learning, and
aggressive nursery turnover.  The remaining gap is probably not candidate
quantity.  The auction can fill 40+ slots with `rtanh` features; the issue is
coordinated hidden-layer training and retention under non-stationarity.

Updated conclusion: Step 2 is decisively ahead of MLP on sparse grammar-aligned
streams and has a plausible, improving path on compositional streams, but the
universal claim should not be made yet.  The strongest honest claim is:
budgeted market/replicator feature discovery beats MLP when useful atoms are in
or near the constructor grammar; persistent rtanh construction reduces the hard
compositional gap, but full recursive representation learning remains open.
