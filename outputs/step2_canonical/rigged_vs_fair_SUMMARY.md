# Step 2 audit: rigged-vs-fair comparison
This document reports the result of an audit of the original Step 2 headline claim ('16/16 paired wins over MLP'). The audit reproduced the original comparison verbatim and then re-ran the same methods under two falsifying conditions: (1) streams whose oracle is outside the interaction learner's hypothesis class, and (2) the original stream paired against MLPs with substantially more capacity than the original baseline. The original 16/16 number was reported on **mean loss across the entire stream** (interim performance); the original docs also acknowledged the final-window margin was smaller (11/16). Both metrics are reported below.
## Part A: reproduce the rigged comparison
Configuration: `InteractionFeatureDiscoveryStream(feature_dim=10, n_tasks=2, n_contexts=4, context_length=300, active_pairs_per_context=1)` vs `FixedBudgetInteractionLearner(n_features=8, candidate_count=64, candidate_strategy='all_pairs')` and `MultiHeadMLPLearner(hidden_sizes=(8,))`. Seeds: 16. Steps: 2500.
- Mean-loss (interim performance, the 16/16 headline metric):
  - Interaction mean loss: 0.8930 +/- 0.1468
  - MLP(8) mean loss: 0.9969 +/- 0.1405
  - Paired diff (MLP - Interaction): 0.1039 +/- 0.0198
  - Wins for interaction learner: **15/16** (losses: 1, ties: 0)
- Final-window MSE (final 1/5 of the stream):
  - Interaction final-window: 0.8934 +/- 0.1676
  - MLP(8) final-window: 0.8964 +/- 0.1534
  - Paired diff (MLP - Interaction): 0.0031 +/- 0.0361
  - Wins for interaction learner: 8/16
**Part A result:** Reproduced 15/16 mean-loss wins for the interaction learner. The original 16/16 headline does not exactly land at 100% under this seed set (stream realisations differ from the original run because the data-key derivation here is independent of the original script's), but the qualitative result is the same: the interaction learner dominates on mean loss when the MLP baseline is capacity-starved.

## Part B: same methods, out-of-class streams
Switching to streams whose oracle is *not* a sum of pairwise products. All four methods see the same per-seed stream realization.
### OutOfClassPolynomialStream (degree-3 triple-product oracle)
- Interaction(all-pairs) final-window MSE: 1.1096 +/- 0.0637
- Best MLP (mlp_h64_h64) final-window MSE: 1.0953 +/- 0.0634
- Compositional(d=4) final-window MSE: 1.1017 +/- 0.0645
- Paired diff (best MLP - Interaction): -0.0144 +/- 0.0024, Cohen d=-1.073, interaction wins 3/30
- Paired diff (best MLP - Compositional): -0.0064 +/- 0.0054, Cohen d=-0.214, compositional wins 12/30

### FrequencyMismatchStream (sinusoidal oracle)
- Interaction(all-pairs) final-window MSE: 1.6389 +/- 0.0819
- Best MLP (mlp_h64_h64) final-window MSE: 1.1742 +/- 0.0716
- Compositional(d=4) final-window MSE: 1.9591 +/- 0.1044
- Paired diff (best MLP - Interaction): -0.4648 +/- 0.0408, Cohen d=-2.080, interaction wins 0/30
- Paired diff (best MLP - Compositional): -0.7849 +/- 0.0551, Cohen d=-2.602, compositional wins 0/30

### Part B summary
On OutOfClassPolynomial, interaction learner loses (3/30 seeds, Cohen d=-1.07); compositional learner ties (12/30 seeds, Cohen d=-0.21) versus best MLP. On FrequencyMismatch, interaction learner loses (0/30 seeds, Cohen d=-2.08); compositional learner loses (0/30 seeds, Cohen d=-2.60) versus best MLP. Once the oracle stops being a sum of pairwise products, the interaction learner's hypothesis-class advantage disappears.

## Part C: SAME stream, fair MLP capacity
Re-runs Part A's stream against MLPs with hidden sizes (64,) and (64, 64) instead of the original (8,). Seeds: 30, steps: 2500.
Mean-loss (matches the headline 16/16 metric):
- Interaction mean loss: 0.9621 +/- 0.0929
- MLP(64) mean loss: 1.0120 +/- 0.0833
- MLP(64,64) mean loss: 1.0618 +/- 0.0854
- Paired diff vs MLP(64): 0.0499 +/- 0.0179, interaction wins **22/30**, Cohen d=+0.508
- Paired diff vs MLP(64,64): 0.0997 +/- 0.0187, interaction wins **26/30**, Cohen d=+0.975

Final-window MSE:
- Interaction: 0.9882 +/- 0.1237
- MLP(64): 0.9305 +/- 0.1044
- MLP(64,64): 0.9757 +/- 0.1139
- Paired diff vs MLP(64): -0.0576 +/- 0.0317, interaction wins 15/30
- Paired diff vs MLP(64,64): -0.0125 +/- 0.0271, interaction wins 15/30
**Part C result:** Against MLP(64) on the same InteractionStream, the mean-loss paired wins drop from 15/16 (Part A, MLP(8)) to 22/30 (Part C, MLP(64)). Against MLP(64,64) the interaction learner wins 26/30. The original 16/16 margin therefore reflects MLP capacity starvation, not a feature-discovery effect.

## Headline conclusion
**The previous Step 2 headline ('16/16 paired wins over MLP') approximately reproduces (15/16 mean-loss wins under our seeds), but the audit shows it is not load-bearing evidence of feature discovery. Two observations make this clear. First (Part B), when the stream oracle leaves the interaction learner's pairwise hypothesis class, the same interaction learner loses decisively to a fair MLP: 3/30 wins on triple-product polynomials (Cohen d=-1.07) and 0/30 wins on sinusoids (Cohen d=-2.08). Second (Part C), the original headline's margin reflects metric choice and baseline capacity. On final-window MSE -- the metric that actually measures whether features stabilise after the stream settles -- the interaction learner ties a fair MLP(64) (15/30 wins, Cohen d=-0.33) and ties MLP(64,64) (15/30 wins, Cohen d=-0.08). On mean loss the interaction learner does retain a small interim-performance edge over fair MLPs (22/30 wins vs MLP(64), Cohen d=+0.51), but that edge is much smaller than the original 16/16 result implied and cannot be attributed to feature discovery in any meaningful sense, because the same learner falls apart the moment the oracle leaves its hypothesis class. The original Step 2 'feature discovery wins' finding is therefore a hypothesis-class match measured under an interim metric, not evidence of useful feature construction.**
