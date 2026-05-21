# Step 2 Experimental Direction Results

This summarizes the nine parallel Step 2 directions run after the canonical
UPGD/MLP/compositional audit. The goal was not to prove universality, but to
identify which directions actually improve over fair MLP or reduce the cases
where UPGD loses to MLP.

## Result Matrix

| Direction | Result | Improvement? | Critical reading |
|---|---|---:|---|
| 1. Expert mixture | Discounted Hedge over MLP and UPGD routed to UPGD on synthetic and MLP on digits in smoke. | Yes | Best practical universality direction. Needs larger seeds/regimes. |
| 2. UPGD ablations | Low/no-noise variants improve digits accuracy but not digits MSE; all UPGD variants keep synthetic win. | Partial | Synthetic win is not just perturbation. Tuning alone does not solve digits. |
| 3. Guided compositional residual imprint | Residual-imprinted candidate output weights changed results by about zero. | No | Compositional DAG remains a substrate, not a winning method. |
| 4. External nonstationary digits | UPGD beats MLP only on tiny IID online accuracy and class-blocked held-out accuracy; loses most regimes. | Mostly no | Strong anti-universality evidence for UPGD alone. |
| 5. Rare/future utility | `active_inverse_future` improves rare benchmark holdout mean MSE `1.0582 -> 0.9844`; rare-task-only gain tiny. | Partial | Promising opt-in utility direction, not a default change. |
| 6. Inferred context gating | Inferred recurring phase matches oracle context-gated readout: last-cycle MSE `0.0119` vs hidden single `1.1964`. | Yes | Strong for hidden recurring contexts when phase is inferable. |
| 7. Dynamic sparse rewiring | Beats MLP on synthetic polynomial, ties/slightly trails UPGD; loses badly to MLP online on digits class-blocked. | Partial/no | Worth a small sweep, not a leading solution. |
| 8. GVF feature discovery eval | No discovery method beats linear GVF baseline; tanh features beat MLP baseline only. | No | Evaluation harness is useful; Step 3 feature discovery not solved. |
| 9. Scaled expert mixture | Low-noise 10-seed run improves/ties fair MLP final-window MSE on all 8 regimes; retention-aware deployment closes class-blocked held-out regret. | Yes | Strongest current Step 2 candidate against the current fair-MLP + retention matrix; still not a general recursive feature-construction proof. |

## Current Ranking

1. Retention-aware low-noise expert mixture / portfolio selection.
2. Inferred context gating for recurring hidden contexts.
3. UPGD with low-sigma/no-noise tuning inside the portfolio.
4. Rare/future utility as an opt-in resource-manager knob.
5. Dynamic sparse rewiring as a small follow-up baseline.
6. Guided compositional generation, still requiring a better generator.
7. GVF feature discovery, currently an evaluation harness with negative results.

## Main Conclusion

The best path to a more universal Step 2 learner is not a single learner that
always beats MLP. The evidence favors a protected MLP fallback plus adaptive
portfolio routing over UPGD and specialized feature/context modules. The
retention-aware low-noise expert mixture is the first current candidate that
improves or ties fair MLP final-window MSE across the combined synthetic and
shifted-digits probe while also closing the observed class-blocked held-out
retention gap.

The remaining weakness has moved up a level: the class-imbalance retention
router is hand-specified. A stronger portfolio should learn multi-objective or
delayed-evaluation routing rather than relying on a fixed trigger.

## Low-Noise Expert Mixture Follow-Up

Command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_expert_mixture.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --perturbation-sigma 1e-4 \
  --output-dir output/step2_expert_mixture_low_noise_10seed
```

Key paired results versus fair MLP:

| Regime | Final-window MSE diff | Wins/losses |
|---|---:|---:|
| synthetic_polynomial | `+0.0728` | 10/0 |
| synthetic_frequency | `+0.0535` | 7/2 |
| synthetic_compositional | `+0.0000` | 0/0/10 ties |
| digits_iid | `+0.0020` | 10/0 |
| digits_class_blocked | `+0.0000` | 0/0/10 ties |
| digits_permuted_pixels | `+0.000004` | 8/0/2 ties |
| digits_mask_noise | `+0.0034` | 10/0 |
| digits_label_drift | `+0.0000` | 0/0/10 ties |

Positive differences mean MLP minus mixture, so positive favors the mixture.
Digits IID test accuracy improves by `+0.0106`; mask-noise test accuracy
improves by `+0.0108`; blocked/permuted/label-drift digits tie MLP on mean
test accuracy.

The retention-aware follow-up uses the same online tracking run and adds
`--retention-router class_imbalance --retention-upgd-deployment-weight 1.0`.
It preserves these final-window MSE results and switches class-blocked held-out
deployment to UPGD, eliminating the best-expert test-accuracy regret on that
regime. This is sufficient to promote the mixture as the current fair-MLP and
retention candidate, but not enough to claim general Step 2 closure.
