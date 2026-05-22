# Step 2 Moonshot Synthesis

This note summarizes ten deliberately small Step 2 pilots. These are not
canonical results; they are triage runs intended to find simple mechanisms
worth scaling.

## Summary Table

| ID | Experiment | Smoke Verdict | Reason |
|---|---|---|---|
| M01 | Centered targets | Weak positive | Final-window accuracy improved by `+0.0025` overall, but code MSE worsened. |
| M02 | ECOC target codes | Negative | ECOC lost paired final-window accuracy `0/6` despite better normalized MSE. |
| M03 | Random Fourier lift | Negative | Frozen RFF readouts lost to raw MLP on final-window and held-out accuracy. |
| M04 | Hashed quadratic lift | Positive, narrow | Beat MLP on interaction stream final-window loss with `3/3` paired wins. |
| M05 | Residual imprint features | Mixed, not promoted | Accuracy improved, but MSE worsened in every paired run. |
| M06 | Fast/slow gated ensemble | Negative | Gated blend improved MSE vs individual MLPs but lost to averaging and did not clear accuracy criterion. |
| M07 | Surprise-triggered reset | Near-neutral | Beat UPGD and slightly beat MLP MSE, but did not improve MLP accuracy. |
| M08 | Prototype context augmentation | Negative | Helped IID slightly, failed drift criteria on class-blocked/permuted pixels. |
| M09 | Hidden orthogonalization | Negative | Increased feature diversity, but hurt final-window MSE and accuracy. |
| M10 | EMA parameter prediction | Interesting negative | Held-out accuracy improved, but label-drift final-window adaptation degraded too much. |

## Candidates To Scale

M04 is the cleanest positive. The hashed quadratic lift beat the MLP on the
interaction stream with lower final-window loss for both tested feature budgets.
This is mathematically unsurprising because the stream is pair-product based,
but that is exactly why it is useful: it shows the current learned MLP can be
beaten by an explicit interaction representation under the same online budget.
The next run should include negative controls where targets are not quadratic,
and a fairer parameter-matched MLP grid.

M01 is a low-cost target-encoding improvement. Centered targets improved
accuracy slightly on IID and label-drift digits, with a very small overall
effect. This should be folded into a broader target-code ablation rather than
promoted alone.

## Deprioritized

M02, M03, M05, M06, M07, M08, M09, and M10 should not be scaled in their
current form. ECOC and RFF were weaker than the fair MLP; residual imprints,
fast/slow gating, and surprise resets did not clear their paired criteria;
prototypes did not solve drift; hard orthogonalization preserved rank at the
cost of useful learned geometry; and EMA prediction improved held-out accuracy
only by sacrificing final-window adaptation.

Their individual scripts and raw outputs have been removed from the active
tree.  This synthesis note keeps the negative results so we do not rediscover
the same failures.

## Recommended Next Experiment

Run a focused three-way suite:

1. Fair MLP baseline with one-hot targets.
2. Fair MLP with centered targets.
3. Hashed quadratic augmentation with centered targets.

Use the interaction stream, a non-quadratic negative-control stream, and digits
permuted-pixels or label-drift. Promote only if the augmented method beats the
fair MLP on the interaction stream without losing on the negative control.
