# Step 2 Moonshot Experiments

Purpose: run very small, falsifiable Step 2 pilots that are mathematically
motivated but cheap enough to discard. A pilot is interesting only if it beats
the existing fair MLP comparator on a paired smoke benchmark and has a clear
mechanism worth scaling.

Each worker owns one script under `examples/The Alberta Plan/Step2/moonshots/`,
one output directory under `outputs/step2_moonshots/`, and one short result note
under `docs/research/step2_moonshots/`.

| ID | Idea | Core Hypothesis | Smoke Success Signal |
|---|---|---|---|
| M01 | Centered target code | MSE one-hot wastes gradient on negative classes; `{-1, +1}` or centered simplex targets may learn online digits faster. | Lower final-window MSE or higher accuracy than one-hot MLP. |
| M02 | ECOC target code | Random redundant error-correcting class codes can make online regression targets better conditioned. | ECOC decode beats one-hot MLP accuracy on digits smoke. |
| M03 | Random Fourier lift | A frozen kernel feature lift can beat learned hidden features under short online horizons. | Linear head on Fourier features beats fair MLP on at least one digits drift regime. |
| M04 | Hashed quadratic lift | Explicit multiplicative features test whether Step 2 is mostly missing second-order interactions. | Quadratic lift beats MLP on interaction stream or digits mask drift. |
| M05 | Residual imprint features | High-error examples define localized corrective features, a one-step matching-pursuit feature discovery rule. | Residual dictionary reduces final-window loss versus MLP on digits drift. |
| M06 | Fast/slow gated ensemble | Nonstationarity needs two time constants; gate by recent loss or margin rather than one learner. | Gated fast/slow blend beats both components and baseline MLP. |
| M07 | Surprise-triggered low-rank reset | Reset only the smallest-utility hidden directions after loss spikes, preserving stable weights. | Reset variant improves over UPGD/MLP without collapse. |
| M08 | Prototype context augmentation | Soft online prototypes can reveal latent regimes; append context probabilities to the input. | Augmented MLP wins on class-blocked/permuted digits. |
| M09 | Hidden-weight orthogonalization | Online feature collapse may be the bottleneck; periodic row orthogonalization preserves representational rank. | Orthogonalized MLP improves final-window loss without destabilization. |
| M10 | EMA parameter prediction | Prediction with slow averaged weights may improve external held-out performance while online weights adapt. | EMA evaluation beats raw MLP on held-out final-phase accuracy. |

Promotion rule: a positive pilot must be rerun with more seeds, paired
statistics, and at least one negative control before changing canonical Step 2.
