# Step 2 Distribution-Free Limits

CLAIM-REJECTED-DISTRIBUTION-FREE-UNIVERSALITY

Step 2 does not prove distribution-free universality for arbitrary online
streams.

COUNTEREXAMPLE-CAUSAL-ONLINE-INDISTINGUISHABILITY

Two streams can share the same causal history and next observation while requiring
different binary targets. Any causal predictor must make the same prediction on
both branches, so one branch suffers at least `1 / 4` square loss.

COUNTEREXAMPLE-ARBITRARY-ADVERSARIAL-DRIFT

An adaptive target can flip after prediction and force constant loss. This
violates a drift model.

COUNTEREXAMPLE-HIDDEN-CONTEXT-ALIASING

If hidden context is not observable, two targets can alias under the same
observation and create irreducible error.

REPLACEMENT-THEOREM-CONDITIONAL-LEARNABILITY

The valid replacement theorem is conditional learnability under
ASSUMPTION-OBSERVATION-SUFFICIENCY, ASSUMPTION-BOUNDED-LOSSES,
ASSUMPTION-ADMITTED-FEATURE-CLASS, ASSUMPTION-MODELED-DRIFT,
ASSUMPTION-RECURRENCE-EVIDENCE, and
ASSUMPTION-REGRET-ESTIMATION-GUARANTEE.

In plain language, the required conditions are modeled drift,
recurrence/evidence, bounded losses, admitted feature class, and
regret/estimation guarantee.

