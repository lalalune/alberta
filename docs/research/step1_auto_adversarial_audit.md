# Step 1 Auto Adversarial Audit

Date: 2026-05-06

Scope: independent audit of whether the current Step 1 production claim can
survive the absence of `Auto (Degris in prep.)`.

## Verdict

The production claim survives only in the narrowed form already used by the
newer package surface:

> Step 1 is complete for public, reproducible methods. The framework implements
> LMS, IDBD, Autostep, AdaGain, Adam, RMSprop, NADALINE, and online
> normalization baselines, but it intentionally does not implement or alias
> `Auto (Degris in prep.)`.

A Sutton-style reviewer should reject any stronger claim that the repo
reconstructs, approximates, or covers `Auto`. The public record does not supply
an implementable `Auto` update rule. AdaGain is a valid public Degris-coauthored
substitute/comparator, but it is not evidence about the unpublished footnote
method unless an author explicitly identifies it as such.

## Sources Checked

Local:

- `ALBERTA_PLAN.md`: local Step 1 statement and copied footnote list.
- `src/alberta_framework/steps/step1.py`: production Step 1 API.
- `src/alberta_framework/core/optimizers.py`: LMS, IDBD, Autostep, AdaGain,
  AutostepGain, and config dispatcher.
- `src/alberta_framework/core/baseline_optimizers.py`: Adam, RMSprop,
  NADALINE.
- `tests/test_config_serialization.py`, `tests/test_production_steps.py`,
  `tests/test_step1_replication.py`, `tests/test_baseline_optimizers.py`,
  `tests/test_optimizers.py`.
- `docs/research/step1_results.md`,
  `docs/research/step1_step2_alberta_plan_assessment.md`,
  `docs/guide/step1-step2-production.md`, `README.md`, `ROADMAP.md`.

Public:

- Sutton, Bowling, Pilarski, *The Alberta Plan for AI Research*,
  https://arxiv.org/abs/2208.11173. The PDF text lists
  `Auto (Degris in prep.)` only in the existing-algorithms footnote, alongside
  NADALINE, IDBD, Autostep, Adam, RMSprop, and Batch Normalization.
- Thomas Degris publication page,
  https://people.bordeaux.inria.fr/degris/publications.html. Relevant public
  entries include Autostep and related RL/robot papers; no public `Auto`
  step-size paper/code was found.
- Mahmood, Sutton, Degris, Pilarski, *Tuning-free Step-size Adaptation*,
  https://people.bordeaux.inria.fr/degris/papers/RupamAutostep.pdf. This is
  the public Autostep specification, not `Auto`.
- Jacobsen, Schlegel, Linke, Degris, White, White, *Meta-descent for Online,
  Continual Prediction*, https://arxiv.org/abs/1907.07751 and
  https://webdocs.cs.ualberta.ca/~whitem/publications/adagain.pdf. This is the
  public AdaGain specification, not `Auto`.

Searches included exact and near-name queries for `"Auto (Degris in prep.)"`,
`"Auto" "Degris" "in prep"`, `"Auto" "Thomas Degris" "step-size"`, and
site-scoped searches under `people.bordeaux.inria.fr/degris`.

## Reconstruction Audit

`Auto` cannot be reconstructed from citable public materials.

What exists:

- The Alberta Plan footnote names `Auto (Degris in prep.)` but gives no
  equations, pseudocode, paper title, venue, code repository, or parameter
  definitions.
- Autostep is public, Degris-coauthored, and fully specified, but it is named
  separately in the same footnote. It cannot also be inferred to be `Auto`.
- AdaGain is public and Degris-coauthored. It has derivations and algorithms,
  but it appeared in 2019 under its own name and objective. It is a legitimate
  comparator/substitute, not a reconstruction of `Auto`.
- `AutostepGain` is clearly documented locally as an experimental hybrid of
  public ingredients. That is acceptable only because it is not exposed as
  `Auto`.

What does not exist in the checked record:

- No public algorithm named `Auto`.
- No Degris-authored pseudocode for `Auto`.
- No public code release or package symbol that can be cited as `Auto`.
- No statement that AdaGain supersedes or renames the Alberta Plan's `Auto`.

## Current Repo Behavior

Acceptable:

- `src/alberta_framework/steps/step1.py` rejects unknown optimizer names and
  explicitly says `Auto (Degris in prep.)` is not accepted.
- `optimizer_from_config({"type": "Auto"})` raises, and tests assert this.
- The public Step 1 production docs now say "public, reproducible methods"
  rather than implying full private-footnote coverage.
- AdaGain and AutostepGain are named separately. This is the right boundary.

Weak spots:

- `Step1OptimizerName` includes `"adagiven"` and `make_step1_optimizer`
  accepts it as an AdaGain alias. This looks like a typo and is not exposed by
  the CLI. It is not an `Auto` bug, but it weakens the "no invented aliases"
  posture.
- `tests/test_step1_replication.py` auto-skips when canonical JSON files are
  absent. That is practical for fresh checkouts, but it means a plain pytest run
  can pass without validating the scientific Step 1 claims.
- `tests/test_step1_replication.py` claims to enforce IDBD robustness, but the
  current robustness assertion only checks finite grid-point count, not the
  documented robustness ratio/working-range claim.
- `tests/test_config_serialization.py` does not currently round-trip Adam,
  RMSprop, or NADALINE through `optimizer_from_config`, even though the lazy
  dispatcher supports them.
- `tests/test_production_steps.py` only smokes the default Autostep production
  path and the `auto` rejection path; it does not smoke every accepted Step 1
  optimizer name.
- `docs/research/step1_results.md` says "Pytest regression covering all of the
  above", but throughput and absent-artifact behavior are not enforced as hard
  scientific gates by ordinary pytest.
- `ROADMAP.md` still says the goal is comparison against the "full set of
  baselines named in footnote 11." That should say "full public/reproducible
  subset, with `Auto` excluded because unpublished."

## Reviewer-Rejection List

A strict reviewer should reject:

- "We implemented Auto."
- "AdaGain is Auto."
- "AutostepGain is Auto-inspired and therefore closes Auto."
- "Full footnote-11 comparison is complete" without the qualifier that the
  unpublished `Auto` item is excluded.
- "IDBD/Autostep beat LMS" as a universal Step 1 statement. The committed
  results support IDBD on Sutton-style sparse relevance tasks and Autostep as
  the robust tuning-light method, not dominance everywhere.
- Any evidence gate that passes a scientific claim after silently skipping
  missing canonical JSON.

## Reviewer-Acceptance List

A strict reviewer could accept:

- The package implements a reproducible Step 1 public-method comparison.
- `Auto` is excluded for source-availability reasons, not because the framework
  forgot a known public algorithm.
- AdaGain is a public Degris-coauthored comparator and can be included under
  its own name.
- Autostep is the paper-faithful public algorithm for the tuning-free
  step-size-adaptation claim.
- The production API is intentionally narrower than the research scripts and
  rejects `Auto` aliases.
- The defensible headline is:

> IDBD reproduces the Sutton sparse-relevance result and beats LMS on the noisy
> Sutton variant; Autostep is the strongest tuning-light method in the tested
> robustness grid; online normalization is necessary for scale-shift settings;
> and Step 1 is complete for public, reproducible methods, not for unpublished
> `Auto`.

## Exact Patches Needed

No core-code patch is required to avoid fabricating `Auto`.

Recommended small follow-up patches:

1. Remove the `"adagiven"` alias from `Step1OptimizerName` and
   `make_step1_optimizer`, or document it as a deprecated typo alias and add a
   test. Prefer removal before release because the code claims no invented
   aliases.
2. Add production smoke parametrization over
   `("lms", "idbd", "autostep", "adagain", "adam", "rmsprop", "nadaline")`.
3. Add `optimizer_from_config` round-trip tests for Adam, RMSprop, and NADALINE.
4. Split Step 1 evidence tests into two modes: fresh-checkout optional tests
   may skip missing JSON, but release/claim validation should fail if canonical
   JSON is missing.
5. Replace the robustness test with assertions matching the documented
   robustness metrics, or soften the docs to match what is actually tested.
6. Patch `ROADMAP.md` wording from "full set of baselines named in footnote 11"
   to "full public/reproducible subset of footnote 11; unpublished `Auto` is
   intentionally excluded."
7. Patch `docs/research/step1_results.md` to avoid saying pytest covers
   throughput and all evidence as hard gates unless those checks are made
   non-skipping and thresholded.

## Final Assessment

The absence of `Auto` is not a production blocker if the claim stays narrow.
It would become a blocker only if the project presents Step 1 as a literal full
footnote-11 replication. The repo is mostly on the right side of that boundary:
it rejects `Auto`, keeps AdaGain under its own name, and documents the caveat.
The remaining risk is wording and gate strength, not a missing implementation.
