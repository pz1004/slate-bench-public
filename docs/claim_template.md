# Declaring a score-level selection claim

A claim that a selection procedure beats a reference on a reused, completed benchmark
table needs the nine fields below (Definition 3.2 and Table 2 of the paper). Six fix the
number that is computed: the artifact, the policy pool, the fixed references, the evidence
unit, the inference rule and the endpoint. The other three decide how it may be read: the
capability set, the cost field and the validation provenance. A claim without them can be
computed but not interpreted. Fill in one line per field before any test label is joined.
Under "Declared when" is the standard by which a claim counts as declaring the field; the
meta-study records in `docs/meta_study/` (protocol, coding scheme and Deviation 6) used it
for an earlier version of the paper. Section 8 and Table 7 of the paper fill in this
template for its own running claim.

## 1. Artifact

- **What must be declared:** which result table is reused, with its version and slice, and
  which of its result rows were imputed rather than measured, saying whether the table
  documents them or they were inferred (for example, from a timing signature).
- **Declared when:** the table is named with an identifier that pins it: a version, a
  snapshot or context name, a date, a DOI, a hash or an immutable repository ID.
- **Declaration:**

## 2. Capability set

- **What must be declared:** which fields of the table the selector and the references may
  read (validation scores, runtimes, meta-features, prediction vectors), with every test
  outcome excluded.
- **Declared when:** the recorded quantities the selector uses are stated, not merely implied
  by how the method works.
- **Declaration:**

## 3. Policy pool

- **What must be declared:** the selector variants or settings considered, and the rule that
  chose the reported one.
- **Declared when:** both the search space and the choice rule are stated; settings fixed in
  advance count as the rule.
- **Declaration:**

## 4. Fixed references

- **What must be declared:** the committed rules the procedure is compared with (single best
  solver, validation-best, defaults, a rule fitted on the same training data) and the oracle.
- **Declared when:** at least one non-adaptive reference and the oracle are each defined. The
  oracle counts when results are reported against the per-dataset best candidate of the
  pool; the best of the compared methods does not count.
- **Declaration:**

## 5. Evidence unit

- **What must be declared:** what counts as an independent observation (dataset, task, fold or
  seed), and the level at which results are aggregated.
- **Declared when:** the unit over which the reported comparison aggregates is stated.
- **Declaration:**

## 6. Inference rule

- **What must be declared:** the estimate and its target (for example the mean difference),
  the test or interval and its unit, the multiplicity family and any equivalence criterion.
  A test of another target, such as a signed-rank test of the typical dataset, is reported
  as such. State each comparison's fitting status (an observation, a contrast between rules
  that fit nothing, one involving a procedure refitted on the evaluation datasets, or a
  procedure fitted once and scored on held-out tasks) and design status (prespecified in a
  public protocol, fixed during the analysis, or specified after the results). Report the
  test outcome, which tests reject, apart from the conclusion it supports, the first that
  applies: observed (an exact property); difference supported (every reported test rejects,
  the uncertainty procedure is justified for the fitting and evaluation design, and the
  comparison was prespecified as primary); no difference detected (no test rejects; for refitted
  procedures the difference between them stays unresolved, and equivalence needs a justified
  interval inside a margin fixed in advance); otherwise difference unresolved. Say which
  estimand the claim targets: the difference realized on this artifact, the procedures
  fitted once and evaluated on new tasks, or the procedures refitted on other collections;
  for procedures fitted on the evaluation datasets, say whether the interval resamples their
  completed losses as if independent or accounts for refitting. Requiring two tests to agree
  is a conservative convention, not a test of validity.
- **Declared when:** a test or interval is named with the unit it is computed over.
- **Declaration:**

## 7. Endpoint

- **What must be declared:** the reported measure, and how outputs outside the candidate set
  (for example prediction ensembles) are scored.
- **Declared when:** the measure is defined, not only named.
- **Declaration:**

## 8. Cost field

- **What must be declared:** what a runtime or cost number measures: the recorded training
  time of the selected candidate, the selector's own overhead, or a budget.
- **Declared when:** the measure is defined. Not applicable to a claim about predictive
  quality only; a claim that compares procedures at a budget or reports a cost needs it.
- **Declaration:**

## 9. Validation provenance

- **What must be declared:** how each candidate's validation and test predictions were
  produced, at two levels: selection (was the stored score chosen across configurations or
  used to fit ensemble weights?) and construction (early stopping or checkpoint selection on
  validation rows; out-of-fold, out-of-bag or leave-one-out validation; fold-averaged, refitted
  or single-model test predictions). Then say whether the candidates' offsets, test minus
  validation error, are comparable (homogeneous). An offset mixes selection optimism with
  construction effects; uniform documented provenance does not by itself establish
  homogeneous offsets. Where possible, store or report a validation score that no
  construction step used, and validation-best on it, so that the effect of reused validation
  evidence can be checked with the candidates fixed.
- **Declared when:** comparability is discussed or controlled. A general remark on
  overfitting does not suffice; the field does not apply when the selector reads no
  validation scores on the dataset where it chooses.
- **Declaration:**

## Validity condition

Validity has two levels. **Fitting level:** no test-side field of an evaluation dataset may
influence any choice or fitted parameter of the procedure (the leakage boundary); a nested
implementation provides this. A claim that violates it is biased, not merely
under-specified. **Design level:** the pool, its grids and the references were not chosen
on the evaluation datasets; nesting cannot restore this once it is lost.

- **Fitting-level separation (nested, held out, or fixed in advance):**
- **Design-level independence (where were the pool, grids and references developed?):**

## Further checks

- Is a single-best or validation-best reference reported?
- Is multiplicity handled when several contrasts are tested, and are test results compared
  only within families of the same size?
- Are observations, contrasts between rules that fit nothing and contrasts involving fitted
  procedures kept apart, and is a difference involving a refitted procedure, or a rule
  specified after the outcomes were seen, reported as unresolved unless inference valid for it
  exists (for example a prespecified evaluation of procedures fitted once on held-out tasks)?
- Are related datasets (variants of one source) grouped in the outer splits, in the
  training-side estimates and in the uncertainty estimate?
- Were the selector's design or hyperparameters tuned on the evaluation datasets without
  nested separation?
