# Declaring a score-level selection claim

A claim that a selection procedure beats a reference on a reused, completed benchmark
table needs the nine fields below (Definition 3.2 and Table 1 of the paper). Six fix the
number that is computed: the artifact, the policy pool, the fixed references, the evidence
unit, the inference rule and the endpoint. The other three decide how it may be read: the
capability set, the cost field and the validation provenance. A claim without them can be
computed but not interpreted. Fill in one line per field before any test label is joined.
Under "Declared when" is the standard the paper's meta-study applies when it codes a
published claim as declaring the field (`docs/meta_study/protocol.md`, coding scheme
and Deviation 6). Section 9 of the paper fills in this template for its own headline
claim.

## 1. Artifact

- **What must be declared:** which result table is reused, with its version and slice.
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
  as such. If the compared procedures were fitted on the evaluation datasets, say whether
  the interval resamples their completed losses as if independent (a nominal interval) or
  accounts for refitting.
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

- **What must be declared:** how each candidate's validation scores were produced, and
  whether they are comparable across candidates (for example, a tuned-and-ensembled
  candidate's validation score is optimistic relative to a default's).
- **Declared when:** comparability is discussed or controlled. A general remark on
  overfitting does not suffice; the field does not apply when the selector reads no
  validation scores on the dataset where it chooses.
- **Declaration:**

## Validity condition

No test-side field of an evaluation dataset may influence any choice or fitted parameter
of the procedure (the leakage boundary). A claim that violates it is biased, not merely
under-specified.

- **Separation of the design from the evaluation datasets (nested, held out, or fixed in
  advance):**

## Further checks

- Is a single-best or validation-best reference reported?
- Is multiplicity handled when several contrasts are tested?
- Were the selector's design or hyperparameters tuned on the evaluation datasets without
  nested separation?
