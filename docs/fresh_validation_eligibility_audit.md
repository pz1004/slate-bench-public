# Eligibility audit, 25 October 2026

This is a post hoc correction of the source check in Section 3.3 of the public
protocol. It does not change the models, partitions, candidates, seeds, endpoints,
contrasts, Holm families, resampling count or equivalence margin. No new model
training was performed. The author confirmed review of all additions previously
excluded from the manuscript's verification disclosure; that confirmation is
separate from the computational and documentary checks recorded here.

The reviewed decision register is `configs/fresh_validation_eligibility_audit.csv`:
one decision for every one of the 181 screened IDs, with original decision,
criterion, source citations and a source-specific rationale. The companion
`analysis-output/v2/fresh_validation/audit_2026-10-25/evidence.json` inventories
complete cached descriptions and feature schemas by SHA-256 and records the
actual target, retained predictors, dropped columns, class counts and duplicate
counts for every original trained task. `scripts/fresh_validation_audit.py`
materializes this evidence and the 46-row cohort manifest. It cannot consult
candidate scores. The trained-task decisions were fixed before the corrected cohort
analyses were calculated; the original results and the colleague's three-task
sensitivity were already known, so this is not a blinded or prespecified audit.

## Decision standard and limits

- **Eligible** means the available source documentation supports a real tabular
  observation and usable target, the retained schema passes inspection, and no
  documented T/S/D/A/M/Q violation was identified after checking the declared
  benchmark sources and the other selected tasks. It is not a guarantee that
  anonymized data contain no undisclosed dependence or leakage.
- **Ineligible** requires a documented criterion failure or direct target/schema
  evidence. A perfect association alone is not automatically leakage: the China
  outcome alias and transaction risk-score thresholds are excluded because of
  their outcome semantics as well as their complete mappings. Near-disjoint
  credit-score ranges in task 46519 are unresolved, not a confirmed failure.
- **Unresolved** means documentation is missing, ambiguous or conflicting for a
  material criterion. It is excluded from the main corrected cohort and included
  only in the explicitly named eligible-plus-unresolved sensitivity.
- M concerns the records and retained predictors. Tabular morphology or wavelet
  summaries do not become raw images merely because images were their source.
  Removing a subject identifier does not remove grouping. Repeated feature
  vectors alone do not establish repeated people; distinct people can have the
  same coarse measurements. These interpretations are recorded post hoc.
- The declared overlap scope is TabArena **v0.1** and TabRepo's **244-dataset
  context**, not every later collection bearing their names. A different
  competition from the same company is not automatically the same raw source.
  Unidentified source collections (including the concealed `rl` task) remain
  unresolved. The source registry and its hash are preserved.
- D decisions refer to later variants of sources already included in the
  original screening order. An earlier included source becoming ineligible does
  not authorize an untrained replacement. All exclusion decisions are recorded,
  including mistaken or unsupported rationales, but no replacement is trained.
- Original excluded tasks have metadata/source inspections only; raw-record
  inspection is limited to the 46 tasks actually trained. Some primary pages
  are inaccessible or do not describe collection/label generation. Those limits
  are recorded as unresolved instead of claiming a complete empirical census.

## Verified failures and unresolved cases

The three reported problems are confirmed: 46527 retains `2019_price_var`, which
completely determines `class`; 40666 retains both Musk identifiers and contains
102 molecule groups; 46864 has 2298 rows for 1373 patients/1641 lesions after
patient/lesion IDs were deleted. PAD-UFES-20's actual target is **biopsed**, with
`diagnostic` among its predictors. The grouped-row failure does not depend on
calling it a cancer-diagnosis target.

Additional confirmed failures include household clustering in IPUMS (382),
repeated daily addresses in BitcoinHeist (45037), generated chess positions (3),
hypothetical mushroom samples (43922), retained outcome information in the
mortgage and SUPPORT tasks (46528, 46860), and target encodings retained in China
credit, transaction risk and the longitudinal survey (46541, 46361, 43892).
The register gives the precise criterion and citations for each. Within-task
partitioning failures are not, by themselves, evidence of dependence between
all task-level means.

Unresolved records include potentially repeated source objects without keys,
concealed or poorly documented collection histories, uncertain synthetic
origins, and inadequately defined targets. The main cohort is the 11 eligible
original tasks; 12 original tasks fail and 23 are unresolved. No choice was made
using a task's model performance or its contribution to C1. A final documentation
check classified untrained task 46520 as unresolved because its feature schema
was unavailable and its citation conflicted with the claimed source; this changed
no trained-task cohort or estimate. The full register has 128 ineligible and 42
unresolved tasks in addition to the 11 eligible tasks.

The original exclusion rationale is explicitly corrected for 43607 (string
categories alone are usable), 1444/1453 (anonymous metrics do not verify NASA
identity), 897 (common institutions do not establish the same survey), 45927
(matching row counts do not prove a test upload), and 42883 (the documented time
series excludes it, even though the original test-upload argument was weak).
These are rationale errors, not six certified eligible replacements.

## Reproduction

```sh
.venv/bin/python scripts/fresh_validation_audit.py
for cohort in original three-exclusion eligible eligible-unresolved; do
  .venv/bin/python scripts/fresh_validation_analysis.py \
    --run-dir analysis-output/v2/fresh_validation \
    --cohort-manifest configs/fresh_validation_cohort_2026-10-25.csv \
    --cohort-selection "$cohort" \
    --output-dir "analysis-output/v2/fresh_validation/analysis_2026-10-25_$cohort"
done
make assets META_FLAGS=--meta-verified-only
```

`original` reproduces all 46 stored tasks; `three-exclusion` removes only
46527/40666/46864; `eligible` is the corrected main cohort; `eligible-unresolved`
removes all confirmed failures. Filtering occurs on OpenML ID before any
choices, ranks, offsets, shares, secondary endpoints or size strata are formed.
Every corrected result records its manifest, candidate-score, configuration and
analysis-code hashes. The manifest links to the full decision and evidence
hashes. Historical protocol, manifests, candidates and original analyses remain
separately addressable. The dated protocol deviation records the changed
population interpretation and post hoc status.

Both tests rejecting is a **test outcome**, not a renewed confirmatory label.
Corrected rejections remain unresolved under the manuscript's design-status
rule. Nonrejection does not establish equivalence. The original 46-task label is
retained only in the historical output; it cannot validate the failed cohort.
