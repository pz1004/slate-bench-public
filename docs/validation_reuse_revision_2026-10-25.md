# Validation-reuse revision and verification record — 25 October 2026

The revised manuscript is `paper/main.tex` and its rendered artifact is
`paper/main.pdf`. This record responds to the current `paper/comments.md`
submission-readiness assessment. The older `paper/revision_traceability.md`
remains a historical record of earlier comments; its old section numbers and
conclusion language are not the current manuscript's claims.

## Main result and scope

All 181 screening decisions were reassessed. Among the 46 trained tasks, 11 are
verified eligible, 12 ineligible and 23 unresolved. Across the full screening
register, the counts are 11 eligible, 128 ineligible and 42 unresolved. These are
source-audit judgments under the documented standard, not proof of undisclosed
independence. The 135 originally excluded tasks were reviewed from metadata and
source documentation; raw-record inspection covers the 46 actually trained. One untrained task (46520)
has no cached feature schema; its conflicting source citation and unavailable
feature endpoint remain explicitly unresolved. This final documentation update
changes no trained cohort or experimental result.

| Cohort | Tasks | C1: reused minus fresh rank regret | 95% bootstrap interval | Test outcome |
|---|---:|---:|---|---|
| Verified eligible, main | 11 | +0.02193 | [-0.01979, 0.07166] | Neither rejects |
| Original | 46 | +0.03414 | [0.01432, 0.05639] | Both reject |
| Three reported exclusions | 43 | +0.03365 | [0.01477, 0.05527] | Both reject |
| Eligible plus unresolved | 34 | +0.03322 | [0.01125, 0.05796] | Both reject |

C2 is not detected in any cohort. Equivalence is not established in the verified
cohort. These results do not establish a difference between cohorts: membership,
size and uncertainty change together. The positive broader-cohort findings cannot
be promoted into a verified-cohort claim. All changed-cohort analyses are post hoc;
the original labels remain only in preserved historical outputs.

No replacement tasks, new large training runs, external experiments, policy
refitting runs or new coverage simulations were added. Existing candidate scores,
replicate intervals and frozen-fit outputs were used for verification.

## Comment-by-comment response

| Assessment comment | Change and verification evidence | Remaining limitation |
|---|---|---|
| §1: journal format, checklist and supported claims | Retained JAIR class, structured abstract and checklist; updated experimental claims, data availability and disclosure; ran the JAIR static checker and PDF build. | This is a technical revision, not a prediction of editorial acceptance. |
| §2: center the frozen-candidate identification problem | New title, abstract, introduction and contributions; experiment now follows concise definitions and related work, before the detailed artifact protocol. | The verified cohort does not retain the proposed positive demonstration; the diagnostic and eligibility dependence are the defensible contribution. |
| §3.1: US Stocks 46527 leaks the target definition | Creator documentation and actual `2019_price_var`/`class` relation checked; Q exclusion recorded. | The task was excluded from corrected analyses, not rebuilt or retrained. |
| §3.2: Musk 40666 has repeated conformations and identifiers | Original UCI documentation and actual retained `molecule_name`/`conformation_name` checked; 102 groups, M/Q failure. | No claim that numeric identifier magnitude itself has a causal meaning. |
| §3.3: PAD 46864 repeats patients and lesions | Exact OpenML description checked against retained schema: 2298 rows, 1373 patients, 1641 lesions; target is `biopsed`, with `diagnostic` retained. M exclusion. | Deleted patient/lesion keys do not make rows independent. Within-task dependence is not automatically dependence among all task means. |
| §3: audit the remaining tasks, not just three | Full 181-row decision register and actual target/predictor inventory for all 46 trained tasks; nine additional trained-task failures; 23 unresolved trained tasks excluded from the main cohort. Original exclusion-rationale mistakes retained. | Inaccessible or ambiguous documentation remains unresolved; no claim of a complete eligible-data census. |
| §3: reproduce the three-exclusion calculation | All seven original CSVs reproduce exactly from frozen candidates; C1 on 43 tasks reproduces +0.03365 [0.01477, 0.05527]. Four separately named analysis directories and manifest/input hashes. | Original and three-exclusion results do not certify the remaining tasks. |
| §4: actual verification and author disclosure | Disclosure records the author's confirmation of personal review of every previously excluded addition. Computational checks include formal tests, source verification, grouped-split boundaries, exact frozen-fit replay, 168 coverage rows, estimands and conclusion rules. | Author confirmation is distinct from AI checks; none is represented as a fresh training replication. |
| §5: incremental significance and internally developed policies | Practical question leads; formal model, artifact diagnoses and nine-field declaration support it. Established selection-bias literature remains credited. Internal policy development is explicit. | The verified data do not establish the stronger proposed finding of harm missed by class offsets. No externally developed policy demonstration was authorized; scientific significance remains an editorial judgment. |
| §6.1: simulation does not validate the exact audit | Retained 8-policy/20-split and shared-fit bridge limits, versus the 45-policy/100-split main design; procedure-level differences remain unresolved. Coverage labels and SE ratios recomputed from stored replicate/truth files. Adequacy-rule history is explicit. | No new calibration of the exact audit; stored configuration records declaration timing, which is not independently certified by rerunning summaries. |
| §6.2: fresh experiment does not identify TabArena causation | Kept the six-family, classification, holdout-construction, 100-configuration boundary. TabArena class offsets remain observational. | No matched construction experiment; no attributed fraction of TabArena regret. |
| §7: clipped central-results table | Replaced with a compact main-text summary and a genuine multipage `longtable` containing the detailed comparison record. Added a multipage task eligibility table. | Original full analysis outputs remain separately available rather than filling the main text. |
| §7: “equally precise” | Replaced with “computed on an equally sized validation set”; explains why adaptive reuse prevents an equal-precision claim. | No estimator-variance equivalence asserted. |
| §7: dense, repetitive structure | Moved extended related work and diagnosis table to supporting appendices; shortened opening and conclusion; consolidated eligibility and inference limits. | Artifact-specific conditions remain where necessary for interpretation. |
| §8: readiness decision | All requested corrections and verification are recorded, including the weaker main finding. | The broader effect cannot substitute for evidence in the verified cohort; this record does not declare the work scientifically ready for acceptance. |

## Verification of the previously excluded additions

- **Formal additions:** inspected Proposition 1(iv)'s conditional Gaussian
  calculation and derivative, Proposition 2's independent-training assumption,
  and the random-offset remark. `tests/test_propositions.py` checks the closed
  forms, unequal-quality minimizer and counterexample with fixed-seed numerical
  integration/Monte Carlo. This is a check of the stated minimal model, not of
  either benchmark's data-generating process.
- **Prediction provenance:** rechecked the primary
  [TabArena paper](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf),
  §§2.1, A.6, C.1–C.4, and the primary
  [TabRepo paper](https://raw.githubusercontent.com/mlresearch/v256/main/assets/salinas24a/salinas24a.pdf),
  §3 and Appendix E. The documented disagreement about forest bagging is retained;
  KNN and FastAI inferences stay labelled. The stored slice check covers 315
  dataset-fold tasks from 105 binary datasets and five configurations; it does
  not prove construction for every one of the 1530 configurations.
- **Grouping:** verified that all 24 recorded outer splits keep declared groups
  apart, the grouping-file hash matches, and the inner grouping setting is
  `group`. Perturbation tests check that a group mate's test errors cannot affect
  an inner target's fitted choice. Group definitions remain partly inferential.
- **Frozen fit:** reproduced `fitted_choices.csv`, `heldout_datasets.csv` and
  `contrasts.csv` exactly using `scripts/tabrepo_frozen_fit.py` on stored outputs.
  Fold 0, seed 20261022, and the unchanged driver hash match the declaration.
  Other folds remain separate. The ensemble-inclusive nested/reference identity
  holds; this is a post hoc frozen-fit result, not refitted-procedure inference.
- **Coverage:** independently recomputed all 168 stored coverage proportions,
  standard-error ratios and final adequacy labels from replicate intervals and
  the appropriate independent truth estimates, including the matched bridge
  subset. No new simulated tables were generated.
- **Estimands and conclusion assignment:** confirmed that C1/C2 average within
  tasks before equal task weighting; every cohort filters before choices, ranks,
  offsets, shares, endpoints and size strata. Corrected decisions explicitly
  carry `confirmatory: false`; any rejection remains unresolved under the
  manuscript's design-status rule. The asset builder checks the decision flags
  against the tests and filters its own candidate-level calculations too.

## Reproduction and integrity

```sh
make fresh-validation-corrected
.venv/bin/python scripts/tabrepo_frozen_fit.py \
  --run analysis-output/v2/score_level_tabrepo_frozen_fit \
  --score-only-run analysis-output/v2/score_level_tabrepo_frozen_fit_score_only \
  --output-dir /tmp/validation_revision_frozen_fit
.venv/bin/python scripts/build_paper_assets.py --meta-verified-only
.venv/bin/python scripts/verify_validation_revision.py \
  --frozen-fit-replay /tmp/validation_revision_frozen_fit
.venv/bin/python -m pytest
.venv/bin/python scripts/static_latex_check.py paper/main.tex \
  --bib paper/main.bib --venue jair --fail-unused-bib
PATH=/home/pz1004/Workspaces/PLGINet/.venv-tinytex/bin/x86_64-linux:$PATH \
  latexmk -cd -pdf -interaction=nonstopmode paper/main.tex
```

`audit_2026-10-25/verification.json` links the audit, cohort manifests, input scores,
analysis code, original outputs, corrected outputs and supporting checks with
SHA-256. The exact protocol at training and the pre-correction protocol are both
preserved. Historical source checks, task lists, training code, configuration,
training outputs and original analyses were not overwritten. Appended deviations
supersede the old source-screening and author-review assertions explicitly.

The full test run exercised 194 tests. It found one setup error in a newly added
asset-builder test (a missing required constructor argument); after fixing that
fixture, all 14 cohort tests passed, along with the asset-builder tests on the
final code. All other 180 tests passed in the full run. The sole suite warning
was an existing Matplotlib deprecation. Test logs are retained locally; the final
verification status is recorded with the rendering checks below.

## Rendered artifact acceptance

The PDF and static checks pass. Render inspection covers the title/abstract,
compact cohort and main-summary tables, both pages of the detailed comparison
record, both pages of the task audit, and the corrected endpoint/choice tables.
The LaTeX log contains no overfull boxes, oversized floats or unresolved
references. PDF bounding-box inspection confirms content stays on the page.
