# Revision and response record — 26 October 2026

The local manuscript is `paper/main.tex`, rendered as `paper/main.pdf`. The main
text, including availability and disclosure, now ends on page 9 rather than
page 26; the full PDF is 54 rather than 56 pages because detailed evidence remains
in appendices. The pre-edit manuscript and PDF are retained locally in
`paper/history/2026-10-25/`. The October 25 audit, decisions and outputs remain
unchanged. This record supersedes the October 25 response's contribution framing,
without rewriting that historical record.

The verified result remains inconclusive: C1 is approximately **+0.37
[−0.34, +1.22] rank positions** among 18 candidates. The interval permits benefit
and appreciable harm. Neither increased regret nor equivalence is established.
The revision does not claim submission readiness or resolution of the colleague's
scientific-significance objection.

## Local delivery and exact identity

- Manuscript: `paper/main.tex`; PDF: `paper/main.pdf`.
- Evidence archive: `dist/validation-reuse-correction-2026-10-26.tar.gz`.
- SHA-256: `083d7679618c65211c6ae99ffb1d6dd8062aaf0358f4183a7c437625838b923f`.
- Checksum sidecar: `dist/validation-reuse-correction-2026-10-26.tar.gz.sha256`.
- Verification: `analysis-output/v2/fresh_validation/packaging_2026-10-26/`.
- Standalone instructions: [package guide](validation_revision_package.md).

The corrected materials accompany this **local revision and are not yet
published**. No push, public release, submission or external communication was
performed. The evidence archive excludes the manuscript, private review comments,
editorial response records, raw task rows and unrelated caches. Its 447 explicit
input paths include previously untracked corrections; the resulting manifest
hashes 457 files, including generated membership lists and package metadata.
The archive checksum also protects the manifest. Dependency versions and applicable
notices are included. No Git history or originating checkout is needed to verify
or reproduce it.

## Comment-by-comment response

| Reassessment point | Completed change and evidence | Remaining limit |
|---|---|---|
| §1: retain the corrections already made | Preserved all 181 audit decisions, the 11/12/23 trained-task classification, post hoc status, original results, unequal-size/precision distinction, and the author's existing personal-review attestation. | Recomputing scores does not independently certify training or source judgments. |
| §2: interpret the verified interval in rank positions | The asset builder now generates both the estimate and interval multiplied by 17. Abstract, introduction, results and conclusion report +0.37 [−0.34, +1.22], explicitly permitting benefit and appreciable harm. | The result establishes neither harm nor equivalence; secondary endpoints do not resolve it. |
| §3: the corrected materials are absent from the public repository | Prepared the dated, checksummed local archive with the complete register, four explicit cohort exports, cached metadata, inspected feature/target records, source URLs, documentation gaps, protocol snapshots, deviations, code and original/corrected outputs. Availability, README and checklist now distinguish the local correction from earlier public materials. | **Public release is pending.** Local packaging does not satisfy a requirement for public access. |
| §4.1: eligibility dependence is not an established general contribution | Removed that claim from the principal contribution, abstract and conclusion. The frozen-candidate comparison leads, with its limited verified estimate; the audit corrects this study's evidential basis. | Cohort comparisons do not identify eligibility as a cause of effect variation. |
| §4.2: validation overfitting is established | Retained the established literature and TabArena's own warning as background. Explained the operational comparison: change selection evidence while keeping deployed candidates fixed. | This implementation does not establish a new general mechanism or a precise average effect. |
| §4.3: the declaration is largely a synthesis | Retained one nine-field declaration table and the two-level distinction between selection and prediction construction. Consolidated overlapping summaries and positioned reporting principles as synthesis. | General methodological utility has not been demonstrated in an independent application. |
| §4.4: policies were developed internally | Kept that limitation in the opening, supporting results and discussion. Preserved TabRepo's exact row-selection identity and its scope to the enumerated pools and completed runs. | No general claim about published selection methods or independent evaluation is made. |
| §4.5: theory is explanatory | Kept the unequal-quality optimum, full propositions, assumptions, proofs, the quality-prior interpretation and the distinction between error-space theory and rank-space correction. The concise model now follows the experiment; details are in Appendix D. | The model supplies neither a validated correction nor a standalone general guarantee. |
| §5: inference limitations should remain explicit | Retained the working independence approximation, refitted versus frozen estimands, post hoc conclusion precedence and all sensitivity results. The eight-policy/20-split simulation and shared-fit bridge remain distinct from the exact 45-policy/100-split audit. | The exact refitted procedure is not validated by the stored simulation. |
| §6 and §7: stronger significance needs substantive evidence | Reframed the discussion and conclusion to state what the current evidence supports. No additional policies, training, simulations, replacement datasets or external evaluation were introduced. | **Scientific significance remains unresolved.** Greater precision or a consequential independent application would require work outside this revision. |
| §7: consolidate the dense manuscript | The experiment remains Section 3 after definitions and related work. Concise provenance, supporting diagnoses and the declaration follow. Detailed protocol, policies, fitting and artifact results moved to appendices. Former overlapping Tables 2, 4 and 7–10 are consolidated into current Tables 2–4 and supporting prose; unique estimates and conditions remain in the detailed record. | The full PDF remains long because technical evidence is retained. |
| §7: “Table O” and “primary inference” | Put the cohort label immediately after its generated longtable caption, removed the outer label, and checked the compiled reference against its caption: both say **Table 15**. Changed the inference sentence to identify the supporting TabArena audit. | No remaining reference or layout failure was detected. |

## Reproduction and verification

The delivered archive was extracted to a separate temporary directory. Bubblewrap
hid the originating checkout behind an empty filesystem mount, exposed the already
installed dependencies through a separate read-only mount, and disabled networking
with a separate network namespace. Python also rejects socket access during replay.
The extracted script ran with isolated Python startup (`-I`). There were no raw
datasets or Git history in the extracted package.

Every cohort reproduced all seven CSV outputs exactly, including choices, all
three endpoints, intervals, adjusted tests, offsets, selection shares and size
strata. Decision JSON fields matched except the invocation command; membership,
input hashes, equivalence status and post hoc status matched. The original
analysis was additionally compared with the preserved historical `analysis/`.

| Cohort | Tasks | C1, normalized rank regret | 95% interval | Declared tests |
|---|---:|---:|---|---|
| Verified eligible | 11 | +0.02193 | [−0.01979, +0.07166] | Neither rejects |
| Original | 46 | +0.03414 | [+0.01432, +0.05639] | Both reject; historical label only |
| Three-exclusion | 43 | +0.03365 | [+0.01477, +0.05527] | Both reject; post hoc |
| Eligible plus unresolved | 34 | +0.03322 | [+0.01125, +0.05796] | Both reject; post hoc |

The isolated partial build produced 1,176 numbers and 14 tables, all identical to
their counterparts in the full build. The full build regenerated 25,363 numbers
and 50 tables; only the included manuscript assets are rendered. The new Make
targets for stored-score reproduction and partial assets also passed locally.

The full suite passed **205 tests** in 241.83 seconds, with one existing
Matplotlib `vert` deprecation warning. After the final portable test-fixture
adjustment, the 19 package/reference tests passed; after excluding editorial
response material from the final allowlist, all 9 package tests passed again.
Tests cover missing/tampered package files, unexpected files, membership exports,
raw-data exclusion, extracted replay, rank conversion, partial-output protection,
network rejection, and missing/duplicate/cyclic LaTeX includes and labels.

JAIR static checks, including unused bibliography and recursively included
references, passed. The PDF build completed. The final log has no overfull boxes,
oversized floats or unresolved references; a bounding-box check finds all words
inside the page. Render inspection covered the opening, experiment, compact
tables, moved protocol/model sections, multipage comparison and cohort tables,
and secondary-endpoint/choice tables. The checksum printed in the PDF matches the
delivered archive. Detailed results and file hashes are in
`packaging_2026-10-26/verification.json`.

The stored-score interface is:

```sh
make fresh-validation-reproduce FV_REPLAY_OUT=/tmp/new-empty-replay
make fresh-validation-assets FV_GENERATED=/tmp/new-fresh-assets
make fresh-validation-package
```

The older `make fresh-validation-corrected` performs raw-data inspection and
rewrites dated analyses; it was not rerun. The October 25 evidence is preserved,
and the October 26 verification has its own output directory. Hash verification
shows the original inputs and results are unchanged. The new archive verifier
uses preserved protocol snapshots instead of the historical verification script's
Git lookups.

## Remaining limits

Public release is still pending. The verified experiment remains too uncertain to
establish increased regret or equivalence, and editorial restructuring cannot
resolve the scientific-significance objection. No submission-readiness claim is
made.
