# Targeted follow-up and response — 26 October 2026

This pass addresses the six remaining textual corrections and prepares the existing
correction archive for repository inclusion. It preserves the manuscript's structure,
experiments, results, audit decisions, post hoc qualifications, theory assumptions,
and the author's personal-review attestation. The verified estimate remains
**+0.37 [−0.34, +1.22] rank positions**: neither increased regret nor equivalence
is established. The scientific-significance concern remains unresolved.

The preceding manuscript and PDF are preserved locally in
`paper/history/2026-10-26-before-followup/`; the October 25 versions remain in
`paper/history/2026-10-25/`. The October 25 and earlier October 26 response and
verification records were not rewritten. The revised local deliverables are
`paper/main.tex` and `paper/main.pdf`.

## Response to the actionable comments

| Comment | Change in this pass | Verification or remaining limit |
|---|---|---|
| §5: model/provenance contribution points to B.1 | Changed its appendix reference to the provenance/model appendix. | The contribution on PDF page 2 now reads “Section 4; Appendix D”; Appendix D begins on page 20. |
| §5: validation-best is grouped with the two committed rules | Used the requested sentence: “References include validation-best and two committed rules: the family-prior rule and a train-selected member of a fixed pool.” | Checked in Section 5 on page 5. The rules and their results are unchanged. |
| §5: fixed deployed-model wording can imply identical choices | Replaced the ambiguity with “holding the trained candidate pool fixed” or “without retraining or modifying any candidate.” Checked the abstract, introduction, experiment design and interpretation, discussion, conclusion, and supporting related-work passage. | The statement that the rules make different choices remains; no selected-model identity is asserted for the experiment. |
| §5: prespecification requirement reads as universal | Began the sentence with “Under our reporting convention, …”. | The inference rule, post hoc qualifications and conclusion precedence are unchanged. |
| §5: editorial consolidation and scientific-sufficiency self-assessment | Removed both editorial sentences. Added a direct pointer to Table 4 and retained the concrete scientific limitations. | Restricted internal pools, lack of an independent application, the synthesized declaration and the explanatory model remain explicit. |
| §5: B.1 describes the former field-mapping table and contains “; The” | Rewrote the passage to describe comparisons, design status, estimates, adjusted tests, conditions and conclusions. Replaced the sentence boundary and retained controlled, coupled and threshold changes. | Page 15 points to Table 4, “Detailed central comparisons and their inferential limits.” Coupled changes are explained without claiming the table maps their fields. |
| §2 and §6: the archive needs an actual retrieval route | Added narrow ignore exceptions for the existing archive and checksum. README, Code and Data Availability, and the checklist identify the repository's `dist/` directory and planned disclosure before submission. | The package remains unpublished. Local verification establishes its identity and consistency, not public accessibility or independent validation of source judgments. |
| §1 and §3: retain the narrowed interpretation and earlier corrections | Preserved all numerical assets, original/sensitivity outputs, the 181-decision audit, 11/12/23 classification, inference limitations and personal-review attestation. | Hash and source comparisons passed; no new empirical evidence was added. |
| §4, §6 and §7: stronger evidence is needed to answer the significance objection | Retained the limits of the selected verified cohort, explanatory theory, internal selector pools and simulation. The suggested routes—more informative evaluation, a consequential independent application, or stronger theory—remain possible future research. | None was undertaken in this textual pass. These edits and archive preparation do not resolve contribution sufficiency or establish submission readiness. |

## Archive access and disclosure plan

The intended retrieval route is the `dist/` directory of
`https://github.com/pz1004/slate-bench-public`, with these two files:

- [Correction archive](../dist/validation-reuse-correction-2026-10-26.tar.gz)
- [Checksum sidecar](../dist/validation-reuse-correction-2026-10-26.tar.gz.sha256)

The author plans to disclose the archive through that repository directory before
submission. Until that occurs, the correction remains **unpublished**; these links
identify the prepared local repository files and do not assert that a remote copy
is already available. Public access remains pending. The existing
[package guide](validation_revision_package.md) describes verification and reproduction.

The archive was preserved byte-for-byte, without rebuilding it. Its SHA-256 remains:

```text
083d7679618c65211c6ae99ffb1d6dd8062aaf0358f4183a7c437625838b923f
```

The dated archive retains its original documentation and evidence snapshots;
this follow-up is a separate record. No review ZIP or embedded attachment was
created, and no commit, push, release, submission or external communication occurred.

## Checks performed in this pass

The new machine-readable records and logs are in
[`analysis-output/v2/fresh_validation/followup_2026-10-26/`](../analysis-output/v2/fresh_validation/followup_2026-10-26/).
They are separate from `audit_2026-10-25/` and `packaging_2026-10-26/`.

- All **10 existing LaTeX reference/static-check tests passed**. The JAIR static
  check passed with recursive reference checks and unused bibliography treated
  as an error. No full test-suite rerun is claimed for this pass.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
  The PDF remains **54 pages**, with no overfull boxes, oversized floats or
  unresolved references in the final log. All 38,685 extracted words lie within
  page bounds. All 17 table labels match their captions; Table 15's eligibility
  caption and reference remain correct.
- Visual inspection covered pages **1–9, 12–16, 20–22, 32, 47–50 and 53–54**:
  the affected passages, adjacent tables, multipage comparison/cohort tables,
  availability statement and checklist. No clipping, overflow, stale targeted
  wording or pagination problem was found.
- `sha256sum -c` passed. The archive was safely extracted to a temporary directory
  and its existing `scripts/validation_revision_package.py verify` command passed
  with isolated Python startup. It verified **457 files**, the 181-decision
  register, 46 inspected-task records, cohort sizes **46/43/11/34**, and 47
  historical hashes. All **447 allowlisted checkout inputs** match the archive.
- All **12,299 preservation hashes** match the pre-edit baseline, including
  **56 generated numerical/table/figure assets** and **11,706 analysis-output
  files**. Before editing, all 11 hashes in the earlier packaging record also
  matched. Original and sensitivity outputs and earlier response records remain
  unchanged. The manuscript's result-macro sequence, tables, theory and proofs,
  structure, and personal-review attestation were checked for preservation.
- Git ignore probes confirmed that only the named archive and checksum are
  admitted under `dist/`; alternative archives, backups, nested copies, ZIPs and
  unrelated files remain excluded. Both prepared files are untracked and eligible
  for inclusion; neither was staged or committed.

This pass did not repeat training, raw-data inspection, source-document auditing,
stored-score reproduction, simulations or asset generation. The earlier replay
and full-suite results remain historical verification, not new checks attributed
to this pass. Hash and package checks establish preservation and internal
consistency; they do not independently certify eligibility judgments or resolve
the experiment's uncertainty.
