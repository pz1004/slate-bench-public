# Meta-study protocol: audit fields in published score-level selection claims

Written 2026-09-28, before any search was run. Changes after that date are listed under
"Deviations" at the end.

## Research question

How often do published claims that a selection procedure beats a reference, where the
claim is evaluated on precomputed candidate results for tabular learning, declare each
of the nine audit fields of the manuscript?

The study is descriptive. It characterises reporting practice. It does not judge the
correctness of any paper's results, and codings describe only what the paper text
declares.

## Eligibility

A **claim** is included when all of the following hold:

1. **Venue and date.** The paper appeared in 2016–2026, either in a peer-reviewed venue
   or on arXiv or OpenReview.
2. **Candidates.** The candidates are supervised tabular learners, configurations or
   pipelines, for classification or regression.
3. **Procedure.** The paper proposes or evaluates a procedure that chooses, ranks,
   weights or combines candidates. Examples: algorithm selection, meta-learned model
   selection, zero-shot or portfolio configuration choice, learned defaults, and post hoc
   ensemble or portfolio construction.
4. **Comparative claim.** The paper states that this procedure improves on at least one
   reference.
5. **Precomputed evaluation.** At least one reported evaluation of that claim reuses
   precomputed candidate results rather than retraining candidates. Examples: a
   performance repository, a meta-dataset, an OpenML run table, TabRepo, TabArena,
   TabZilla, AMLB results, or an HPO benchmark with tabular tasks.

**Exclusions:**
- papers whose only evaluation reruns full AutoML systems end to end;
- papers proposing a learner rather than a selector;
- non-tabular candidates;
- surveys and position papers without their own claim;
- theses duplicating an included paper.

When a paper contains several claims, the headline claim from the abstract or
contributions list is coded, so that each paper contributes one claim.

## Search

- **Sources:**
  - Semantic Scholar Graph API keyword search;
  - arXiv listing search;
  - general web search;
  - forward citations of TabRepo (Salinas & Erickson 2024), TabArena (Erickson et al.
    2025), TabZilla (McElfresh et al. 2023), ASlib (Bischl et al. 2016) and Auto-sklearn
    2.0 (Feurer et al. 2022).
- **Query strings** (combined with "tabular" where the engine allows):
  - "zero-shot AutoML";
  - "portfolio" with "configuration" and "AutoML";
  - "meta-learning model selection";
  - "algorithm selection" with "machine learning" and "OpenML";
  - "post hoc ensembling";
  - "learned defaults" and "multiple defaults";
  - "meta-features" with "model selection";
  - "performance repository"; "TabRepo"; "TabArena";
  - "LLM" with "model selection" and "tabular".
- **Stopping rule.** Screening stops when 20 claims are included or when every source
  has been screened to its first 50 records, whichever comes first.
- **Screening log.** Every screened record goes in `screening.csv`, with its source,
  its decision and the reason for any exclusion.

## Coding scheme

Each field is coded **D** (declared), **P** (partial), **A** (absent) or **NA** (not
applicable), with a pointer to the section, table or figure that supports the code and a
one-sentence paraphrase. No text is copied from the paper.

1. **Artifact**: the completed result table that is reused.
   - D: named, with a version, snapshot, date or hash.
   - P: named without a version.
   - A: not identifiable.
2. **Capability set**: the fields the selector may read (validation scores, runtimes,
   meta-features, prediction vectors), with test outcomes excluded.
   - D: inputs listed explicitly.
   - P: inputs implied by the method description only.
   - A: not stated.
3. **Policy pool**: the selector variants and hyperparameters that were considered, and
   how the reported one was chosen.
   - D: search space and choice rule both stated.
   - P: final configuration only.
   - A: not stated.
4. **Fixed references**: non-adaptive comparators, such as the single best solver,
   validation-best, defaults or random choice, and the oracle (virtual best).
   - D: at least one non-adaptive reference and the oracle, each defined.
   - P: only one of the two, or undefined.
   - A: none.
5. **Evidence unit**: what counts as an independent observation in the comparison
   (dataset, task, fold or seed).
   - D: stated, with aggregation performed at that level.
   - P: implied by the tables.
   - A: not identifiable.
6. **Inference rule**: the interval or test supporting the improvement claim, and any
   multiplicity control.
   - D: test or interval named with its unit.
   - P: significance mentioned without its unit or correction, or a critical-difference
     diagram only.
   - A: point estimates only.
7. **Endpoint**: the reported measure (metric, normalized score, rank, regret), and how
   outputs outside the candidate set are scored.
   - D: defined.
   - P: named only.
   - A: not stated.
8. **Cost field**: the runtime or cost measure and what it means (recorded training
   time, selector overhead, budget).
   - D: defined.
   - P: reported without semantics.
   - A: a cost claim with no measure.
   - NA: no cost claim.
9. **Validation provenance**: whether candidates' validation scores are comparable, for
   example validation computed after candidate-specific tuning or ensembling on the same
   folds.
   - D: discussed or controlled.
   - P: validation protocol described without addressing comparability.
   - A: not addressed, although the selector reads validation scores.
   - NA: the selector reads no validation scores.

**Extra items** (yes, no or unclear):

- **E1:** selector design or hyperparameters were tuned on the datasets used to evaluate
  it, with no nested separation;
- **E2:** a single-best-solver or validation-best reference is present;
- **E3:** multiplicity is handled when several contrasts are tested.

## Verification

Claude does the first coding. The author checks every coding against the source and
marks it `verified` in `claims.csv`, correcting it where needed. Only verified rows are
reported in the manuscript. Disagreements are resolved by the author.

## Reporting

- **Manuscript content:**
  - the number of records screened and included;
  - for each field, the share of D, P and A codes across included claims;
  - for each extra item, the share of yes codes.
- **Appendix:** a per-paper table with the codes and pointers.
- **Wording:** "not declared in the text" rather than any judgement of the work.

## Deviations

Recorded 2026-09-29, before the author's verification.

1. **Stopping rule not reached.** Screening stopped after 25 records (14 included, 9
   excluded, 2 not coded), before either stopping condition was met (20 included claims,
   or every source screened to its first 50 records). The sample is therefore not
   exhaustive, and the manuscript says so in Section 8 and in its limitations.
2. **Two records not assessed.** The first coder could not retrieve the full text of S24
   (MetaBu) or S25 ("To Select or Not to Select?"). The screening log marks them
   `not coded`, a decision this protocol does not define. The author decides their
   eligibility at verification by the criteria above; an included record is coded in the
   same way as the others.
3. **Recording verification.** To report agreement between the first coding and the
   verified coding, the first codes are kept in the `*_draft` columns of `claims.csv`,
   and the author edits only the main code columns. The manuscript reports how many of the
   drafted codes the author changed. A claim the author finds ineligible is marked
   `verification=excluded`, with the reason in `verification_note`, and its decision in
   `screening.csv` is changed to `exclude` with the same reason. Excluded claims are
   left out of every reported count.
4. **Submission build.** `make assets META_FLAGS=--meta-verified-only` refuses to run while
   any claim is still `unverified`, so no claim can leave the reported sample silently.

Recorded 2026-09-30, at verification.

5. **Verification by a second AI coding.** At the author's request, the verification
   described above was not done by the author. Each claim was coded a second time with
   Claude, blind to the first coding (the second coder read only this protocol and the
   paper), and each disagreement was resolved against the source text under the
   conventions in item 6. The second codes are kept in the `*_check` columns of
   `claims.csv`, next to the first codes in `*_draft`; the main columns hold the resolved
   codes. The manuscript reports the agreement between the two codings and how many first
   codes changed, and states that no human coder was involved.
6. **Conventions adopted to resolve disagreements.** Where the definitions above leave room,
   these rules were applied to every claim:
   - **Artifact:** D needs an identifier that pins the reused table: a version, a snapshot
     or context name, a date, a DOI, a hash or an immutable repository ID. An unreleased
     table that the paper describes is P.
   - **Capability set:** D needs the text to state which recorded quantities the selector
     uses. Inputs that follow only from equations or from how the method works are P.
   - **Policy pool:** D needs the variants or settings considered and the rule for the
     reported one. An explicit statement that settings were fixed in advance, for example
     a prior implementation's defaults, counts as the rule.
   - **Fixed references:** the oracle counts when results are reported against the
     per-dataset best candidate of the pool, as a comparator or as the zero point of a
     regret. An anchor that is the best of the compared methods does not count.
   - **Evidence unit:** D needs the text to state the unit over which the reported
     comparison aggregates.
   - **Inference rule:** D needs a test or interval named with the unit it is computed
     over. Multiplicity is coded separately, in E3.
   - **Validation provenance:** the field applies only when the selector reads validation
     scores on the dataset where it chooses; a selector that reads only meta-features there
     is NA. D needs the text to discuss or control whether candidates' validation scores
     are comparable. Discussing validation quality or overfitting in general is P.
   - **E1:** no only when the text states nested or held-out separation for the design
     choices, or that settings were fixed in advance; unclear when it does not say how
     settings were chosen.
   - **E2:** a single best solver, the best single model among those compared, or a
     validation-best choice over the pool or a stated part of it (such as one model family).
   - **E3:** a family-wise procedure: Holm, Bonferroni, or a Nemenyi-based
     critical-difference diagram.
7. **Eligibility at verification.**
   - S04 is excluded on criterion 5. Its claim is evaluated only by retraining the
     learners on held-out datasets, and the stored results only train surrogates.
   - S14 stays included. It compares its meta-model with a logistic-regression baseline,
     so criterion 4 holds.
   - S25 was assessed by two independent coders. Both found that it fails criterion 3, so
     it is excluded.
   - The full text of S24 is blocked for automated access (OpenReview and HAL bot
     checks), so neither coder could settle criterion 5. It is recorded as
     `not retrieved`, and the manuscript reports it as such. This supersedes item 2.

Recorded 2026-10-01, before the human coding.

8. **Human coding of a subset.** After a colleague's review, the author codes a declared
   subset of the cells by hand. Neither AI coding is shown during this coding.
   - **Cells.** Fixed references, validation provenance and E1 for all 13 verified claims,
     plus all twelve items for 4 of those claims: 75 cells in all.
   - **Draw.** The 4 claims are the first four record ids when the 13 are sorted by the
     SHA-256 hex digest of `20261001:<record_id>`. `scripts/meta_study_sheet.py human-draw`
     applies this rule and writes it to `human_subset.json`, together with a hash of the
     AI-resolved codes at the time of the draw. Claims drawn: S02, S08, S11 and S14.
   - **Material.** The author codes from each paper, this protocol and
     `human_coding_sheet.md`. The sheet lists the cells and the claim to code, but no AI
     code, evidence pointer or note. The manuscript's aggregate counts and its per-paper
     code table were available to the author before this coding.
   - **Resolution.** Where the author's code differs from the AI-resolved code, the author's
     code is final. The AI-resolved codes stay in the `*_ai` columns of `claims.csv`, the
     author's codes go in `*_human`, and the main columns hold the final codes.
   - **Reporting.** The manuscript reports:
     - the agreement between the author's codes and the AI-resolved codes, raw and as
       Cohen's kappa, overall and for each item coded on every claim;
     - how many final codes the author changed.
   - **Enforcement.** The submission build refuses to run until every declared cell is
     coded and merged.

Recorded 2026-10-01, after Deviation 8 and before the subset was coded.

9. **Third AI coding instead of the author's.** The author decided not to code the declared
   subset by hand and asked Claude to code it. The declared cells, the draw and the frozen
   AI-resolved codes of Deviation 8 are unchanged, and the declaration's `coder` field
   records the change. The files are renamed `recheck_subset.json`, `recheck_coding.csv`
   and `recheck_coding_sheet.md`.
   - **Coders.** Fresh Claude sessions coded the 75 cells. They read only each paper, this
     protocol's coding scheme and conventions, and the claim to code, and were shown no
     earlier code, evidence pointer or note.
   - **Records.** The codes are in `recheck_coding.csv` and in the `*_recheck` columns of
     `claims.csv`.
   - **Resolution.** Where a third code differs from the AI-resolved code, the difference
     is resolved against the source text under the conventions of item 6. The resolved code
     and the reason are recorded in `recheck_coding.csv`, and the main columns hold the
     final codes.
   - **Reporting.** The manuscript reports:
     - the agreement between the third coding and the AI-resolved codes, raw and as Cohen's
       kappa;
     - how many final codes the resolution changed;
     - that no human coder was involved.
   - This supersedes the material, resolution and reporting rules of item 8.

Recorded 2026-10-02, after the recheck was merged.

10. **Check of the recheck resolutions.** The author asked Claude to check the 11 resolutions
    of item 9. A fresh Claude session reread each against the paper and the conventions of
    item 6, recording its own code for every cell before it saw the resolution. The
    session that made the resolutions then reread the passages behind every point raised.
    No human coder was involved.
    - **S07, validation provenance: D changed to P.** Sec. 2 of the paper defines a model's
      true performance as its expected cross-validation error. The Sec. 3.1 statement that
      each entry of the error matrix is an unbiased estimate of that performance therefore
      concerns the noise of each entry. It does not address whether candidates' validation
      errors are comparable, so it is validation quality in general (P), as for S01 and S06.
    - **The other ten resolutions stand.** Seven notes were corrected to cite the passages
      exactly: S01, S02, S05 and S07 on E1, S07 on fixed references, and S11 on the
      capability set and the cost field.
    - **Cost field: when NA applies.** Every coding read the NA code ("no cost claim") as
      follows. The field applies when a claim compares procedures at a stated budget or
      reports a cost, and is NA only when the claim does neither. All codings coded S01,
      S02, S05, S06 and S13 this way. The reading is written down here because the third
      coding read S08 the other way; S08 keeps D, since its comparison is made at a matched
      budget of loss evaluations.
    - **Evidence pointers.** The merge used to write only codes, so three final codes the
      resolution changed (S07 fixed references, S11 capability set and cost field) still
      showed the pointer of the earlier code. Now the AI-resolved pointer is frozen in
      `*_evidence_ai`, and for each resolved disagreement the merge writes the reason for
      the resolution into `*_evidence`. The submission build checks both.

Recorded 2026-10-03, before the author's check.

11. **Author's check of the codes behind the abstract's counts.** After a further colleague
    review, the author checks against the papers the fixed-references and
    validation-provenance codes of every verified claim: 26 cells, the codes whose counts
    the abstract reports.
    - **Preparation.** Claude writes `author_check.csv` and `author_check_sheet.md`
      (`make meta-author-sheet`). For each cell they give the final code, its evidence
      pointer, section, page or table locators of the passages that bear on the code, a
      one-line paraphrase of those passages, and points that need attention. No text is
      copied from the papers.
    - **Check.** The author reads the located passages and decides under the coding scheme
      and the conventions of item 6: `agree`, or `change` with the correct code and a
      reason with its locator. The author has seen the AI codes and the counts in the
      manuscript, so the check is neither blind nor an independent coding.
    - **Merge.** `make meta-author-merge` writes the decisions into `*_author` and
      `*_author_note`. A changed code becomes the final code, and its reason the field's
      evidence pointer. The `*_draft`, `*_check`, `*_ai` and `*_recheck` columns are left
      unchanged.
    - **Reporting.** The manuscript reports how many of the 26 codes the author changed,
      and keeps the limitation that no independent human coder was involved. The
      submission build refuses to run until every cell is checked.

Recorded 2026-10-04, before the check of item 11 was made.

12. **Check of the 26 cells by Claude instead of the author.** The author decided not to make
    the check of item 11 and asked Claude to make it. The cells, the prepared locators and
    paraphrases, and the merge rules of item 11 are unchanged. The files are renamed
    `count_check.csv` and `count_check_sheet.md`, and the merged columns of `claims.csv`
    are `*_countcheck` and `*_countcheck_note`.
    - **Checkers.** Four fresh Claude sessions, each covering three or four claims. Each
      first coded the two fields of its claims from the papers, given only the claim, the
      coding scheme, the conventions of item 6 and the locators of the passages to start
      from, and reported these codes before it was shown anything else; they are kept in the
      `first_code` column of `count_check.csv`. It was then shown
      the final codes, their evidence pointers, the prepared paraphrases and the attention
      notes, and decided for each cell: `agree`, or `change` with the correct code and a
      reason with its locator.
    - **Resolution.** The session that prepared the sheet reread the passages behind every
      proposed change, and behind every cell where a checker's first code differed from the
      final code, and decided under the conventions of item 6. Its decision and reason are
      recorded in `count_check.csv`, next to each checker's own reason (`checker_reason`).
    - **Evidence pointers.** A check note is given for every changed code, as its reason, and
      for a confirmed code whose evidence pointer the checkers found inaccurate, as the
      corrected pointer. In both cases the note becomes the field's evidence pointer.
    - **Reporting.** The manuscript reports how many of the 26 codes the check changed, and
      that no human coder was involved in any coding or check. This supersedes the check
      and reporting rules of item 11.
