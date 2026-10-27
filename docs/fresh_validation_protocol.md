# Fresh-validation experiment: prespecified protocol

This protocol was fixed and published (commit adding this file to `pz1004/slate-bench-public`) before any data of the experiment's tasks were downloaded and before the experiment or the TabRepo frozen-fit check was run. Every later change is listed under *Deviations*, with its date and reason. The results are reported under this plan whatever they are. An AI assistant (Claude, Anthropic) drafted the protocol and the code and made the source check of Section 3.3 under the author's direction; the author has not verified the source check.

## 1. Question

A selector that reads a reused score table, such as validation-best, treats each candidate's validation score as a comparable estimate of its test score. For tuned and ensembled candidates that score has been reused: by early stopping, by the choice among configurations and by the ensemble weights. On TabArena, tuned-and-ensembled candidates' test-minus-validation offsets exceed defaults' on every dataset, and validation-best, which favours them, has a high regret; the audit does not identify how much of that regret the reuse causes. This experiment holds the deployed candidates fixed and changes only the validation evidence used to choose among them.

- **C1.** Does validation-best on fresh selection-validation scores, from a set as large as the construction-validation set, have a different test regret from validation-best on the reused construction-validation scores?
- **C2.** Does the *defaults rule* (choose the learner family whose default has the lowest validation error, deploy that family's tuned-and-ensembled candidate), found after the main results on TabArena, have a different test regret from validation-best on the reused scores?

Both questions are two-sided. Either answer is interpretable: a difference in C1 attributes part of validation-best's regret to the reuse of construction-validation evidence with candidate quality and deployment fixed; no difference, together with the equivalence analysis, bounds that contribution in this setting.

## 2. Estimand

For a task drawn from the population of Section 3, the expected difference in normalized rank regret between two rules applied to the same 18 frozen candidates of a unit, averaged over the task's 5 repetitions. No rule is fitted across tasks, so the tasks' differences are independent draws, and tests over tasks target this mean. The tasks are a census of the eligible, untouched OpenML datasets at the catalogue's retrieval date, not a sample of a larger population; the inference treats them as draws from the population of such tasks.

## 3. Tasks

### 3.1 Eligibility (metadata only)

`scripts/fresh_validation_tasks.py catalogue` downloads OpenML's list of active datasets with their metadata qualities (REST API `data/list/status/active`) and records the file's SHA-256 (`analysis-output/v2/fresh_validation_tasks/catalogue_summary.json`). A dataset is eligible if it

- is active and not in sparse format;
- has a nominal default target with 2 to 10 classes;
- has 1,000 to 100,000 rows and at most 500 features besides the target;
- has at least 100 rows in its smallest class;
- does not have a name of a generator family of artificial data (patterns in the script), a machine-made subsample name (`<name>_seed_<k>_nrows_<n>_…`) or the exchange-rate series family `FOREX_…`;
- is not one of TabArena-v0.1's 51 datasets or TabRepo's 244-dataset context by normalized name or by OpenML ID (`configs/fresh_validation_excluded.csv`);
- is the latest version among eligible datasets of the same name.

The two name families of the fifth item were added after a first pass over the catalogue showed 346 machine-made subsamples and 128 exchange-rate series among the eligible datasets; no task had been chosen and no data were downloaded.

### 3.2 Order

The eligible datasets, sorted by OpenML ID, are put in the order of `numpy.random.default_rng(20261022).permutation` (`analysis-output/v2/fresh_validation_tasks/eligible_in_random_order.csv`).

### 3.3 Source check (before any data are seen)

`scripts/fresh_validation_tasks.py sheet` downloads, for every eligible dataset, its OpenML description and feature list (metadata only). Each dataset is decided in order from its name, description, tags and metadata (`configs/fresh_validation_source_check.csv`), and excluded if it is

- **T** a TabArena-v0.1 or TabRepo-244 dataset under another name or ID;
- **S** from the same source as one of them (another version, subset or processing of the same data);
- **D** from the same source as an earlier included dataset;
- **A** artificially generated or simulated;
- **M** images, text, sequences, time series, or rows grouped by subject;
- **Q** without a usable target or features (test uploads, documented label leakage, a target derived from the features).

The evidence for each exclusion is coded C (OpenML description or tags), N (name), M (metadata: shape, class sizes) or K (background knowledge). Of 181 eligible datasets, 46 are included; all 46 are the candidate tasks (`configs/fresh_validation_tasks.csv`).

### 3.4 Automatic checks after download

`scripts/fresh_validation_tasks.py download` stores each candidate's data file in its own new directory (`data/fresh_validation/openml/<id>/`, not published), records its SHA-256 and runs these checks: the file parses; the default target and the attributes to drop exist; after dropping rows without a label there are 2 to 10 classes with at least 100 rows each; 1 to 500 features remain. Every candidate that passes is a task; the others are recorded with the failed check. The row-identifier and ignore attributes of the OpenML description are dropped.

## 4. Per task and repetition

- **Rows.** Tasks with more than 20,000 rows are reduced to a stratified subsample of 20,000, the same for every repetition (seed from the master seed and the task ID).
- **Splits.** Five repetitions, each a stratified split into training rows (48%), construction-validation rows (16%), selection-validation rows (16%) and test rows (20%), with seeds from the master seed 20261022, the task ID and the repetition.
- **Preprocessing,** fitted on the training rows only. Numeric and boolean columns are numeric, all others categorical. Trees: ordinal codes for categories (a missing category is a level of its own, an unseen one is missing); histogram gradient boosting treats categories with at most 250 levels as categorical; random forest and extra trees use median-imputed values. Logistic regression, KNN and MLP: median imputation with missingness indicators and standard scaling for numeric columns, one-hot encoding with at most 20 levels (rarer levels grouped) for categorical ones.
- **Families and configurations** (`configs/fresh_validation_experiment.yaml`, scikit-learn 1.9.0): histogram gradient boosting, random forest and extra trees (300 trees), logistic regression, KNN and an MLP. Each family has scikit-learn's default configuration and 100 random configurations from the declared search space, drawn once (seed from the master seed and the family) and used for every task. Every configuration is fitted on the training rows. Histogram gradient boosting stops early on the construction-validation loss (patience 10, at most 1,000 iterations); the MLP is trained by epochs and restored to the epoch with the lowest construction-validation log loss (patience 10, at most 100 epochs).
- **Candidates (18 per unit).** Per family: the *default*; the *tuned* candidate, the configuration with the lowest construction-validation log loss among the default and the 100 random ones (exact ties to the earlier configuration); and the *tuned + ensemble* candidate, greedy ensemble selection with replacement over all configurations' construction-validation predictions, 40 steps, exact ties to the earlier configuration, whose predictions elsewhere are the weighted means of its members'. A configuration that raises an error or gives non-finite probabilities is dropped from the search; if a family's default fails, the family is dropped from that unit for every rule. Identical candidates (for example a tuned candidate that is the default) remain separate rows.
- **Scores.** Log loss on the construction-validation, selection-validation and test rows, with probabilities floored at 10^-15 and renormalized. Each candidate's fit time is recorded (for an ensemble, the sum over its members).

## 5. Rules and endpoints

No rule is fitted across tasks.

- **Validation-best, reused**: the candidate with the lowest construction-validation log loss, then the lowest fit time, then the earlier candidate in the fixed order (families as listed, then default, tuned, tuned + ensemble).
- **Validation-best, fresh**: the same on the selection-validation log loss.
- **Defaults rule**: the family whose default has the lowest construction-validation log loss (same tie-break), deployed as that family's tuned + ensemble candidate.
- **Validation-best among defaults** (reused scores) and the **oracle** (lowest test log loss) as references.

The primary endpoint is the normalized rank regret of the chosen candidate's test log loss among the unit's candidates, ranked by the audit's rule (`tabular_sota_benchmark.ranking.rank_against_candidates`, rule `consistent`: values within 10^-12(1+|v|) are tied and get the average rank; regret = (rank − 1)/(K − 1)). Secondary endpoints: the median-normalized error regret (test loss minus the unit's best, over the median candidate's minus the best) and the test log loss minus the unit's best. A task's value is the mean over its repetitions.

## 6. Analysis

- **Primary family** (Holm, α = 0.05): C1, reused validation-best minus fresh validation-best; C2, reused validation-best minus the defaults rule; both on normalized rank regret. Tests: the two-sided percentile bootstrap test over tasks (10,000 resamples, seed 20260608) and the paired *t*-test, each Holm-adjusted within the family (`tabular_sota_benchmark.inference.paired_contrast`, `contrast_table`).
- **Conclusions** follow the manuscript's rules for a prespecified contrast between rules that fit nothing across independently sampled tasks: *difference supported* if both tests reject; *no difference detected* if neither rejects; *difference unresolved* if they disagree. For C1, if no difference is detected, equivalence within ±0.5 rank positions (±0.5/17 in normalized rank regret) is supported if the 90% *t* interval lies inside that margin. The minimum detectable effect is reported as an illustration.
- **Secondary analyses** (all reported; *p*-values unadjusted unless stated):
  1. Offsets by class: per task, the mean over repetitions of test minus construction-validation and test minus selection-validation log loss for each variant; tuned + ensemble minus default and tuned minus default, with sign tests across tasks. The reused offsets are expected to differ by class as on TabArena, the fresh ones not to (a control).
  2. Selection shares by variant for each rule and the oracle, and the share of units in which reused and fresh validation-best choose the same candidate.
  3. A secondary Holm family: fresh validation-best minus the defaults rule, and validation-best among defaults minus reused validation-best.
  4. The primary and secondary contrasts on the two secondary endpoints (own Holm families).
  5. C1 and C2 within the halves of tasks below and above the median number of rows after the cap (descriptive).
- Any other analysis is exploratory and labelled so.

Code: `scripts/fresh_validation_experiment.py` (runs; one part per task, repetition and family), `scripts/fresh_validation_analysis.py` (this section), `src/tabular_sota_benchmark/fresh_validation.py`, tests in `tests/test_fresh_validation.py`.

## 7. Pilot

Before this protocol was fixed, the code ran on the eight TabArena classification datasets of the manuscript's compact retraining check (already inspected; local copies), with two repetitions, 200 random configurations per family (TabArena's search size) and otherwise the settings above (`analysis-output/v2/fresh_validation_pilot/`). The pilot set only computational settings. Projected from its fit times and from timings of single configurations on its largest dataset, 200 configurations would take about 230 core-hours for the 46 candidate tasks (the random forests on the widest tasks dominate), beyond the compute planned for the experiment; 100 configurations take about half. Its results are exploratory and are not part of the analysis.

## 8. TabRepo frozen-fit check

A second prespecified analysis gives the audit's fitted procedures one evaluation whose inference is valid for a fixed fit.

- **Run.** The fully grouped driver, unedited (`scripts/grouped_inner_splits.py`, SHA-256 recorded by the run), with `--config configs/score_level_tabrepo_nested.yaml --repeats 1 --split-seed 20261022`: one stratified 4-fold split of TabRepo's source groups (`configs/tabrepo_source_groups.csv`), with whole groups left out of every inner estimate. The score-only pool comes from `scripts/score_level_policy_pool_ablation.py` on the same run.
- **Declared evaluation: fold 0.** Every procedure is fitted on fold 0's training groups and frozen; its held-out groups, which no fitted quantity saw, are the evaluation tasks. The other three folds are computed because the driver runs whole repeats; each is reported as its own evaluation and never pooled with fold 0.
- **Contrasts** (`scripts/tabrepo_frozen_fit.py`; validation-best is the literal rule of `scripts/literal_validation_best.py`): in the ensemble-inclusive pool, nested selection minus the train-selected rule over the ensemble-inclusive pool, minus the train-selected rule over the score-only pool, minus validation-best and minus the family-prior rule; in the score-only pool, nested selection minus the train-selected rule, validation-best and the family-prior rule.
- **Tests**: the dataset bootstrap over the held-out datasets (10,000 resamples, seed 20260608), the paired *t*-test, and the bootstrap over held-out source groups, each Holm-adjusted within the pool's family.
- **Status.** The design is fixed here before it is run, but TabRepo's outcomes under other splits were known, so the manuscript classifies the check as post hoc: a rejection is a difference unresolved, and no rejection is no difference detected for this fit.

## 9. Software and hardware

CPython 3.12.13, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0, pyarrow 20.0.0; one workstation with 24 cores and 125 GiB RAM, about 21 worker processes with one thread each. The run records the SHA-256 of this protocol, the configuration, the task list and the code, and the git commit.

## 10. Deviation policy

Any change after publication (a bug fix, a failed task, a changed setting) is added below with its date and reason, and the manuscript reports it. Results are reported whatever they are.

## Deviations

- **2026-10-22, run.** The main run used 18 worker processes instead of about 21 (Section 9), because the TabRepo frozen-fit check (Section 8) ran alongside it with 4. Fit times depend on the load and enter only the tie-breaks between candidates with equal validation log loss. The code, configuration and task list are as published (commit `bc9bca3`); the task list's `use` column was set by the automatic checks as the file states.

- **2026-10-25, eligibility correction (post hoc).** After the original results were
  known, all 181 screening decisions were reassessed using source documentation
  and cached metadata; the actual targets and retained predictors were inspected
  for all 46 trained tasks. The decision register is
  `configs/fresh_validation_eligibility_audit.csv`; the main cohort uses only the
  11 verified eligible trained tasks, excluding 12 confirmed failures and 23
  unresolved tasks. The original 46-task analysis, the three-reported-exclusion
  sensitivity (43), and eligible-plus-unresolved sensitivity (34) are retained
  separately. No untrained replacement task is added, and this cohort is not a
  census or random sample of all eligible datasets. The source audit is unblinded
  and cannot recover the original prespecified status.
- **2026-10-25, analysis interface and interpretation.** Optional cohort-manifest
  and cohort-selection arguments filter stored candidate scores by dataset ID
  before every analysis. Original settings, seeds, 10,000 bootstrap resamples,
  contrasts, Holm families and the C1 equivalence margin remain unchanged. No
  training is repeated. Corrected outputs use `analysis_2026-10-25_*` directories;
  their decisions distinguish test outcomes from post hoc status and do not
  inherit confirmatory difference/equivalence labels. The original manifests,
  candidate scores and `analysis/` outputs are unchanged. The pre-correction
  protocol is preserved as `audit_2026-10-25/protocol_before_correction.md` under
  the run directory. Audit and score hashes are linked through each cohort
  manifest and decision file, with the reproduction checks recorded in
  `audit_2026-10-25/verification.json`.

- **2026-10-25, verification disclosure.** The author confirms personal review of
  all additions previously excluded from the manuscript's verification account,
  including the experiment and source screening. This supersedes the historical
  opening statement about author review, without asserting that the original
  screening was correct. The audit records its errors and unresolved cases;
  computational checks and stored-score replay are distinguished from training.
