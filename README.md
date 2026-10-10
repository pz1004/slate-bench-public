# Auditing score-level selection claims in tabular benchmark artifacts

Code, configurations and results for the paper *Auditing Score-Level Selection Claims in
Tabular Benchmark Artifacts* (Sooyoung Jang, Hanbat National University).

The paper asks when a claim that a selector beats a baseline in a reused benchmark table
means what it appears to mean. It formalizes such claims as a declaration of nine fields.
It then runs a nested, dataset-level audit on the TabArena results artifact (51 datasets,
45 policies, 100 outer splits) and the same protocol on TabRepo (200 datasets). Finally, it
codes 13 published selection claims against the nine fields.

Every number, table and figure in the paper is generated from the outputs in
`analysis-output/v2/`. `scripts/build_paper_assets.py` writes the numbers to
`paper/generated/numbers.tex`, which the paper reads as `\res{key}`, and the tables to
`paper/generated/tables/`; the analysis scripts write the figures to `paper/figures/`. The
`paper/` folder holds the manuscript and is not part of this repository, so these files are
created when you run the scripts.

## Contents

| Path | Contents |
|---|---|
| `src/tabular_sota_benchmark/` | Library: artifact loaders, selectors, the shared rank rule (`ranking.py`) and the inference layer (`inference.py`) |
| `scripts/` | The nested audit, the analyses, the figure script and the paper-asset builder |
| `configs/` | One YAML file per run. `tabrepo_D244_F3_C1530_200_datasets.txt` pins the TabRepo slice; the two `*_smoke.yaml` files are small versions for a quick check, and `score_level_policy.yaml` is the benchmark script's default, which the paper does not use |
| `analysis-output/v2/` | Outputs of every run the paper uses, including the stored training results of the compact offline check (`tabarena_ict_retraining_extended/`) |
| `docs/meta_study/` | Meta-study protocol, screening log, codes, the blind recoding of a declared subset, the check of the codes behind the fixed-reference and validation-provenance counts, and the verification sheet |
| `docs/claim_template.md` | Blank declaration of the nine fields for a new score-level claim |
| `docs/positioning_coverage.csv` | Which of the nine fields each existing reporting instrument asks for, with the section or item each code rests on |
| `data/tabarena/` | Copy of the TabArena results artifact and dataset metadata (see `NOTICE.md`) |
| `tests/` | Unit tests |

## Setup

Tested with CPython 3.12.13 on Ubuntu 22.04.5 LTS (Linux 6.8), on one workstation with an
Intel Core Ultra 7 270K Plus (24 cores) and 125 GiB of RAM.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

The TabRepo runs also need TabRepo at the commit used for the paper, checked out in
`.deps/tabrepo`:

```bash
git clone https://github.com/autogluon/tabrepo.git .deps/tabrepo
git -C .deps/tabrepo checkout 3785b6f7a7853d992627cf97908081d54d908bd3
.venv/bin/pip install -e .deps/tabrepo
```

On first use, TabRepo downloads the context `D244_F3_C1530_200` into
`.deps/tabrepo/data`. With prediction vectors this is about 134 GB. For that download the
stored runs used a checkout whose `tabrepo/utils/download.py` retries a failed file up to
five times; this changes no data, so an unmodified checkout computes the same results. The
TabArena runs need only the files in `data/tabarena/`.

## Quick check

```bash
make test
make smoke
make assets META_FLAGS=--meta-verified-only
```

- `make test` runs the unit tests (about a minute).
- `make smoke` runs the nested TabArena pipeline on 3 outer splits of 12 datasets (about a
  minute) and writes to `analysis-output/v2/smoke/`, which is not published; the paper uses
  none of it.
- `make assets` regenerates the paper's numbers and tables in `paper/generated/` from the
  stored outputs in a few seconds, and stops with an error if a verdict or comparison the
  paper states no longer holds for the numbers. `paper/generated/numbers.tex` lists every
  number the paper prints, under the key the paper uses.

## Reproducing the results

To rerun everything, in order (about six days on 24 cores, most of it the TabRepo nested run
and the two jackknife runs):

```bash
make all JOBS=20 TABREPO_WORKERS=14
```

Or run the steps one at a time. The wall times below were measured for the stored outputs,
with the worker counts given; most of these runs shared the 24 cores with another run, so
a dedicated machine is faster.

| Target | What it runs | Wall time (workers) |
|---|---|---|
| `make test` | Unit tests | 1 min |
| `make tabarena` | Nested TabArena audit (100 splits) and dataset-level review | 2 h (12) |
| `make tabarena-attribution` | The double-counting tie rule (`rank_rule: legacy`), in-sample weight choice and the shared inner fit (implementation-conventions appendix) | 3.6 h (12) |
| `make tabarena-extended` | Meta-feature rows inside the nested pool | 3.3 h (10) |
| `make tabarena-shrink` | Eight-policy mitigation grid | 1.4 h (10) |
| `make tabarena-analyses` | Validation provenance, $2^5$ factorial, grid diagnostics, MCS/SPA/PBO, evidence units, figures (in `paper/figures/`) | 6 min |
| `make tabrepo` | TabRepo nested audit, score-only panel, reviews, split-half check | 70 h (14) |
| `make tabrepo-attribution` | The TabRepo run with the shared inner fit, and its score-only panel | 1.6 h (14) |
| `make tabrepo-analyses` | The same analyses on both TabRepo panels, and the tie-rule sensitivity of the fixed ensembles | 30 min (14) |
| `make pooled overlap compact` | Pooled cross-artifact diagnostic, TabRepo contrasts without the datasets shared with TabArena, and compact offline check | 1 min |
| `make jackknife` | Delete-d jackknife over the whole nested procedure: TabArena, then TabRepo | 12 h + 47 h (10) |
| `make assets` | `paper/generated/numbers.tex`, `labels.tex` and `tables/` | 10 s |
| `make compress` | Lossless zstd recompression of the nested runs' `unit_scores.parquet` files (see Outputs) | 5 min |

**Design and seeds:**
- 100 task-stratified outer splits (seed 20260604, held-out fraction 0.25).
- One shared matrix of 10,000 dataset bootstrap resamples per artifact (seed 20260608).
- Jackknife subsets with common random numbers (seed 20260620).
- The TabRepo overlap sensitivity resamples its 182 datasets from the shared matrix's seed,
  salted with `exclude_tabarena_overlap`.
- The meta-study subset that was recoded blind was drawn with seed 20261001
  (`docs/meta_study/protocol.md`, Deviations 8 and 9).
- All configs set `rank_rule: consistent` and `weight_selection: lodo`, and use the default
  `inner_source_fit: per_target`: on the training side, an adaptive policy's row on a
  dataset is built from source rows fitted without that dataset. The attribution configs
  give the alternative conventions: two use `weight_selection: resubstitution` (one also
  `rank_rule: legacy`), and the two `*_attribution_inner_shared` configs use
  `inner_source_fit: shared`, the earlier fit in which an inner target entered the source
  rows that its adaptive policies read.
- The jackknife uses cyclic splits (`--split-design cyclic`): every kept dataset is held out
  once in every cycle of four splits.

## Inference

Every contrast is a mean dataset-level paired difference (`scripts/score_level_review.py`,
`src/tabular_sota_benchmark/inference.py`), and each column of `contrasts.csv` names its
target:
- **The mean:** the 95% percentile bootstrap interval (`ci_low`, `ci_high`) and the
  two-sided test that inverts it (`bootstrap_p`, Holm-adjusted within the contrast's family
  as `bootstrap_holm_p`), which gives the separation verdict. The paired t-test
  (`ttest_p`, `ttest_holm_p`), the TOST equivalence tests and the minimum detectable
  effects also target the mean.
- **The typical dataset:** the Wilcoxon signed-rank test (`wilcoxon_p`,
  `wilcoxon_holm_p`) and its location estimate, the Hodges–Lehmann pseudo-median
  (`hodges_lehmann_difference`).
- **The win probability:** the sign test and the win/tie/loss counts.

The bootstrap resamples the completed dataset-level differences as independent draws,
with the fitted procedures held fixed. Each dataset also trains the procedures evaluated on
the others, so this is a working independence approximation: the intervals and verdicts
are nominal and give no guarantee for the procedure. The delete-d jackknife
(`scripts/procedure_level_jackknife.py`) refits the procedures on subsets of datasets; it
is a refitting-sensitivity diagnostic, not a standard error with known coverage.

## Inputs

- **TabArena:** `data/tabarena/df_results.csv`, from
  <https://huggingface.co/datasets/TabArena/benchmark_results>
  (sha256 `7c8c225a9ff8ec48658c5d211184a429e4506f05d981393afb1ceb6964b32853`), and
  `data/tabarena/tabarena_dataset_metadata.csv`
  (sha256 `c68918c9bd1d98cfe5b74d5f21eb360e14dbc1c31430e402fe01725a215f8753`).
  The Hugging Face dataset was no longer publicly accessible on 2026-10-08; the copy
  here is the unmodified file that every TabArena run used.
- **TabRepo:** commit `3785b6f`, context `D244_F3_C1530_200`, restricted to the 200
  datasets listed in `configs/tabrepo_D244_F3_C1530_200_datasets.txt`.
- **Compact offline check:** the stored results in
  `analysis-output/v2/tabarena_ict_retraining_extended/`, trained on 2026-06-02 by the package's
  experiment loop (`python -m tabular_sota_benchmark.cli run-loop`). They are the input of
  the paper's compact check. The training reads the eight OpenML datasets from a local
  parquet cache, which is not redistributed here. The cache mirrors OpenML's parquet store:
  dataset `<id>` (IDs in `data/tabarena/tabarena_dataset_metadata.csv`) is read from
  `data/tabarena/openml_parquet/0004/<id>/dataset_<id>.pq`, the file at
  `https://data.openml.org/datasets/0004/<id>/dataset_<id>.pq`. To train them again, run the
  training step alone (about 4 minutes):

  ```bash
  .venv/bin/python -m tabular_sota_benchmark.cli run-experiment --config configs/tabarena_ict_retraining_extended.yaml
  ```

  This replaces the stored files
  (`git restore analysis-output/v2/tabarena_ict_retraining_extended/` brings them back). The loop's later
  steps check documents from an earlier stage of the project that are not part of this
  repository, so `run-loop` would report failures after training. In the environment of
  `requirements.txt`, a retraining reproduces the decision-tree, random-forest and
  extra-trees rows exactly. The histogram-gradient-boosting rows, and the selector rows
  built on them, differ by up to 0.003 in log loss, and the recorded times always differ.
  The compact check's verdicts do not change: the local ensemble still lowers log loss and
  does not lower rank regret. Its rank difference then comes from two datasets instead of
  one.

## Outputs

`analysis-output/v2/` holds one directory per run or analysis, each with a
`metadata.json`:
- **Nested runs:** the command, the configuration path and its hash, the input hashes, the
  package versions, the git commit of the tree the run was made from (none for a clean
  copy), and `source`: the SHA-256 of every Python file under `src/` and `scripts/` of that
  tree, taken when the run wrote its metadata at its end.
- **Jackknife runs:** the command, the configuration path, the design, the git commit and
  `source`.
- **Analyses:** the command and the seeds.

To check a run's code, compare the hashes in `source.files` with this repository for the
files the run executes: the run script, `scripts/_path_setup.py`, and the package modules
`__init__.py`, `artifacts.py`, `markdown.py` and `ranking.py` (a jackknife run also loads
`scripts/score_level_policy_benchmark.py`). These match for every stored nested run and
jackknife run. The combined hash `source.sha256` does not, because it also covers files the
runs never import, such as the asset builder, which changed after the runs, and in the
development tree a manuscript check that is not published here.

The four TabArena runs of `make tabarena` and `make tabarena-attribution` were made on
2026-10-07 from a clean copy of this repository, so they record no commit. They reproduced
the earlier development runs of the same configurations exactly: every data file of the
four nested runs and their reviews, the CSV files byte for byte and the parquet tables value
for value. The other runs were made from an uncommitted working tree of the development
repository, so their recorded commit does not identify their code; the per-file hashes do.
The TabRepo runs were not repeated in a clean copy, which would take about three days, but
they ran the published pipeline files, as their hashes show.

Two things differ from what the runs wrote:
- The nested runs' `unit_scores.parquet` files (one row per split, unit and method) are
  stored with zstd compression. This is a lossless recompression that keeps every file
  within GitHub's size limits; any recent pyarrow reads them. `make compress` does it and
  checks that every table is unchanged; a nested run writes the file uncompressed again.
- The per-split resume caches (`split_parts/`) and run logs are not included. A rerun
  writes them again.

Every nested run's configuration file matches the hash recorded in its `metadata.json`.

## Meta-study

`docs/meta_study/` holds the protocol, written before the search, and the screening log
(25 records). It also holds the codes (`claims.csv`) and a verification sheet. Each code
moves through up to six column stages:
- the first AI coding (`*_draft` columns);
- a second AI coding made blind to the first (`*_check`);
- the AI-resolved code, frozen when the recheck subset was drawn (`*_ai`);
- a third coding of a declared subset of 75 cells (`*_recheck`), made blind to both
  earlier codings;
- a check of the fixed-references and validation-provenance codes of every claim, the
  codes behind the paper's main counts (`*_countcheck`, with a reason in
  `*_countcheck_note` where the check changed a code; `count_check.csv`, Deviations 11
  and 12);
- the final code (the main columns): the AI-resolved code, unless a disagreement with the
  third coding was resolved against the source to another code, or the check changed it.

Each field's evidence pointer is frozen with the AI-resolved code (`*_evidence_ai`). The
`*_evidence` columns hold that pointer, or for a resolved disagreement with the third
coding, or for a code the check changed, the reason for the resolution or the change.

The protocol's Deviations section records the conventions used to resolve the
disagreements and the rule that drew the subset (`recheck_subset.json`). The third codes,
their pointers and the resolution of each disagreement, checked once more in a separate AI
session (Deviation 10), are in `recheck_coding.csv`. `count_check.csv` and
`count_check_sheet.md` hold the check of the 26 codes behind those counts, with
the section and page locators prepared for it, each checker's own code made before it saw
the code under check (`first_code`), and the decisions (`make meta-count-sheet`,
`make meta-count-merge`). At the author's request this check was made by fresh Claude
sessions rather than by the author (Deviation 12). All codings and checks were made with
Claude (Anthropic), and no human coder was involved; the paper reports this as a
limitation.

## Licence and credits

- **Code:** MIT License (`LICENSE`).
- **Outputs and records:** the outputs in `analysis-output/` and the meta-study and coverage
  records in `docs/` are also released under the MIT License. The outputs derive from the TabArena benchmark results (CC BY 4.0) and
  from TabRepo (Apache 2.0); credit both when reusing them.
- **Data copies:** the files in `data/tabarena/` keep their own licences; see
  `data/tabarena/NOTICE.md`.

## Use of AI tools

The analysis code, the manuscript, all three meta-study codings and the check of the codes
behind its fixed-reference and validation-provenance counts were prepared with
extensive assistance from Claude (Anthropic) under the author's direction; see the
paper's Use of AI Tools section.

## Citation

To cite the code or the stored results, use `CITATION.cff` (GitHub shows it as "Cite this
repository"). The paper's reference will be added here on publication.

## Contact

Sooyoung Jang, Hanbat National University (syjang@hanbat.ac.kr).
