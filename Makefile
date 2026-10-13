# Reproduce every number, table and figure of the paper (v2 design).
#
#   make all            # full pipeline: runs, analyses and paper assets
#   make tabarena JOBS=20
#   make assets         # regenerate paper/generated/ from the stored outputs
#
# Every output lands under $(OUT). Wall times were measured on 24 cores with the stated
# workers, mostly while another run shared the machine.

PYTHON  ?= .venv/bin/python
OUT     ?= analysis-output/v2
JOBS    ?= 20
LOGS    := $(OUT)/logs

# One BLAS thread per worker process; the scripts parallelise across splits.
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

.PHONY: all test tabarena tabarena-attribution tabarena-extended tabarena-shrink \
        tabarena-analyses tabrepo tabrepo-attribution tabrepo-analyses pooled compact jackknife checks compress \
        meta-sheet assets meta-recheck-draw meta-recheck-sheet meta-recheck-check meta-recheck-merge \
        meta-count-sheet meta-count-check meta-count-merge overlap smoke

all: test tabarena tabarena-attribution tabarena-extended tabarena-shrink tabarena-analyses \
     tabrepo tabrepo-attribution tabrepo-analyses pooled overlap compact jackknife checks assets

$(LOGS):
	mkdir -p $(LOGS)

test:
	$(PYTHON) -m pytest -q

# Quick check of the nested pipeline on TabArena: 3 outer splits of 12 datasets (~1 min).
# It writes to $(OUT)/smoke/, which is not published; the paper uses none of it.
smoke:
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_smoke.yaml --jobs $(JOBS)

# ---------------------------------------------------------------- TabArena (~2 h at 12 jobs)
tabarena: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena \
	  --config configs/review_tabarena.yaml --output-dir $(OUT)/score_level_tabarena_review

# Implementation-conventions appendix: the legacy tie rule with in-sample weights, the
# consistent tie rule with in-sample weights, and the reported conventions with the shared
# inner fit of the adaptive policies; the reported conventions are the main run.
# (~1.5-2 h for each per-target run and ~15 min for the shared-fit run, at 12 jobs)
tabarena-attribution: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_legacy.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/attribution/legacy_v1 \
	  --config configs/review_tabarena_attribution.yaml --output-dir $(OUT)/attribution/legacy_v1_review
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_rank_fix.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/attribution/rank_fix_only \
	  --config configs/review_tabarena_attribution.yaml --output-dir $(OUT)/attribution/rank_fix_only_review
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_attribution_inner_shared.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/attribution/inner_shared \
	  --config configs/review_tabarena.yaml --output-dir $(OUT)/attribution/inner_shared_review
	$(PYTHON) scripts/grid_diagnostics.py --nested-output $(OUT)/attribution/inner_shared \
	  --review-output $(OUT)/attribution/inner_shared_review --output-dir $(OUT)/attribution/inner_shared_grid_diagnostics

# Search-space sensitivity: meta-feature rows inside the nested pool. (~3.3 h at 10 jobs)
tabarena-extended: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_extended_pool.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena_extended_pool \
	  --config configs/review_tabarena_extended.yaml --output-dir $(OUT)/score_level_tabarena_extended_pool_review

# Eight-policy mitigation grid. (~1.4 h at 10 jobs)
tabarena-shrink: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_shrink.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena_shrink \
	  --config configs/review_tabarena_shrink.yaml --output-dir $(OUT)/score_level_tabarena_shrink_review
	$(PYTHON) scripts/grid_diagnostics.py --nested-output $(OUT)/score_level_tabarena_shrink \
	  --review-output $(OUT)/score_level_tabarena_shrink_review \
	  --output-dir $(OUT)/score_level_tabarena_shrink_grid_diagnostics --prefix-counts 8

# Provenance, factorial, grid diagnostics, search-aware inference, evidence units. (~6 min)
tabarena-analyses: | $(LOGS)
	$(PYTHON) scripts/validation_provenance.py --benchmark-config configs/score_level_tabarena.yaml \
	  --nested-output $(OUT)/score_level_tabarena --output-dir $(OUT)/tabarena_provenance \
	  --figure paper/figures/validation_calibration.pdf
	$(PYTHON) scripts/family_prior_factorial.py --benchmark-config configs/score_level_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_factorial --jobs $(JOBS)
	$(PYTHON) scripts/grid_diagnostics.py --nested-output $(OUT)/score_level_tabarena \
	  --review-output $(OUT)/score_level_tabarena_review --output-dir $(OUT)/tabarena_grid_diagnostics
	$(PYTHON) scripts/search_aware_inference.py --review-output $(OUT)/score_level_tabarena_review \
	  --nested-output $(OUT)/score_level_tabarena --output-dir $(OUT)/tabarena_search_inference
	$(PYTHON) scripts/evidence_unit_exhibit.py --nested-output $(OUT)/score_level_tabarena \
	  --review-output $(OUT)/score_level_tabarena_review --output-dir $(OUT)/tabarena_evidence_unit
	$(PYTHON) scripts/paper_figures.py --review $(OUT)/score_level_tabarena_review --figures paper/figures
	$(PYTHON) scripts/validation_best_tie_rule.py --benchmark-config configs/score_level_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_tie_rule

# ---------------------------------------------------------------- TabRepo (~70 h at 14 workers)
# The score-only panel removes the ensemble rows from the nested pool of the same run.
TABREPO_WORKERS ?= 12
tabrepo: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabrepo_nested.yaml --jobs $(TABREPO_WORKERS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_nested \
	  --config configs/review_tabrepo.yaml --output-dir $(OUT)/score_level_tabrepo_nested_review
	$(PYTHON) scripts/score_level_policy_pool_ablation.py --nested-output $(OUT)/score_level_tabrepo_nested \
	  --output-dir $(OUT)/score_level_tabrepo_score_only
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_score_only \
	  --config configs/review_tabrepo_score_only.yaml --output-dir $(OUT)/score_level_tabrepo_score_only_review
	$(PYTHON) scripts/tabrepo_split_half_replication.py --output-dir $(OUT)/tabrepo_split_half

# The TabRepo run with the shared inner fit, for the implementation-conventions appendix.
# (~1.6 h at 14 workers)
tabrepo-attribution: | $(LOGS)
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabrepo_attribution_inner_shared.yaml --jobs $(TABREPO_WORKERS)
	$(PYTHON) scripts/score_level_policy_pool_ablation.py --nested-output $(OUT)/attribution/inner_shared_tabrepo \
	  --output-dir $(OUT)/attribution/inner_shared_tabrepo_score_only
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/attribution/inner_shared_tabrepo_score_only \
	  --config configs/review_tabrepo_score_only.yaml --output-dir $(OUT)/attribution/inner_shared_tabrepo_score_only_review

# (~30 min at 14 jobs, dominated by the factorial)
tabrepo-analyses: | $(LOGS)
	$(PYTHON) scripts/family_prior_factorial.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --output-dir $(OUT)/tabrepo_factorial --jobs $(JOBS)
	$(PYTHON) scripts/validation_provenance.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --nested-output $(OUT)/score_level_tabrepo_nested --output-dir $(OUT)/tabrepo_provenance
	for panel in score_level_tabrepo_nested score_level_tabrepo_score_only; do \
	  $(PYTHON) scripts/grid_diagnostics.py --nested-output $(OUT)/$$panel \
	    --review-output $(OUT)/$${panel}_review --output-dir $(OUT)/$${panel}_grid_diagnostics && \
	  $(PYTHON) scripts/search_aware_inference.py --review-output $(OUT)/$${panel}_review \
	    --nested-output $(OUT)/$$panel --output-dir $(OUT)/$${panel}_search_inference || exit 1; \
	done
	$(PYTHON) scripts/evidence_unit_exhibit.py --nested-output $(OUT)/score_level_tabrepo_nested \
	  --review-output $(OUT)/score_level_tabrepo_nested_review \
	  --output-dir $(OUT)/score_level_tabrepo_nested_evidence_unit \
	  --contrasts nested_policy_selected:meta_selector_tabrepo_ensemble_k3 \
	    nested_policy_selected:train_selected_fixed_reference__score_only \
	    nested_policy_selected:meta_selector_single
	$(PYTHON) scripts/evidence_unit_exhibit.py --nested-output $(OUT)/score_level_tabrepo_score_only \
	  --review-output $(OUT)/score_level_tabrepo_score_only_review \
	  --output-dir $(OUT)/score_level_tabrepo_score_only_evidence_unit \
	  --contrasts nested_policy_selected:meta_selector_single \
	    nested_policy_selected:train_selected_fixed_reference__score_only
	$(PYTHON) scripts/rank_rule_sensitivity.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --nested-output $(OUT)/score_level_tabrepo_nested --output-dir $(OUT)/tabrepo_rank_rule_sensitivity
	$(PYTHON) scripts/validation_best_tie_rule.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --output-dir $(OUT)/tabrepo_tie_rule

# ---------------------------------------------------------------- cross-artifact, compact
pooled:
	$(PYTHON) scripts/pooled_cross_artifact.py --tabarena-review $(OUT)/score_level_tabarena_review \
	  --tabrepo-review $(OUT)/score_level_tabrepo_score_only_review --output-dir $(OUT)/pooled_cross_artifact

# TabRepo contrasts without the 16 datasets that share a name with a TabArena dataset and
# the two wine-quality halves of TabArena's wine_quality (needs `pooled`).
overlap:
	$(PYTHON) scripts/tabrepo_overlap_sensitivity.py --overlap $(OUT)/pooled_cross_artifact/overlapping_datasets.csv \
	  --review ensemble=$(OUT)/score_level_tabrepo_nested_review \
	  --review score_only=$(OUT)/score_level_tabrepo_score_only_review --output-dir $(OUT)/tabrepo_overlap_sensitivity

compact:
	$(PYTHON) scripts/third_artifact_validation.py --input analysis-output/v2/tabarena_ict_retraining_extended/results.csv \
	  --output-dir $(OUT)/third_artifact_validation

# ---------------------------------------------------------------- jackknife (~12 h + ~47 h at 10 workers)
# Refitting sensitivity. Cyclic splits hold out every kept dataset once per cycle of four.
jackknife: | $(LOGS)
	$(PYTHON) scripts/procedure_level_jackknife.py --benchmark-config configs/score_level_tabarena.yaml \
	  --review-config configs/review_tabarena.yaml --reference-pool declared_fixed \
	  --output-dir $(OUT)/tabarena_jackknife --subsets 50 --splits 12 --drop 10 --split-design cyclic --jobs $(JOBS)
	$(PYTHON) scripts/procedure_level_jackknife.py --benchmark-config configs/score_level_tabrepo_score_only.yaml \
	  --review-config configs/review_tabrepo.yaml --reference-pool score_only \
	  --output-dir $(OUT)/tabrepo_jackknife --subsets 20 --splits 4 --drop 40 --split-design cyclic --jobs $(TABREPO_WORKERS)

# ---------------------------------------------------------------- Robustness checks (~1.5 h at 12 jobs)
# TabArena's copied rows and the main audit rerun without them; the corrected resampled
# t-test for every review; how much the running claim depends on single datasets; and a
# deployable error-space optimism correction. Needs the TabArena and TabRepo runs above.
checks: | $(LOGS)
	$(PYTHON) scripts/imputed_rows.py --output-dir $(OUT)/tabarena_imputed
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_imputed.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena_no_imputed \
	  --config configs/review_tabarena.yaml --output-dir $(OUT)/score_level_tabarena_no_imputed_review
	$(PYTHON) scripts/validation_provenance.py --benchmark-config configs/score_level_tabarena_no_imputed.yaml \
	  --nested-output $(OUT)/score_level_tabarena_no_imputed --output-dir $(OUT)/tabarena_imputed/provenance
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_no_imputed_review \
	  --nested-output $(OUT)/score_level_tabarena_no_imputed --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_imputed/corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_review \
	  --nested-output $(OUT)/score_level_tabarena --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_extended_pool_review \
	  --nested-output $(OUT)/score_level_tabarena_extended_pool --review-config configs/review_tabarena_extended.yaml \
	  --output-dir $(OUT)/tabarena_extended_corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_shrink_review \
	  --nested-output $(OUT)/score_level_tabarena_shrink --review-config configs/review_tabarena_shrink.yaml \
	  --output-dir $(OUT)/tabarena_shrink_corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_nested_review \
	  --nested-output $(OUT)/score_level_tabrepo_nested --review-config configs/review_tabrepo.yaml \
	  --output-dir $(OUT)/tabrepo_corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_score_only_review \
	  --nested-output $(OUT)/score_level_tabrepo_score_only --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_score_only_corrected_test
	$(PYTHON) scripts/influence_check.py --review-output $(OUT)/score_level_tabarena_review \
	  --nested-output $(OUT)/score_level_tabarena --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_influence
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabarena.yaml \
	  --review-output $(OUT)/score_level_tabarena_review --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_error_space
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabarena.yaml \
	  --review-output $(OUT)/score_level_tabarena_review --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_error_space_median --scale median
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(OUT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_error_space --family-offsets
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(OUT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_error_space_median --family-offsets --scale median
	$(PYTHON) scripts/compress_unit_scores.py $(OUT)/score_level_tabarena_no_imputed/unit_scores.parquet

# Recompress the nested runs' unit_scores.parquet files losslessly with zstd, so that every
# file stays within GitHub's size limits. Run it after a nested run, before committing.
compress:
	$(PYTHON) scripts/compress_unit_scores.py --out $(OUT)

# ---------------------------------------------------------------- paper assets
# Writes paper/generated/numbers.tex, labels.tex and tables/*.tex; paper/ holds the manuscript
# and is not part of the public repository. For the submission build add
# META_FLAGS=--meta-verified-only once every meta-study coding is verified.
# Rewrite docs/meta_study/verification_sheet.md from claims.csv (after editing codes).
meta-sheet:
	$(PYTHON) scripts/meta_study_sheet.py

# Third, blind coding of the declared subset (docs/meta_study/protocol.md, Deviations 8-9):
# draw the subset once, write the blank coding files, check progress, then merge the codes.
RECHECK_SEED ?= 20261001
meta-recheck-draw:
	$(PYTHON) scripts/meta_study_sheet.py recheck-draw --seed $(RECHECK_SEED)

meta-recheck-sheet:
	$(PYTHON) scripts/meta_study_sheet.py recheck-sheet

meta-recheck-check:
	$(PYTHON) scripts/meta_study_sheet.py recheck-merge --dry-run

meta-recheck-merge:
	$(PYTHON) scripts/meta_study_sheet.py recheck-merge

# The check of the codes behind the abstract's counts (protocol.md, Deviations 11 and 12;
# made by fresh Claude sessions at the author's request): write the check file and its
# readable sheet, check progress, then merge the decisions.
meta-count-sheet:
	$(PYTHON) scripts/meta_study_sheet.py count-check-sheet

meta-count-check:
	$(PYTHON) scripts/meta_study_sheet.py count-check-merge --dry-run

meta-count-merge:
	$(PYTHON) scripts/meta_study_sheet.py count-check-merge

META_FLAGS ?=
assets:
	$(PYTHON) scripts/build_paper_assets.py --out $(OUT) $(META_FLAGS)

# Optional local targets, not part of the repository.
-include local.mk
