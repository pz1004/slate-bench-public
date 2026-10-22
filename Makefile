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
        tabrepo-grouped tabrepo-grouped-inner literal-vb calibration fresh-validation-tasks fresh-validation-data \
        fresh-validation-pilot fresh-validation tabrepo-frozen-fit \
        meta-sheet assets meta-recheck-draw meta-recheck-sheet meta-recheck-check meta-recheck-merge \
        meta-count-sheet meta-count-check meta-count-merge overlap smoke

all: test tabarena tabarena-attribution tabarena-extended tabarena-shrink tabarena-analyses \
     tabrepo tabrepo-attribution tabrepo-analyses pooled overlap compact jackknife tabrepo-grouped checks \
     tabrepo-grouped-inner literal-vb \
     calibration assets

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
# the two wine-quality halves of TabArena's wine_quality (needs `pooled`); then also without
# the datasets that share a documented source with TabArena (dna, a processed splice, and
# the NASA defect sets from jm1's program).
overlap:
	$(PYTHON) scripts/tabrepo_overlap_sensitivity.py --overlap $(OUT)/pooled_cross_artifact/overlapping_datasets.csv \
	  --review ensemble=$(OUT)/score_level_tabrepo_nested_review \
	  --review score_only=$(OUT)/score_level_tabrepo_score_only_review --output-dir $(OUT)/tabrepo_overlap_sensitivity
	$(PYTHON) scripts/tabrepo_overlap_sensitivity.py --overlap $(OUT)/pooled_cross_artifact/overlapping_datasets.csv \
	  --related wine-quality-red wine-quality-white dna kc1 kc2 mc1 pc1 pc2 pc3 pc4 \
	  --review ensemble=$(OUT)/score_level_tabrepo_nested_review \
	  --review score_only=$(OUT)/score_level_tabrepo_score_only_review --output-dir $(OUT)/tabrepo_overlap_sensitivity_source

# ---------------------------------------------------------------- TabRepo, grouped splits (~10 h at 12 workers)
# The TabRepo audit on 24 outer splits that keep groups of related datasets together
# (configs/tabrepo_source_groups.csv), the cluster bootstrap over those groups, and the
# grouped run against random 24-split subsets of the original splits.
tabrepo-grouped: | $(LOGS)
	$(PYTHON) scripts/grouped_outer_splits.py --config configs/score_level_tabrepo_nested.yaml --repeats 6 \
	  --jobs $(TABREPO_WORKERS) --resume --output-dir $(OUT)/score_level_tabrepo_grouped
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_grouped \
	  --config configs/review_tabrepo.yaml --output-dir $(OUT)/score_level_tabrepo_grouped_review
	$(PYTHON) scripts/score_level_policy_pool_ablation.py --nested-output $(OUT)/score_level_tabrepo_grouped \
	  --output-dir $(OUT)/score_level_tabrepo_grouped_score_only
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_grouped_score_only \
	  --config configs/review_tabrepo_score_only.yaml --output-dir $(OUT)/score_level_tabrepo_grouped_score_only_review
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_grouped_review \
	  --nested-output $(OUT)/score_level_tabrepo_grouped --review-config configs/review_tabrepo.yaml \
	  --output-dir $(OUT)/tabrepo_grouped_corrected_test --unequal-splits
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_grouped_score_only_review \
	  --nested-output $(OUT)/score_level_tabrepo_grouped_score_only --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_grouped_score_only_corrected_test --unequal-splits
	$(PYTHON) scripts/tabrepo_source_groups.py --review ensemble=$(OUT)/score_level_tabrepo_nested_review \
	  --review score_only=$(OUT)/score_level_tabrepo_score_only_review \
	  --review ensemble_grouped=$(OUT)/score_level_tabrepo_grouped_review \
	  --review score_only_grouped=$(OUT)/score_level_tabrepo_grouped_score_only_review --output-dir $(OUT)/tabrepo_source_groups
	$(PYTHON) scripts/tabrepo_grouped_comparison.py --score-only-random $(OUT)/score_level_tabrepo_score_only_review \
	  --score-only-grouped $(OUT)/score_level_tabrepo_grouped_score_only_review \
	  --ensemble-random $(OUT)/score_level_tabrepo_nested_review --ensemble-grouped $(OUT)/score_level_tabrepo_grouped_review \
	  --output-dir $(OUT)/tabrepo_grouped_comparison

# ---------------------------------------------------------------- TabRepo, grouped inner estimates (~10 h at 12 workers)
# The same 24 outer splits with each training dataset's whole group also left out of every
# inner estimate (scripts/grouped_inner_splits.py patches the frozen pipeline in its own
# process only), its reviews and corrected tests, the cluster bootstrap, the comparison with
# the random splits, and the paired comparison with the outer-grouped run.
tabrepo-grouped-inner: | $(LOGS)
	$(PYTHON) scripts/grouped_inner_splits.py --config configs/score_level_tabrepo_nested.yaml --repeats 6 \
	  --jobs $(TABREPO_WORKERS) --resume --output-dir $(OUT)/score_level_tabrepo_grouped_inner
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_grouped_inner \
	  --config configs/review_tabrepo.yaml --output-dir $(OUT)/score_level_tabrepo_grouped_inner_review
	$(PYTHON) scripts/score_level_policy_pool_ablation.py --nested-output $(OUT)/score_level_tabrepo_grouped_inner \
	  --output-dir $(OUT)/score_level_tabrepo_grouped_inner_score_only
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabrepo_grouped_inner_score_only \
	  --config configs/review_tabrepo_score_only.yaml --output-dir $(OUT)/score_level_tabrepo_grouped_inner_score_only_review
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_grouped_inner_review \
	  --nested-output $(OUT)/score_level_tabrepo_grouped_inner --review-config configs/review_tabrepo.yaml \
	  --output-dir $(OUT)/tabrepo_grouped_inner_corrected_test --unequal-splits
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabrepo_grouped_inner_score_only_review \
	  --nested-output $(OUT)/score_level_tabrepo_grouped_inner_score_only --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_grouped_inner_score_only_corrected_test --unequal-splits
	$(PYTHON) scripts/tabrepo_source_groups.py --review ensemble_grouped_inner=$(OUT)/score_level_tabrepo_grouped_inner_review \
	  --review score_only_grouped_inner=$(OUT)/score_level_tabrepo_grouped_inner_score_only_review \
	  --output-dir $(OUT)/tabrepo_source_groups_inner
	$(PYTHON) scripts/tabrepo_grouped_comparison.py --score-only-random $(OUT)/score_level_tabrepo_score_only_review \
	  --score-only-grouped $(OUT)/score_level_tabrepo_grouped_inner_score_only_review \
	  --ensemble-random $(OUT)/score_level_tabrepo_nested_review --ensemble-grouped $(OUT)/score_level_tabrepo_grouped_inner_review \
	  --grouped-splits $(OUT)/score_level_tabrepo_grouped_inner/grouped_splits.json --output-dir $(OUT)/tabrepo_grouped_inner_comparison
	$(PYTHON) scripts/tabrepo_grouped_inner_comparison.py \
	  --outer score_only=$(OUT)/score_level_tabrepo_grouped_score_only_review --outer ensemble=$(OUT)/score_level_tabrepo_grouped_review \
	  --inner score_only=$(OUT)/score_level_tabrepo_grouped_inner_score_only_review --inner ensemble=$(OUT)/score_level_tabrepo_grouped_inner_review \
	  --output-dir $(OUT)/tabrepo_grouped_inner_vs_outer

# ---------------------------------------------------------------- literal validation-best (~10 min)
# The pool's validation-best row adds 1e-6 log(1 + fit seconds) to the validation error; on
# TabRepo that changes the choice in 7 of 600 units. Every TabRepo contrast takes the literal
# rule (validation error, then fit time, then candidate order) as its reference: the reviews
# are rebuilt with the frozen review code, the pool row renamed meta_selector_single_runtime,
# and the analyses that read them rerun into $(LIT). Also checked: the literal rule inside the
# pool, and TabArena's restricted validation-best rows under the same tie rule. Needs the
# TabArena, TabRepo, grouped, attribution, overlap and checks outputs above.
LIT := $(OUT)/literal_vb
LITERAL_RUNS := score_level_tabrepo_nested:review_tabrepo score_level_tabrepo_score_only:review_tabrepo_score_only \
  score_level_tabrepo_grouped:review_tabrepo score_level_tabrepo_grouped_score_only:review_tabrepo_score_only \
  score_level_tabrepo_grouped_inner:review_tabrepo score_level_tabrepo_grouped_inner_score_only:review_tabrepo_score_only \
  attribution/inner_shared_tabrepo_score_only:review_tabrepo_score_only
literal-vb: | $(LOGS)
	$(PYTHON) scripts/literal_validation_best.py units --benchmark-config configs/score_level_tabarena.yaml \
	  --nested-output $(OUT)/score_level_tabarena --output $(LIT)/tabarena_units.parquet
	$(PYTHON) scripts/literal_validation_best.py units --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --nested-output $(OUT)/score_level_tabrepo_nested --output $(LIT)/tabrepo_units.parquet
	$(PYTHON) scripts/literal_validation_best.py tabarena-ties --benchmark-config configs/score_level_tabarena.yaml \
	  --output-dir $(LIT)
	for pair in $(LITERAL_RUNS); do run=$${pair%%:*}; config=$${pair##*:}; \
	  $(PYTHON) scripts/literal_validation_best.py review --nested-output $(OUT)/$$run --review-config configs/$$config.yaml \
	    --units $(LIT)/tabrepo_units.parquet --original-review $(OUT)/$${run}_review --output-dir $(LIT)/$${run}_review || exit 1; \
	done
	for run in nested grouped grouped_inner; do \
	  if [ $$run = nested ]; then source=score_level_tabrepo_nested; review=score_level_tabrepo_nested; scored=score_level_tabrepo_score_only; \
	  else source=score_level_tabrepo_$$run; review=$$source; scored=$${source}_score_only; fi; \
	  $(PYTHON) scripts/literal_validation_best.py substitute --nested-output $(OUT)/$$source --review-config configs/review_tabrepo.yaml \
	    --units $(LIT)/tabrepo_units.parquet --original-review $(OUT)/$${review}_review --literal-review $(LIT)/$${review}_review \
	    --output-dir $(LIT)/pool_substitution/ensemble_$$run || exit 1; \
	  $(PYTHON) scripts/literal_validation_best.py substitute --nested-output $(OUT)/$$source --score-only \
	    --stored-splits $(OUT)/$$scored/split_results.csv --review-config configs/review_tabrepo_score_only.yaml \
	    --units $(LIT)/tabrepo_units.parquet --original-review $(OUT)/$${scored}_review --literal-review $(LIT)/$${scored}_review \
	    --output-dir $(LIT)/pool_substitution/score_only_$$run || exit 1; \
	done
	$(PYTHON) scripts/literal_validation_best.py substitute --nested-output $(OUT)/attribution/inner_shared_tabrepo --score-only \
	  --stored-splits $(OUT)/attribution/inner_shared_tabrepo_score_only/split_results.csv --review-config configs/review_tabrepo_score_only.yaml \
	  --units $(LIT)/tabrepo_units.parquet --original-review $(OUT)/attribution/inner_shared_tabrepo_score_only_review \
	  --literal-review $(LIT)/attribution/inner_shared_tabrepo_score_only_review --output-dir $(LIT)/pool_substitution/score_only_attribution_inner_shared
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(LIT)/score_level_tabrepo_nested_review \
	  --nested-output $(OUT)/score_level_tabrepo_nested --review-config configs/review_tabrepo.yaml \
	  --output-dir $(LIT)/tabrepo_corrected_test
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(LIT)/score_level_tabrepo_score_only_review \
	  --nested-output $(OUT)/score_level_tabrepo_score_only --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(LIT)/tabrepo_score_only_corrected_test
	for run in grouped grouped_inner; do \
	  $(PYTHON) scripts/corrected_resampled_test.py --review-output $(LIT)/score_level_tabrepo_$${run}_review \
	    --nested-output $(OUT)/score_level_tabrepo_$$run --review-config configs/review_tabrepo.yaml \
	    --output-dir $(LIT)/tabrepo_$${run}_corrected_test --unequal-splits || exit 1; \
	  $(PYTHON) scripts/corrected_resampled_test.py --review-output $(LIT)/score_level_tabrepo_$${run}_score_only_review \
	    --nested-output $(OUT)/score_level_tabrepo_$${run}_score_only --review-config configs/review_tabrepo_score_only.yaml \
	    --output-dir $(LIT)/tabrepo_$${run}_score_only_corrected_test --unequal-splits || exit 1; \
	done
	$(PYTHON) scripts/tabrepo_source_groups.py --review ensemble=$(LIT)/score_level_tabrepo_nested_review \
	  --review score_only=$(LIT)/score_level_tabrepo_score_only_review \
	  --review ensemble_grouped=$(LIT)/score_level_tabrepo_grouped_review \
	  --review score_only_grouped=$(LIT)/score_level_tabrepo_grouped_score_only_review --output-dir $(LIT)/tabrepo_source_groups
	$(PYTHON) scripts/tabrepo_source_groups.py --review ensemble_grouped_inner=$(LIT)/score_level_tabrepo_grouped_inner_review \
	  --review score_only_grouped_inner=$(LIT)/score_level_tabrepo_grouped_inner_score_only_review \
	  --output-dir $(LIT)/tabrepo_source_groups_inner
	$(PYTHON) scripts/tabrepo_grouped_comparison.py --score-only-random $(LIT)/score_level_tabrepo_score_only_review \
	  --score-only-grouped $(LIT)/score_level_tabrepo_grouped_score_only_review \
	  --ensemble-random $(LIT)/score_level_tabrepo_nested_review --ensemble-grouped $(LIT)/score_level_tabrepo_grouped_review \
	  --output-dir $(LIT)/tabrepo_grouped_comparison
	$(PYTHON) scripts/tabrepo_grouped_comparison.py --score-only-random $(LIT)/score_level_tabrepo_score_only_review \
	  --score-only-grouped $(LIT)/score_level_tabrepo_grouped_inner_score_only_review \
	  --ensemble-random $(LIT)/score_level_tabrepo_nested_review --ensemble-grouped $(LIT)/score_level_tabrepo_grouped_inner_review \
	  --grouped-splits $(OUT)/score_level_tabrepo_grouped_inner/grouped_splits.json --output-dir $(LIT)/tabrepo_grouped_inner_comparison
	$(PYTHON) scripts/tabrepo_grouped_inner_comparison.py \
	  --outer score_only=$(LIT)/score_level_tabrepo_grouped_score_only_review --outer ensemble=$(LIT)/score_level_tabrepo_grouped_review \
	  --inner score_only=$(LIT)/score_level_tabrepo_grouped_inner_score_only_review --inner ensemble=$(LIT)/score_level_tabrepo_grouped_inner_review \
	  --output-dir $(LIT)/tabrepo_grouped_inner_vs_outer
	$(PYTHON) scripts/tabrepo_overlap_sensitivity.py --overlap $(OUT)/pooled_cross_artifact/overlapping_datasets.csv \
	  --review ensemble=$(LIT)/score_level_tabrepo_nested_review \
	  --review score_only=$(LIT)/score_level_tabrepo_score_only_review --output-dir $(LIT)/tabrepo_overlap_sensitivity
	$(PYTHON) scripts/tabrepo_overlap_sensitivity.py --overlap $(OUT)/pooled_cross_artifact/overlapping_datasets.csv \
	  --related wine-quality-red wine-quality-white dna kc1 kc2 mc1 pc1 pc2 pc3 pc4 \
	  --review ensemble=$(LIT)/score_level_tabrepo_nested_review \
	  --review score_only=$(LIT)/score_level_tabrepo_score_only_review --output-dir $(LIT)/tabrepo_overlap_sensitivity_source
	for panel in score_level_tabrepo_nested score_level_tabrepo_score_only; do \
	  $(PYTHON) scripts/search_aware_inference.py --review-output $(LIT)/$${panel}_review --nested-output $(OUT)/$$panel \
	    --pool-alias meta_selector_single=meta_selector_single_runtime --output-dir $(LIT)/$${panel}_search_inference || exit 1; \
	done
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(LIT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(LIT)/tabrepo_error_space --family-offsets --provenance-output $(OUT)/tabrepo_provenance --tie-rule lexicographic
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(LIT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(LIT)/tabrepo_error_space_median --family-offsets --scale median --tie-rule lexicographic

# ---------------------------------------------------------------- fresh-validation experiment
# The prespecified experiment of docs/fresh_validation_protocol.md. `fresh-validation-tasks`
# rebuilds the eligibility table, the random order and the source-check sheet from OpenML's
# metadata (the frozen task list is configs/fresh_validation_tasks.csv); `fresh-validation-data`
# downloads the candidates' data files (about 60 MB) and runs the automatic checks. The pilot
# runs on the compact check's eight TabArena datasets (local copies).
FV_WORKERS ?= 21
FV_OUT := $(OUT)/fresh_validation
fresh-validation-tasks:
	$(PYTHON) scripts/fresh_validation_tasks.py catalogue
	$(PYTHON) scripts/fresh_validation_tasks.py sheet --first 181
fresh-validation-data:
	$(PYTHON) scripts/fresh_validation_tasks.py download
fresh-validation-pilot: | $(LOGS)
	$(PYTHON) scripts/fresh_validation_experiment.py --tasks configs/fresh_validation_pilot_tasks.csv --repetitions 2 \
	  --output-dir $(OUT)/fresh_validation_pilot --jobs $(FV_WORKERS) --resume
	$(PYTHON) scripts/fresh_validation_analysis.py --run-dir $(OUT)/fresh_validation_pilot \
	  --output-dir $(OUT)/fresh_validation_pilot/analysis
fresh-validation: | $(LOGS)
	$(PYTHON) scripts/fresh_validation_experiment.py --tasks configs/fresh_validation_tasks.csv \
	  --protocol docs/fresh_validation_protocol.md --output-dir $(FV_OUT) --jobs $(FV_WORKERS) --resume
	$(PYTHON) scripts/fresh_validation_analysis.py --run-dir $(FV_OUT) --output-dir $(FV_OUT)/analysis

# TabRepo frozen-fit check (the protocol's Section 8): one 4-fold repeat of the fully grouped
# design with a new split seed; fold 0 is the declared evaluation. (~5 h at 4 workers)
tabrepo-frozen-fit: | $(LOGS)
	$(PYTHON) scripts/grouped_inner_splits.py --config configs/score_level_tabrepo_nested.yaml --repeats 1 \
	  --split-seed 20261022 --jobs 4 --resume --output-dir $(OUT)/score_level_tabrepo_frozen_fit
	$(PYTHON) scripts/score_level_policy_pool_ablation.py --nested-output $(OUT)/score_level_tabrepo_frozen_fit \
	  --output-dir $(OUT)/score_level_tabrepo_frozen_fit_score_only
	$(PYTHON) scripts/tabrepo_frozen_fit.py --run $(OUT)/score_level_tabrepo_frozen_fit \
	  --score-only-run $(OUT)/score_level_tabrepo_frozen_fit_score_only --output-dir $(OUT)/tabrepo_frozen_fit

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

# ---------------------------------------------------------------- Robustness checks (~4 h at 20 jobs)
# TabArena's imputed rows and the main audit rerun without them, with the main run's own
# choices rescored and its fitted rules reapplied on the reduced candidate sets, and again
# without only the rows the metadata documents; the corrected resampled t-test for every
# review; how much the running claim depends on single datasets, trimming and dataset size;
# a rule that selects on defaults and deploys tuned + ensemble; a deployable error-space
# optimism correction with the spread-to-noise ratio on its own offsets; and the declared and
# reduced pools in matched Holm families. Needs the TabArena and TabRepo runs above.
checks: | $(LOGS)
	$(PYTHON) scripts/tabrepo_construction_check.py --output-dir $(OUT)/tabrepo_construction
	$(PYTHON) scripts/imputed_rows.py --output-dir $(OUT)/tabarena_imputed
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_imputed.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena_no_imputed \
	  --config configs/review_tabarena.yaml --output-dir $(OUT)/score_level_tabarena_no_imputed_review
	$(PYTHON) scripts/validation_provenance.py --benchmark-config configs/score_level_tabarena_no_imputed.yaml \
	  --nested-output $(OUT)/score_level_tabarena_no_imputed --output-dir $(OUT)/tabarena_imputed/provenance
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_no_imputed_review \
	  --nested-output $(OUT)/score_level_tabarena_no_imputed --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_imputed/corrected_test
	$(PYTHON) scripts/imputed_rows_rescore.py --main-output $(OUT)/score_level_tabarena \
	  --main-review $(OUT)/score_level_tabarena_review --rerun-output $(OUT)/score_level_tabarena_no_imputed \
	  --rerun-review $(OUT)/score_level_tabarena_no_imputed_review \
	  --candidates $(OUT)/tabarena_imputed/df_results_no_imputed.csv --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_imputed/rescored
	$(PYTHON) scripts/provenance_reference.py --benchmark-config configs/score_level_tabarena.yaml \
	  --review-output $(OUT)/score_level_tabarena_review --review-config configs/review_tabarena.yaml \
	  --nested-output $(OUT)/score_level_tabarena --output-dir $(OUT)/tabarena_provenance_reference
	$(PYTHON) scripts/provenance_reference.py --benchmark-config configs/score_level_tabarena_no_imputed.yaml \
	  --review-output $(OUT)/score_level_tabarena_no_imputed_review --review-config configs/review_tabarena.yaml \
	  --nested-output $(OUT)/score_level_tabarena_no_imputed --output-dir $(OUT)/tabarena_imputed/provenance_reference
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
	  --output-dir $(OUT)/tabarena_error_space --provenance-output $(OUT)/tabarena_provenance
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabarena.yaml \
	  --review-output $(OUT)/score_level_tabarena_review --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_error_space_median --scale median
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(OUT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_error_space --family-offsets --provenance-output $(OUT)/tabrepo_provenance
	$(PYTHON) scripts/error_space_correction.py --benchmark-config configs/score_level_tabrepo_nested.yaml \
	  --review-output $(OUT)/score_level_tabrepo_score_only_review --review-config configs/review_tabrepo_score_only.yaml \
	  --output-dir $(OUT)/tabrepo_error_space_median --family-offsets --scale median
	$(PYTHON) scripts/compress_unit_scores.py $(OUT)/score_level_tabarena_no_imputed/unit_scores.parquet
	$(PYTHON) scripts/imputed_rows_reapply.py --main-output $(OUT)/score_level_tabarena \
	  --main-review $(OUT)/score_level_tabarena_review --rerun-review $(OUT)/score_level_tabarena_no_imputed_review \
	  --rescored-output $(OUT)/tabarena_imputed/rescored --review-config configs/review_tabarena.yaml \
	  --check-splits 100 --check-only --jobs $(JOBS) --output-dir $(OUT)/tabarena_imputed/reapplied
	$(PYTHON) scripts/imputed_rows_reapply.py --main-output $(OUT)/score_level_tabarena \
	  --main-review $(OUT)/score_level_tabarena_review --rerun-review $(OUT)/score_level_tabarena_no_imputed_review \
	  --rescored-output $(OUT)/tabarena_imputed/rescored --review-config configs/review_tabarena.yaml \
	  --jobs $(JOBS) --output-dir $(OUT)/tabarena_imputed/reapplied
	$(PYTHON) scripts/imputed_rows.py --mask metadata --output-dir $(OUT)/tabarena_imputed_flagged
	$(PYTHON) scripts/score_level_policy_benchmark.py --config configs/score_level_tabarena_no_flagged.yaml --jobs $(JOBS)
	$(PYTHON) scripts/score_level_review.py --nested-output $(OUT)/score_level_tabarena_no_flagged \
	  --config configs/review_tabarena.yaml --output-dir $(OUT)/score_level_tabarena_no_flagged_review
	$(PYTHON) scripts/imputed_rows_rescore.py --main-output $(OUT)/score_level_tabarena \
	  --main-review $(OUT)/score_level_tabarena_review --rerun-output $(OUT)/score_level_tabarena_no_flagged \
	  --rerun-review $(OUT)/score_level_tabarena_no_flagged_review \
	  --candidates $(OUT)/tabarena_imputed_flagged/df_results_no_flagged.csv --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_imputed_flagged/rescored
	$(PYTHON) scripts/imputed_rows_reapply.py --reduced-config configs/score_level_tabarena_no_flagged.yaml \
	  --main-output $(OUT)/score_level_tabarena --main-review $(OUT)/score_level_tabarena_review \
	  --rerun-review $(OUT)/score_level_tabarena_no_flagged_review --rescored-output $(OUT)/tabarena_imputed_flagged/rescored \
	  --review-config configs/review_tabarena.yaml --jobs $(JOBS) --output-dir $(OUT)/tabarena_imputed_flagged/reapplied
	$(PYTHON) scripts/corrected_resampled_test.py --review-output $(OUT)/score_level_tabarena_no_flagged_review \
	  --nested-output $(OUT)/score_level_tabarena_no_flagged --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_imputed_flagged/corrected_test
	$(PYTHON) scripts/compress_unit_scores.py $(OUT)/score_level_tabarena_no_flagged/unit_scores.parquet
	$(PYTHON) scripts/pool_matched_family.py --declared-review $(OUT)/score_level_tabarena_review \
	  --declared-nested $(OUT)/score_level_tabarena --reduced-review $(OUT)/score_level_tabarena_shrink_review \
	  --reduced-nested $(OUT)/score_level_tabarena_shrink --review-config configs/review_tabarena.yaml \
	  --output-dir $(OUT)/tabarena_pool_matched

# ---------------------------------------------------------------- coverage study (~11 h at 22 workers)
# Simulated TabArena-shaped score tables run through the audit's own code (paper App. K): how
# well the audit's tests of the mean cover a known procedure-level contrast.
calibration: | $(LOGS)
	$(PYTHON) scripts/calibration_study.py --stage estimate
	$(PYTHON) scripts/calibration_study.py --stage facts
	$(PYTHON) scripts/calibration_study.py --stage pilot
	$(PYTHON) scripts/calibration_study.py --stage truth --jobs $(JOBS)
	$(PYTHON) scripts/calibration_study.py --stage truth_shared --jobs $(JOBS)
	$(PYTHON) scripts/calibration_study.py --stage replicates --jobs $(JOBS)
	$(PYTHON) scripts/calibration_study.py --stage bridge --jobs $(JOBS)
	$(PYTHON) scripts/calibration_study.py --stage summarize

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
