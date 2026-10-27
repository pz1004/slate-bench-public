"""Cohort correction must change membership everywhere, never the analysis design."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import fresh_validation_analysis as analysis


@pytest.fixture
def run_inputs(tmp_path):
    families = ["knn", "logistic_regression"]
    rows = []
    rng = np.random.default_rng(22)
    for did in range(100, 108):
        for rep in range(2):
            for family in families:
                for variant in ["default", "tuned", "tuned + ensemble"]:
                    rows.append(dict(dataset_id=did, dataset=f"task-{did}", repetition=rep,
                                     n_rows=1000 * (did - 99), family=family, variant=variant,
                                     candidate=f"{family} ({variant})", fit_seconds=1,
                                     construction_loss=rng.uniform(.1, 1),
                                     selection_loss=rng.uniform(.1, 1), test_loss=rng.uniform(.1, 1)))
    candidates = pd.DataFrame(rows)
    # Use the implementation's public naming convention, not a guessed separator.
    from tabular_sota_benchmark.fresh_validation import candidate_order
    names = list(candidate_order(families))
    candidates["candidate"] = names * 16
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidates.to_parquet(run_dir / "candidates.parquet", index=False)
    (run_dir / "metadata.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(dict(families={k: {} for k in families},
                                         analysis=dict(bootstrap_seed=20260608, bootstrap_resamples=99,
                                                       equivalence_rank_positions=.5))))
    manifest = tmp_path / "cohort.csv"
    pd.DataFrame(dict(dataset_id=range(100, 108),
                      status=["eligible"] * 4 + ["unresolved"] * 2 + ["ineligible"] * 2)).to_csv(manifest, index=False)
    return candidates, run_dir, config, manifest


def test_membership_requires_verified_eligibility(run_inputs):
    candidates, _, _, manifest = run_inputs
    selected, meta = analysis.select_cohort(candidates, manifest, "eligible")
    assert set(selected.dataset_id) == {100, 101, 102, 103}
    assert meta["analysis_status"] == "post hoc eligibility correction"
    broad, _ = analysis.select_cohort(candidates, manifest, "eligible-unresolved")
    assert set(broad.dataset_id) == set(range(100, 106))
    original, meta = analysis.select_cohort(candidates, manifest, "original")
    pd.testing.assert_frame_equal(original, candidates)
    assert meta["analysis_status"] == "original reproduction"


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "missing", "status", "fractional"])
def test_rejects_invalid_manifest(run_inputs, problem):
    candidates, _, _, manifest = run_inputs
    frame = pd.read_csv(manifest)
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "unknown":
        frame.loc[0, "dataset_id"] = 999
    elif problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "status":
        frame.loc[0, "status"] = "probably eligible"
    else:
        frame["dataset_id"] = frame.dataset_id.astype(float)
        frame.loc[0, "dataset_id"] = 100.5
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        analysis.select_cohort(candidates, manifest, "eligible")


def test_every_output_matches_a_run_with_only_selected_inputs(run_inputs, tmp_path):
    candidates, run_dir, config, manifest = run_inputs
    corrected = tmp_path / "corrected"
    analysis.run(Namespace(config=config, run_dir=run_dir, output_dir=corrected,
                           cohort_manifest=manifest, cohort_selection="eligible"))
    direct_run = tmp_path / "direct-run"
    direct_run.mkdir()
    candidates[candidates.dataset_id.lt(104)].to_parquet(direct_run / "candidates.parquet", index=False)
    (direct_run / "metadata.json").write_text("{}")
    direct = tmp_path / "direct"
    analysis.run(Namespace(config=config, run_dir=direct_run, output_dir=direct))
    for path in direct.glob("*.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(corrected / path.name))
    result = json.loads((corrected / "decisions.json").read_text())
    assert result["n_datasets"] == 4 and result["n_units"] == 8
    assert result["cohort"]["selected_dataset_ids"] == [100, 101, 102, 103]
    assert result["input_sha256"]["candidates"] == analysis.sha256(run_dir / "candidates.parquet")
    assert all(d["confirmatory"] is False for d in result["decisions"].values())
    assert sum(pd.read_csv(corrected / "by_size.csv").query("contrast == 'C1'").n_datasets) == 4


def test_corrected_rejections_do_not_inherit_confirmatory_labels():
    row = pd.Series(dict(bootstrap_holm_p=.001, ttest_holm_p=.002,
                         tost_90_ci_low=-.01, tost_90_ci_high=.01))
    assert analysis.decide(row, .03)["conclusion"] == "difference supported"
    corrected = analysis.decide(row, .03, corrected=True)
    assert corrected["test_outcome"] == "both tests reject"
    assert corrected["conclusion"].startswith("difference unresolved")
    assert corrected["equivalence_supported"] is False


def test_cannot_overwrite_historical_analysis(run_inputs):
    _, run_dir, config, manifest = run_inputs
    with pytest.raises(ValueError, match="overwrite"):
        analysis.run(Namespace(config=config, run_dir=run_dir, output_dir=run_dir / "analysis",
                               cohort_manifest=manifest, cohort_selection="eligible"))


def test_committed_audit_covers_all_screening_decisions_and_cohort():
    root = Path(__file__).resolve().parents[1]
    audit = pd.read_csv(root / "configs/fresh_validation_eligibility_audit.csv")
    original = pd.read_csv(root / "configs/fresh_validation_source_check.csv", comment="#")
    cohort = pd.read_csv(root / "configs/fresh_validation_cohort_2026-10-25.csv")
    assert len(audit) == 181 and audit.dataset_id.is_unique
    assert set(audit.dataset_id) == set(original.did)
    assert len(cohort) == 46 and cohort.dataset_id.is_unique
    assert set(cohort.dataset_id) == set(original.loc[original.decision.eq("include"), "did"])
    assert set(analysis.REPORTED_FAILURES) <= set(cohort.loc[cohort.status.eq("ineligible"), "dataset_id"])
    assert cohort.audit_sha256.eq(analysis.sha256(root / "configs/fresh_validation_eligibility_audit.csv")).all()


def test_three_exclusion_uses_ids_and_requires_all_three(run_inputs):
    candidates, _, _, manifest = run_inputs
    with pytest.raises(ValueError, match="all three"):
        analysis.select_cohort(candidates, manifest, "three-exclusion")
    mapping = {100: 46527, 101: 40666, 102: 46864}
    candidates["dataset_id"] = candidates.dataset_id.replace(mapping)
    frame = pd.read_csv(manifest)
    frame["dataset_id"] = frame.dataset_id.replace(mapping)
    frame.to_csv(manifest, index=False)
    selected, meta = analysis.select_cohort(candidates, manifest, "three-exclusion")
    assert set(selected.dataset_id) == set(range(103, 108))
    assert meta["excluded_dataset_ids"] == sorted(analysis.REPORTED_FAILURES)


@pytest.mark.parametrize("missing", ["cohort_manifest", "cohort_selection"])
def test_cohort_arguments_are_paired(run_inputs, tmp_path, missing):
    _, run_dir, config, manifest = run_inputs
    kwargs = dict(config=config, run_dir=run_dir, output_dir=tmp_path / "out",
                  cohort_manifest=manifest, cohort_selection="eligible")
    kwargs[missing] = None
    with pytest.raises(ValueError, match="supplied together"):
        analysis.run(Namespace(**kwargs))


def test_asset_builder_filters_exploratory_pool_and_rejects_stale_labels(run_inputs, tmp_path):
    from scripts import build_paper_assets as builder
    candidates, run_dir, config, manifest = run_inputs
    metadata = json.dumps(dict(n_parts=32, job_seconds_this_invocation=3600))
    (run_dir / "metadata.json").write_text(metadata)
    analysis.run(Namespace(config=config, run_dir=run_dir, output_dir=run_dir / "corrected",
                           cohort_manifest=manifest, cohort_selection="eligible"))
    direct_run = tmp_path / "direct-run"
    direct_run.mkdir()
    candidates[candidates.dataset_id.lt(104)].to_parquet(direct_run / "candidates.parquet", index=False)
    (direct_run / "metadata.json").write_text(metadata)
    analysis.run(Namespace(config=config, run_dir=direct_run, output_dir=direct_run / "analysis"))
    corrected, direct = builder.Assets(tmp_path, allow_missing=False), builder.Assets(tmp_path, allow_missing=False)
    builder.build_fresh_validation(corrected, "check", "run/corrected")
    builder.build_fresh_validation(direct, "check", "direct-run/analysis")
    # Includes all numeric outputs, particularly within-family ensemble comparisons,
    # which are computed by the builder directly from candidate scores.
    assert corrected.numbers == direct.numbers
    assert "Difference supported" not in corrected.tables["fresh_validation_check_endpoints"]
    path = run_dir / "corrected/decisions.json"
    result = json.loads(path.read_text())
    result["decisions"]["C1"]["confirmatory"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="confirmatory labels"):
        builder.build_fresh_validation(builder.Assets(tmp_path, allow_missing=False), "check", "run/corrected")
