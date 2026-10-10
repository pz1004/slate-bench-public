from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_assets.py"
SPEC = importlib.util.spec_from_file_location("build_paper_assets", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
assets_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets_module
SPEC.loader.exec_module(assets_module)

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tabular_sota_benchmark import meta_study  # noqa: E402

CODED = meta_study.ITEMS


def _claim(record: str, key: str, verification: str) -> dict[str, str]:
    row = {"record_id": record, "paper": f"Author {record} (Tag)", "cite_key": key, "verification": verification}
    for column in CODED:
        value = "yes" if column in meta_study.EXTRA_COLUMNS else "D"
        row[column] = value
        row[f"{column}_draft"] = value
        row[f"{column}_check"] = value
        if column in meta_study.FIELD_COLUMNS:
            row[f"{column}_evidence"] = f"Sec. 1 of {record}"
    return row


def _write_meta_study(root: Path, claims: list[dict[str, str]]) -> None:
    directory = root / "docs" / "meta_study"
    directory.mkdir(parents=True)
    pd.DataFrame(claims).to_csv(directory / "claims.csv", index=False)
    records = [claim["record_id"] for claim in claims]
    pd.DataFrame(
        {"record_id": records + ["S98", "S99"], "decision": ["include"] * len(records) + ["exclude", "not retrieved"]}
    ).to_csv(directory / "screening.csv", index=False)


def _declare(
    root: Path,
    *,
    seed: int = 7,
    n_full: int = 1,
    recheck: dict[tuple[str, str], str] | None = None,
    resolved: dict[tuple[str, str], str] | None = None,
    count_check: bool | dict[tuple[str, str], str] = True,
) -> dict:
    """Draw the recheck subset of the written claims and fill it, by default with the AI codes.

    ``recheck`` overrides third codes; ``resolved`` gives the code a disagreement was resolved
    to against the source (the final code), with a note. ``count_check`` adds the count check:
    ``True`` confirms every checked code, a dict changes the codes it names (with a reason),
    and ``False`` leaves the check out.
    """
    directory = root / "docs" / "meta_study"
    claims = meta_study.read_claims(directory / "claims.csv")
    population = sorted(claims.loc[claims["verification"].eq("verified"), "record_id"])
    declaration = {
        "population": population,
        "all_claims_items": list(meta_study.RECHECK_ALL_CLAIM_ITEMS),
        "seed": seed,
        "n_full": n_full,
        "full_claims": meta_study.draw_full_claims(population, seed, n_full),
        "ai_codes_sha256": meta_study.ai_codes_digest(claims, population),
    }
    for item in CODED:
        claims[f"{item}_ai"] = claims[item]
        claims[f"{item}_recheck"] = ""
    for field in meta_study.FIELD_COLUMNS:
        claims[f"{field}_evidence_ai"] = claims[f"{field}_evidence"]
    index = {record: position for position, record in enumerate(claims["record_id"])}
    sheet = []
    for record, item in meta_study.declared_cells(declaration):
        ai = claims.at[index[record], f"{item}_ai"]
        code = (recheck or {}).get((record, item), ai)
        final = (resolved or {}).get((record, item), "")
        note = "resolved against the source" if final else ""
        claims.at[index[record], f"{item}_recheck"] = code
        if code and code != ai and final:
            claims.at[index[record], item] = final
            if item in meta_study.FIELD_COLUMNS:
                claims.at[index[record], f"{item}_evidence"] = note
        sheet.append({"record_id": record, "item": item, "code": code, "resolved": final, "resolution_note": note})
    if count_check is not False:
        changes = count_check if isinstance(count_check, dict) else {}
        for item in meta_study.COUNT_CHECK_ITEMS:
            claims[f"{item}_countcheck"] = ""
            claims[f"{item}_countcheck_note"] = ""
        check_rows = []
        for record, item in meta_study.count_check_cells(population):
            row = index[record]
            under_check = claims.at[row, item]
            code = changes.get((record, item), under_check)
            check_rows.append({"record_id": record, "item": item, "code": under_check, "first_code": under_check})
            claims.at[row, f"{item}_countcheck"] = code
            if code != claims.at[row, item]:
                claims.at[row, f"{item}_countcheck_note"] = "changed by the count check"
                claims.at[row, item] = code
                if item in meta_study.FIELD_COLUMNS:
                    claims.at[row, f"{item}_evidence"] = "changed by the count check"
    if count_check is not False:
        pd.DataFrame(check_rows).to_csv(directory / meta_study.COUNT_CHECK_FILE, index=False)
    claims.to_csv(directory / "claims.csv", index=False)
    pd.DataFrame(sheet).to_csv(directory / meta_study.RECHECK_CODING_FILE, index=False)
    (directory / meta_study.RECHECK_SUBSET_FILE).write_text(json.dumps(declaration), encoding="utf-8")
    return declaration


def _build(root: Path, *, submission: bool = True) -> dict[str, str]:
    assets = assets_module.Assets(root, allow_missing=False, meta_verified_only=submission)
    assets_module.build_meta_study(assets, root)
    return assets.numbers


def test_meta_study_counts_changes_and_leaves_out_excluded_claims(tmp_path: Path) -> None:
    changed = _claim("S01", "alpha", "verified")
    changed["fixed_references"] = "P"  # the adjudication corrected one drafted code
    changed["fixed_references_check"] = "P"  # after the blind second coding disagreed
    disputed = _claim("S02", "beta", "verified")
    disputed["endpoint_check"] = "A"  # a disagreement resolved in favour of the draft
    claims = [changed, disputed, _claim("S03", "gamma", "excluded")]
    _write_meta_study(tmp_path, claims)
    _declare(tmp_path)
    numbers = _build(tmp_path)
    assert numbers["meta.included"] == "2"
    assert numbers["meta.screened"] == "5"
    assert numbers["meta.notretrieved"] == "1"
    assert numbers["meta.excludedatverification"] == "1"
    assert numbers["meta.codes.checked"] == str(2 * len(CODED))
    assert numbers["meta.codes.agreed"] == str(2 * len(CODED) - 2)
    assert numbers["meta.codes.changed"] == "1"
    assert numbers["meta.codes.nine.checked"] == "18"
    assert numbers["meta.codes.nine.agreed"] == "16"
    assert numbers["meta.codes.disagreed"] == "2"
    assert numbers["meta.codes.disagreed.check"] == "1"
    assert numbers["meta.codes.disagreed.draft"] == "1"
    assert numbers["meta.fixedreferences.declared"] == "1"
    assert numbers["meta.recheck.cells"] == "15"
    assert numbers["meta.recheck.changed"] == "0"
    assert numbers["meta.recheck.disagreed"] == "0"
    assert "gamma" not in numbers["meta.citations"]


def test_submission_build_refuses_unverified_claims(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "unverified")])
    assets = assets_module.Assets(tmp_path, allow_missing=False, meta_verified_only=True)
    with pytest.raises(ValueError, match="S02"):
        assets_module.build_meta_study(assets, tmp_path)


def test_verified_claim_needs_a_second_pass_code(tmp_path: Path) -> None:
    claim = _claim("S01", "alpha", "verified")
    claim["endpoint_check"] = ""
    _write_meta_study(tmp_path, [claim])
    assets = assets_module.Assets(tmp_path, allow_missing=False, meta_verified_only=True)
    with pytest.raises(ValueError, match="S01:endpoint"):
        assets_module.build_meta_study(assets, tmp_path)


def test_draft_build_marks_agreement_pending_before_verification(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "unverified")])
    assets = assets_module.Assets(tmp_path, allow_missing=False)
    assets_module.build_meta_study(assets, tmp_path)
    assert "pending" in assets.numbers["meta.codes.changed"]
    assert assets.numbers["meta.included"] == "1"
    assert "pending" in assets.numbers["meta.codes.nine.kappa"]
    assert "pending" in assets.numbers["meta.recheck.agreed"]


def test_kappa_matches_scikit_learn_and_marks_the_degenerate_case() -> None:
    from sklearn.metrics import cohen_kappa_score

    first = ["D", "P", "A", "NA", "D", "D", "P", "A", "D", "P"]
    second = ["D", "P", "P", "NA", "D", "A", "P", "A", "D", "D"]
    observed, _, kappa = meta_study.cohen_kappa(first, second)
    assert observed == pytest.approx(0.7)
    assert kappa == pytest.approx(cohen_kappa_score(first, second))
    _, _, degenerate = meta_study.cohen_kappa(["D", "D"], ["D", "D"])
    assert assets_module.num(degenerate, 2) == "--"


def test_applicable_counts_leave_out_not_applicable_codes(tmp_path: Path) -> None:
    reads_meta_features = _claim("S02", "beta", "verified")
    reads_meta_features["validation_provenance"] = "NA"
    reads_meta_features["validation_provenance_draft"] = "NA"
    reads_meta_features["validation_provenance_check"] = "NA"
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), reads_meta_features])
    _declare(tmp_path)
    numbers = _build(tmp_path)
    assert numbers["meta.validationprovenance.applicable"] == "1"
    assert numbers["meta.validationprovenance.declared"] == "1"
    assert numbers["meta.artifact.applicable"] == "2"


def test_recheck_disagreement_resolved_against_the_source_sets_the_final_code(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    cell = ("S01", "fixed_references")
    _declare(tmp_path, recheck={cell: "P"}, resolved={cell: "P"})
    numbers = _build(tmp_path)
    assert numbers["meta.recheck.disagreed"] == "1"
    assert numbers["meta.recheck.changed"] == "1"
    assert numbers["meta.recheck.fixedreferences.changed"] == "1"
    assert numbers["meta.recheck.fixedreferences.cells"] == "2"
    assert numbers["meta.fixedreferences.declared"] == "1"
    assert numbers["meta.fixedreferences.declared.ai"] == "2"
    # The earlier AI resolution changed no first code; the recheck resolution is counted apart.
    assert numbers["meta.codes.changed"] == "0"


def test_recheck_disagreement_can_be_resolved_back_to_the_ai_code(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    cell = ("S01", "fixed_references")
    _declare(tmp_path, recheck={cell: "P"}, resolved={cell: "D"})
    numbers = _build(tmp_path)
    assert numbers["meta.recheck.disagreed"] == "1"
    assert numbers["meta.recheck.changed"] == "0"
    assert numbers["meta.fixedreferences.declared"] == "2"


def test_submission_build_refuses_without_the_declared_recheck(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified")])
    with pytest.raises(ValueError, match="recheck_subset.json"):
        _build(tmp_path)
    # A draft build still runs and marks the recheck keys pending.
    assert "pending" in _build(tmp_path, submission=False)["meta.recheck.agreed"]


def test_submission_build_names_uncoded_and_unresolved_recheck_cells(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    _declare(tmp_path, recheck={("S02", "validation_provenance"): "", ("S01", "fixed_references"): "A"})
    with pytest.raises(ValueError, match="S02:validation_provenance.*S01:fixed_references"):
        _build(tmp_path)
    numbers = _build(tmp_path, submission=False)
    assert numbers["meta.recheck.coded"] == str(int(numbers["meta.recheck.cells"]) - 1)
    assert "pending" in numbers["meta.recheck.kappa"]


@pytest.mark.parametrize("tamper", ["undeclared", "final", "draw", "ai", "unmerged", "evidence"])
def test_inconsistent_recheck_coding_is_refused_in_every_build(tmp_path: Path, tamper: str) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    declaration = _declare(tmp_path)
    directory = tmp_path / "docs" / "meta_study"
    claims = meta_study.read_claims(directory / "claims.csv")
    partial = next(record for record in declaration["population"] if record not in declaration["full_claims"])
    row = claims.index[claims["record_id"].eq(partial)][0]
    if tamper == "undeclared":
        claims.at[row, "endpoint_recheck"] = "P"
    elif tamper == "final":
        claims.at[row, "fixed_references"] = "A"
    elif tamper == "ai":
        claims.at[row, "endpoint_ai"] = "A"
        claims.at[row, "endpoint"] = "A"
    elif tamper == "evidence":
        claims.at[row, "endpoint_evidence"] = "edited by hand"
    elif tamper == "unmerged":
        sheet = pd.read_csv(directory / meta_study.RECHECK_CODING_FILE, keep_default_na=False, dtype=str)
        sheet.loc[sheet["record_id"].eq(partial) & sheet["item"].eq("fixed_references"), "code"] = "A"
        sheet.to_csv(directory / meta_study.RECHECK_CODING_FILE, index=False)
    else:
        declaration["full_claims"] = [partial]
        (directory / meta_study.RECHECK_SUBSET_FILE).write_text(json.dumps(declaration), encoding="utf-8")
    claims.to_csv(directory / "claims.csv", index=False)
    with pytest.raises(ValueError, match="Recheck coding is inconsistent"):
        _build(tmp_path, submission=False)


def test_top_tenth_takes_at_least_five_candidates() -> None:
    table = pd.DataFrame({"val_rank": list(range(44, 0, -1))})
    assert sorted(assets_module.top_tenth(table)["val_rank"]) == [1, 2, 3, 4, 5]
    assert len(assets_module.top_tenth(pd.DataFrame({"val_rank": range(1530)}))) == 153


def test_pvalue_shows_which_side_of_the_threshold_a_value_falls() -> None:
    assert assets_module.pvalue(0.050175) == "0.0502"
    assert assets_module.pvalue(0.04951) == "0.0495"
    assert assets_module.pvalue(0.05) == "0.050"
    assert assets_module.pvalue(0.0512) == "0.051"
    assert assets_module.pvalue(0.0123) == "0.012"
    assert assets_module.pvalue(0.0004) == "$<$0.001"


def test_mean_test_variants_separate_skew_from_scale() -> None:
    from tabular_sota_benchmark.inference import ArtifactResampler

    # A few large gains: the tests that follow the skew give smaller p-values than the
    # symmetric ones, and the percentile test matches the pipeline's bootstrap p.
    skewed = pd.Series([-0.30, -0.25, -0.20] + [-0.01] * 20 + [0.01] * 18, index=[f"d{i:02d}" for i in range(41)])
    resampler = ArtifactResampler(list(skewed.index), seed=7, n_resamples=4000)
    values = resampler.align(skewed)
    p = assets_module.mean_test_variants(values, resampler.indices)
    assert max(p["perc"], p["bteq"]) < min(p["t"], p["btsym"])
    assert p["perc"] == assets_module.bootstrap_mean_p(values[resampler.indices].mean(axis=1))
    # Without skew the two bootstrap-t tests agree closely.
    symmetric = pd.Series([-0.02, 0.01] * 20, index=[f"d{i:02d}" for i in range(40)]) - 0.004
    resampler = ArtifactResampler(list(symmetric.index), seed=7, n_resamples=4000)
    q = assets_module.mean_test_variants(resampler.align(symmetric), resampler.indices)
    assert abs(q["bteq"] - q["btsym"]) < 0.05
    # No variation: nothing to test.
    flat = assets_module.mean_test_variants(np.zeros(10), np.zeros((5, 10), dtype=int))
    assert flat == {"perc": 1.0, "bteq": 1.0, "btsym": 1.0, "t": 1.0}


def test_contrast_verdicts_name_each_target() -> None:
    row = pd.Series(
        {
            "mean_difference": -0.05,
            "ci_low": -0.10,
            "ci_high": -0.01,
            "bootstrap_holm_p": 0.03,
            "ttest_holm_p": 0.07,
            "wilcoxon_holm_p": 0.12,
            "hodges_lehmann_difference": -0.03,
            "tost_p_0p010": 0.20,
        }
    )
    verdicts = assets_module.contrast_verdicts(row)
    assert verdicts == {
        "interval": "excludes zero",
        "mean": "better",
        "t": "not separated",
        "rank": "not separated",
        "equiv0p010": "not equivalent",
    }
    worse = row.copy()
    worse[["mean_difference", "ci_low", "ci_high", "hodges_lehmann_difference"]] = [0.05, 0.01, 0.10, 0.03]
    worse["wilcoxon_holm_p"] = 0.01
    assert assets_module.contrast_verdicts(worse)["rank"] == "worse"


def test_stated_verdicts_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = assets_module.Assets(Path("."), allow_missing=False)
    assets.verdicts["ta.primary.nested.vb"] = {"mean": "better", "rank": "not separated"}
    monkeypatch.setattr(assets_module, "STATED_VERDICTS", {"ta.primary.nested.vb": {"mean": "better"}})
    assert assets_module.check_stated_verdicts(assets) == []
    monkeypatch.setattr(
        assets_module, "STATED_VERDICTS", {"ta.primary.nested.vb": {"rank": "better"}, "ta.x.y.z": {"mean": "better"}}
    )
    problems = assets_module.check_stated_verdicts(assets)
    assert len(problems) == 2 and "the text says 'better'" in problems[0]


def test_count_check_counts_changes_and_is_required_for_submission(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    _declare(tmp_path, count_check=False)
    with pytest.raises(ValueError, match="count check"):
        _build(tmp_path)
    assert _build(tmp_path, submission=False)["meta.countcheck.changed"] == "\\textbf{[pending]}"
    shutil.rmtree(tmp_path / "docs")
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified"), _claim("S02", "beta", "verified")])
    _declare(tmp_path, count_check={("S02", "fixed_references"): "A"})
    numbers = _build(tmp_path)
    assert numbers["meta.countcheck.cells"] == "4"
    assert numbers["meta.countcheck.changed"] == "1"
    assert numbers["meta.countcheck.fixedreferences.changed"] == "1"
    assert numbers["meta.countcheck.validationprovenance.changed"] == "0"
    assert numbers["meta.countcheck.firstagreed"] == "4"
    # The changed code is the final one: the declared-count keys follow it.
    assert numbers["meta.fixedreferences.absent"] == "1"


def test_count_check_change_needs_a_reason(tmp_path: Path) -> None:
    _write_meta_study(tmp_path, [_claim("S01", "alpha", "verified")])
    _declare(tmp_path, count_check={("S01", "validation_provenance"): "A"})
    path = tmp_path / "docs" / "meta_study" / "claims.csv"
    claims = meta_study.read_claims(path)
    claims["validation_provenance_countcheck_note"] = ""
    claims.to_csv(path, index=False)
    with pytest.raises(ValueError, match="without a reason|evidence does not match"):
        _build(tmp_path, submission=False)
