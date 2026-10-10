from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "meta_study_sheet.py"
SPEC = importlib.util.spec_from_file_location("meta_study_sheet", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sheet_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sheet_module
SPEC.loader.exec_module(sheet_module)

from tabular_sota_benchmark import meta_study  # noqa: E402

LEAK = "LEAKTOKEN"
PROTOCOL = """# Protocol

## Coding scheme

1. **Artifact**: the reused table.

## Verification

Text.

## Deviations

6. **Conventions adopted to resolve disagreements.** Rules.
7. **Eligibility at verification.** S14 stays included.
"""


def _write(directory: Path, records: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = {
            "record_id": record,
            "paper": f"Author {record}",
            "cite_key": record.lower(),
            "claim_paraphrase": f"Selector {record} beats a baseline.",
            "verification": "verified",
            "verification_note": f"{LEAK} note",
        }
        for item in meta_study.ITEMS:
            code = "yes" if item in meta_study.EXTRA_COLUMNS else "P"
            row[item] = code
            row[f"{item}_draft"] = code
            row[f"{item}_check"] = code
            if item in meta_study.FIELD_COLUMNS:
                row[f"{item}_evidence"] = f"{LEAK} evidence"
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "claims.csv", index=False)
    pd.DataFrame(
        {
            "record_id": records,
            "title": [f"Title {record}" for record in records],
            "year": ["2024"] * len(records),
            "url": [f"https://example.org/{record}" for record in records],
            "decision": ["include"] * len(records),
            "reason": [""] * len(records),
        }
    ).to_csv(directory / "screening.csv", index=False)
    (directory / "protocol.md").write_text(PROTOCOL, encoding="utf-8")


RECORDS = [f"S{number:02d}" for number in range(1, 14)]


def test_draw_is_reproducible_and_declares_seventy_five_cells() -> None:
    shuffled = RECORDS[:]
    random.Random(3).shuffle(shuffled)
    assert meta_study.draw_order(shuffled, 20261001) == meta_study.draw_order(RECORDS, 20261001)
    full = meta_study.draw_full_claims(RECORDS, 20261001, 4)
    declaration = {"population": RECORDS, "all_claims_items": meta_study.RECHECK_ALL_CLAIM_ITEMS, "full_claims": full}
    assert len(meta_study.declared_cells(declaration)) == 3 * 13 + 9 * 4


def test_draw_freezes_the_ai_codes_and_is_declared_once(tmp_path: Path) -> None:
    _write(tmp_path, RECORDS)
    declaration = sheet_module.recheck_draw(tmp_path, 20261001, 4)
    claims = meta_study.read_claims(tmp_path / "claims.csv")
    for item in meta_study.ITEMS:
        assert claims[f"{item}_ai"].equals(claims[item].rename(f"{item}_ai"))
        assert claims[f"{item}_recheck"].eq("").all()
    for field in meta_study.FIELD_COLUMNS:
        assert claims[f"{field}_evidence_ai"].equals(claims[f"{field}_evidence"].rename(f"{field}_evidence_ai"))
    assert declaration["n_cells"] == 75
    assert json.loads((tmp_path / meta_study.RECHECK_SUBSET_FILE).read_text())["full_claims"] == declaration["full_claims"]
    with pytest.raises(SystemExit):
        sheet_module.recheck_draw(tmp_path, 20261001, 4)


def test_blank_sheet_shows_no_ai_code_evidence_or_note(tmp_path: Path) -> None:
    _write(tmp_path, RECORDS)
    sheet_module.recheck_draw(tmp_path, 20261001, 4)
    sheet_module.recheck_sheet(tmp_path, force=False)
    coding = pd.read_csv(tmp_path / meta_study.RECHECK_CODING_FILE, keep_default_na=False, dtype=str)
    assert list(coding.columns) == sheet_module.SHEET_COLUMNS
    assert coding["code"].eq("").all() and len(coding) == 75
    for path in [tmp_path / meta_study.RECHECK_CODING_FILE, tmp_path / meta_study.RECHECK_SHEET_FILE]:
        text = path.read_text(encoding="utf-8")
        assert LEAK not in text
        assert "_draft" not in text and "_check" not in text and "_ai" not in text
    assert "S14 stays included" not in (tmp_path / meta_study.RECHECK_SHEET_FILE).read_text(encoding="utf-8")


def test_sheet_with_codes_is_not_overwritten(tmp_path: Path) -> None:
    _write(tmp_path, RECORDS)
    sheet_module.recheck_draw(tmp_path, 20261001, 4)
    sheet_module.recheck_sheet(tmp_path, force=False)
    path = tmp_path / meta_study.RECHECK_CODING_FILE
    coding = pd.read_csv(path, keep_default_na=False, dtype=str)
    coding.loc[0, "code"] = "D"
    coding.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        sheet_module.recheck_sheet(tmp_path, force=False)
    sheet_module.recheck_sheet(tmp_path, force=True)


def _fill(directory: Path, code_for: dict[str, str]) -> pd.DataFrame:
    path = directory / meta_study.RECHECK_CODING_FILE
    coding = pd.read_csv(path, keep_default_na=False, dtype=str)
    coding["code"] = [code_for.get(item, "p" if item in meta_study.FIELD_COLUMNS else "Yes") for item in coding["item"]]
    coding.to_csv(path, index=False)
    return coding


def test_merge_writes_recheck_codes_and_resolutions_and_is_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, RECORDS)
    sheet_module.recheck_draw(tmp_path, 20261001, 4)
    sheet_module.recheck_sheet(tmp_path, force=False)
    coding = _fill(tmp_path, {"fixed_references": " d "})
    before = (tmp_path / "claims.csv").read_text(encoding="utf-8")
    sheet_module.recheck_merge(tmp_path, dry_run=True)
    assert (tmp_path / "claims.csv").read_text(encoding="utf-8") == before
    sheet_module.recheck_merge(tmp_path, dry_run=False)
    claims = meta_study.read_claims(tmp_path / "claims.csv")
    assert claims["fixed_references_recheck"].eq("D").all()
    # A disagreement keeps the AI-resolved code until it is resolved against the source.
    assert claims["fixed_references"].eq("P").all()
    assert claims["fixed_references_ai"].eq("P").all()
    assert claims["fixed_references_evidence"].eq(f"{LEAK} evidence").all()
    declaration = meta_study.load_declaration(tmp_path)
    sheet = pd.read_csv(tmp_path / meta_study.RECHECK_CODING_FILE, keep_default_na=False, dtype=str)
    with pytest.raises(ValueError, match="unresolved"):
        meta_study.check_recheck_state(claims, declaration, sheet=sheet, require_complete=True)
    # Resolve half of the disagreements to the third code and half back to the AI code.
    rows = coding.index[coding["item"].eq("fixed_references")]
    for number, row in enumerate(rows):
        coding.loc[row, "resolved"] = "D" if number % 2 == 0 else "P"
        coding.loc[row, "resolution_note"] = "reread the evaluation section"
    coding.to_csv(tmp_path / meta_study.RECHECK_CODING_FILE, index=False)
    sheet_module.recheck_merge(tmp_path, dry_run=False)
    claims = meta_study.read_claims(tmp_path / "claims.csv")
    expected = ["D" if number % 2 == 0 else "P" for number in range(len(rows))]
    assert claims["fixed_references"].tolist() == expected
    assert claims["fixed_references_draft"].eq("P").all()
    # A resolved disagreement takes the reason as its evidence, whichever code it kept.
    assert claims["fixed_references_evidence"].eq("reread the evaluation section").all()
    assert claims["fixed_references_evidence_ai"].eq(f"{LEAK} evidence").all()
    assert claims["endpoint_evidence"].eq(f"{LEAK} evidence").all()
    merged = (tmp_path / "claims.csv").read_text(encoding="utf-8")
    sheet_module.recheck_merge(tmp_path, dry_run=False)
    assert (tmp_path / "claims.csv").read_text(encoding="utf-8") == merged
    sheet = pd.read_csv(tmp_path / meta_study.RECHECK_CODING_FILE, keep_default_na=False, dtype=str)
    coded, uncoded, unresolved = meta_study.check_recheck_state(claims, declaration, sheet=sheet, require_complete=True)
    assert len(coded) == 75 and not uncoded and not unresolved


@pytest.mark.parametrize("problem", ["code", "extra", "missing"])
def test_merge_rejects_invalid_or_mismatched_cells(tmp_path: Path, problem: str) -> None:
    _write(tmp_path, RECORDS)
    sheet_module.recheck_draw(tmp_path, 20261001, 4)
    sheet_module.recheck_sheet(tmp_path, force=False)
    coding = _fill(tmp_path, {})
    if problem == "code":
        coding.loc[0, "code"] = "maybe"
    elif problem == "extra":
        partial = next(record for record in RECORDS if record not in meta_study.draw_full_claims(RECORDS, 20261001, 4))
        extra = coding.iloc[[0]].copy()
        extra["record_id"], extra["item"] = partial, "endpoint"
        coding = pd.concat([coding, extra], ignore_index=True)
    else:
        coding = coding.iloc[1:]
    coding.to_csv(tmp_path / meta_study.RECHECK_CODING_FILE, index=False)
    with pytest.raises(SystemExit, match="not valid"):
        sheet_module.recheck_merge(tmp_path, dry_run=False)


def test_verification_sheet_gains_recheck_columns_after_the_draw(tmp_path: Path) -> None:
    _write(tmp_path, RECORDS)
    sheet_module.write_sheet(tmp_path)
    assert "| AI resolved |" not in (tmp_path / "verification_sheet.md").read_text(encoding="utf-8")
    sheet_module.recheck_draw(tmp_path, 20261001, 4)
    sheet_module.write_sheet(tmp_path)
    text = (tmp_path / "verification_sheet.md").read_text(encoding="utf-8")
    assert "| AI resolved | Recheck |" in text
    assert "0 of 75 cells coded" in text


def test_claim_template_lists_the_nine_fields_in_protocol_order() -> None:
    text = (ROOT / "docs" / "claim_template.md").read_text(encoding="utf-8")
    headings = [line[3:].split(". ", 1)[1] for line in text.splitlines() if line.startswith("## ") and line[3].isdigit()]
    assert headings == [label for _, label in meta_study.FIELDS]


def _recheck_done(directory: Path) -> None:
    """Draw, code and merge the recheck with every third code equal to the AI code."""
    _write(directory, RECORDS)
    sheet_module.recheck_draw(directory, 20261001, 4)
    sheet_module.recheck_sheet(directory, force=False)
    _fill(directory, {})
    sheet_module.recheck_merge(directory, dry_run=False)


def test_count_check_sheet_lists_the_checked_cells_and_keeps_prepared_columns(tmp_path: Path) -> None:
    _recheck_done(tmp_path)
    sheet_module.count_check_sheet(tmp_path, reset=False)
    check = pd.read_csv(tmp_path / meta_study.COUNT_CHECK_FILE, keep_default_na=False, dtype=str)
    assert list(check.columns) == sheet_module.COUNT_CHECK_SHEET_COLUMNS
    assert len(check) == 2 * len(RECORDS)
    assert set(check["item"]) == set(meta_study.COUNT_CHECK_ITEMS)
    assert check["code"].eq("P").all() and check["decision"].eq("").all()
    # Prepared locators survive a rerun, which re-renders the readable sheet from them.
    check.loc[0, "locator"] = "Sec. 4.2, Table 3"
    check.to_csv(tmp_path / meta_study.COUNT_CHECK_FILE, index=False)
    sheet_module.count_check_sheet(tmp_path, reset=False)
    assert "Sec. 4.2, Table 3" in (tmp_path / meta_study.COUNT_CHECK_SHEET_FILE).read_text(encoding="utf-8")
    check.loc[0, "decision"] = "agree"
    check.to_csv(tmp_path / meta_study.COUNT_CHECK_FILE, index=False)
    with pytest.raises(SystemExit, match="decisions"):
        sheet_module.count_check_sheet(tmp_path, reset=True)


def test_count_check_merge_applies_changes_with_reasons_and_is_idempotent(tmp_path: Path) -> None:
    _recheck_done(tmp_path)
    sheet_module.count_check_sheet(tmp_path, reset=False)
    path = tmp_path / meta_study.COUNT_CHECK_FILE
    check = pd.read_csv(path, keep_default_na=False, dtype=str)
    check["decision"] = "agree"
    check["first_code"] = check["code"]
    changed = check.index[check["record_id"].eq("S02") & check["item"].eq("validation_provenance")][0]
    check.loc[changed, ["decision", "check_code", "check_note"]] = ["change", "D", "Sec. 5 compares the candidates' validation protocols"]
    # A confirmed code whose pointer was inaccurate gets the corrected pointer as its note.
    repointed = check.index[check["record_id"].eq("S03") & check["item"].eq("fixed_references")][0]
    check.loc[repointed, "check_note"] = "Sec. 4 and Table 2: single best only, no oracle"
    check.to_csv(path, index=False)
    sheet_module.count_check_merge(tmp_path, dry_run=False)
    claims = meta_study.read_claims(tmp_path / "claims.csv").set_index("record_id")
    assert claims.at["S02", "validation_provenance"] == "D"
    assert claims.at["S02", "validation_provenance_countcheck"] == "D"
    assert claims.at["S02", "validation_provenance_evidence"] == "Sec. 5 compares the candidates' validation protocols"
    assert claims.at["S02", "validation_provenance_ai"] == "P"
    assert claims.at["S03", "validation_provenance"] == "P" and claims.at["S03", "fixed_references_countcheck"] == "P"
    assert claims.at["S03", "fixed_references_evidence"] == "Sec. 4 and Table 2: single best only, no oracle"
    merged = (tmp_path / "claims.csv").read_text(encoding="utf-8")
    sheet_module.count_check_merge(tmp_path, dry_run=False)
    assert (tmp_path / "claims.csv").read_text(encoding="utf-8") == merged
    # Rerunning the recheck merge keeps the count check's change.
    sheet_module.recheck_merge(tmp_path, dry_run=False)
    assert (tmp_path / "claims.csv").read_text(encoding="utf-8") == merged
    checked, unchecked = meta_study.check_count_check_state(meta_study.read_claims(tmp_path / "claims.csv"), require_complete=True)
    assert len(checked) == 2 * len(RECORDS) and not unchecked


@pytest.mark.parametrize("problem", ["no reason", "same code", "bad decision", "missing cell", "no first code"])
def test_count_check_merge_rejects_invalid_decisions(tmp_path: Path, problem: str) -> None:
    _recheck_done(tmp_path)
    sheet_module.count_check_sheet(tmp_path, reset=False)
    path = tmp_path / meta_study.COUNT_CHECK_FILE
    check = pd.read_csv(path, keep_default_na=False, dtype=str)
    check["first_code"] = check["code"]
    if problem == "no first code":
        check.loc[0, ["first_code", "decision"]] = ["", "agree"]
    elif problem == "no reason":
        check.loc[0, ["decision", "check_code"]] = ["change", "A"]
    elif problem == "same code":
        check.loc[0, ["decision", "check_code", "check_note"]] = ["change", "P", "no change"]
    elif problem == "bad decision":
        check.loc[0, "decision"] = "maybe"
    else:
        check = check.iloc[1:]
    check.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="not valid"):
        sheet_module.count_check_merge(tmp_path, dry_run=False)
