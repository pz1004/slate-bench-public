"""Shared definitions for the meta-study in docs/meta_study.

One place for the coded items and their allowed codes, Cohen's kappa, and the declared
subset that a third, blind AI coding rechecks (protocol.md, Deviations 8 and 9).

Each claim's codes move through six column stages in ``claims.csv``:

* ``{item}_draft``: the first AI coding;
* ``{item}_check``: the blind second AI coding;
* ``{item}_ai``: the AI-resolved code, frozen when the recheck subset was drawn;
* ``{item}_recheck``: the third, blind AI coding, for the declared cells only;
* ``{item}_countcheck`` and ``{item}_countcheck_note``: the check of the codes behind the
  abstract's counts (fixed references and validation provenance of every verified claim;
  protocol.md, Deviations 11 and 12), with the reason where the check changed a code;
* ``{item}``: the final code. It is the AI-resolved code unless the recheck disagreed and
  the disagreement was resolved against the source to another code, as recorded in
  ``recheck_coding.csv``, or unless the count check changed it.

Each field also has an evidence pointer. ``{field}_evidence_ai`` freezes the pointer of the
AI-resolved code with it. ``{field}_evidence`` is that pointer, except where the recheck
disagreed and the disagreement was resolved (there it is the reason for the resolution)
and where the count check gave a note (the reason for a change, or the corrected pointer of a
confirmed code).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

FIELDS: list[tuple[str, str]] = [
    ("artifact", "Artifact"),
    ("capability_set", "Capability set"),
    ("policy_pool", "Policy pool"),
    ("fixed_references", "Fixed references"),
    ("evidence_unit", "Evidence unit"),
    ("inference_rule", "Inference rule"),
    ("endpoint", "Endpoint"),
    ("cost_field", "Cost field"),
    ("validation_provenance", "Validation provenance"),
]
EXTRAS: list[tuple[str, str]] = [
    ("e1_selector_tuned_on_evaluation", "E1 selector tuned on the evaluation datasets"),
    ("e2_single_best_or_validation_best", "E2 single-best or validation-best reference"),
    ("e3_multiplicity_handled", "E3 multiplicity handled"),
]
FIELD_COLUMNS = [column for column, _ in FIELDS]
EXTRA_COLUMNS = [column for column, _ in EXTRAS]
ITEMS = FIELD_COLUMNS + EXTRA_COLUMNS
LABELS = dict(FIELDS + EXTRAS)
FIELD_CODES = ("D", "P", "A", "NA")
EXTRA_CODES = ("yes", "no", "unclear")
# Keys of the extra items in \res{...}; the fields use their column name without underscores.
EXTRA_KEYS = {
    "e1_selector_tuned_on_evaluation": "eone",
    "e2_single_best_or_validation_best": "etwo",
    "e3_multiplicity_handled": "ethree",
}

# The declared recheck subset: these items for every verified claim, and every item for
# claims drawn at random (protocol.md, Deviations 8 and 9).
RECHECK_ALL_CLAIM_ITEMS = ["fixed_references", "validation_provenance", "e1_selector_tuned_on_evaluation"]
RECHECK_SUBSET_FILE = "recheck_subset.json"
RECHECK_CODING_FILE = "recheck_coding.csv"
RECHECK_SHEET_FILE = "recheck_coding_sheet.md"
DRAW_RULE = "sort record ids by the SHA-256 hex digest of '<seed>:<record_id>' and take the first n_full"

# The check of the codes behind the abstract's counts (protocol.md, Deviations 11 and 12):
# these items for every verified claim.
COUNT_CHECK_ITEMS = ["fixed_references", "validation_provenance"]
COUNT_CHECK_FILE = "count_check.csv"
COUNT_CHECK_SHEET_FILE = "count_check_sheet.md"
COUNT_CHECK_DECISIONS = ("agree", "change")


def item_key(item: str) -> str:
    """The key of an item in \\res{...}: ``fixedreferences``, ``eone``."""
    return EXTRA_KEYS.get(item, item.replace("_", ""))


def allowed_codes(item: str) -> tuple[str, ...]:
    if item in FIELD_COLUMNS:
        return FIELD_CODES
    if item in EXTRA_COLUMNS:
        return EXTRA_CODES
    raise KeyError(f"Unknown meta-study item: {item}")


def normalize_code(item: str, value: str) -> str:
    """Strip a code and give it the case the protocol uses (D/P/A/NA, yes/no/unclear)."""
    text = str(value).strip()
    if not text:
        return ""
    return text.upper() if item in FIELD_COLUMNS else text.lower()


def read_claims(path: Path) -> pd.DataFrame:
    # "NA" (not applicable) is a code, not a missing value.
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def cohen_kappa(first: Iterable[str], second: Iterable[str]) -> tuple[float, float, float]:
    """Observed agreement, chance agreement and Cohen's kappa of two codings of the same cells.

    Chance agreement uses the pooled marginals of each coding. Kappa is NaN when chance
    agreement is 1, that is, when both codings use one and the same code throughout.
    """
    a = [str(value) for value in first]
    b = [str(value) for value in second]
    if not a or len(a) != len(b):
        raise ValueError("Kappa needs two codings of the same, non-empty set of cells")
    n = len(a)
    observed = sum(left == right for left, right in zip(a, b)) / n
    chance = sum((a.count(code) / n) * (b.count(code) / n) for code in sorted(set(a) | set(b)))
    kappa = float("nan") if chance >= 1.0 else (observed - chance) / (1.0 - chance)
    return observed, chance, kappa


def draw_order(record_ids: Iterable[str], seed: int) -> list[str]:
    """Record ids in the order of the declared draw; independent of the input order."""
    return sorted(set(record_ids), key=lambda record: hashlib.sha256(f"{seed}:{record}".encode()).hexdigest())


def draw_full_claims(population: Sequence[str], seed: int, n_full: int) -> list[str]:
    if n_full > len(set(population)):
        raise ValueError(f"Cannot draw {n_full} claims from {len(set(population))}")
    return sorted(draw_order(population, seed)[:n_full])


def declared_cells(declaration: dict) -> list[tuple[str, str]]:
    """Every declared (record, item) cell of the recheck, in record and item order."""
    full = set(declaration["full_claims"])
    shared = set(declaration["all_claims_items"])
    return [
        (record, item)
        for record in sorted(declaration["population"])
        for item in ITEMS
        if record in full or item in shared
    ]


def resolved_codes(frame: pd.DataFrame, item: str) -> pd.Series:
    """The AI-resolved codes: the frozen ``_ai`` column once it exists, otherwise the main one."""
    column = f"{item}_ai"
    return frame[column] if column in frame.columns else frame[item]


def ai_codes_digest(claims: pd.DataFrame, population: Sequence[str]) -> str:
    """SHA-256 of the AI-resolved codes of the population, so later edits are detected."""
    rows = claims.set_index("record_id")
    lines = [
        f"{record},{item},{resolved_codes(rows.loc[[record]], item).iloc[0]}"
        for record in sorted(population)
        for item in ITEMS
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def recheck_final(row: pd.Series, item: str, sheet_entry: pd.Series | None) -> tuple[str, str]:
    """A cell's code and evidence pointer after the recheck, before the count check.

    The AI-resolved code and pointer, unless the recheck disagreed and the disagreement was
    resolved against the source (``resolved`` with a ``resolution_note`` in
    ``recheck_coding.csv``); then the resolved code and, for a field, the note.
    """
    ai = str(row.get(f"{item}_ai", row[item]))
    evidence = str(row.get(f"{item}_evidence_ai", row.get(f"{item}_evidence", ""))) if item in FIELD_COLUMNS else ""
    recheck = str(row.get(f"{item}_recheck", ""))
    if sheet_entry is None or not recheck or recheck == ai:
        return ai, evidence
    resolved = normalize_code(item, sheet_entry.get("resolved", ""))
    note = str(sheet_entry.get("resolution_note", "")).strip()
    if resolved and note:
        return resolved, (note if item in FIELD_COLUMNS else "")
    return ai, evidence


def count_check_cells(population: Iterable[str]) -> list[tuple[str, str]]:
    """Every (record, item) cell of the count check, in record and item order."""
    return [(record, item) for record in sorted(set(population)) for item in COUNT_CHECK_ITEMS]


def _count_check_code(rows: pd.DataFrame, record: str, item: str) -> tuple[str, str]:
    """The count check's code and note of a cell, or empty strings before the check."""
    column = f"{item}_countcheck"
    if column not in rows.columns:
        return "", ""
    return str(rows.at[record, column]), str(rows.at[record, f"{item}_countcheck_note"])


def check_count_check_state(claims: pd.DataFrame, *, require_complete: bool) -> tuple[list[tuple[str, str]], list[str]]:
    """Check the count-check columns of the verified claims; return checked cells and unchecked ones.

    Raises ``ValueError`` when a count-check code is invalid, sits outside the checked items or
    claims, or changes a code without a reason, and, with ``require_complete``, when a cell
    of the check is still unchecked.
    """
    verified = claims[claims["verification"].eq("verified")]
    population = sorted(verified["record_id"])
    cells = count_check_cells(population)
    stray = [
        column
        for column in claims.columns
        if column.endswith("_countcheck") and column[: -len("_countcheck")] not in COUNT_CHECK_ITEMS
    ]
    problems = [f"claims.csv has the count-check column {column} outside the checked items" for column in stray]
    columns = [f"{item}_countcheck{suffix}" for item in COUNT_CHECK_ITEMS for suffix in ("", "_note")]
    present = [column for column in columns if column in claims.columns]
    if present and len(present) != len(columns):
        problems.append(f"claims.csv has only some of the count-check columns: {present}")
    if not present:
        if problems:
            raise ValueError("Count check is inconsistent: " + "; ".join(problems))
        if require_complete:
            raise ValueError("Submission build needs the count check (docs/meta_study/count_check.csv); merge it first")
        return [], [f"{record}:{item}" for record, item in cells]
    unverified = claims[~claims["verification"].eq("verified")]
    for item in COUNT_CHECK_ITEMS:
        filled = unverified.loc[unverified[f"{item}_countcheck"].ne(""), "record_id"].tolist()
        if filled:
            problems.append(f"{item}: count-check codes on claims that are not verified: {filled}")
    rows = verified.set_index("record_id")
    checked: list[tuple[str, str]] = []
    unchecked: list[str] = []
    for record, item in cells:
        code, note = _count_check_code(rows, record, item)
        if not code:
            unchecked.append(f"{record}:{item}")
            if note:
                problems.append(f"{record}:{item} has a count-check note but no count-check code")
            continue
        if code not in allowed_codes(item):
            problems.append(f"{record}:{item} has the invalid count-check code {code!r}")
        if rows.at[record, item] != code:
            problems.append(f"{record}:{item} final code {rows.at[record, item]!r} should be the count check's {code!r}")
        checked.append((record, item))
    if problems:
        raise ValueError("Count check is inconsistent: " + "; ".join(problems))
    if require_complete and unchecked:
        raise ValueError(f"Submission build needs every cell of the count check; unchecked: {unchecked}")
    return checked, unchecked


def load_declaration(directory: Path) -> dict | None:
    path = directory / RECHECK_SUBSET_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_recheck_state(
    claims: pd.DataFrame,
    declaration: dict | None,
    *,
    sheet: pd.DataFrame | None = None,
    require_complete: bool,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Check the recheck columns of the verified claims against the declaration and the sheet.

    Returns the declared cells that hold a recheck code, the uncoded declared cells and the
    disagreements not yet resolved against the source (both as ``record:item``). Raises
    ``ValueError`` when the columns, the declaration or the sheet are inconsistent, and,
    with ``require_complete``, when a declared cell is uncoded or a disagreement unresolved.
    """
    recheck_columns = [f"{item}_recheck" for item in ITEMS]
    has_columns = all(column in claims.columns for column in recheck_columns)
    if declaration is None:
        if has_columns and claims[recheck_columns].ne("").any().any():
            raise ValueError("claims.csv holds recheck codes but docs/meta_study has no recheck_subset.json")
        if require_complete:
            raise ValueError(
                "Submission build needs the declared recheck coding (docs/meta_study/recheck_subset.json); "
                "draw and code the subset first"
            )
        return [], [], []
    problems: list[str] = []
    verified = claims[claims["verification"].eq("verified")]
    population = sorted(declaration["population"])
    if sorted(verified["record_id"]) != population:
        problems.append(f"the declared population {population} is not the set of verified claims")
    expected = draw_full_claims(population, int(declaration["seed"]), int(declaration["n_full"]))
    if expected != sorted(declaration["full_claims"]):
        problems.append(f"the declared draw {declaration['full_claims']} does not reproduce from its seed ({expected})")
    evidence_columns = [f"{field}_evidence{stage}" for field in FIELD_COLUMNS for stage in ("", "_ai")]
    required = recheck_columns + [f"{item}_ai" for item in ITEMS] + evidence_columns
    missing_columns = [column for column in required if column not in claims]
    if missing_columns:
        problems.append(f"claims.csv lacks the columns {missing_columns}")
    if sheet is None:
        problems.append(f"docs/meta_study has no {RECHECK_CODING_FILE}")
    if problems:
        raise ValueError("Recheck coding is inconsistent: " + "; ".join(problems))
    if ai_codes_digest(claims, population) != declaration["ai_codes_sha256"]:
        problems.append("the AI-resolved codes (*_ai) changed after the draw")
    declared = set(declared_cells(declaration))
    resolved: dict[tuple[str, str], str] = {}
    notes: dict[tuple[str, str], str] = {}
    sheet_codes: dict[tuple[str, str], str] = {}
    for _, entry in sheet.iterrows():
        cell = (str(entry["record_id"]).strip(), str(entry["item"]).strip())
        if cell[1] not in ITEMS:
            continue
        sheet_codes[cell] = normalize_code(cell[1], entry.get("code", ""))
        code = normalize_code(cell[1], entry.get("resolved", ""))
        if code:
            if code not in allowed_codes(cell[1]):
                problems.append(f"{cell[0]}:{cell[1]} has the invalid resolved code {entry['resolved']!r}")
            resolved[cell] = code
        if str(entry.get("resolution_note", "")).strip():
            notes[cell] = str(entry["resolution_note"])
    coded: list[tuple[str, str]] = []
    uncoded: list[str] = []
    unresolved: list[str] = []
    rows = verified.set_index("record_id")
    for record in population:
        for item in ITEMS:
            recheck = rows.at[record, f"{item}_recheck"]
            final = rows.at[record, item]
            ai = rows.at[record, f"{item}_ai"]
            cell = (record, item)
            expected_final = ai
            evidence_from_resolution = False
            if recheck:
                if cell not in declared:
                    problems.append(f"{record}:{item} has a recheck code but is not a declared cell")
                elif recheck not in allowed_codes(item):
                    problems.append(f"{record}:{item} has the invalid recheck code {recheck!r}")
                else:
                    coded.append(cell)
                if sheet_codes.get(cell, "") != recheck:
                    problems.append(f"{record}:{item} in {RECHECK_CODING_FILE} is not merged into claims.csv")
                if recheck != ai:
                    if cell in resolved and cell in notes:
                        expected_final = resolved[cell]
                        evidence_from_resolution = True
                    else:
                        unresolved.append(f"{record}:{item}")
            else:
                if cell in declared:
                    uncoded.append(f"{record}:{item}")
                if sheet_codes.get(cell, ""):
                    problems.append(f"{record}:{item} in {RECHECK_CODING_FILE} is not merged into claims.csv")
            if cell in resolved and (not recheck or recheck == ai):
                problems.append(f"{record}:{item} has a resolved code although the recheck did not disagree")
            # The count check comes last: a code it changed is final, and a check note, given
            # for every change and wherever a confirmed code's pointer was inaccurate, is the
            # field's evidence (protocol.md, Deviations 11 and 12).
            counted, counted_note = _count_check_code(rows, record, item)
            by_count_check = bool(counted) and counted != expected_final
            if by_count_check:
                expected_final = counted
                if not counted_note:
                    problems.append(f"{record}:{item} was changed by the count check without a reason")
            if final != expected_final:
                problems.append(f"{record}:{item} final code {final!r} should be {expected_final!r}")
            if item in FIELD_COLUMNS:
                if counted and counted_note:
                    expected_evidence, source = counted_note, "the count check's note"
                elif evidence_from_resolution:
                    expected_evidence, source = notes[cell], "the resolution note"
                else:
                    expected_evidence, source = rows.at[record, f"{item}_evidence_ai"], f"{item}_evidence_ai"
                if rows.at[record, f"{item}_evidence"] != expected_evidence:
                    problems.append(f"{record}:{item} evidence does not match {source}")
    if problems:
        raise ValueError("Recheck coding is inconsistent: " + "; ".join(problems))
    if require_complete and (uncoded or unresolved):
        raise ValueError(
            "Submission build needs every declared recheck cell coded and every disagreement resolved; "
            f"uncoded: {uncoded}; unresolved: {unresolved}"
        )
    return coded, uncoded, unresolved
