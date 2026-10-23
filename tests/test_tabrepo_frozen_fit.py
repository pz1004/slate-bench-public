"""The TabRepo frozen-fit check keeps the declarations of its published protocol."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _constants() -> dict[str, object]:
    tree = ast.parse((ROOT / "scripts" / "tabrepo_frozen_fit.py").read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def test_declared_fold_and_seed_match_the_protocol():
    protocol = (ROOT / "docs" / "fresh_validation_protocol.md").read_text(encoding="utf-8")
    assert _constants()["DECLARED_FOLD"] == 0
    assert "Declared evaluation: fold 0" in protocol
    assert "--repeats 1 --split-seed 20261022" in protocol
    assert "--repeats 1 \\\n\t  --split-seed 20261022" in (ROOT / "Makefile").read_text(encoding="utf-8")


def test_contrasts_match_the_protocol():
    source = (ROOT / "scripts" / "tabrepo_frozen_fit.py").read_text(encoding="utf-8")
    for reference in [
        "train_selected_fixed_reference__ensemble_inclusive",
        "train_selected_fixed_reference__score_only",
        "meta_selector_family_prior_budgeted",
    ]:
        assert reference in source
    protocol = (ROOT / "docs" / "fresh_validation_protocol.md").read_text(encoding="utf-8")
    assert "minus the train-selected rule over the ensemble-inclusive pool" in protocol
    assert "nested selection minus the train-selected rule, validation-best and the family-prior rule" in protocol
