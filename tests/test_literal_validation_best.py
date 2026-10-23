"""The literal validation-best rule against the pool's runtime-term row (scripts/literal_validation_best.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("literal_validation_best_test", ROOT / "scripts" / "literal_validation_best.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lvb = _module()


def _candidates() -> pd.DataFrame:
    rows = []
    # Unit a: the runtime term reorders two validation errors closer than it.
    rows += [("a", "m1", 0.1, 100.0), ("a", "m2", 0.1000001, 1.0)]
    # Unit b: an exact tie in validation error goes to the lower fit time under both rules.
    rows += [("b", "m1", 0.2, 5.0), ("b", "m2", 0.2, 3.0)]
    # Unit c: an exact tie in both goes to the earlier candidate.
    rows += [("c", "m1", 0.3, 2.0), ("c", "m2", 0.3, 2.0)]
    frame = pd.DataFrame(rows, columns=["dataset", "method", "val_primary_metric", "fit_seconds"])
    for column, value in [("fold", 0), ("seed", 0), ("task", "binary"), ("primary_metric_name", "log_loss")]:
        frame[column] = value
    return frame


def test_runtime_term_reorders_a_near_tie_that_the_literal_rule_keeps():
    frame = _candidates()
    runtime = lvb.choose(frame, rule="runtime", runtime_weight=1e-6).set_index("dataset")["method"]
    literal = lvb.choose(frame, rule="literal", runtime_weight=1e-6).set_index("dataset")["method"]
    assert runtime.to_dict() == {"a": "m2", "b": "m2", "c": "m1"}
    assert literal.to_dict() == {"a": "m1", "b": "m2", "c": "m1"}


def test_with_values_replaces_unit_values_and_keeps_dtypes():
    unit = list(lvb.UNIT)
    base = pd.DataFrame(
        {
            **{column: [value] * 2 for column, value in zip(unit, ["d", 0, 0, "binary", "log_loss"])},
            "split_id": [0, 1],
            **{column: [0.0, 0.0] for column in lvb.FLOATS},
            "selected_member": ["x", "x"],
            "n_candidates": [3, 3],
        }
    )
    values = base.iloc[[0]].drop(columns=["split_id"]).copy()
    values["selected_member"] = "y"
    values["normalized_rank_regret"] = 0.5
    out = lvb._with_values(base, values)
    assert out["selected_member"].tolist() == ["y", "y"]
    assert out["normalized_rank_regret"].tolist() == [0.5, 0.5]
    assert out["split_id"].tolist() == [0, 1]
    assert out["n_candidates"].dtype == base["n_candidates"].dtype
