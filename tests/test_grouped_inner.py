"""Checks of scripts/grouped_inner_splits.py: group-aware inner estimates by patching.

With every dataset its own group the patched benchmark must reproduce the unpatched one
exactly; with a group, no test error of one member may change a training-side selection on
another member, while unpatched it can.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gis = _load("grouped_inner_splits", ROOT / "scripts" / "grouped_inner_splits.py")
helpers = _load("benchmark_test_helpers", ROOT / "tests" / "test_score_level_policy_benchmark.py")
artifacts = gis.artifacts
METHODS = ["A (default)", "A (tuned)", "B (default)", "B (tuned)"]


@pytest.fixture(autouse=True)
def _restore():
    yield
    gis.restore_patches()


@pytest.fixture
def forced_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unpatched benchmark on the fast leave-one-dataset-out path, which the driver forces."""
    monkeypatch.setattr(
        artifacts, "_calibrated_selector", functools.partial(artifacts._calibrated_selector, fast_path_min_rows=0)
    )


def _units(artifact: dict) -> pd.DataFrame:
    datasets = sorted(artifact["candidates"]["dataset"].unique())
    train = [name for name in datasets if name not in helpers.HELDOUT]
    _, _, _, units = gis.bench._evaluate_artifact_split(
        artifact, split_id=0, train_datasets=train, heldout_datasets=helpers.HELDOUT, split_mode="task_stratified"
    )
    return units


def _selections(artifact: dict, phase: str) -> pd.Series:
    units = _units(artifact)
    frame = units[units["phase"].eq(phase) & units["role"].eq("pool")]
    return frame.set_index(["dataset", "fold", "method"])["selected_member"].sort_index()


def test_singleton_groups_reproduce_the_unpatched_benchmark(forced_fast: None) -> None:
    artifact = helpers._synthetic_artifact(inner_source_fit="per_target")
    unpatched = _units(artifact)
    gis.apply_patches({})
    pd.testing.assert_frame_equal(_units(artifact), unpatched)


def test_the_fast_path_chooses_as_the_slow_path() -> None:
    # The driver forces the fast path; on the synthetic artifact the default is the slow one.
    artifact = helpers._synthetic_artifact(inner_source_fit="per_target")
    slow = _selections(artifact, "train")
    gis.apply_patches({})
    assert _selections(artifact, "train").equals(slow)


@pytest.mark.parametrize("group", [("bin1", "bin2"), ("bin3", "reg3")])
def test_a_members_test_errors_change_no_training_selection_on_another(group: tuple[str, str], forced_fast: None) -> None:
    artifact = helpers._synthetic_artifact(inner_source_fit="per_target")
    first, second = group
    changed_unpatched = 0
    for source, target in [(first, second), (second, first)]:
        for method in METHODS:
            favoured = helpers._favour(artifact, source, method)
            changed_unpatched += helpers._changed_on(
                target, _selections(artifact, "train"), _selections(favoured, "train")
            )
    gis.apply_patches({first: "g", second: "g"})
    for source, target in [(first, second), (second, first)]:
        base = _selections(artifact, "train")
        for method in METHODS:
            favoured = helpers._favour(artifact, source, method)
            assert helpers._changed_on(target, base, _selections(favoured, "train")) == 0, (source, target, method)
    if group == ("bin1", "bin2"):
        # The check has power: without the patch a sibling's test errors do reach its group-mate.
        assert changed_unpatched > 0


def test_the_leakage_boundary_still_holds_with_groups() -> None:
    artifact = helpers._synthetic_artifact(inner_source_fit="per_target")
    gis.apply_patches({"bin1": "g", "bin2": "g"})
    base_train = _selections(artifact, "train")
    base_held = _selections(artifact, "heldout")
    for target, method in [("bin1", "A (default)"), ("bin0", "B (tuned)"), ("reg2", "A (tuned)")]:
        assert helpers._changed_on(target, base_train, _selections(helpers._favour(artifact, target, method), "train")) == 0
    assert _selections(helpers._favour(artifact, "bin4", "B (default)"), "heldout").equals(base_held)


def test_guards_on_a_member_have_no_group_mate_among_their_neighbours() -> None:
    artifact = helpers._synthetic_artifact(inner_source_fit="per_target")
    gis.apply_patches({"bin1": "g", "bin2": "g"})
    units = _units(artifact)
    guards = units[units["phase"].eq("train") & units["method"].astype(str).str.contains("guard")]
    assert not guards.empty
    for dataset, mate in [("bin1", "bin2"), ("bin2", "bin1")]:
        details = guards.loc[guards["dataset"].eq(dataset), "ensemble_weights"].astype(str)
        neighbours = details.str.extract(r"nearest_datasets=([^;]*)")[0].dropna()
        assert not neighbours.empty
        assert not neighbours.str.split(",").map(lambda names: mate in names).any()


def test_rebuilt_functions_change_only_the_named_lines() -> None:
    import inspect

    for module, name in [(artifacts, "_task_adaptive_selector_from_selectors"), (gis.bench, "_per_target_adaptive_rows")]:
        source = inspect.getsource(getattr(module, name))
        for old, new in gis.GROUP_EDITS[name]:
            assert source.count(old) == 1
            assert new not in source
    gis.apply_patches({"bin1": "g", "bin2": "g"})
    assert gis.group_label("bin1") == gis.group_label("bin2") == "group:g"
    assert gis.group_label("bin3") == "single:bin3"
    assert gis.group_members("bin2") == frozenset({"bin1", "bin2"})
    gis.restore_patches()
    assert not hasattr(artifacts, "_group_members")
    assert artifacts._factorize_columns.__module__ == artifacts.__name__
