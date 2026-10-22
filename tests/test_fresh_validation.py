"""Tests of the fresh-validation experiment (src/tabular_sota_benchmark/fresh_validation.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_sota_benchmark import fresh_validation as fv

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = {"hgb_max_iter": 50, "hgb_patience": 5, "forest_trees": 20, "lr_max_iter": 300, "max_epochs": 5, "patience": 2}
FAMILIES = {
    "hist_gradient_boosting": {
        "default": {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0},
        "space": {
            "learning_rate": ["loguniform", 0.01, 0.3],
            "max_leaf_nodes": ["choice", [7, 15, 31]],
            "min_samples_leaf": ["choice", [5, 10, 20]],
            "l2_regularization": ["choice", [0.0, 0.1, 1.0]],
            "max_features": ["uniform", 0.4, 1.0],
        },
    },
    "logistic_regression": {
        "default": {"C": 1.0, "class_weight": None},
        "space": {"C": ["loguniform", 0.001, 1000.0], "class_weight": ["choice", [None, "balanced"]]},
    },
    "knn": {
        "default": {"n_neighbors": 5, "weights": "uniform", "p": 2},
        "space": {"n_neighbors": ["choice", [1, 3, 7, 15]], "weights": ["choice", ["uniform", "distance"]], "p": ["choice", [1, 2]]},
    },
    "mlp": {
        "default": {"hidden_layer_sizes": [16], "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 64},
        "space": {
            "hidden_layer_sizes": ["choice", [[8], [16]]],
            "alpha": ["loguniform", 1e-6, 0.1],
            "learning_rate_init": ["loguniform", 1e-4, 1e-2],
            "batch_size": ["choice", [32, 64]],
        },
    },
}
FRACTIONS = {"train": 0.48, "construction": 0.16, "selection": 0.16, "test": 0.20}


def _synthetic(n: int = 900, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    colour = rng.choice(["red", "green", "blue"], size=n)
    logits = np.stack([x1 + 0.5 * (colour == "red"), x2, -x1 - x2], axis=1)
    y = np.array([rng.choice(3, p=np.exp(row) / np.exp(row).sum()) for row in logits])
    x1[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({"x1": x1, "x2": x2, "colour": pd.Categorical(colour), "noise": rng.normal(size=n)}), y


def _unit(features: pd.DataFrame, y: np.ndarray, labels: dict[str, np.ndarray] | None = None, n_random: int = 3):
    parts = fv.four_way_split(y, FRACTIONS, seed=11)
    data = fv.encode(features, parts)
    ys = {name: y[index] for name, index in parts.items()}
    if labels:
        ys.update(labels)
    frames = []
    for family_index, (family, spec) in enumerate(FAMILIES.items()):
        configurations = fv.family_configurations(spec, n_random, seed=100 + family_index)
        seeds = [fv.derived_seed(7, family_index, index) for index in range(len(configurations))]
        searched, candidates = fv.family_candidates(
            family, configurations, data, ys, 3, seeds=seeds, schedule=SCHEDULE, ensemble_steps=10
        )
        frames.append((searched, candidates))
    searched = pd.concat([frame[0] for frame in frames], ignore_index=True)
    candidates = pd.concat([frame[1] for frame in frames], ignore_index=True)
    candidates["order"] = candidates["candidate"].map(fv.candidate_order(list(FAMILIES)))
    candidates["dataset"], candidates["repetition"] = "synthetic", 0
    return parts, searched, candidates


def test_four_way_split_is_a_stratified_partition_with_equal_validation_sets():
    _, y = _synthetic()
    parts = fv.four_way_split(y, FRACTIONS, seed=3)
    together = np.concatenate([parts[name] for name in fv.SPLITS])
    assert sorted(together.tolist()) == list(range(len(y)))
    assert abs(len(parts["construction"]) - len(parts["selection"])) <= 1
    for name in fv.SPLITS:
        shares = np.bincount(y[parts[name]], minlength=3) / len(parts[name])
        assert np.allclose(shares, np.bincount(y, minlength=3) / len(y), atol=0.05)


def test_capped_rows_is_stratified_and_deterministic():
    _, y = _synthetic(n=900)
    first, second = fv.capped_rows(y, 300, seed=5), fv.capped_rows(y, 300, seed=5)
    assert len(first) == 300 and np.array_equal(first, second)
    assert np.all(np.diff(first) > 0)
    assert np.array_equal(fv.capped_rows(y, 5000, seed=5), np.arange(len(y)))


def test_test_and_selection_labels_shape_no_construction_and_no_reused_choice():
    features, y = _synthetic()
    parts, searched, candidates = _unit(features, y)
    rng = np.random.default_rng(1)
    shuffled_test = rng.permutation(y[parts["test"]])
    shuffled_selection = rng.permutation(y[parts["selection"]])
    _, searched_t, candidates_t = _unit(features, y, {"test": shuffled_test})
    _, searched_s, candidates_s = _unit(features, y, {"selection": shuffled_selection})
    for other in [candidates_t, candidates_s]:
        # The tuned configuration and the ensemble weights depend on construction labels only.
        assert other["detail"].tolist() == candidates["detail"].tolist()
        assert np.allclose(other["construction_loss"], candidates["construction_loss"])
    assert np.allclose(searched_t["construction_loss"], searched["construction_loss"])
    assert np.allclose(candidates_t["selection_loss"], candidates["selection_loss"])
    unit = ["dataset", "repetition"]
    choices = fv.rule_choices(candidates, unit).set_index("rule")["candidate"]
    for other, unchanged in [(candidates_t, ["validation_best_reused", "validation_best_fresh", "defaults_rule"]),
                             (candidates_s, ["validation_best_reused", "defaults_rule", "validation_best_defaults"])]:
        other_choices = fv.rule_choices(other, unit).set_index("rule")["candidate"]
        for rule in unchanged:
            assert other_choices[rule] == choices[rule]


def test_units_are_deterministic_apart_from_timing():
    features, y = _synthetic()
    _, first, candidates_a = _unit(features, y)
    _, second, candidates_b = _unit(features, y)
    columns = ["construction_loss", "selection_loss", "test_loss"]
    assert np.array_equal(first[columns].to_numpy(), second[columns].to_numpy())
    assert candidates_a["detail"].tolist() == candidates_b["detail"].tolist()
    assert len(candidates_a) == 3 * len(FAMILIES)


def test_greedy_ensemble_prefers_the_better_member_and_breaks_ties_to_the_earlier_one():
    y = np.array([0, 1, 1, 0])
    good = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])
    bad = np.full((4, 2), 0.5)
    weights = fv.greedy_ensemble([bad, good], y, steps=10)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] > weights[0]
    tied = fv.greedy_ensemble([good, good.copy()], y, steps=4)
    assert tied.tolist() == [1.0, 0.0]


def test_lexicographic_choice_breaks_ties_by_fit_time_then_order():
    frame = pd.DataFrame(
        {
            "dataset": ["d"] * 3,
            "repetition": [0] * 3,
            "candidate": ["a", "b", "c"],
            "construction_loss": [0.5, 0.5, 0.5],
            "fit_seconds": [2.0, 1.0, 1.0],
            "order": [0, 2, 1],
        }
    )
    chosen = fv.lexicographic_choice(frame, "construction_loss", ["dataset", "repetition"])
    assert chosen["candidate"].tolist() == ["c"]


def test_log_loss_matches_scikit_learn():
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=50)
    y = rng.integers(0, 3, size=50)
    assert fv.log_loss(y, proba) == pytest.approx(log_loss(y, proba, labels=[0, 1, 2]))


def _tasks_module():
    spec = importlib.util.spec_from_file_location("fresh_validation_tasks", ROOT / "scripts" / "fresh_validation_tasks.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_eligibility_rule_on_synthetic_metadata():
    tasks = _tasks_module()
    rule = {"min_classes": 2, "max_classes": 10, "min_rows": 1000, "max_rows": 100000, "max_features": 500, "min_class_rows": 100}
    rows = [
        {"did": 900001, "name": "fine_one", "version": 1},
        {"did": 900002, "name": "too_small", "version": 1, "NumberOfInstances": 500},
        {"did": 900003, "name": "regression_like", "version": 1, "NumberOfClasses": 0},
        {"did": 900004, "name": "BNG(something)", "version": 1},
        {"did": 900005, "name": "Amazon_employee_access", "version": 1},
        {"did": 900006, "name": "fine_one", "version": 2},
        {"did": 900007, "name": "rare_class", "version": 1, "MinorityClassSize": 50},
        {"did": 900008, "name": "wide", "version": 1, "NumberOfFeatures": 600},
    ]
    base = {"status": "active", "format": "ARFF", "NumberOfInstances": 5000, "NumberOfFeatures": 20, "NumberOfClasses": 2, "MinorityClassSize": 1000}
    frame = pd.DataFrame([{**base, **row} for row in rows])
    table = tasks.eligible(frame, rule).set_index("did")["exclusion"]
    assert table[900001] == "earlier version of an eligible name" and table[900006] == ""
    assert table[900002] == "rows" and table[900003] == "classes" and table[900004] == "synthetic generator"
    assert table[900005] == "TabArena or TabRepo name"
    assert table[900007] == "smallest class" and table[900008] == "features"
