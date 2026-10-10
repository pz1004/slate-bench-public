from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# One BLAS thread per process: splits run in forked workers, and numpy is not the
# bottleneck. Must happen before numpy is imported.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.artifacts import (
    ARTIFACT_OUTPUT_COLUMNS,
    ARTIFACT_UNIT_COLUMNS,
    build_split_score_artifact_selector_rows,
    build_static_selector_rows,
    load_dataset_descriptors,
    load_standard_meta_features,
    load_tabarena_results,
    load_tabrepo_repository,
    load_tabrepo_results,
    selector_provenance,
)
from tabular_sota_benchmark.artifacts import (
    _candidate_rows_from_tabarena,
    _candidate_rows_from_tabrepo,
)
from tabular_sota_benchmark.markdown import dataframe_to_markdown
from tabular_sota_benchmark.ranking import (
    get_rank_rule,
    oracle_rank,
    rank_against_candidates,
    set_rank_rule,
)


NESTED_SELECTED_METHOD = "nested_policy_selected"
VALIDATION_ONLY_METHOD = "meta_selector_ablate_validation_only"
TASK_ADAPTIVE_METHOD = "meta_selector_task_adaptive_budgeted"
FAMILY_PRIOR_METHOD = "meta_selector_family_prior_budgeted"
CALIBRATED_BUDGET_METHOD = "meta_selector_calibrated_budget_10000s"

DEFAULT_FAMILY_PRIOR_GRID = [
    {
        "validation_weight": 1.0,
        "task_method_prior_weight": 0.0,
        "task_family_prior_weight": 0.0,
        "global_method_prior_weight": 0.0,
        "runtime_weight": 0.0,
    },
    {
        "validation_weight": 0.75,
        "task_method_prior_weight": 0.0,
        "task_family_prior_weight": 0.15,
        "global_method_prior_weight": 0.10,
        "runtime_weight": 0.0,
    },
    {
        "validation_weight": 0.60,
        "task_method_prior_weight": 0.0,
        "task_family_prior_weight": 0.25,
        "global_method_prior_weight": 0.15,
        "runtime_weight": 0.0,
    },
    {
        "validation_weight": 0.45,
        "task_method_prior_weight": 0.0,
        "task_family_prior_weight": 0.35,
        "global_method_prior_weight": 0.15,
        "runtime_weight": 0.05,
    },
    {
        "validation_weight": 0.30,
        "task_method_prior_weight": 0.0,
        "task_family_prior_weight": 0.50,
        "global_method_prior_weight": 0.15,
        "runtime_weight": 0.05,
    },
]


def _safe_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _runtime_code(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def _margin_code(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def _family_budget_method_name(budget_seconds: float) -> str:
    if int(budget_seconds) == 10000:
        return FAMILY_PRIOR_METHOD
    return f"meta_selector_family_prior_budgeted_{int(budget_seconds)}s"


def _guard_method_name(margin: float, top_k: int, runtime_penalty: float) -> str:
    code = _margin_code(margin)
    if int(top_k) == 8 and math.isclose(float(runtime_penalty), 0.03):
        return f"meta_selector_meta_feature_guard_{code}"
    return (
        f"meta_selector_meta_feature_guard_{code}"
        f"_k{int(top_k)}_rt{_runtime_code(float(runtime_penalty))}"
    )


REFERENCE_ROLE = "reference"
POOL_ROLE = "pool"


def _declared_grid(selector_config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    grid = selector_config.get(key)
    if not grid:
        raise ValueError(
            f"Selector config must declare `{key}`; the benchmark no longer falls back to an "
            "implicit weight grid."
        )
    return [dict(item) for item in grid]


def _declared_selector_entries(
    selector_config: dict[str, Any],
    key: str,
    *,
    weight_selection: str,
) -> list[dict[str, Any]]:
    """Explicit selector specs from the config (reference rows or extra pool rows)."""
    entries = []
    for raw_entry in selector_config.get(key, []) or []:
        spec = dict(raw_entry)
        if "method_name" not in spec or "kind" not in spec:
            raise ValueError(f"Every `{key}` entry needs `kind` and `method_name`.")
        if spec["kind"] in {"calibrated", "budgeted_calibrated"}:
            if "weights" in spec:
                spec["weight_grid"] = [spec.pop("weights")]
            if not spec.get("weight_grid"):
                raise ValueError(f"`{spec['method_name']}` must declare `weights` or `weight_grid`.")
            spec.setdefault("leave_one_dataset_out", True)
            spec.setdefault("budget_feature", "fit_seconds")
            spec.setdefault("weight_selection", weight_selection)
            budget = spec.get("max_budget_seconds")
            spec["max_budget_seconds"] = float(budget) if budget is not None else None
        entries.append(spec)
    return entries


def _selector_grid(
    selector_config: dict[str, Any],
    artifact_kind: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Derived selector specs, the nested policy pool, and the reference-only rows.

    Reference rows (``reference_selectors`` and the legacy ``decomposition_selectors``) are
    scored in both phases and written to every output, but never enter the nested choice.
    ``extra_pool_selectors`` are appended to the pool; they exist for the extended-pool
    sensitivity run.
    """
    grid = dict(selector_config.get("policy_grid", {}))
    weight_selection = str(selector_config.get("weight_selection", "lodo"))
    family_weight_grid = selector_config.get("family_prior_weight_grid", DEFAULT_FAMILY_PRIOR_GRID)
    calibrated_weight_grid = _declared_grid(selector_config, "calibrated_weight_grid")
    family_budgets = [float(value) for value in grid.get("family_prior_budgets", [100, 500, 1000, 5000, 10000])]
    calibrated_budget = float(grid.get("calibrated_budget", 10000))
    guard_margins = [float(value) for value in grid.get("guard_margins", [0.02, 0.05, 0.10, 0.20])]
    guard_top_k = [int(value) for value in grid.get("guard_top_k", [4, 8, 12])]
    guard_runtime_penalties = [
        float(value) for value in grid.get("guard_runtime_penalties", [0.0, 0.03, 0.1])
    ]
    ensemble_sizes = [int(value) for value in grid.get("tabrepo_ensemble_sizes", [3, 5, 10])]

    derived: list[dict[str, Any]] = [
        {
            "kind": "validation_rank",
            "method_name": VALIDATION_ONLY_METHOD,
        }
    ]
    policy_methods = [
        str(selector_config.get("selector_method_name", "meta_selector_single")),
        VALIDATION_ONLY_METHOD,
    ]

    family_methods = []
    for budget_seconds in family_budgets:
        method_name = _family_budget_method_name(budget_seconds)
        family_methods.append(method_name)
        policy_methods.append(method_name)
        derived.append(
            {
                "kind": "budgeted_calibrated",
                "method_name": method_name,
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": budget_seconds,
                "weight_grid": family_weight_grid,
                "weight_selection": weight_selection,
            }
        )

    calibrated_method = f"meta_selector_calibrated_budget_{int(calibrated_budget)}s"
    derived.append(
        {
            "kind": "budgeted_calibrated",
            "method_name": calibrated_method,
            "leave_one_dataset_out": True,
            "budget_feature": "fit_seconds",
            "max_budget_seconds": calibrated_budget,
            "weight_grid": calibrated_weight_grid,
            "weight_selection": weight_selection,
        }
    )
    policy_methods.append(calibrated_method)

    task_sources = list(dict.fromkeys([*family_methods, calibrated_method]))
    derived.append(
        {
            "kind": "task_adaptive_selector",
            "method_name": TASK_ADAPTIVE_METHOD,
            "candidate_selector_methods": task_sources,
            "choice_scope": "task",
            "leave_one_dataset_out": True,
        }
    )
    policy_methods.append(TASK_ADAPTIVE_METHOD)

    guard_sources = list(dict.fromkeys([TASK_ADAPTIVE_METHOD, *family_methods, calibrated_method]))
    for margin in guard_margins:
        for top_k in guard_top_k:
            for runtime_penalty in guard_runtime_penalties:
                method_name = _guard_method_name(margin, top_k, runtime_penalty)
                derived.append(
                    {
                        "kind": "meta_feature_adaptive_selector",
                        "method_name": method_name,
                        "candidate_selector_methods": guard_sources,
                        "choice_scope": "task",
                        "leave_one_dataset_out": True,
                        "top_k": top_k,
                        "distance_power": 1.0,
                        "runtime_tradeoff_weight": runtime_penalty,
                        "default_selector_method": TASK_ADAPTIVE_METHOD,
                        "min_improvement_over_default": margin,
                        # The declared pool guards use score-artifact meta-features only;
                        # standard dataset descriptors are reserved for the meta-feature reference rows.
                        "exclude_feature_column_prefixes": ["descriptor_"],
                    }
                )
                policy_methods.append(method_name)

    if artifact_kind == "tabrepo_repository":
        for ensemble_size in ensemble_sizes:
            method_name = f"meta_selector_tabrepo_ensemble_k{ensemble_size}"
            derived.append(
                {
                    "kind": "tabrepo_prediction_ensemble",
                    "method_name": method_name,
                    "top_k": int(grid.get("tabrepo_ensemble_top_k", 5)),
                    "ensemble_size": ensemble_size,
                    "budget_feature": "fit_seconds",
                    "max_budget_seconds": float(grid.get("tabrepo_ensemble_budget", 10000)),
                    "runtime_weight": float(grid.get("tabrepo_ensemble_runtime_penalty", 0.03)),
                    "backend": str(grid.get("tabrepo_ensemble_backend", "native")),
                }
            )
            policy_methods.append(method_name)

    extra_pool = _declared_selector_entries(
        selector_config,
        "extra_pool_selectors",
        weight_selection=weight_selection,
    )
    derived.extend(extra_pool)
    policy_methods.extend(str(entry["method_name"]) for entry in extra_pool)

    reference_entries = _declared_selector_entries(
        selector_config,
        "reference_selectors",
        weight_selection=weight_selection,
    )
    for entry in selector_config.get("decomposition_selectors", []) or []:
        spec = dict(entry)
        spec.setdefault("kind", "budgeted_calibrated")
        reference_entries.extend(
            _declared_selector_entries(
                {"decomposition": [spec]},
                "decomposition",
                weight_selection=weight_selection,
            )
        )
    derived.extend(reference_entries)
    reference_methods = [str(entry["method_name"]) for entry in reference_entries]

    names = [str(entry["method_name"]) for entry in derived]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate selector method names: {duplicates}")
    policy_methods = list(dict.fromkeys(policy_methods))
    overlap = sorted(set(policy_methods) & set(reference_methods))
    if overlap:
        raise ValueError(f"Selectors cannot be both pool and reference rows: {overlap}")
    return derived, policy_methods, reference_methods


def _selector_configs_require_tabrepo_predictions(selectors: list[dict[str, Any]]) -> bool:
    return any(str(selector.get("kind")) == "tabrepo_prediction_ensemble" for selector in selectors)


def _limit_datasets(frame: pd.DataFrame, *, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame
    datasets = np.array(sorted(str(value) for value in frame["dataset"].unique()), dtype=object)
    if int(limit) >= len(datasets):
        return frame
    if int(limit) < 3:
        raise ValueError("dataset_limit must be at least 3.")
    rng = np.random.default_rng(seed)
    selected = set(str(value) for value in rng.choice(datasets, size=int(limit), replace=False))
    return frame[frame["dataset"].astype(str).isin(selected)].copy()


def _load_artifact(
    artifact_config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    artifact_source = dict(artifact_config["artifact_source"])
    selector_config = dict(artifact_config.get("selector", {}))
    artifact_kind = str(artifact_source.get("kind", "tabarena_results"))
    derived_selectors, policy_methods, reference_methods = _selector_grid(
        selector_config,
        artifact_kind,
    )
    repository = None
    if artifact_kind == "tabarena_results":
        raw_frame = load_tabarena_results(artifact_source)
        candidates = _candidate_rows_from_tabarena(raw_frame)
    elif artifact_kind == "tabrepo_repository":
        if _selector_configs_require_tabrepo_predictions(derived_selectors):
            artifact_source["load_predictions"] = True
        repository = load_tabrepo_repository(artifact_source)
        raw_frame = load_tabrepo_results(artifact_source, repository=repository)
        candidates = _candidate_rows_from_tabrepo(raw_frame)
    else:
        raise ValueError(f"Unknown artifact kind: {artifact_kind}")

    candidates = _limit_datasets(
        candidates,
        limit=(
            int(artifact_config["dataset_limit"])
            if artifact_config.get("dataset_limit") is not None
            else None
        ),
        seed=seed,
    )
    if candidates.empty:
        raise ValueError(f"Artifact `{artifact_config['name']}` has no candidate rows.")
    candidates = candidates[ARTIFACT_OUTPUT_COLUMNS].reset_index(drop=True)
    descriptors = load_dataset_descriptors(artifact_config.get("dataset_descriptors"))
    standard_meta_features = load_standard_meta_features(
        artifact_config.get("standard_meta_features"),
        datasets=sorted(str(value) for value in candidates["dataset"].unique()),
        repository=repository,
    )
    if standard_meta_features is not None:
        descriptors = (
            standard_meta_features
            if descriptors is None
            else descriptors.merge(standard_meta_features, on="dataset", how="outer")
        )
    static_rows = build_static_selector_rows(
        candidates,
        runtime_weight=float(selector_config.get("runtime_weight", 0.0)),
        selector_method_name=str(selector_config.get("selector_method_name", "meta_selector_single")),
        derived_selectors=derived_selectors,
        ensemble_repository=repository,
    )
    return {
        "name": str(artifact_config["name"]),
        "kind": artifact_kind,
        "source": artifact_source,
        "candidates": candidates,
        "repository": repository,
        "selector": selector_config,
        "derived_selectors": derived_selectors,
        "policy_methods": policy_methods,
        "reference_methods": reference_methods,
        "dataset_descriptors": descriptors,
        "static_selector_rows": static_rows,
    }


def _task_by_dataset(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        candidate_rows[["dataset", "task"]]
        .drop_duplicates()
        .sort_values(["dataset", "task"])
        .groupby("dataset", as_index=False)
        .agg(task=("task", "first"))
    )


def _global_split(
    datasets: list[str],
    *,
    heldout_fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    n_heldout = max(2, int(round(len(datasets) * float(heldout_fraction))))
    if n_heldout >= len(datasets):
        raise ValueError("heldout_fraction leaves no training datasets.")
    return sorted(str(value) for value in rng.choice(np.array(datasets, dtype=object), size=n_heldout, replace=False))


def _split_datasets(
    candidate_rows: pd.DataFrame,
    *,
    n_splits: int,
    heldout_fraction: float,
    seed: int,
) -> list[tuple[int, list[str], list[str], str]]:
    dataset_task = _task_by_dataset(candidate_rows)
    datasets = sorted(dataset_task["dataset"].astype(str).tolist())
    if len(datasets) < 3:
        raise ValueError("Nested policy benchmark requires at least three datasets.")
    n_heldout_global = max(2, int(round(len(datasets) * float(heldout_fraction))))
    splits = []
    rng = np.random.default_rng(seed)
    task_groups = {
        str(task): sorted(group["dataset"].astype(str).tolist())
        for task, group in dataset_task.groupby("task", sort=True)
    }
    stratified_possible = (
        len(task_groups) > 1
        and len(task_groups) <= n_heldout_global
        and all(len(values) >= 2 for values in task_groups.values())
    )
    for split_id in range(int(n_splits)):
        mode = "global"
        if stratified_possible:
            heldout: list[str] = []
            for task, task_datasets in task_groups.items():
                n_task_heldout = max(1, int(round(len(task_datasets) * float(heldout_fraction))))
                n_task_heldout = min(n_task_heldout, len(task_datasets) - 1)
                heldout.extend(
                    str(value)
                    for value in rng.choice(
                        np.array(task_datasets, dtype=object),
                        size=n_task_heldout,
                        replace=False,
                    )
                )
            heldout = sorted(set(heldout))
            if len(heldout) >= len(datasets) or not heldout:
                heldout = _global_split(datasets, heldout_fraction=heldout_fraction, rng=rng)
            else:
                mode = "task_stratified"
        else:
            heldout = _global_split(datasets, heldout_fraction=heldout_fraction, rng=rng)
        heldout_set = set(heldout)
        train = [dataset for dataset in datasets if dataset not in heldout_set]
        if not train:
            raise ValueError("Outer split has no training datasets.")
        splits.append((split_id, train, heldout, mode))
    return splits


def _rank_against_candidates(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["total_seconds"] = (
        frame["fit_seconds"].astype(float) + frame["predict_seconds"].astype(float)
    )
    return rank_against_candidates(frame, unit_columns=ARTIFACT_UNIT_COLUMNS)


def _aggregate_policy_scores(
    ranked: pd.DataFrame,
    *,
    artifact: str,
    split_id: int,
    phase: str,
    methods: list[str],
    reference_methods: list[str] | None = None,
) -> pd.DataFrame:
    reference_methods = [method for method in (reference_methods or []) if method not in methods]
    all_methods = [*methods, *reference_methods]
    rows = ranked[
        ranked["method_type"].eq("selector")
        & ranked["method"].astype(str).isin(all_methods)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    summary = (
        rows.groupby("method", as_index=False)
        .agg(
            mean_rank=("candidate_rank", "mean"),
            tied_best_rate=("tied_best", "mean"),
            mean_total_seconds=("total_seconds", "mean"),
            mean_rank_regret=("rank_regret", "mean"),
            mean_normalized_rank_regret=("normalized_rank_regret", "mean"),
            n_units=("candidate_rank", "size"),
            n_datasets=("dataset", "nunique"),
        )
        .sort_values(["mean_normalized_rank_regret", "mean_total_seconds", "method"])
    )
    # Pool rows keep their stable policy order (the final nested tie-break); reference rows
    # get negative orders so no consumer can mistake them for selectable policies.
    order = {method: index for index, method in enumerate(methods)}
    order.update({method: -10 - index for index, method in enumerate(reference_methods)})
    summary["artifact"] = artifact
    summary["split_id"] = int(split_id)
    summary["phase"] = phase
    summary["method_order"] = summary["method"].map(order)
    summary["role"] = np.where(
        summary["method"].isin(set(reference_methods)), REFERENCE_ROLE, POOL_ROLE
    )
    return summary[
        [
            "artifact",
            "split_id",
            "phase",
            "method",
            "method_order",
            "role",
            "mean_rank",
            "tied_best_rate",
            "mean_total_seconds",
            "mean_rank_regret",
            "mean_normalized_rank_regret",
            "n_units",
            "n_datasets",
        ]
    ]


def _oracle_rows(ranked: pd.DataFrame) -> pd.DataFrame:
    """One row per unit for the non-deployable oracle, ranked by the shared rule.

    The oracle output is the unit's best test error; when k candidates tie at the best it
    receives the average-tie rank 1 + (k - 1) / 2, like any selector choosing one of them.
    """
    candidates = ranked[ranked["method_type"].eq("candidate")].copy()
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    for _, group in candidates.groupby(ARTIFACT_UNIT_COLUMNS, sort=False):
        best = group.sort_values(["primary_metric", "total_seconds", "method"]).iloc[0].copy()
        candidate_values = group["primary_metric"].to_numpy(dtype=float)
        best["selected_member"] = str(best["method"])
        best["method"] = "oracle"
        best["method_type"] = "selector"
        best["candidate_rank"] = oracle_rank(candidate_values)
        best["tied_best"] = 1.0
        rows.append(best)
    oracle = pd.DataFrame(rows)
    oracle["rank_regret"] = oracle["candidate_rank"] - 1.0
    denominator = oracle["n_candidates"].astype(float) - 1.0
    oracle["normalized_rank_regret"] = np.where(
        denominator > 0.0,
        oracle["rank_regret"] / denominator.where(denominator > 0.0, 1.0),
        0.0,
    )
    return oracle.reset_index(drop=True)


def _oracle_score(
    ranked: pd.DataFrame,
    *,
    artifact: str,
    split_id: int,
    phase: str,
) -> pd.DataFrame:
    oracle = _oracle_rows(ranked)
    if oracle.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "artifact": artifact,
                "split_id": int(split_id),
                "phase": phase,
                "method": "oracle",
                "method_order": -2,
                "role": "oracle",
                "mean_rank": float(oracle["candidate_rank"].mean()),
                "tied_best_rate": 1.0,
                "mean_total_seconds": float(oracle["total_seconds"].mean()),
                "mean_rank_regret": float(oracle["rank_regret"].mean()),
                "mean_normalized_rank_regret": float(oracle["normalized_rank_regret"].mean()),
                "n_units": int(oracle.shape[0]),
                "n_datasets": int(oracle["dataset"].nunique()),
            }
        ]
    )


def _choose_policy(training_scores: pd.DataFrame, policy_methods: list[str]) -> pd.Series:
    candidates = training_scores[training_scores["method"].isin(policy_methods)].copy()
    if candidates.empty:
        raise ValueError("No policy methods are available for nested selection.")
    candidates["method_order"] = candidates["method"].map(
        {method: index for index, method in enumerate(policy_methods)}
    )
    return candidates.sort_values(
        ["mean_normalized_rank_regret", "mean_total_seconds", "method_order"],
        ascending=True,
    ).iloc[0]


def _with_nested_selected_alias(
    heldout_scores: pd.DataFrame,
    *,
    selected_method: str,
) -> pd.DataFrame:
    selected = heldout_scores[heldout_scores["method"].astype(str).eq(selected_method)].copy()
    if selected.empty:
        raise ValueError(f"Held-out scores missing selected policy `{selected_method}`.")
    selected["original_method"] = selected["method"]
    selected["method"] = NESTED_SELECTED_METHOD
    selected["method_order"] = -1
    selected["role"] = "alias"
    heldout_scores = heldout_scores.copy()
    heldout_scores["original_method"] = ""
    return pd.concat([heldout_scores, selected], ignore_index=True)


UNIT_SCORE_COLUMNS = [
    "artifact",
    "split_id",
    "phase",
    *ARTIFACT_UNIT_COLUMNS,
    "method",
    "original_method",
    "role",
    "selected_member",
    "ensemble_members",
    "ensemble_weights",
    "source_selector",
    "base_selector",
    "primary_metric",
    "val_primary_metric",
    "candidate_rank",
    "rank_regret",
    "normalized_rank_regret",
    "tied_best",
    "total_seconds",
    "n_candidates",
    "candidate_min_error",
    "candidate_median_error",
    "candidate_max_error",
]


def _unit_score_rows(
    ranked: pd.DataFrame,
    *,
    artifact: str,
    split_id: int,
    phase: str,
    methods: list[str],
    reference_methods: list[str],
    selected_method: str | None = None,
) -> pd.DataFrame:
    """Per-unit rows for every scored selector, the oracle and the nested alias."""
    rows = ranked[
        ranked["method_type"].eq("selector")
        & ranked["method"].astype(str).isin(set(methods) | set(reference_methods))
    ].copy()
    rows["original_method"] = ""
    rows["role"] = np.where(rows["method"].isin(set(reference_methods)), REFERENCE_ROLE, POOL_ROLE)
    frames = [rows]
    oracle = _oracle_rows(ranked)
    if not oracle.empty:
        oracle["original_method"] = ""
        oracle["role"] = "oracle"
        frames.append(oracle)
    if selected_method is not None:
        alias = rows[rows["method"].astype(str).eq(selected_method)].copy()
        alias["original_method"] = alias["method"]
        alias["method"] = NESTED_SELECTED_METHOD
        alias["role"] = "alias"
        frames.append(alias)
    units = pd.concat(frames, ignore_index=True)
    provenance = [
        selector_provenance(method, weights)
        for method, weights in zip(
            units["original_method"].where(units["original_method"].astype(str) != "", units["method"]),
            units["ensemble_weights"],
        )
    ]
    units["source_selector"] = [value[0] for value in provenance]
    units["base_selector"] = [value[1] for value in provenance]
    units.loc[units["role"].eq("oracle"), ["source_selector", "base_selector"]] = ""
    units["artifact"] = artifact
    units["split_id"] = int(split_id)
    units["phase"] = phase
    for column in ["selected_member", "ensemble_members", "ensemble_weights"]:
        units[column] = units[column].fillna("").astype(str)
    return units[UNIT_SCORE_COLUMNS].reset_index(drop=True)


# Selector kinds that choose among other selectors' rows on the training datasets.
ADAPTIVE_KINDS = frozenset({"task_adaptive_selector", "meta_feature_adaptive_selector"})
# `per_target`: every training-side row of an adaptive policy on an inner target dataset
# is built from selector rows fitted without that dataset (the leakage boundary of
# Section 3.4). `shared`: the adaptive rows read source rows fitted once per split on all
# training datasets, as in the earlier runs, so the inner target enters the priors and
# weight choices of the source rows on the other datasets (kept for the attribution run).
INNER_SOURCE_FITS = ("per_target", "shared")


def _inner_source_fit(selector_config: dict[str, Any]) -> str:
    value = str(selector_config.get("inner_source_fit", "per_target"))
    if value not in INNER_SOURCE_FITS:
        raise ValueError(f"inner_source_fit must be one of {INNER_SOURCE_FITS}, not {value!r}.")
    return value


def _per_target_adaptive_rows(
    train_candidates: pd.DataFrame,
    builder_options: dict[str, Any],
) -> pd.DataFrame | None:
    """Training-side adaptive rows with each inner target excluded from every fitted quantity.

    For each training dataset d, the sources an adaptive policy reads (and the
    task-adaptive rows that guards use as their default) are rebuilt leave-one-dataset-out
    on the other training datasets, so that the source row on every other dataset d' is
    fitted on neither d nor d'. The adaptive rows on d are then built from those rows.
    Adaptive policies here choose within the task type, so the source rows are rebuilt
    only on datasets of d's task; every other dataset still enters the priors.
    """
    derived = list(builder_options.get("derived_selectors") or [])
    adaptive = [spec for spec in derived if str(spec["kind"]) in ADAPTIVE_KINDS]
    if not adaptive:
        return None
    adaptive_names = {str(spec["method_name"]) for spec in adaptive}
    if any(str(spec.get("choice_scope", "task")) != "task" for spec in adaptive):
        raise ValueError("The per-target rebuild assumes adaptive policies that choose within the task type.")
    sources = {str(name) for spec in adaptive for name in spec.get("candidate_selector_methods", [])}
    history_specs = [spec for spec in derived if str(spec["method_name"]) in sources]
    target_specs = [spec for spec in derived if str(spec["method_name"]) in sources | adaptive_names]
    keys = train_candidates[["dataset", "task"]].drop_duplicates()
    task_of = dict(zip(keys["dataset"].astype(str), keys["task"].astype(str)))
    frames = []
    for dataset in sorted(task_of):
        mask = train_candidates["dataset"].astype(str).eq(dataset)
        history = train_candidates[~mask]
        target = train_candidates[mask]
        same_task = {name for name, task in task_of.items() if task == task_of[dataset] and name != dataset}
        # With fewer than two other same-task datasets the adaptive choices fall back to
        # every task, so the source rows are then rebuilt on every dataset.
        lodo_targets = same_task if len(same_task) >= 2 else None
        history_rows, _ = build_split_score_artifact_selector_rows(
            history,
            history,
            **{**builder_options, "derived_selectors": history_specs},
            lodo_targets=lodo_targets,
        )
        target_rows, _ = build_split_score_artifact_selector_rows(
            history,
            target,
            **{**builder_options, "derived_selectors": target_specs},
            precomputed_history_selectors=history_rows,
            skip_history_kinds=frozenset({"meta_feature_adaptive_selector"}),
        )
        frames.append(target_rows[target_rows["method"].astype(str).isin(adaptive_names)])
    return pd.concat(frames, ignore_index=True)


def _replace_selector_rows(selectors: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
    """``selectors`` with the rows of ``replacement``'s methods replaced, in place by unit."""
    key = [*ARTIFACT_UNIT_COLUMNS, "method"]
    frame = selectors.reset_index(drop=True)
    replaced = frame["method"].astype(str).isin(set(replacement["method"].astype(str)))
    old_keys = pd.MultiIndex.from_frame(frame.loc[replaced, key].astype(str))
    new = replacement[ARTIFACT_OUTPUT_COLUMNS].copy()
    new.index = pd.MultiIndex.from_frame(new[key].astype(str))
    if new.index.has_duplicates or set(new.index) != set(old_keys):
        raise ValueError("The per-target adaptive rows do not cover the same units as the shared rows.")
    # Each new row takes the position of the row it replaces, so row order is unchanged.
    new = new.loc[old_keys].reset_index(drop=True)
    new.index = frame.index[replaced]
    result = pd.concat([frame.loc[~replaced, ARTIFACT_OUTPUT_COLUMNS], new]).sort_index(kind="stable")
    return result.reset_index(drop=True)


def _evaluate_artifact_split(
    artifact: dict[str, Any],
    *,
    split_id: int,
    train_datasets: list[str],
    heldout_datasets: list[str],
    split_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = artifact["candidates"]
    train_candidates = candidates[
        candidates["dataset"].astype(str).isin(set(train_datasets))
    ].copy()
    heldout_candidates = candidates[
        candidates["dataset"].astype(str).isin(set(heldout_datasets))
    ].copy()
    selector_config = dict(artifact.get("selector", {}))
    builder_options = {
        "runtime_weight": float(selector_config.get("runtime_weight", 0.0)),
        "selector_method_name": str(selector_config.get("selector_method_name", "meta_selector_single")),
        "derived_selectors": artifact["derived_selectors"],
        "dataset_descriptors": artifact.get("dataset_descriptors"),
        "ensemble_repository": artifact.get("repository"),
        "static_selector_rows": artifact.get("static_selector_rows"),
    }
    train_selectors, _ = build_split_score_artifact_selector_rows(
        train_candidates,
        train_candidates,
        **builder_options,
    )
    # The held-out call reuses the leave-one-dataset-out history rows built above instead
    # of rebuilding every history selector; the rows are identical by construction. For a
    # held-out target these shared rows are out of sample.
    heldout_selectors, _ = build_split_score_artifact_selector_rows(
        train_candidates,
        heldout_candidates,
        precomputed_history_selectors=train_selectors,
        **builder_options,
    )
    # For the training-side estimate an inner target must not enter the source rows its
    # adaptive policies read; rebuild those rows per inner target.
    if _inner_source_fit(selector_config) == "per_target":
        per_target = _per_target_adaptive_rows(train_candidates, builder_options)
        if per_target is not None:
            train_selectors = _replace_selector_rows(train_selectors, per_target)

    train_ranked = _rank_against_candidates(
        pd.concat([train_candidates, train_selectors], ignore_index=True)
    )
    heldout_ranked = _rank_against_candidates(
        pd.concat([heldout_candidates, heldout_selectors], ignore_index=True)
    )
    policy_methods = artifact["policy_methods"]
    reference_methods = list(artifact.get("reference_methods", []))
    train_scores = _aggregate_policy_scores(
        train_ranked,
        artifact=artifact["name"],
        split_id=split_id,
        phase="train",
        methods=policy_methods,
        reference_methods=reference_methods,
    )
    selected = _choose_policy(train_scores, policy_methods)
    selected_method = str(selected["method"])
    heldout_scores = _aggregate_policy_scores(
        heldout_ranked,
        artifact=artifact["name"],
        split_id=split_id,
        phase="heldout",
        methods=policy_methods,
        reference_methods=reference_methods,
    )
    heldout_scores = pd.concat(
        [
            heldout_scores,
            _oracle_score(
                heldout_ranked,
                artifact=artifact["name"],
                split_id=split_id,
                phase="heldout",
            ),
        ],
        ignore_index=True,
    )
    heldout_scores = _with_nested_selected_alias(
        heldout_scores,
        selected_method=selected_method,
    )
    unit_scores = pd.concat(
        [
            _unit_score_rows(
                train_ranked,
                artifact=artifact["name"],
                split_id=split_id,
                phase="train",
                methods=policy_methods,
                reference_methods=reference_methods,
            ),
            _unit_score_rows(
                heldout_ranked,
                artifact=artifact["name"],
                split_id=split_id,
                phase="heldout",
                methods=policy_methods,
                reference_methods=reference_methods,
                selected_method=selected_method,
            ),
        ],
        ignore_index=True,
    )
    selected_heldout = heldout_scores[
        heldout_scores["method"].eq(NESTED_SELECTED_METHOD)
    ].iloc[0]
    split_result = pd.DataFrame(
        [
            {
                "artifact": artifact["name"],
                "split_id": int(split_id),
                "split_mode": split_mode,
                "selected_policy": selected_method,
                "n_train_datasets": int(len(train_datasets)),
                "n_heldout_datasets": int(len(heldout_datasets)),
                "train_mean_rank": float(selected["mean_rank"]),
                "train_mean_total_seconds": float(selected["mean_total_seconds"]),
                "train_mean_normalized_rank_regret": float(
                    selected["mean_normalized_rank_regret"]
                ),
                "heldout_mean_rank": float(selected_heldout["mean_rank"]),
                "heldout_tied_best_rate": float(selected_heldout["tied_best_rate"]),
                "heldout_mean_total_seconds": float(
                    selected_heldout["mean_total_seconds"]
                ),
                "heldout_mean_rank_regret": float(
                    selected_heldout["mean_rank_regret"]
                ),
                "heldout_mean_normalized_rank_regret": float(
                    selected_heldout["mean_normalized_rank_regret"]
                ),
                "heldout_n_units": int(selected_heldout["n_units"]),
            }
        ]
    )
    return split_result, train_scores, heldout_scores, unit_scores


PART_KINDS = {
    "split_results": ".csv",
    "training_policy_scores": ".csv",
    "heldout_policy_scores": ".csv",
    "unit_scores": ".parquet",
}


def _part_path(output_dir: Path, kind: str, artifact: str, split_id: int) -> Path:
    suffix = PART_KINDS.get(kind, ".csv")
    return output_dir / "split_parts" / kind / f"{_safe_slug(artifact)}_split_{int(split_id):04d}{suffix}"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _write_split_parts(
    output_dir: Path,
    *,
    artifact: str,
    split_id: int,
    split_result: pd.DataFrame,
    train_scores: pd.DataFrame,
    heldout_scores: pd.DataFrame,
    unit_scores: pd.DataFrame,
) -> None:
    # The unit part is written first: a split counts as complete only when the three
    # summary parts exist as well, so an interrupted write is simply recomputed.
    _atomic_write_parquet(unit_scores, _part_path(output_dir, "unit_scores", artifact, split_id))
    _atomic_write_csv(split_result, _part_path(output_dir, "split_results", artifact, split_id))
    _atomic_write_csv(train_scores, _part_path(output_dir, "training_policy_scores", artifact, split_id))
    _atomic_write_csv(heldout_scores, _part_path(output_dir, "heldout_policy_scores", artifact, split_id))


def _completed_parts(output_dir: Path, artifact: str, expected_split_ids: list[int]) -> set[int]:
    completed = set()
    for split_id in expected_split_ids:
        if all(
            _part_path(output_dir, kind, artifact, split_id).exists()
            for kind in PART_KINDS
        ):
            completed.add(split_id)
    return completed


def _load_parts(output_dir: Path, kind: str) -> pd.DataFrame:
    part_dir = output_dir / "split_parts" / kind
    suffix = PART_KINDS.get(kind, ".csv")
    paths = sorted(part_dir.glob(f"*{suffix}")) if part_dir.exists() else []
    if not paths:
        return pd.DataFrame()
    reader = pd.read_parquet if suffix == ".parquet" else pd.read_csv
    return pd.concat([reader(path) for path in paths], ignore_index=True)


def _summary(heldout_scores: pd.DataFrame) -> pd.DataFrame:
    if heldout_scores.empty:
        return pd.DataFrame()
    return (
        heldout_scores.groupby(["artifact", "method"], as_index=False)
        .agg(
            mean_rank=("mean_rank", "mean"),
            tied_best_rate=("tied_best_rate", "mean"),
            mean_total_seconds=("mean_total_seconds", "mean"),
            mean_rank_regret=("mean_rank_regret", "mean"),
            mean_normalized_rank_regret=("mean_normalized_rank_regret", "mean"),
            n_splits=("split_id", "nunique"),
        )
        .sort_values(["artifact", "mean_normalized_rank_regret", "mean_total_seconds", "method"])
    )


def _selection_counts(split_results: pd.DataFrame) -> pd.DataFrame:
    if split_results.empty:
        return pd.DataFrame()
    return (
        split_results.groupby(["artifact", "selected_policy"], as_index=False)
        .size()
        .rename(columns={"size": "n_splits"})
        .sort_values(["artifact", "n_splits", "selected_policy"], ascending=[True, False, True])
    )


def _holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(values) - rank) * float(values[index])
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _paired_tests(heldout_scores: pd.DataFrame, references: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for artifact, artifact_scores in heldout_scores.groupby("artifact", sort=False):
        left = artifact_scores[artifact_scores["method"].eq(NESTED_SELECTED_METHOD)][
            [
                "split_id",
                "mean_rank",
                "mean_total_seconds",
                "mean_normalized_rank_regret",
            ]
        ].rename(
            columns={
                "mean_rank": "method_rank",
                "mean_total_seconds": "method_seconds",
                "mean_normalized_rank_regret": "method_norm_regret",
            }
        )
        for reference in references:
            right = artifact_scores[artifact_scores["method"].eq(reference)][
                [
                    "split_id",
                    "mean_rank",
                    "mean_total_seconds",
                    "mean_normalized_rank_regret",
                ]
            ].rename(
                columns={
                    "mean_rank": "reference_rank",
                    "mean_total_seconds": "reference_seconds",
                    "mean_normalized_rank_regret": "reference_norm_regret",
                }
            )
            paired = left.merge(right, on="split_id", how="inner")
            if paired.empty:
                continue
            rank_diff = paired["method_rank"] - paired["reference_rank"]
            time_diff = paired["method_seconds"] - paired["reference_seconds"]
            regret_diff = paired["method_norm_regret"] - paired["reference_norm_regret"]
            nonzero = rank_diff[rank_diff.ne(0.0)]
            p_value = (
                1.0
                if len(nonzero) == 0
                else float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue)
            )
            p_values.append(p_value)
            rows.append(
                {
                    "artifact": artifact,
                    "method": NESTED_SELECTED_METHOD,
                    "reference_method": reference,
                    "n_splits": int(len(paired)),
                    "mean_rank_difference": float(rank_diff.mean()),
                    "mean_normalized_regret_difference": float(regret_diff.mean()),
                    "mean_runtime_difference_seconds": float(time_diff.mean()),
                    "rank_wilcoxon_p_value": p_value,
                }
            )
    adjusted = _holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["rank_holm_p_value"] = value
    return pd.DataFrame(rows)


def _write_figures(
    output_dir: Path,
    *,
    split_results: pd.DataFrame,
    heldout_scores: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not split_results.empty:
        counts = _selection_counts(split_results)
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        labels = counts["artifact"].astype(str) + ": " + counts["selected_policy"].astype(str)
        ax.barh(labels, counts["n_splits"], color="#4C78A8")
        ax.invert_yaxis()
        ax.set_xlabel("Outer splits selected")
        ax.set_title("Nested policy selections")
        fig.tight_layout()
        fig.savefig(figure_dir / "selected_policy_frequency.png", dpi=220)
        plt.close(fig)

    regret = heldout_scores[
        heldout_scores["method"].isin(
            [
                NESTED_SELECTED_METHOD,
                TASK_ADAPTIVE_METHOD,
                "meta_selector_meta_feature_guard_005",
                "meta_selector_tabrepo_ensemble_k5",
                "oracle",
            ]
        )
    ].copy()
    if not regret.empty:
        regret["label"] = regret["artifact"].astype(str) + ": " + regret["method"].astype(str)
        labels = sorted(regret["label"].unique())
        values = [
            regret[regret["label"].eq(label)]["mean_normalized_rank_regret"].to_numpy(dtype=float)
            for label in labels
        ]
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        try:
            ax.boxplot(values, tick_labels=labels, vert=False, showfliers=False)
        except TypeError:
            ax.boxplot(values, labels=labels, vert=False, showfliers=False)
        ax.set_xlabel("Held-out mean normalized rank regret")
        ax.set_title("Held-out regret across outer splits")
        fig.tight_layout()
        fig.savefig(figure_dir / "heldout_normalized_regret.png", dpi=220)
        plt.close(fig)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        plot = summary[
            summary["method"].isin(
                [
                    NESTED_SELECTED_METHOD,
                    TASK_ADAPTIVE_METHOD,
                    FAMILY_PRIOR_METHOD,
                    "meta_selector_meta_feature_guard_005",
                    "meta_selector_tabrepo_ensemble_k5",
                    "oracle",
                ]
            )
        ].copy()
        for _, row in plot.iterrows():
            ax.scatter(
                row["mean_total_seconds"],
                row["mean_normalized_rank_regret"],
                s=70 if row["method"] == NESTED_SELECTED_METHOD else 42,
                marker="D" if row["method"] == NESTED_SELECTED_METHOD else "o",
                edgecolor="black",
                linewidth=0.6,
            )
            ax.annotate(
                f"{row['artifact']}: {row['method']}",
                (row["mean_total_seconds"], row["mean_normalized_rank_regret"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Mean total time (s, log)")
        ax.set_ylabel("Mean normalized rank regret")
        ax.set_title("Nested regret/runtime tradeoff")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.55)
        fig.tight_layout()
        fig.savefig(figure_dir / "heldout_regret_runtime.png", dpi=220)
        plt.close(fig)


def _write_report(
    output_dir: Path,
    *,
    split_results: pd.DataFrame,
    summary: pd.DataFrame,
    selection_counts: pd.DataFrame,
    paired: pd.DataFrame,
    command: str,
) -> None:
    lines = [
        "# Fully Nested Score-Level Policy Benchmark",
        "",
        f"- Command: `{command}`",
        f"- Completed artifact/split rows: {split_results.shape[0]}",
        "- Protocol: selector rows are rebuilt from outer-training datasets only; held-out datasets provide validation/runtime artifacts for target-side selection and test metrics for final evaluation only.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Selected Policies",
        "",
        dataframe_to_markdown(selection_counts),
        "",
        "## Paired Tests",
        "",
        dataframe_to_markdown(paired),
        "",
    ]
    (output_dir / "analysis-report.md").write_text("\n".join(lines), encoding="utf-8")


def _references_from_config(config: dict[str, Any]) -> list[str]:
    references = config.get(
        "references",
        [
            TASK_ADAPTIVE_METHOD,
            "meta_selector_meta_feature_guard_005",
            FAMILY_PRIOR_METHOD,
            VALIDATION_ONLY_METHOD,
            "meta_selector_single",
            "meta_selector_tabrepo_ensemble_k5",
        ],
    )
    return [str(value) for value in references]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _source_digest() -> dict[str, Any]:
    """SHA-256 of every Python source file of the pipeline, and one digest over all of them.

    The git commit recorded next to it need not exist in a published history; these digests
    can be checked against any copy of the code.
    """
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        path
        for pattern in ("src/tabular_sota_benchmark/*.py", "scripts/*.py")
        for path in root.glob(pattern)
    )
    hashes = {str(path.relative_to(root)): _sha256_file(path) for path in files}
    combined = hashlib.sha256("".join(f"{name}:{digest}\n" for name, digest in hashes.items()).encode()).hexdigest()
    return {"sha256": combined, "files": hashes}


def _artifact_input_hashes(artifact: dict[str, Any]) -> dict[str, str]:
    source = dict(artifact.get("source", {}))
    hashes: dict[str, str] = {}
    local_path = source.get("local_path")
    if local_path and Path(local_path).exists():
        hashes[str(local_path)] = _sha256_file(Path(local_path))
    if artifact.get("kind") == "tabrepo_repository":
        root = Path(__file__).resolve().parents[1] / ".deps" / "tabrepo" / "data" / "results"
        for path in sorted(root.glob("*/configs.parquet")):
            hashes[str(path.relative_to(root.parents[2]))] = _sha256_file(path)
        hashes["tabrepo_context"] = str(source.get("context_name", ""))
    return hashes


def _package_versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


_WORKER_ARTIFACT: dict[str, Any] | None = None


def _evaluate_split_task(
    task: tuple[int, list[str], list[str], str],
) -> tuple[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Worker entry point. The artifact is inherited through fork, never pickled."""
    if _WORKER_ARTIFACT is None:
        raise RuntimeError("Worker artifact is not initialised.")
    split_id, train_datasets, heldout_datasets, split_mode = task
    return split_id, _evaluate_artifact_split(
        _WORKER_ARTIFACT,
        split_id=split_id,
        train_datasets=train_datasets,
        heldout_datasets=heldout_datasets,
        split_mode=split_mode,
    )


def evaluate_splits(
    artifact: dict[str, Any],
    tasks: list[tuple[int, list[str], list[str], str]],
    *,
    jobs: int,
    on_result: Any,
) -> None:
    """Evaluate outer splits sequentially or in forked worker processes.

    Splits are drawn by the caller, and each split's computation is deterministic, so the
    written parts do not depend on ``jobs``.
    """
    global _WORKER_ARTIFACT
    if int(jobs) <= 1 or len(tasks) <= 1:
        for task in tasks:
            split_id, frames = _evaluate_split_task_inline(artifact, task)
            on_result(split_id, frames)
        return
    _WORKER_ARTIFACT = artifact
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as pool:
            for split_id, frames in pool.map(_evaluate_split_task, tasks):
                on_result(split_id, frames)
    finally:
        _WORKER_ARTIFACT = None


def _evaluate_split_task_inline(
    artifact: dict[str, Any],
    task: tuple[int, list[str], list[str], str],
) -> tuple[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    split_id, train_datasets, heldout_datasets, split_mode = task
    return split_id, _evaluate_artifact_split(
        artifact,
        split_id=split_id,
        train_datasets=train_datasets,
        heldout_datasets=heldout_datasets,
        split_mode=split_mode,
    )


def run(args: argparse.Namespace) -> None:
    started = time.time()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    output_dir = Path(args.output_dir or config.get("output_dir", "analysis-output/score_level_policy"))
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = dict(config.get("benchmark", {}))
    n_splits = int(args.splits if args.splits is not None else benchmark.get("splits", 100))
    heldout_fraction = float(
        args.heldout_fraction
        if args.heldout_fraction is not None
        else benchmark.get("heldout_fraction", 0.25)
    )
    seed = int(args.seed if args.seed is not None else benchmark.get("seed", 20260604))
    set_rank_rule(str(benchmark.get("rank_rule", "consistent")))

    artifacts = [_load_artifact(dict(item), seed=seed) for item in config.get("artifacts", [])]
    if not artifacts:
        raise ValueError("Nested policy config must define at least one artifact.")

    for artifact in artifacts:
        splits = _split_datasets(
            artifact["candidates"],
            n_splits=n_splits,
            heldout_fraction=heldout_fraction,
            seed=seed,
        )
        split_lookup = {
            split_id: (split_id, train, heldout, split_mode)
            for split_id, train, heldout, split_mode in splits
        }
        expected_ids = [split_id for split_id, _, _, _ in splits]
        completed = (
            _completed_parts(output_dir, artifact["name"], expected_ids)
            if bool(args.resume)
            else set()
        )
        pending = [
            split_lookup[split_id]
            for split_id in expected_ids
            if split_id not in completed
        ]
        if completed:
            print(
                f"{artifact['name']}: resuming with {len(completed)} completed and "
                f"{len(pending)} pending split(s).",
                flush=True,
            )
        print(
            f"{artifact['name']}: {len(pending)} nested split(s), jobs={int(args.jobs)}, "
            f"rank_rule={get_rank_rule()}, "
            f"weight_selection={artifact['selector'].get('weight_selection', 'lodo')}, "
            f"inner_source_fit={_inner_source_fit(artifact['selector'])}",
            flush=True,
        )

        def write_result(
            split_id: int,
            frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
            artifact_name: str = artifact["name"],
        ) -> None:
            split_result, train_scores, heldout_scores, unit_scores = frames
            _write_split_parts(
                output_dir,
                artifact=artifact_name,
                split_id=split_id,
                split_result=split_result,
                train_scores=train_scores,
                heldout_scores=heldout_scores,
                unit_scores=unit_scores,
            )
            print(f"{artifact_name}: finished split {split_id + 1}/{n_splits}", flush=True)

        evaluate_splits(artifact, pending, jobs=int(args.jobs), on_result=write_result)

    split_results = _load_parts(output_dir, "split_results").sort_values(
        ["artifact", "split_id"]
    )
    training_scores = _load_parts(output_dir, "training_policy_scores").sort_values(
        ["artifact", "split_id", "method_order", "method"]
    )
    heldout_scores = _load_parts(output_dir, "heldout_policy_scores").sort_values(
        ["artifact", "split_id", "method_order", "method"]
    )
    unit_scores = _load_parts(output_dir, "unit_scores")
    summary = _summary(heldout_scores)
    selection_counts = _selection_counts(split_results)
    paired = _paired_tests(heldout_scores, _references_from_config(config))

    split_results.to_csv(output_dir / "split_results.csv", index=False)
    training_scores.to_csv(output_dir / "training_policy_scores.csv", index=False)
    heldout_scores.to_csv(output_dir / "heldout_policy_scores.csv", index=False)
    unit_scores.to_parquet(output_dir / "unit_scores.parquet", index=False)
    selection_counts.to_csv(output_dir / "policy_selection_counts.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_tests.csv", index=False)
    metadata = {
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "command": [str(value) for value in getattr(args, "command", sys.argv)],
        "git": _git_state(),
        "source": _source_digest(),
        "packages": _package_versions(),
        "generated_at_epoch": time.time(),
        "wall_seconds": time.time() - started,
        "jobs": int(args.jobs),
        "n_splits": n_splits,
        "heldout_fraction": heldout_fraction,
        "seed": seed,
        "rank_rule": get_rank_rule(),
        "artifacts": [
            {
                "name": artifact["name"],
                "kind": artifact["kind"],
                "n_datasets": int(artifact["candidates"]["dataset"].nunique()),
                "n_candidate_methods": int(artifact["candidates"]["method"].nunique()),
                "n_policy_methods": int(len(artifact["policy_methods"])),
                "policy_methods": list(artifact["policy_methods"]),
                "reference_methods": list(artifact.get("reference_methods", [])),
                "weight_selection": str(artifact["selector"].get("weight_selection", "lodo")),
                "inner_source_fit": _inner_source_fit(artifact["selector"]),
                "input_sha256": _artifact_input_hashes(artifact),
            }
            for artifact in artifacts
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_figures(
        output_dir,
        split_results=split_results,
        heldout_scores=heldout_scores,
        summary=summary,
    )
    _write_report(
        output_dir,
        split_results=split_results,
        summary=summary,
        selection_counts=selection_counts,
        paired=paired,
        command=" ".join(metadata["command"]),
    )
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fully nested score-level selector-policy benchmark."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/score_level_policy.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.command = sys.argv
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
