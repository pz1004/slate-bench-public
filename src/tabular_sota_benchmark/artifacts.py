from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from tabular_sota_benchmark.ranking import grouped_within_unit_ranks

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


TABARENA_RESULTS_URL = (
    "https://huggingface.co/datasets/TabArena/benchmark_results/resolve/main/df_results.csv"
)

TABREPO_DEFAULT_CONTEXT = "D244_F3_C1530_30"

TABARENA_DATASET_METADATA_URL = (
    "https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/main/"
    "dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv"
)

TABARENA_DATASET_CURATION_BASE_URL = (
    "https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/main/"
    "dataset_creation_scripts/datasets"
)

DATAHUB_OPENML_DATAPACKAGE_URL = (
    "https://datahub.io/core/openml-datasets/_r/-/datapackage.json"
)

DATAHUB_OPENML_RESOURCE_BASE_URL = "https://datahub.io/core/openml-datasets/_r/-"

OPENML_PARQUET_BASE_URL = "https://data.openml.org/datasets"

DEFAULT_DESCRIPTOR_NUMERIC_COLUMNS = [
    "num_features",
    "num_instances",
    "num_classes",
    "percentage_cat_features",
    "year",
]

DEFAULT_DESCRIPTOR_LOG_COLUMNS = [
    "num_features",
    "num_instances",
]

DEFAULT_DESCRIPTOR_BOOLEAN_COLUMNS = [
    "can_run_tabpfnv2",
    "can_run_tabicl",
]

DEFAULT_DESCRIPTOR_CATEGORICAL_COLUMNS = [
    "data_source",
    "domain",
]

RAW_DATA_MISSING_VALUES = ["?", "NA", "N/A", "nan", "NaN", "null", "None", ""]

TABARENA_REQUIRED_COLUMNS = {
    "dataset",
    "fold",
    "method",
    "metric_error",
    "time_train_s",
    "time_infer_s",
    "metric_error_val",
    "seed",
    "problem_type",
    "metric",
}

ARTIFACT_OUTPUT_COLUMNS = [
    "dataset",
    "fold",
    "seed",
    "task",
    "method",
    "method_type",
    "primary_metric_name",
    "primary_metric",
    "val_primary_metric",
    "fit_seconds",
    "predict_seconds",
    "selected_member",
    "ensemble_members",
    "ensemble_weights",
    "source",
    "problem_type",
    "metric",
    "metric_error",
    "metric_error_val",
    "time_train_s",
    "time_infer_s",
]

ARTIFACT_UNIT_COLUMNS = ["dataset", "fold", "seed", "task", "primary_metric_name"]

CALIBRATION_WEIGHT_COLUMNS = [
    "validation_weight",
    "task_method_prior_weight",
    "task_family_prior_weight",
    "global_method_prior_weight",
    "runtime_weight",
]

BUDGET_TUNING_COLUMNS = [
    "budget_feature",
    "max_budget_seconds",
    "n_target_units",
    "n_target_candidates",
    "n_target_budget_fallback_units",
    "n_training_candidates",
]

FILTER_TUNING_COLUMNS = [
    "candidate_filter",
    "n_unfiltered_candidates",
    "n_filtered_candidates",
]

ADAPTIVE_TUNING_COLUMNS = [
    "candidate_selector_methods",
    "selected_selector_method",
    "adaptive_objective",
    "adaptive_mean_total_seconds",
    "adaptive_choice_scope",
    "n_candidate_selectors",
    "meta_feature_objective",
    "meta_feature_runtime_norm",
    "meta_feature_runtime_tradeoff_weight",
    "meta_feature_columns",
    "meta_feature_groups",
    "nearest_datasets",
    "n_neighbor_datasets",
    "nearest_distance_min",
    "nearest_distance_mean",
    "default_selector_method",
    "min_improvement_over_default",
]

RISK_TUNING_COLUMNS = [
    "task_reference_method",
    "family_reference_method",
    "fallback_selector_methods",
    "risk_noninferiority_margin_rank",
    "risk_noninferiority_alpha",
    "risk_min_training_datasets",
    "risk_require_runtime_gain",
    "risk_min_runtime_gain_seconds",
    "risk_task_upper_one_sided_ci",
    "risk_family_upper_one_sided_ci",
    "risk_task_mean_rank_difference",
    "risk_family_mean_rank_difference",
    "risk_task_mean_runtime_difference_seconds",
    "risk_gate_passed",
    "risk_fallback_reason",
    "risk_objective",
    "risk_runtime_norm",
]

CONFIDENCE_TUNING_COLUMNS = [
    "validation_rank_margin",
    "validation_rank_norm_margin",
    "confidence_set_mean_size",
    "confidence_set_min_size",
    "confidence_set_max_size",
    "confidence_budget_feature",
    "confidence_max_budget_seconds",
    "confidence_budget_fallback_units",
]

MetaFeatureGroup = tuple[str, list[str], float, bool]


def _configured_string_list(
    config: dict[str, Any],
    key: str,
    default: list[str],
) -> list[str]:
    value = config.get(key)
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def ensure_remote_file(url: str, local_path: str | Path, download: bool = True) -> Path:
    path = Path(local_path)
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"Artifact file does not exist and download is disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        with NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    return path


def load_tabarena_results(artifact_config: dict[str, Any]) -> pd.DataFrame:
    url = str(artifact_config.get("url", TABARENA_RESULTS_URL))
    local_path = ensure_remote_file(
        url,
        artifact_config.get("local_path", "data/tabarena/df_results.csv"),
        download=bool(artifact_config.get("download", True)),
    )
    frame = pd.read_csv(local_path)
    missing = TABARENA_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"TabArena artifact missing columns: {sorted(missing)}")

    dataset_allowlist = artifact_config.get("dataset_allowlist")
    if dataset_allowlist:
        frame = frame[frame["dataset"].isin(dataset_allowlist)]

    method_allowlist = artifact_config.get("method_allowlist")
    if method_allowlist:
        frame = frame[frame["method"].isin(method_allowlist)]

    problem_types = artifact_config.get("problem_types")
    if problem_types:
        frame = frame[frame["problem_type"].isin(problem_types)]

    frame = frame.dropna(
        subset=["metric_error", "metric_error_val", "time_train_s", "time_infer_s"]
    )
    if frame.empty:
        raise ValueError("TabArena artifact has no rows after filtering.")
    return frame.reset_index(drop=True)


def _patch_tabrepo_rank_scorer_readonly_sort() -> None:
    try:
        from tabrepo.utils import rank_utils  # type: ignore
    except Exception:
        return

    rank_scorer = getattr(rank_utils, "RankScorer", None)
    if rank_scorer is None or getattr(rank_scorer, "_slate_readonly_sort_patch", False):
        return

    def patched_init(
        self: Any,
        df_results: pd.DataFrame,
        tasks: list[str],
        metric_error_col: str = "metric_error",
        task_col: str = "task",
        framework_col: str = "framework",
        ties_win: bool = False,
        pct: bool = False,
        include_partial: bool = True,
    ) -> None:
        assert all(col in df_results for col in [metric_error_col, task_col, framework_col])
        all_datasets = set(df_results[task_col].unique())
        for task in tasks:
            assert task in all_datasets, f"{task_col} {task} not present in passed evaluations"
        self.ties_win = ties_win
        self.pct = pct
        self.include_partial = include_partial
        df_pivot = df_results.pivot_table(
            values=metric_error_col,
            index=task_col,
            columns=framework_col,
        )
        sorted_values = np.sort(df_pivot.to_numpy(copy=True), axis=1)
        sorted_pivot = pd.DataFrame(sorted_values, index=df_pivot.index)
        self.error_dict = {
            dataset: sorted_pivot.loc[dataset].dropna().tolist()
            for dataset in tasks
        }

    rank_scorer.__init__ = patched_init
    rank_scorer._slate_readonly_sort_patch = True


def _repo_local_tabrepo_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    local_path = repo_root / ".deps" / "tabrepo"
    if (local_path / "tabrepo" / "__init__.py").exists():
        return local_path
    return None


def _load_tabrepo_loader() -> Any:
    try:
        from tabarena import load_repository  # type: ignore

        return load_repository
    except ImportError:
        pass

    try:
        from tabrepo import load_repository  # type: ignore

        return load_repository
    except ImportError:
        pass

    local_path = _repo_local_tabrepo_path()
    if local_path is not None:
        local_text = str(local_path)
        if local_text not in sys.path:
            sys.path.insert(0, local_text)
        try:
            from tabrepo import load_repository  # type: ignore

            return load_repository
        except ImportError:
            pass

    raise ImportError(
        "TabRepo artifact loading requires `tabarena.load_repository`, "
        "`tabrepo.load_repository`, or the repo-local `.deps/tabrepo` checkout. "
        "In Anaconda, install it in the active environment or run from this "
        "repo with `.deps/tabrepo` present. For example: "
        "`python -m pip install -e .deps/tabrepo`."
    )


def load_tabrepo_repository(artifact_config: dict[str, Any]) -> Any:
    cache_dir = artifact_config.get("cache_dir")
    if cache_dir:
        os.environ["TABREPO_CACHE"] = str(Path(cache_dir))
    context_name = str(artifact_config.get("context_name", TABREPO_DEFAULT_CONTEXT))
    load_predictions = bool(artifact_config.get("load_predictions", False))
    cache = artifact_config.get("cache", False)
    prediction_format = str(artifact_config.get("prediction_format", "memmap"))

    load_repository = _load_tabrepo_loader()
    _patch_tabrepo_rank_scorer_readonly_sort()
    try:
        return load_repository(
            context_name,
            load_predictions=load_predictions,
            cache=cache,
            prediction_format=prediction_format,
        )
    except TypeError:
        return load_repository(context_name)


def load_tabrepo_results(
    artifact_config: dict[str, Any],
    repository: Any | None = None,
) -> pd.DataFrame:
    repo = repository if repository is not None else load_tabrepo_repository(artifact_config)
    datasets = artifact_config.get("dataset_allowlist")
    allowlist_file = artifact_config.get("dataset_allowlist_file")
    if allowlist_file:
        # The repository's dataset list depends on whether predictions are loaded (the
        # D244_F3_C1530_200 context has 244 datasets of scores, 200 with predictions), so
        # analyses that skip predictions pin the slice with an explicit list.
        if datasets:
            raise ValueError("Set dataset_allowlist or dataset_allowlist_file, not both.")
        lines = Path(str(allowlist_file)).read_text(encoding="utf-8").splitlines()
        datasets = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if datasets:
        dataset_names = [str(value) for value in datasets]
    else:
        dataset_names = [str(value) for value in repo.datasets()]

    folds = artifact_config.get("folds")
    fold_values = [int(value) for value in folds] if folds is not None else None
    configs = artifact_config.get("method_allowlist") or artifact_config.get("config_allowlist")
    config_values = [str(value) for value in configs] if configs else None
    config_limit = artifact_config.get("config_limit")
    if config_values is None and config_limit is not None:
        config_values = [str(value) for value in repo.configs()[: int(config_limit)]]

    metrics = repo.metrics(
        datasets=dataset_names,
        folds=fold_values,
        configs=config_values,
    )
    frame = metrics.reset_index()
    rename = {"framework": "method", "config": "method"}
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame})
    required = {
        "dataset",
        "fold",
        "method",
        "metric_error",
        "metric_error_val",
        "time_train_s",
        "time_infer_s",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"TabRepo metrics missing columns: {sorted(missing)}")

    info_cache: dict[str, dict[str, Any]] = {}
    for dataset in sorted(frame["dataset"].astype(str).unique()):
        info_cache[dataset] = dict(repo.dataset_info(dataset))
    frame["problem_type"] = frame["dataset"].astype(str).map(
        lambda value: str(info_cache[value].get("problem_type", "unknown"))
    )
    frame["metric"] = frame["dataset"].astype(str).map(
        lambda value: str(info_cache[value].get("metric", "metric_error"))
    )
    frame["seed"] = frame["fold"].astype(int)

    problem_types = artifact_config.get("problem_types")
    if problem_types:
        frame = frame[frame["problem_type"].isin(problem_types)]

    frame = frame.dropna(
        subset=["metric_error", "metric_error_val", "time_train_s", "time_infer_s"]
    )
    if frame.empty:
        raise ValueError("TabRepo artifact has no rows after filtering.")
    return frame[
        [
            "dataset",
            "fold",
            "method",
            "metric_error",
            "time_train_s",
            "time_infer_s",
            "metric_error_val",
            "seed",
            "problem_type",
            "metric",
        ]
    ].reset_index(drop=True)


def load_dataset_descriptors(descriptor_config: dict[str, Any] | None) -> pd.DataFrame | None:
    if not descriptor_config:
        return None

    url = str(descriptor_config.get("url", TABARENA_DATASET_METADATA_URL))
    local_path = ensure_remote_file(
        url,
        descriptor_config.get("local_path", "data/tabarena/tabarena_dataset_metadata.csv"),
        download=bool(descriptor_config.get("download", True)),
    )
    frame = pd.read_csv(local_path)
    dataset_column = str(descriptor_config.get("dataset_column", "dataset_name"))
    if dataset_column not in frame.columns:
        raise ValueError(f"Dataset descriptor file missing dataset column: {dataset_column}")

    prepared = pd.DataFrame({"dataset": frame[dataset_column].astype(str)})
    numeric_columns = _configured_string_list(
        descriptor_config,
        "numeric_columns",
        DEFAULT_DESCRIPTOR_NUMERIC_COLUMNS,
    )
    log_columns = set(
        _configured_string_list(
            descriptor_config,
            "log_columns",
            DEFAULT_DESCRIPTOR_LOG_COLUMNS,
        )
    )
    for column in numeric_columns:
        if column not in frame.columns:
            raise ValueError(f"Dataset descriptor file missing numeric column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        safe_column = _safe_feature_name(column)
        prepared[f"descriptor_{safe_column}"] = values.astype(float)
        if column in log_columns:
            nonnegative = values.clip(lower=0.0)
            prepared[f"descriptor_log_{safe_column}"] = np.log1p(nonnegative)

    boolean_columns = _configured_string_list(
        descriptor_config,
        "boolean_columns",
        DEFAULT_DESCRIPTOR_BOOLEAN_COLUMNS,
    )
    for column in boolean_columns:
        if column not in frame.columns:
            raise ValueError(f"Dataset descriptor file missing boolean column: {column}")
        values = frame[column]
        numeric = values.map(
            lambda value: np.nan
            if pd.isna(value)
            else float(str(value).strip().lower() in {"1", "true", "yes", "y"})
        )
        prepared[f"descriptor_{_safe_feature_name(column)}"] = numeric.astype(float)

    categorical_columns = _configured_string_list(
        descriptor_config,
        "categorical_columns",
        DEFAULT_DESCRIPTOR_CATEGORICAL_COLUMNS,
    )
    for column in categorical_columns:
        if column not in frame.columns:
            raise ValueError(f"Dataset descriptor file missing categorical column: {column}")
        values = frame[column].fillna("missing").map(_safe_feature_name).astype(str)
        dummies = pd.get_dummies(
            values,
            prefix=f"descriptor_{_safe_feature_name(column)}",
            dtype=float,
        )
        prepared = pd.concat([prepared, dummies], axis=1)

    prepared["descriptor_missing"] = 0.0
    prepared = prepared.drop_duplicates(subset=["dataset"], keep="first")
    if bool(descriptor_config.get("include_curation_yaml", False)):
        curation_descriptors = _load_curation_yaml_descriptors(frame, descriptor_config)
        prepared = prepared.merge(curation_descriptors, on="dataset", how="left")
    if descriptor_config.get("quality_mirror"):
        quality_descriptors = _load_quality_mirror_descriptors(frame, descriptor_config)
        prepared = prepared.merge(quality_descriptors, on="dataset", how="left")
    return prepared.sort_values("dataset").reset_index(drop=True)


def _load_curation_yaml_descriptors(
    metadata: pd.DataFrame,
    descriptor_config: dict[str, Any],
) -> pd.DataFrame:
    dataset_column = str(descriptor_config.get("dataset_column", "dataset_name"))
    base_url = str(
        descriptor_config.get("curation_base_url", TABARENA_DATASET_CURATION_BASE_URL)
    ).rstrip("/")
    local_dir = Path(
        descriptor_config.get("curation_local_dir", "data/tabarena/curation_yamls")
    )
    download = bool(descriptor_config.get("download", True))
    rows: list[dict[str, Any]] = []
    for _, metadata_row in metadata.iterrows():
        dataset = str(metadata_row[dataset_column])
        filename = f"{dataset}.yaml"
        local_path = local_dir / dataset / filename
        url_dataset = quote(dataset, safe="")
        url = f"{base_url}/{url_dataset}/{url_dataset}.yaml"
        path = ensure_remote_file(url, local_path, download=download)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected curation YAML structure for dataset: {dataset}")
        categorical_features = raw.get("categorical_features") or []
        if isinstance(categorical_features, str):
            categorical_features = [categorical_features]
        categorical_count = float(len(categorical_features))
        raw_feature_count = pd.to_numeric(
            pd.Series([metadata_row.get("num_features")]),
            errors="coerce",
        ).iloc[0]
        feature_count = float(raw_feature_count) - 1.0 if pd.notna(raw_feature_count) else np.nan
        if np.isfinite(feature_count):
            feature_count = max(feature_count, 0.0)
        numeric_count = (
            max(feature_count - categorical_count, 0.0)
            if np.isfinite(feature_count)
            else np.nan
        )
        categorical_ratio = (
            categorical_count / feature_count
            if np.isfinite(feature_count) and feature_count > 0.0
            else np.nan
        )
        problem_type = str(metadata_row.get("problem_type", raw.get("problem_type", "")))
        raw_num_classes = pd.to_numeric(
            pd.Series([metadata_row.get("num_classes")]),
            errors="coerce",
        ).iloc[0]
        target_cardinality = (
            float(raw_num_classes)
            if pd.notna(raw_num_classes) and problem_type != "regression"
            else 0.0
        )
        curation_text = "\n".join(
            str(raw.get(key, ""))
            for key in ["curation_comments", "reference", "original_dataset_source"]
        )
        rows.append(
            {
                "dataset": dataset,
                "descriptor_curation_feature_count": feature_count,
                "descriptor_curation_categorical_feature_count": categorical_count,
                "descriptor_curation_numeric_feature_count": numeric_count,
                "descriptor_curation_categorical_feature_ratio": categorical_ratio,
                "descriptor_target_cardinality": target_cardinality,
                "descriptor_target_is_regression": float(problem_type == "regression"),
                "descriptor_curation_mentions_missing": float(
                    _curation_text_mentions_missing(curation_text)
                ),
                "descriptor_curation_mentions_duplicate": float(
                    "duplicate" in curation_text.lower()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _curation_text_mentions_missing(text: str) -> bool:
    normalized = text.lower()
    if any(term in normalized for term in ["missing", "nan", "null", "not_applicable"]):
        return True
    return bool(re.search(r"(^|[^a-z0-9])na([^a-z0-9]|$)", normalized))


def _load_quality_mirror_descriptors(
    metadata: pd.DataFrame,
    descriptor_config: dict[str, Any],
) -> pd.DataFrame:
    quality_config = dict(descriptor_config.get("quality_mirror") or {})
    kind = str(quality_config.get("kind", "datahub_openml_datasets"))
    if kind == "datahub_openml_datasets":
        return _load_datahub_quality_mirror_descriptors(metadata, descriptor_config)
    if kind == "openml_parquet":
        return _load_openml_parquet_quality_descriptors(metadata, descriptor_config)
    raise ValueError(f"Unsupported quality mirror kind: {kind}")


def _load_openml_parquet_quality_descriptors(
    metadata: pd.DataFrame,
    descriptor_config: dict[str, Any],
) -> pd.DataFrame:
    quality_config = dict(descriptor_config.get("quality_mirror") or {})
    dataset_column = str(descriptor_config.get("dataset_column", "dataset_name"))
    dataset_id_column = str(quality_config.get("dataset_id_column", "dataset_id"))
    if dataset_id_column not in metadata.columns:
        raise ValueError(f"Dataset descriptor file missing OpenML dataset id column: {dataset_id_column}")
    cache_dir = Path(quality_config.get("parquet_cache_dir", "data/tabarena/openml_parquet"))
    base_url = str(quality_config.get("parquet_base_url", OPENML_PARQUET_BASE_URL)).rstrip("/")
    download = bool(quality_config.get("download", descriptor_config.get("download", True)))

    fallback_to_datahub = bool(quality_config.get("fallback_to_datahub", False))
    datahub_resources: dict[str, str] | None = None
    if fallback_to_datahub:
        datahub_resources = _datahub_resource_paths_by_key(quality_config, descriptor_config)

    rows: list[dict[str, Any]] = []
    for _, metadata_row in metadata.iterrows():
        dataset = str(metadata_row[dataset_column])
        dataset_id = pd.to_numeric(pd.Series([metadata_row.get(dataset_id_column)]), errors="coerce").iloc[0]
        if pd.notna(dataset_id):
            openml_id = int(dataset_id)
            prefix = f"{openml_id // 10000:04d}"
            local_path = cache_dir / prefix / str(openml_id) / f"dataset_{openml_id}.pq"
            url = f"{base_url}/{prefix}/{openml_id}/dataset_{openml_id}.pq"
            try:
                path = ensure_remote_file(url, local_path, download=download)
                rows.append(_quality_descriptor_row_from_parquet(path, metadata_row, dataset))
                continue
            except Exception:
                if not fallback_to_datahub:
                    rows.append(_missing_quality_descriptor_row(dataset))
                    continue

        if datahub_resources is not None:
            row = _datahub_quality_descriptor_for_metadata_row(
                metadata_row,
                dataset,
                descriptor_config,
                datahub_resources,
            )
            rows.append(row)
        else:
            rows.append(_missing_quality_descriptor_row(dataset))
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _load_datahub_quality_mirror_descriptors(
    metadata: pd.DataFrame,
    descriptor_config: dict[str, Any],
) -> pd.DataFrame:
    quality_config = dict(descriptor_config.get("quality_mirror") or {})
    resource_by_key = _datahub_resource_paths_by_key(quality_config, descriptor_config)
    dataset_column = str(descriptor_config.get("dataset_column", "dataset_name"))

    rows: list[dict[str, Any]] = []
    for _, metadata_row in metadata.iterrows():
        dataset = str(metadata_row[dataset_column])
        rows.append(
            _datahub_quality_descriptor_for_metadata_row(
                metadata_row,
                dataset,
                descriptor_config,
                resource_by_key,
            )
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _datahub_resource_paths_by_key(
    quality_config: dict[str, Any],
    descriptor_config: dict[str, Any],
) -> dict[str, str]:
    datapackage_path = ensure_remote_file(
        str(quality_config.get("datapackage_url", DATAHUB_OPENML_DATAPACKAGE_URL)),
        quality_config.get(
            "datapackage_local_path",
            "data/tabarena/openml_datahub/datapackage.json",
        ),
        download=bool(quality_config.get("download", descriptor_config.get("download", True))),
    )
    with datapackage_path.open("r", encoding="utf-8") as handle:
        datapackage = json.load(handle)

    resources = datapackage.get("resources", [])
    resource_by_key: dict[str, str] = {}
    for resource in resources:
        path = str(resource.get("path", ""))
        if not path.endswith(".csv"):
            continue
        parts = path.split("/")
        names = {Path(path).stem}
        if len(parts) >= 2:
            names.add(parts[-2])
        for name in names:
            resource_by_key.setdefault(_normalize_resource_key(name), path)
    return resource_by_key


def _datahub_quality_descriptor_for_metadata_row(
    metadata_row: pd.Series,
    dataset: str,
    descriptor_config: dict[str, Any],
    resource_by_key: dict[str, str],
) -> dict[str, float | str]:
    quality_config = dict(descriptor_config.get("quality_mirror") or {})
    dataset_column = str(descriptor_config.get("dataset_column", "dataset_name"))
    raw_cache_dir = Path(
        quality_config.get("raw_cache_dir", "data/tabarena/openml_datahub")
    )
    resource_base_url = str(
        quality_config.get("resource_base_url", DATAHUB_OPENML_RESOURCE_BASE_URL)
    ).rstrip("/")
    download = bool(quality_config.get("download", descriptor_config.get("download", True)))
    candidate_names = [
        metadata_row.get(dataset_column, ""),
        metadata_row.get("openml_dataset_name", ""),
        metadata_row.get("dataset_name", ""),
    ]
    resource_path = _find_quality_resource_path(candidate_names, resource_by_key)
    if resource_path is None:
        return _missing_quality_descriptor_row(dataset)

    local_path = raw_cache_dir / resource_path
    url = f"{resource_base_url}/{resource_path}"
    try:
        path = ensure_remote_file(url, local_path, download=download)
    except FileNotFoundError:
        return _missing_quality_descriptor_row(dataset)
    return _quality_descriptor_row_from_csv(path, metadata_row, dataset)


def _normalize_resource_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_quality_resource_path(
    candidate_names: list[object],
    resource_by_key: dict[str, str],
) -> str | None:
    for name in candidate_names:
        key = _normalize_resource_key(name)
        if key in resource_by_key:
            return resource_by_key[key]
    return None


def _missing_quality_descriptor_row(dataset: str) -> dict[str, float | str]:
    return {
        "dataset": dataset,
        "descriptor_quality_raw_coverage": 0.0,
        "descriptor_quality_label_coverage": 0.0,
        "descriptor_quality_n_instances": np.nan,
        "descriptor_quality_n_features": np.nan,
        "descriptor_quality_missing_value_fraction": np.nan,
        "descriptor_quality_instances_with_missing_fraction": np.nan,
        "descriptor_quality_features_with_missing_fraction": np.nan,
        "descriptor_quality_target_missing_fraction": np.nan,
        "descriptor_quality_majority_class_fraction": np.nan,
        "descriptor_quality_minority_class_fraction": np.nan,
        "descriptor_quality_label_imbalance_ratio": np.nan,
        "descriptor_quality_label_entropy_norm": np.nan,
    }


def _quality_descriptor_row_from_csv(
    path: Path,
    metadata_row: pd.Series,
    dataset: str,
) -> dict[str, float | str]:
    data = pd.read_csv(path, na_values=RAW_DATA_MISSING_VALUES, keep_default_na=True)
    return _quality_descriptor_row_from_frame(data, metadata_row, dataset)


def _quality_descriptor_row_from_parquet(
    path: Path,
    metadata_row: pd.Series,
    dataset: str,
) -> dict[str, float | str]:
    data = pd.read_parquet(path)
    return _quality_descriptor_row_from_frame(data, metadata_row, dataset)


def _quality_descriptor_row_from_frame(
    data: pd.DataFrame,
    metadata_row: pd.Series,
    dataset: str,
) -> dict[str, float | str]:
    target_column = _resolve_target_column(data.columns, str(metadata_row.get("target_feature", "")))
    feature_data = data.drop(columns=[target_column], errors="ignore") if target_column else data
    n_instances = float(len(data))
    n_features = float(len(feature_data.columns))
    if n_instances > 0.0 and n_features > 0.0:
        missing_mask = feature_data.isna()
        missing_value_fraction = float(missing_mask.to_numpy().sum() / (n_instances * n_features))
        instances_with_missing_fraction = float(missing_mask.any(axis=1).mean())
        features_with_missing_fraction = float(missing_mask.any(axis=0).mean())
    else:
        missing_value_fraction = np.nan
        instances_with_missing_fraction = np.nan
        features_with_missing_fraction = np.nan

    if target_column and n_instances > 0.0:
        target = data[target_column]
        target_missing_fraction = float(target.isna().mean())
    else:
        target = None
        target_missing_fraction = np.nan

    problem_type = str(metadata_row.get("problem_type", ""))
    label_stats = _classification_label_quality_stats(target, problem_type)
    return {
        "dataset": dataset,
        "descriptor_quality_raw_coverage": 1.0,
        "descriptor_quality_label_coverage": label_stats.pop("label_coverage"),
        "descriptor_quality_n_instances": n_instances,
        "descriptor_quality_n_features": n_features,
        "descriptor_quality_missing_value_fraction": missing_value_fraction,
        "descriptor_quality_instances_with_missing_fraction": instances_with_missing_fraction,
        "descriptor_quality_features_with_missing_fraction": features_with_missing_fraction,
        "descriptor_quality_target_missing_fraction": target_missing_fraction,
        **{
            f"descriptor_quality_{key}": value
            for key, value in label_stats.items()
        },
    }


def _resolve_target_column(columns: pd.Index, target_feature: str) -> str | None:
    column_names = [str(column) for column in columns]
    candidates = [
        target_feature,
        unquote(target_feature),
        quote(target_feature, safe=""),
        target_feature.replace(" ", "_"),
    ]
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    normalized = {_normalize_resource_key(column): column for column in column_names}
    for candidate in candidates:
        key = _normalize_resource_key(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in ["class", "Class", "target", "label", "y"]:
        if candidate in column_names:
            return candidate
    return None


def _classification_label_quality_stats(
    target: pd.Series | None,
    problem_type: str,
) -> dict[str, float]:
    empty = {
        "label_coverage": 0.0,
        "majority_class_fraction": np.nan,
        "minority_class_fraction": np.nan,
        "label_imbalance_ratio": np.nan,
        "label_entropy_norm": np.nan,
    }
    if target is None or problem_type == "regression":
        return empty
    counts = target.dropna().astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0.0 or len(counts) <= 1:
        return empty
    fractions = counts.astype(float) / total
    majority = float(fractions.max())
    minority = float(fractions.min())
    entropy = float(-(fractions * np.log(fractions)).sum())
    entropy_norm = entropy / float(np.log(len(fractions))) if len(fractions) > 1 else np.nan
    return {
        "label_coverage": 1.0,
        "majority_class_fraction": majority,
        "minority_class_fraction": minority,
        "label_imbalance_ratio": majority / minority if minority > 0.0 else np.nan,
        "label_entropy_norm": entropy_norm,
    }


def _candidate_rows_from_tabarena(frame: pd.DataFrame) -> pd.DataFrame:
    return _candidate_rows_from_score_frame(
        frame,
        source="tabarena_benchmark_results",
        metric_prefix="tabarena_metric_error",
    )


def _candidate_rows_from_tabrepo(frame: pd.DataFrame) -> pd.DataFrame:
    return _candidate_rows_from_score_frame(
        frame,
        source="tabrepo_repository",
        metric_prefix="tabrepo_metric_error",
    )


def _candidate_rows_from_score_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    metric_prefix: str,
) -> pd.DataFrame:
    rows = frame.copy()
    rows["task"] = rows["problem_type"]
    rows["method_type"] = "candidate"
    rows["primary_metric_name"] = metric_prefix + ":" + rows["metric"].astype(str)
    rows["primary_metric"] = rows["metric_error"].astype(float)
    rows["val_primary_metric"] = rows["metric_error_val"].astype(float)
    rows["fit_seconds"] = rows["time_train_s"].astype(float)
    rows["predict_seconds"] = rows["time_infer_s"].astype(float)
    rows["selected_member"] = ""
    rows["ensemble_members"] = ""
    rows["ensemble_weights"] = ""
    rows["source"] = source
    return rows[ARTIFACT_OUTPUT_COLUMNS].copy()


_TABREPO_CONFIG_PATTERN = re.compile(r"^(?P<family>.+?)_(?:c|r)\d+(?:_BAG_L\d+)?$")


def _method_family(method: object) -> str:
    """Base-learner family of a candidate.

    TabArena names carry the family before a parenthesised variant, e.g.
    ``"MNCA (tuned + ensemble)"``. TabRepo configuration names such as
    ``"CatBoost_r12_BAG_L1"`` have no parenthesis; their family is the learner prefix, so
    that every configuration of one base learner shares a family as the manuscript states.
    """
    name = str(method).strip()
    if "(" in name:
        return name.split("(", maxsplit=1)[0].strip()
    match = _TABREPO_CONFIG_PATTERN.match(name)
    if match:
        return match.group("family")
    return name


def _method_variant(method: object) -> str:
    """Validation-provenance variant from a TabArena name: default, tuned, tuned + ensemble.

    Names without a parenthesised variant (TabRepo configurations, which are single fits)
    return ``"n/a"``.
    """
    name = str(method)
    if "(" not in name:
        return "n/a"
    inner = name.split("(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0].strip().lower()
    return " ".join(inner.replace("+", " + ").split())


_ADAPTIVE_CHOICE_PATTERN = re.compile(r"(?:^|;)adaptive_choice=([^;]+)")


def selector_provenance(method: object, ensemble_weights: object) -> tuple[str, str]:
    """Return ``(source_selector, base_selector)`` for one selector row.

    Adaptive selectors copy a source selector's row and append their own
    ``adaptive_choice=<source>`` to ``ensemble_weights``, so the choices are recorded
    innermost first: the last entry is the row's direct source and the first is the
    non-adaptive selector that actually chose the candidate. Non-adaptive rows have no
    entry; their base selector is the row itself.
    """
    text_value = "" if ensemble_weights is None or pd.isna(ensemble_weights) else str(ensemble_weights)
    choices = _ADAPTIVE_CHOICE_PATTERN.findall(text_value)
    if not choices:
        return "", str(method)
    return str(choices[-1]), str(choices[0])


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _filter_candidate_rows(
    candidate_rows: pd.DataFrame,
    selector_config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    rows = candidate_rows.copy()
    rows["_method_family"] = _map_method_family(rows["method"])
    descriptions: list[str] = []

    task_allowlist = set(_as_string_list(selector_config.get("task_allowlist")))
    if task_allowlist:
        rows = rows[rows["task"].astype(str).isin(task_allowlist)]
        descriptions.append(f"task_allowlist={','.join(sorted(task_allowlist))}")

    family_allowlist = set(_as_string_list(selector_config.get("method_family_allowlist")))
    if family_allowlist:
        rows = rows[rows["_method_family"].astype(str).isin(family_allowlist)]
        descriptions.append(f"method_family_allowlist={','.join(sorted(family_allowlist))}")

    family_blocklist = set(_as_string_list(selector_config.get("method_family_blocklist")))
    if family_blocklist:
        rows = rows[~rows["_method_family"].astype(str).isin(family_blocklist)]
        descriptions.append(f"method_family_blocklist={','.join(sorted(family_blocklist))}")

    variant_allowlist = set(_as_string_list(selector_config.get("method_variant_allowlist")))
    if variant_allowlist:
        rows = rows[rows["method"].map(_method_variant).isin(variant_allowlist)]
        descriptions.append(f"method_variant_allowlist={','.join(sorted(variant_allowlist))}")

    include_substrings = _as_string_list(selector_config.get("method_name_include_contains"))
    if include_substrings:
        mask = rows["method"].astype(str).apply(
            lambda method: any(substring in method for substring in include_substrings)
        )
        rows = rows[mask]
        descriptions.append(
            f"method_name_include_contains={','.join(sorted(include_substrings))}"
        )

    exclude_substrings = _as_string_list(selector_config.get("method_name_exclude_contains"))
    if exclude_substrings:
        mask = rows["method"].astype(str).apply(
            lambda method: not any(substring in method for substring in exclude_substrings)
        )
        rows = rows[mask]
        descriptions.append(
            f"method_name_exclude_contains={','.join(sorted(exclude_substrings))}"
        )

    rows = rows.drop(columns=["_method_family"])
    if rows.empty:
        raise ValueError(f"Selector `{selector_config['method_name']}` has no candidates after filtering.")
    return rows.reset_index(drop=True), ";".join(descriptions) if descriptions else "none"


def _map_method_family(methods: pd.Series) -> pd.Series:
    """``methods.map(_method_family)``, with the family computed once per distinct method."""
    if methods.isna().any():
        return methods.map(_method_family)
    return methods.map({value: _method_family(value) for value in pd.unique(methods)})


def _with_rank_normalized_validation(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows = candidate_rows.copy()
    rows["method_family"] = _map_method_family(rows["method"])
    grouped = rows.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)
    # Validation ranks are a selector input and keep exact ties. Test ranks feed priors and
    # training objectives, so they follow the endpoint's tie convention (ranking.py).
    rows["val_rank"] = grouped["val_primary_metric"].rank(method="average", ascending=True)
    rows["test_rank"] = grouped_within_unit_ranks(
        rows,
        unit_columns=ARTIFACT_UNIT_COLUMNS,
        value_column="primary_metric",
    )
    group_sizes = grouped["method"].transform("size").astype(float)
    rows["val_rank_norm"] = np.where(
        group_sizes > 1,
        (rows["val_rank"] - 1.0) / (group_sizes - 1.0),
        0.0,
    )
    rows["test_rank_norm"] = np.where(
        group_sizes > 1,
        (rows["test_rank"] - 1.0) / (group_sizes - 1.0),
        0.0,
    )
    log_time = np.log1p(rows["fit_seconds"].astype(float))
    rows["log_fit_seconds"] = log_time
    min_time = grouped["log_fit_seconds"].transform("min")
    max_time = grouped["log_fit_seconds"].transform("max")
    span = max_time - min_time
    rows["fit_time_norm"] = np.where(span > 1e-12, (log_time - min_time) / span, 0.0)
    return rows


def _candidate_budget_seconds(rows: pd.DataFrame, budget_feature: str) -> pd.Series:
    if budget_feature == "fit_seconds":
        return rows["fit_seconds"].astype(float)
    if budget_feature == "predict_seconds":
        return rows["predict_seconds"].astype(float)
    if budget_feature == "fit_plus_predict_seconds":
        return rows["fit_seconds"].astype(float) + rows["predict_seconds"].astype(float)
    raise ValueError(
        "budget_feature must be one of: fit_seconds, predict_seconds, "
        "fit_plus_predict_seconds"
    )


def _apply_candidate_budget(
    ranked: pd.DataFrame,
    *,
    max_budget_seconds: float | None,
    budget_feature: str,
) -> pd.DataFrame:
    rows = ranked.copy()
    rows["budget_feature"] = budget_feature
    rows["budget_seconds"] = _candidate_budget_seconds(rows, budget_feature)
    rows["max_budget_seconds"] = (
        float(max_budget_seconds) if max_budget_seconds is not None else np.inf
    )
    rows["budget_fallback"] = False
    if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds)):
        return rows

    selected_indices: list[int] = []
    fallback_indices: set[int] = set()
    for _, group in rows.groupby(ARTIFACT_UNIT_COLUMNS, sort=False):
        eligible = group[group["budget_seconds"] <= float(max_budget_seconds)]
        if eligible.empty:
            fallback_index = int(group["budget_seconds"].idxmin())
            selected_indices.append(fallback_index)
            fallback_indices.add(fallback_index)
        else:
            selected_indices.extend(int(index) for index in eligible.index)

    filtered = rows.loc[selected_indices].copy()
    filtered["budget_fallback"] = filtered.index.isin(fallback_indices)
    return filtered.reset_index(drop=True)


def _selector_rows_from_utility(
    candidate_rows: pd.DataFrame,
    *,
    method_name: str,
    utility: pd.Series,
    details: pd.Series | str | None = None,
) -> pd.DataFrame:
    rows = candidate_rows.copy()
    utility_series = pd.Series(utility, index=rows.index).astype(float)
    rows["_selector_utility"] = utility_series.to_numpy()
    best_indices = [
        group.index[int(np.argmin(group["_selector_utility"].to_numpy(dtype=float)))]
        for _, group in rows.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)
    ]
    selected = rows.loc[best_indices].copy()
    selected_members = selected["method"].astype(str).to_numpy()
    selected["method"] = method_name
    selected["method_type"] = "selector"
    selected["selected_member"] = selected_members
    if isinstance(details, pd.Series):
        detail_values = details.reindex(selected.index).fillna("").astype(str).to_numpy()
    elif isinstance(details, str):
        detail_values = np.asarray(
            [
                f"utility={float(value):.6g};{details}"
                for value in selected["_selector_utility"].to_numpy(dtype=float)
            ],
            dtype=object,
        )
    else:
        detail_values = np.asarray([""] * len(selected), dtype=object)
    selected["ensemble_weights"] = [
        str(detail) if str(detail) else f"utility={float(value):.6g}"
        for detail, value in zip(
            detail_values,
            selected["_selector_utility"].to_numpy(dtype=float),
        )
    ]
    return selected[ARTIFACT_OUTPUT_COLUMNS].reset_index(drop=True)


def _tabrepo_ensemble_error_and_weights(
    repository: Any,
    *,
    dataset: str,
    fold: int,
    configs: list[str],
    ensemble_size: int,
    backend: str,
) -> tuple[float, dict[str, float]]:
    try:
        output = repository.evaluate_ensemble(
            datasets=[dataset],
            configs=configs,
            ensemble_size=ensemble_size,
            rank=False,
            folds=[fold],
            backend=backend,
        )
    except TypeError:
        output = repository.evaluate_ensemble(
            dataset=dataset,
            fold=fold,
            configs=configs,
            ensemble_size=ensemble_size,
        )

    if isinstance(output, tuple) and len(output) == 2:
        errors, weights = output
    else:
        errors, weights = output, None

    if isinstance(errors, pd.Series):
        try:
            error = float(errors.loc[(dataset, fold)])
        except Exception:
            error = float(errors.iloc[0])
    elif isinstance(errors, pd.DataFrame):
        error = float(errors.to_numpy(dtype=float).ravel()[0])
    elif isinstance(errors, dict):
        error = float(next(iter(errors.values())))
    else:
        error = float(np.asarray(errors, dtype=float).ravel()[0])

    weight_map: dict[str, float] = {}
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        try:
            row = weights.loc[(dataset, fold)]
        except Exception:
            row = weights.iloc[0]
        weight_map = {
            str(config): float(weight)
            for config, weight in row.items()
            if np.isfinite(float(weight)) and float(weight) > 0.0
        }
    elif isinstance(weights, pd.Series):
        weight_map = {
            str(config): float(weight)
            for config, weight in weights.items()
            if np.isfinite(float(weight)) and float(weight) > 0.0
        }
    elif isinstance(weights, dict):
        weight_map = {
            str(config): float(weight)
            for config, weight in weights.items()
            if np.isfinite(float(weight)) and float(weight) > 0.0
        }
    return error, weight_map


def _tabrepo_prediction_ensemble_selector(
    candidate_rows: pd.DataFrame,
    *,
    repository: Any,
    method_name: str,
    top_k: int,
    ensemble_size: int,
    runtime_weight: float,
    max_budget_seconds: float | None,
    budget_feature: str,
    backend: str,
) -> pd.DataFrame:
    ranked = _with_rank_normalized_validation(candidate_rows)
    budgeted = _apply_candidate_budget(
        ranked,
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    budgeted = budgeted.copy()
    budgeted["_ensemble_utility"] = (
        budgeted["val_rank_norm"].astype(float)
        + runtime_weight * budgeted["fit_time_norm"].astype(float)
    )
    rows: list[dict[str, Any]] = []
    grouped_units = list(budgeted.groupby(ARTIFACT_UNIT_COLUMNS, sort=False))
    if len(grouped_units) >= 100:
        print(
            f"TabRepo ensemble selector `{method_name}`: "
            f"{len(grouped_units)} units, top_k={int(top_k)}, ensemble_size={int(ensemble_size)}",
            flush=True,
        )
    for unit_index, (unit_key, group) in enumerate(grouped_units, start=1):
        if len(grouped_units) >= 100 and (
            unit_index == 1 or unit_index % 50 == 0 or unit_index == len(grouped_units)
        ):
            print(
                f"TabRepo ensemble selector `{method_name}` progress: "
                f"{unit_index}/{len(grouped_units)} units",
                flush=True,
            )
        dataset, fold, _seed, _task, _metric_name = unit_key
        selected = group.sort_values(
            ["_ensemble_utility", "val_rank", "fit_seconds", "method"],
            ascending=True,
        ).head(max(1, int(top_k)))
        configs = [str(value) for value in selected["method"].tolist()]
        error, weight_map = _tabrepo_ensemble_error_and_weights(
            repository,
            dataset=str(dataset),
            fold=int(fold),
            configs=configs,
            ensemble_size=int(ensemble_size),
            backend=backend,
        )
        selected_by_method = selected.set_index("method", drop=False)
        if weight_map:
            weighted_configs = [
                config for config in configs if config in weight_map and config in selected_by_method.index
            ]
            weight_values = np.asarray([weight_map[config] for config in weighted_configs])
            val_values = selected_by_method.loc[weighted_configs, "val_primary_metric"].to_numpy(
                dtype=float,
            )
            val_primary = (
                float(np.average(val_values, weights=weight_values))
                if float(weight_values.sum()) > 0.0
                else float(selected["val_primary_metric"].mean())
            )
            weights_text = ";".join(
                f"{config}:{weight_map[config]:.6f}"
                for config in configs
                if config in weight_map
            )
        else:
            val_primary = float(selected["val_primary_metric"].mean())
            weights_text = "unavailable"
        base = selected.iloc[0].to_dict()
        base["method"] = method_name
        base["method_type"] = "selector"
        base["primary_metric"] = error
        base["val_primary_metric"] = val_primary
        base["fit_seconds"] = float(selected["fit_seconds"].astype(float).sum())
        base["predict_seconds"] = float(selected["predict_seconds"].astype(float).sum())
        base["selected_member"] = configs[0]
        base["ensemble_members"] = ";".join(configs)
        base["ensemble_weights"] = (
            f"top_k={int(top_k)};ensemble_size={int(ensemble_size)};"
            f"runtime_weight={float(runtime_weight):.6g};"
            f"budget_feature={budget_feature};"
            f"max_budget_seconds={max_budget_seconds};weights={weights_text}"
        )
        rows.append(base)
    return pd.DataFrame(rows)[ARTIFACT_OUTPUT_COLUMNS]


def _raw_validation_selector(
    candidate_rows: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
) -> pd.DataFrame:
    utilities = candidate_rows["val_primary_metric"].astype(float) + float(
        runtime_weight
    ) * np.log1p(candidate_rows["fit_seconds"].astype(float))
    return _selector_rows_from_utility(
        candidate_rows,
        method_name=selector_method_name,
        utility=pd.Series(utilities, index=candidate_rows.index),
        details=f"lambda={runtime_weight:.6g};scale=raw",
    )


def _validation_rank_selector(
    candidate_rows: pd.DataFrame,
    *,
    method_name: str,
) -> pd.DataFrame:
    ranked = _with_rank_normalized_validation(candidate_rows)
    details = (
        "utility="
        + ranked["val_rank_norm"].map(lambda value: f"{float(value):.6g}")
        + ";lambda=0;scale=val_rank_norm"
    )
    return _selector_rows_from_utility(
        ranked,
        method_name=method_name,
        utility=ranked["val_rank_norm"],
        details=details,
    )


def _objective_for_lambda(
    rows: pd.DataFrame,
    *,
    runtime_weight: float,
    tuning_runtime_weight: float,
) -> float:
    utility = rows["val_rank_norm"] + runtime_weight * rows["fit_time_norm"]
    selected = rows.assign(_utility=utility).loc[
        rows.assign(_utility=utility).groupby(ARTIFACT_UNIT_COLUMNS)["_utility"].idxmin()
    ]
    objective = selected["val_rank_norm"] + tuning_runtime_weight * selected["fit_time_norm"]
    return float(objective.mean())


def _tune_task_conditioned_weights(
    ranked: pd.DataFrame,
    *,
    runtime_weight_grid: list[float],
    tuning_runtime_weight: float,
    leave_one_dataset_out: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_frame in ranked.groupby("task", sort=False):
        datasets = sorted(str(dataset) for dataset in task_frame["dataset"].unique())
        for dataset in datasets:
            if leave_one_dataset_out:
                training = task_frame[task_frame["dataset"] != dataset]
                fallback_used = False
                if training.empty:
                    training = task_frame
                    fallback_used = True
            else:
                training = task_frame
                fallback_used = False
            scores = [
                (
                    runtime_weight,
                    _objective_for_lambda(
                        training,
                        runtime_weight=float(runtime_weight),
                        tuning_runtime_weight=tuning_runtime_weight,
                    ),
                )
                for runtime_weight in runtime_weight_grid
            ]
            best_runtime_weight, best_objective = min(scores, key=lambda item: (item[1], item[0]))
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "runtime_weight": float(best_runtime_weight),
                    "validation_objective": float(best_objective),
                    "tuning_runtime_weight": float(tuning_runtime_weight),
                    "n_training_units": int(training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]),
                    "leave_one_dataset_out": bool(leave_one_dataset_out),
                    "fallback_used": bool(fallback_used),
                }
            )
    return pd.DataFrame(rows)


def _task_conditioned_selector(
    candidate_rows: pd.DataFrame,
    *,
    method_name: str,
    runtime_weight_grid: list[float],
    tuning_runtime_weight: float,
    leave_one_dataset_out: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = _with_rank_normalized_validation(candidate_rows)
    tuning = _tune_task_conditioned_weights(
        ranked,
        runtime_weight_grid=runtime_weight_grid,
        tuning_runtime_weight=tuning_runtime_weight,
        leave_one_dataset_out=leave_one_dataset_out,
    )
    weight_lookup = {
        (str(row["task"]), str(row["dataset"])): float(row["runtime_weight"])
        for _, row in tuning.iterrows()
    }
    weights = ranked.apply(
        lambda row: weight_lookup[(str(row["task"]), str(row["dataset"]))],
        axis=1,
    )
    utility = ranked["val_rank_norm"] + weights * ranked["fit_time_norm"]
    details = (
        "utility="
        + utility.map(lambda value: f"{float(value):.6g}")
        + ";lambda="
        + weights.map(lambda value: f"{float(value):.6g}")
        + ";scale=val_rank_norm;fit_time=log_norm"
    )
    selector = _selector_rows_from_utility(
        ranked,
        method_name=method_name,
        utility=utility,
        details=details,
    )
    return selector, tuning


def _default_calibration_weight_grid() -> list[dict[str, float]]:
    return [
        {
            "validation_weight": 1.0,
            "task_method_prior_weight": 0.0,
            "task_family_prior_weight": 0.0,
            "global_method_prior_weight": 0.0,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.75,
            "task_method_prior_weight": 0.20,
            "task_family_prior_weight": 0.05,
            "global_method_prior_weight": 0.0,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.60,
            "task_method_prior_weight": 0.30,
            "task_family_prior_weight": 0.05,
            "global_method_prior_weight": 0.05,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.45,
            "task_method_prior_weight": 0.35,
            "task_family_prior_weight": 0.10,
            "global_method_prior_weight": 0.10,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.30,
            "task_method_prior_weight": 0.45,
            "task_family_prior_weight": 0.15,
            "global_method_prior_weight": 0.10,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.20,
            "task_method_prior_weight": 0.55,
            "task_family_prior_weight": 0.15,
            "global_method_prior_weight": 0.10,
            "runtime_weight": 0.0,
        },
        {
            "validation_weight": 0.45,
            "task_method_prior_weight": 0.30,
            "task_family_prior_weight": 0.10,
            "global_method_prior_weight": 0.10,
            "runtime_weight": 0.05,
        },
        {
            "validation_weight": 0.30,
            "task_method_prior_weight": 0.40,
            "task_family_prior_weight": 0.15,
            "global_method_prior_weight": 0.10,
            "runtime_weight": 0.05,
        },
    ]


def _calibration_weight_grid(
    configured_grid: list[dict[str, Any]] | None,
) -> list[dict[str, float]]:
    grid = configured_grid or _default_calibration_weight_grid()
    normalized: list[dict[str, float]] = []
    for item in grid:
        weights = {column: float(item.get(column, 0.0)) for column in CALIBRATION_WEIGHT_COLUMNS}
        if not any(weights.values()):
            raise ValueError("Calibrated selector weight candidates cannot all be zero.")
        normalized.append(weights)
    return normalized


def _calibrated_prior_features(
    rows: pd.DataFrame,
    prior_source: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    if prior_source.empty:
        features = rows.copy()
        for column in [
            "task_method_prior_rank_norm",
            "task_family_prior_rank_norm",
            "global_method_prior_rank_norm",
        ]:
            features[column] = 0.5
        return features, True

    features = rows.copy()
    global_mean = float(prior_source["test_rank_norm"].mean())

    task_means = (
        prior_source.groupby("task", as_index=False)["test_rank_norm"]
        .mean()
        .rename(columns={"test_rank_norm": "task_prior_mean"})
    )
    task_method_priors = (
        prior_source.groupby(["task", "method"], as_index=False)["test_rank_norm"]
        .mean()
        .rename(columns={"test_rank_norm": "task_method_prior_rank_norm"})
    )
    task_family_priors = (
        prior_source.groupby(["task", "method_family"], as_index=False)["test_rank_norm"]
        .mean()
        .rename(columns={"test_rank_norm": "task_family_prior_rank_norm"})
    )
    global_method_priors = (
        prior_source.groupby("method", as_index=False)["test_rank_norm"]
        .mean()
        .rename(columns={"test_rank_norm": "global_method_prior_rank_norm"})
    )
    global_family_priors = (
        prior_source.groupby("method_family", as_index=False)["test_rank_norm"]
        .mean()
        .rename(columns={"test_rank_norm": "global_family_prior_rank_norm"})
    )

    features = features.merge(task_means, on="task", how="left")
    features = features.merge(task_method_priors, on=["task", "method"], how="left")
    features = features.merge(task_family_priors, on=["task", "method_family"], how="left")
    features = features.merge(global_method_priors, on="method", how="left")
    features = features.merge(global_family_priors, on="method_family", how="left")

    features["task_prior_mean"] = features["task_prior_mean"].fillna(global_mean)
    missing_task_method = features["task_method_prior_rank_norm"].isna()
    missing_task_family = features["task_family_prior_rank_norm"].isna()
    missing_global_method = features["global_method_prior_rank_norm"].isna()
    fallback_used = bool(
        missing_task_method.any() or missing_task_family.any() or missing_global_method.any()
    )

    features["global_method_prior_rank_norm"] = features[
        "global_method_prior_rank_norm"
    ].fillna(global_mean)
    features["task_method_prior_rank_norm"] = (
        features["task_method_prior_rank_norm"]
        .fillna(features["global_method_prior_rank_norm"])
        .fillna(features["task_prior_mean"])
        .fillna(global_mean)
    )
    features["task_family_prior_rank_norm"] = (
        features["task_family_prior_rank_norm"]
        .fillna(features["global_family_prior_rank_norm"])
        .fillna(features["task_prior_mean"])
        .fillna(global_mean)
    )
    features = features.drop(columns=["task_prior_mean", "global_family_prior_rank_norm"])
    return features, fallback_used


def _calibrated_utility(features: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        weights["validation_weight"] * features["val_rank_norm"]
        + weights["task_method_prior_weight"] * features["task_method_prior_rank_norm"]
        + weights["task_family_prior_weight"] * features["task_family_prior_rank_norm"]
        + weights["global_method_prior_weight"] * features["global_method_prior_rank_norm"]
        + weights["runtime_weight"] * features["fit_time_norm"]
    )


def _calibration_objective(
    features: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    utility = _calibrated_utility(features, weights)
    scored = features.assign(_utility=utility)
    selected = scored.loc[scored.groupby(ARTIFACT_UNIT_COLUMNS)["_utility"].idxmin()]
    return float(selected["test_rank_norm"].mean())


WEIGHT_SELECTION_MODES = ("lodo", "resubstitution")


def _validate_weight_selection(weight_selection: str) -> str:
    if weight_selection not in WEIGHT_SELECTION_MODES:
        raise ValueError(
            f"weight_selection must be one of {WEIGHT_SELECTION_MODES}, got {weight_selection!r}"
        )
    return weight_selection


def _loo_prior_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Priors for every row computed from ``rows`` without that row's own dataset.

    Row-for-row equal (up to floating-point summation order) to
    ``_calibrated_prior_features(rows[d], rows[dataset != d])`` for each dataset ``d``,
    including its fallback chain and the 0.5 value for an empty prior source.
    """
    features = rows.copy()
    if rows.empty:
        for column in [
            "task_method_prior_rank_norm",
            "task_family_prior_rank_norm",
            "global_method_prior_rank_norm",
        ]:
            features[column] = pd.Series(dtype=float)
        return features, False
    values = rows["test_rank_norm"].to_numpy(dtype=float)
    dataset_codes, _ = _factorize_columns(rows, ["dataset"])
    n_datasets = int(dataset_codes.max()) + 1

    def loo_mean(columns: list[str]) -> np.ndarray:
        codes, _ = _factorize_columns(rows, columns)
        totals, counts, sums_by_dataset, counts_by_dataset = _aggregate_code_dataset(
            values,
            codes,
            dataset_codes,
            n_codes=int(codes.max()) + 1,
            n_datasets=n_datasets,
        )
        sums = totals[codes] - sums_by_dataset[codes, dataset_codes]
        remaining = counts[codes] - counts_by_dataset[codes, dataset_codes]
        means = np.full(codes.shape[0], np.nan, dtype=float)
        valid = remaining > 0
        means[valid] = sums[valid] / remaining[valid]
        return means

    dataset_sums = np.bincount(dataset_codes, weights=values, minlength=n_datasets)
    dataset_counts = np.bincount(dataset_codes, minlength=n_datasets)
    source_counts = values.shape[0] - dataset_counts[dataset_codes]
    empty_source = source_counts <= 0
    global_mean = np.full(values.shape[0], np.nan, dtype=float)
    global_mean[~empty_source] = (
        values.sum() - dataset_sums[dataset_codes][~empty_source]
    ) / source_counts[~empty_source]

    task_mean = loo_mean(["task"])
    task_method = loo_mean(["task", "method"])
    task_family = loo_mean(["task", "method_family"])
    global_method = loo_mean(["method"])
    global_family = loo_mean(["method_family"])
    fallback_used = bool(
        np.isnan(task_method[~empty_source]).any()
        or np.isnan(task_family[~empty_source]).any()
        or np.isnan(global_method[~empty_source]).any()
        or empty_source.any()
    )

    task_mean = np.where(np.isfinite(task_mean), task_mean, global_mean)
    global_method = np.where(np.isfinite(global_method), global_method, global_mean)
    task_method = np.where(np.isfinite(task_method), task_method, global_method)
    task_method = np.where(np.isfinite(task_method), task_method, task_mean)
    task_method = np.where(np.isfinite(task_method), task_method, global_mean)
    task_family = np.where(np.isfinite(task_family), task_family, global_family)
    task_family = np.where(np.isfinite(task_family), task_family, task_mean)
    task_family = np.where(np.isfinite(task_family), task_family, global_mean)

    features["task_method_prior_rank_norm"] = np.where(empty_source, 0.5, task_method)
    features["task_family_prior_rank_norm"] = np.where(empty_source, 0.5, task_family)
    features["global_method_prior_rank_norm"] = np.where(empty_source, 0.5, global_method)
    return features, fallback_used


def _select_calibration_weights(
    training: pd.DataFrame,
    weight_grid: list[dict[str, float]],
    *,
    weight_selection: str,
) -> tuple[dict[str, float], float]:
    """Choose the grid tuple with the best objective on ``training``.

    ``"lodo"`` scores each training dataset with priors estimated without it, which is the
    leave-one-dataset-out rule the manuscript states. ``"resubstitution"`` reuses each
    dataset's own test ranks in its prior; it is kept for legacy pipelines and for the
    two attribution runs only.
    """
    _validate_weight_selection(weight_selection)
    if training.empty:
        return weight_grid[0], float("nan")
    if weight_selection == "lodo":
        training_features, _ = _loo_prior_features(training)
    else:
        training_features, _ = _calibrated_prior_features(training, training)
    scored = [
        (_calibration_objective(training_features, weights), index, weights)
        for index, weights in enumerate(weight_grid)
    ]
    objective, _, weights = min(scored, key=lambda item: (item[0], item[1]))
    return weights, float(objective)


def _factorize_columns(rows: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, Any]:
    if len(columns) == 1:
        values = rows[columns[0]].astype(str)
    else:
        values = pd.MultiIndex.from_frame(rows[columns].astype(str))
    codes, uniques = pd.factorize(values, sort=False)
    return codes.astype(np.int64, copy=False), uniques


def _aggregate_code_dataset(
    values: np.ndarray,
    codes: np.ndarray,
    dataset_codes: np.ndarray,
    *,
    n_codes: int,
    n_datasets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sums_by_dataset = np.zeros((n_codes, n_datasets), dtype=float)
    counts_by_dataset = np.zeros((n_codes, n_datasets), dtype=np.int64)
    np.add.at(sums_by_dataset, (codes, dataset_codes), values)
    np.add.at(counts_by_dataset, (codes, dataset_codes), 1)
    return (
        sums_by_dataset.sum(axis=1),
        counts_by_dataset.sum(axis=1),
        sums_by_dataset,
        counts_by_dataset,
    )


def _calibrated_selector_fast_lodo(
    ranked: pd.DataFrame,
    *,
    method_name: str,
    weight_grid: list[dict[str, float]],
    max_budget_seconds: float | None,
    budget_feature: str,
    candidate_filter: str,
    n_unfiltered_candidates: int | None,
    original_candidate_count: int,
    weight_selection: str,
    lodo_targets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_weight_selection(weight_selection)
    rows = ranked.reset_index(drop=True).copy()
    n_rows = int(rows.shape[0])
    dataset_codes, dataset_uniques = _factorize_columns(rows, ["dataset"])
    unit_codes, _ = _factorize_columns(rows, ARTIFACT_UNIT_COLUMNS)
    task_codes, _ = _factorize_columns(rows, ["task"])
    method_codes, _ = _factorize_columns(rows, ["method"])
    family_codes, _ = _factorize_columns(rows, ["method_family"])
    task_method_codes, _ = _factorize_columns(rows, ["task", "method"])
    task_family_codes, _ = _factorize_columns(rows, ["task", "method_family"])

    dataset_names = [str(value) for value in dataset_uniques]
    dataset_code_lookup = {name: code for code, name in enumerate(dataset_names)}
    # Rows are built for every dataset, or only for ``lodo_targets``; the priors and the
    # weight objective still use every dataset except the one being scored.
    dataset_order = sorted(
        name for name in dataset_names if lodo_targets is None or name in lodo_targets
    )
    verbose = lodo_targets is None
    n_datasets = len(dataset_names)
    n_units = int(unit_codes.max()) + 1 if n_rows else 0
    unit_dataset_codes = np.full(n_units, -1, dtype=np.int64)
    unit_dataset_codes[unit_codes] = dataset_codes

    test_rank_norm = rows["test_rank_norm"].to_numpy(dtype=float)
    val_rank_norm = rows["val_rank_norm"].to_numpy(dtype=float)
    fit_time_norm = rows["fit_time_norm"].to_numpy(dtype=float)
    dataset_sums = np.bincount(
        dataset_codes,
        weights=test_rank_norm,
        minlength=n_datasets,
    ).astype(float)
    dataset_counts = np.bincount(dataset_codes, minlength=n_datasets).astype(np.int64)
    global_sum = float(test_rank_norm.sum())
    global_count = int(test_rank_norm.shape[0])

    aggregates = {
        "task": _aggregate_code_dataset(
            test_rank_norm,
            task_codes,
            dataset_codes,
            n_codes=int(task_codes.max()) + 1,
            n_datasets=n_datasets,
        ),
        "method": _aggregate_code_dataset(
            test_rank_norm,
            method_codes,
            dataset_codes,
            n_codes=int(method_codes.max()) + 1,
            n_datasets=n_datasets,
        ),
        "family": _aggregate_code_dataset(
            test_rank_norm,
            family_codes,
            dataset_codes,
            n_codes=int(family_codes.max()) + 1,
            n_datasets=n_datasets,
        ),
        "task_method": _aggregate_code_dataset(
            test_rank_norm,
            task_method_codes,
            dataset_codes,
            n_codes=int(task_method_codes.max()) + 1,
            n_datasets=n_datasets,
        ),
        "task_family": _aggregate_code_dataset(
            test_rank_norm,
            task_family_codes,
            dataset_codes,
            n_codes=int(task_family_codes.max()) + 1,
            n_datasets=n_datasets,
        ),
    }

    group_codes = {
        "task": task_codes,
        "method": method_codes,
        "family": family_codes,
        "task_method": task_method_codes,
        "task_family": task_family_codes,
    }
    # A row's priors depend on it only through its task-method code (task, method, family
    # and task-family are functions of it) and its dataset, so the priors of the weight
    # objective are computed once per (task-method, dataset) pair, with the same arithmetic
    # as per row, and gathered to the rows.
    pair_keys = task_method_codes * np.int64(n_datasets) + dataset_codes
    _, pair_first_row, row_pair = np.unique(pair_keys, return_index=True, return_inverse=True)
    row_pair = np.asarray(row_pair, dtype=np.int64).reshape(-1)
    pair_dataset_codes = dataset_codes[pair_first_row]
    pair_group_codes = {name: codes[pair_first_row] for name, codes in group_codes.items()}
    # Aggregates with each pair's own dataset removed, reused by every target's objective.
    own_removed = {
        name: (
            aggregates[name][0][codes] - aggregates[name][2][codes, pair_dataset_codes],
            aggregates[name][1][codes] - aggregates[name][3][codes, pair_dataset_codes],
        )
        for name, codes in pair_group_codes.items()
    }
    own_removed_global_sum = global_sum - dataset_sums[pair_dataset_codes]
    own_removed_global_count = global_count - dataset_counts[pair_dataset_codes]
    # Rows sorted by unit (stably, so rows keep their order within a unit) for the
    # per-unit choice of the objective.
    unit_order = np.argsort(unit_codes, kind="stable")
    sorted_unit_codes = unit_codes[unit_order]
    unit_starts = np.flatnonzero(np.r_[True, sorted_unit_codes[1:] != sorted_unit_codes[:-1]])
    unit_sizes = np.diff(np.r_[unit_starts, n_rows])
    sorted_dataset_codes = dataset_codes[unit_order]
    sorted_row_pair = row_pair[unit_order]
    sorted_val_rank_norm = val_rank_norm[unit_order]
    sorted_fit_time_norm = fit_time_norm[unit_order]
    finite_features = {
        "validation_weight": bool(np.isfinite(val_rank_norm).all()),
        "runtime_weight": bool(np.isfinite(fit_time_norm).all()),
    }

    def _safe_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        means = np.full(np.shape(sums), np.nan, dtype=float)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        return means

    def _fill_prior_chain(
        *,
        global_mean: np.ndarray | float,
        task_mean: np.ndarray,
        task_method: np.ndarray,
        task_family: np.ndarray,
        global_method: np.ndarray,
        global_family: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Same order as `_calibrated_prior_features`: global method -> global mean; task
        # method -> global method -> task mean -> global mean; task family -> global
        # family -> task mean -> global mean.
        task_mean = np.where(np.isfinite(task_mean), task_mean, global_mean)
        global_method = np.where(np.isfinite(global_method), global_method, global_mean)
        task_method = np.where(np.isfinite(task_method), task_method, global_method)
        task_method = np.where(np.isfinite(task_method), task_method, task_mean)
        task_method = np.where(np.isfinite(task_method), task_method, global_mean)
        task_family = np.where(np.isfinite(task_family), task_family, global_family)
        task_family = np.where(np.isfinite(task_family), task_family, task_mean)
        task_family = np.where(np.isfinite(task_family), task_family, global_mean)
        return task_method, task_family, global_method

    def code_means_excluding(name: str, dataset_code: int) -> np.ndarray:
        """Mean of each group without one dataset, per group code (NaN for an empty group)."""
        total_sums, total_counts, sums_by_dataset, counts_by_dataset = aggregates[name]
        sums = total_sums - sums_by_dataset[:, dataset_code]
        counts = total_counts - counts_by_dataset[:, dataset_code]
        means = np.full(sums.shape[0], np.nan, dtype=float)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        return means

    def priors_for_dataset(
        dataset_code: int,
        row_indices: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Priors from every dataset except the target, for ``row_indices`` (all rows if None)."""
        training_sum = global_sum - float(dataset_sums[dataset_code])
        training_count = global_count - int(dataset_counts[dataset_code])
        global_mean = training_sum / training_count if training_count > 0 else float("nan")
        code_means = {name: code_means_excluding(name, dataset_code) for name in group_codes}
        # Every code occurs in some row, so a code-level NaN is a row-level NaN.
        fallback_used = bool(
            np.isnan(code_means["task_method"]).any()
            or np.isnan(code_means["task_family"]).any()
            or np.isnan(code_means["method"]).any()
        )
        means = {
            name: code_means[name][codes if row_indices is None else codes[row_indices]]
            for name, codes in group_codes.items()
        }
        task_method_prior, task_family_prior, global_method_prior = _fill_prior_chain(
            global_mean=global_mean,
            task_mean=means["task"],
            task_method=means["task_method"],
            task_family=means["task_family"],
            global_method=means["method"],
            global_family=means["family"],
        )
        if training_count <= 0:
            size = n_rows if row_indices is None else int(row_indices.shape[0])
            task_method_prior = np.full(size, 0.5)
            task_family_prior = np.full(size, 0.5)
            global_method_prior = np.full(size, 0.5)
            fallback_used = True
        return task_method_prior, task_family_prior, global_method_prior, fallback_used

    def objective_priors_for_dataset(
        dataset_code: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Priors for the weight objective on non-target rows, in unit order.

        Under leave-one-dataset-out weight selection each non-target row is scored with
        priors that exclude both the target and the row's own dataset; under
        resubstitution only the target is excluded (the v1 behaviour).
        """
        if weight_selection == "resubstitution":
            task_method_prior, task_family_prior, global_method_prior, _ = priors_for_dataset(
                dataset_code, None
            )
            return (
                task_method_prior[unit_order],
                task_family_prior[unit_order],
                global_method_prior[unit_order],
            )
        means = {}
        for name, codes in pair_group_codes.items():
            sums = own_removed[name][0] - aggregates[name][2][codes, dataset_code]
            counts = own_removed[name][1] - aggregates[name][3][codes, dataset_code]
            means[name] = _safe_means(sums, counts)
        source_count = own_removed_global_count - int(dataset_counts[dataset_code])
        global_mean = _safe_means(
            own_removed_global_sum - float(dataset_sums[dataset_code]),
            source_count,
        )
        task_method_prior, task_family_prior, global_method_prior = _fill_prior_chain(
            global_mean=global_mean,
            task_mean=means["task"],
            task_method=means["task_method"],
            task_family=means["task_family"],
            global_method=means["method"],
            global_family=means["family"],
        )
        empty_source = source_count <= 0
        return (
            np.where(empty_source, 0.5, task_method_prior)[sorted_row_pair],
            np.where(empty_source, 0.5, task_family_prior)[sorted_row_pair],
            np.where(empty_source, 0.5, global_method_prior)[sorted_row_pair],
        )

    def first_index_per_unit(ordered: np.ndarray, valid_units: np.ndarray) -> np.ndarray:
        """Lowest row index at each valid unit's minimum utility (utilities in unit order)."""
        minima = np.minimum.reduceat(ordered, unit_starts)
        at_minimum = np.flatnonzero(ordered == np.repeat(minima, unit_sizes))
        units = sorted_unit_codes[at_minimum]
        first = np.r_[True, units[1:] != units[:-1]] if units.size else np.zeros(0, dtype=bool)
        first_by_unit = np.full(n_units, n_rows, dtype=np.int64)
        # The stable sort keeps row order within a unit, so the first row at the minimum
        # in sorted order is the lowest row index at the minimum.
        first_by_unit[units[first]] = unit_order[at_minimum[first]]
        selected = first_by_unit[valid_units]
        return selected[selected < n_rows]

    def objective_for_weights(
        *,
        weights: dict[str, float],
        features: list[tuple[str, np.ndarray, bool]],
        sorted_target_indices: np.ndarray,
        valid_units: np.ndarray,
    ) -> float:
        # The terms are added in the order of the utility formula. A zero-weight term of
        # finite values adds a zero to every utility and changes no comparison, so it is
        # skipped.
        utility = None
        for key, values, finite in features:
            weight = weights[key]
            if weight == 0.0 and finite:
                continue
            term = weight * values
            utility = term if utility is None else utility + term
        if utility is None:
            utility = np.zeros(n_rows, dtype=float)
        utility[sorted_target_indices] = np.inf
        selected = first_index_per_unit(utility, valid_units)
        return float(np.mean(test_rank_norm[selected])) if selected.size else float("nan")

    selector_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    if verbose:
        print(
            f"Fast LODO calibrated selector `{method_name}`: "
            f"{n_rows} candidate rows across {n_datasets} datasets",
            flush=True,
        )
    for dataset_index, dataset in enumerate(dataset_order, start=1):
        if verbose and (dataset_index == 1 or dataset_index % 25 == 0 or dataset_index == len(dataset_order)):
            print(
                f"Fast LODO calibrated selector `{method_name}` progress: "
                f"{dataset_index}/{len(dataset_order)} datasets",
                flush=True,
            )
        dataset_code = dataset_code_lookup[dataset]
        target_mask = dataset_codes == dataset_code
        target_indices = np.flatnonzero(target_mask)
        (
            target_task_method_prior,
            target_task_family_prior,
            target_global_method_prior,
            prior_fallback_used,
        ) = priors_for_dataset(dataset_code, target_indices)

        if len(weight_grid) == 1:
            weights = weight_grid[0]
            calibration_objective = float("nan")
        else:
            objective_task_method, objective_task_family, objective_global_method = (
                objective_priors_for_dataset(dataset_code)
            )
            features = [
                ("validation_weight", sorted_val_rank_norm, finite_features["validation_weight"]),
                ("task_method_prior_weight", objective_task_method, bool(np.isfinite(objective_task_method).all())),
                ("task_family_prior_weight", objective_task_family, bool(np.isfinite(objective_task_family).all())),
                ("global_method_prior_weight", objective_global_method, bool(np.isfinite(objective_global_method).all())),
                ("runtime_weight", sorted_fit_time_norm, finite_features["runtime_weight"]),
            ]
            sorted_target_indices = np.flatnonzero(sorted_dataset_codes == dataset_code)
            valid_units = unit_dataset_codes != dataset_code
            scored_weights = [
                (
                    objective_for_weights(
                        weights=weights,
                        features=features,
                        sorted_target_indices=sorted_target_indices,
                        valid_units=valid_units,
                    ),
                    index,
                    weights,
                )
                for index, weights in enumerate(weight_grid)
            ]
            calibration_objective, _, weights = min(
                scored_weights,
                key=lambda item: (item[0], item[1]),
            )

        target = rows.iloc[target_indices].copy()
        utility = (
            weights["validation_weight"] * target["val_rank_norm"].to_numpy(dtype=float)
            + weights["task_method_prior_weight"] * target_task_method_prior
            + weights["task_family_prior_weight"] * target_task_family_prior
            + weights["global_method_prior_weight"] * target_global_method_prior
            + weights["runtime_weight"] * target["fit_time_norm"].to_numpy(dtype=float)
        )
        max_budget_label = (
            "inf"
            if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
            else f"{float(max_budget_seconds):.6g}"
        )
        details = pd.Series(
            [
                "utility="
                + f"{float(value):.6g}"
                + f";w_val={weights['validation_weight']:.6g}"
                + f";w_task_method={weights['task_method_prior_weight']:.6g}"
                + f";w_task_family={weights['task_family_prior_weight']:.6g}"
                + f";w_global_method={weights['global_method_prior_weight']:.6g}"
                + f";w_runtime={weights['runtime_weight']:.6g}"
                + f";budget_feature={budget_feature}"
                + f";budget_seconds={float(budget_seconds):.6g}"
                + f";max_budget_seconds={max_budget_label}"
                + f";budget_fallback={int(bool(budget_fallback))}"
                + ";scale=fast_calibrated_lodo"
                for value, budget_seconds, budget_fallback in zip(
                    utility,
                    target["budget_seconds"].to_numpy(dtype=float),
                    target["budget_fallback"].to_numpy(dtype=bool),
                )
            ],
            index=target.index,
        )
        selector_frames.append(
            _selector_rows_from_utility(
                target,
                method_name=method_name,
                utility=pd.Series(utility, index=target.index),
                details=details,
            )
        )
        valid_training_units = unit_dataset_codes != dataset_code
        target_unit_count = int(np.unique(unit_codes[target_indices]).shape[0])
        tuning_rows.append(
            {
                "task": str(target["task"].iloc[0]),
                "dataset": dataset,
                "calibration_objective": calibration_objective,
                "n_training_units": int(valid_training_units.sum()),
                "n_training_datasets": int(n_datasets - 1),
                "budget_feature": budget_feature,
                "max_budget_seconds": (
                    np.inf
                    if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                    else float(max_budget_seconds)
                ),
                "n_target_units": target_unit_count,
                "n_target_candidates": int(target.shape[0]),
                "n_target_budget_fallback_units": int(
                    target.loc[target["budget_fallback"], ARTIFACT_UNIT_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_training_candidates": int(n_rows - target.shape[0]),
                "candidate_filter": candidate_filter,
                "n_unfiltered_candidates": (
                    int(n_unfiltered_candidates)
                    if n_unfiltered_candidates is not None
                    else int(original_candidate_count)
                ),
                "n_filtered_candidates": int(original_candidate_count),
                "leave_one_dataset_out": True,
                "fallback_used": bool(prior_fallback_used),
                "n_weight_candidates": int(len(weight_grid)),
                **weights,
            }
        )

    selector = pd.concat(selector_frames, ignore_index=True)
    return selector, pd.DataFrame(tuning_rows)


def _calibrated_selector(
    candidate_rows: pd.DataFrame,
    *,
    method_name: str,
    weight_grid: list[dict[str, float]],
    leave_one_dataset_out: bool,
    weight_selection: str,
    max_budget_seconds: float | None = None,
    budget_feature: str = "fit_seconds",
    candidate_filter: str = "none",
    n_unfiltered_candidates: int | None = None,
    fast_path_min_rows: int = 100_000,
    lodo_targets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-dataset-out rows of a calibrated selector (one row per unit).

    ``lodo_targets`` restricts the datasets that get rows; every other dataset still
    contributes to the priors and the weight objective of the targets.
    """
    _validate_weight_selection(weight_selection)
    if lodo_targets is not None and not leave_one_dataset_out:
        raise ValueError("lodo_targets needs leave-one-dataset-out rows.")
    ranked = _apply_candidate_budget(
        _with_rank_normalized_validation(candidate_rows),
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    if (
        leave_one_dataset_out
        and ranked.shape[0] >= int(fast_path_min_rows)
        and ranked["dataset"].nunique() > 1
    ):
        return _calibrated_selector_fast_lodo(
            ranked,
            method_name=method_name,
            weight_grid=weight_grid,
            max_budget_seconds=max_budget_seconds,
            budget_feature=budget_feature,
            candidate_filter=candidate_filter,
            n_unfiltered_candidates=n_unfiltered_candidates,
            original_candidate_count=int(candidate_rows.shape[0]),
            weight_selection=weight_selection,
            lodo_targets=lodo_targets,
        )
    scored_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    for dataset in sorted(str(value) for value in ranked["dataset"].unique()):
        if lodo_targets is not None and dataset not in lodo_targets:
            continue
        target = ranked[ranked["dataset"].astype(str) == dataset].copy()
        if leave_one_dataset_out:
            training = ranked[ranked["dataset"].astype(str) != dataset].copy()
            fallback_used = False
            if training.empty:
                training = ranked.copy()
                fallback_used = True
        else:
            training = ranked.copy()
            fallback_used = False

        if len(weight_grid) == 1:
            weights = weight_grid[0]
            calibration_objective = float("nan")
        else:
            weights, calibration_objective = _select_calibration_weights(
                training,
                weight_grid,
                weight_selection=weight_selection,
            )
        features, prior_fallback_used = _calibrated_prior_features(target, training)
        utility = _calibrated_utility(features, weights)
        features["calibrated_utility"] = utility
        features["selector_details"] = (
            "utility="
            + utility.map(lambda value: f"{float(value):.6g}")
            + ";w_val="
            + f"{weights['validation_weight']:.6g}"
            + ";w_task_method="
            + f"{weights['task_method_prior_weight']:.6g}"
            + ";w_task_family="
            + f"{weights['task_family_prior_weight']:.6g}"
            + ";w_global_method="
            + f"{weights['global_method_prior_weight']:.6g}"
            + ";w_runtime="
            + f"{weights['runtime_weight']:.6g}"
            + ";task_method_prior="
            + features["task_method_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";task_family_prior="
            + features["task_family_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";global_method_prior="
            + features["global_method_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";budget_feature="
            + budget_feature
            + ";budget_seconds="
            + features["budget_seconds"].map(lambda value: f"{float(value):.6g}")
            + ";max_budget_seconds="
            + (
                "inf"
                if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                else f"{float(max_budget_seconds):.6g}"
            )
            + ";budget_fallback="
            + features["budget_fallback"].map(lambda value: str(int(bool(value))))
            + ";scale="
            + (
                "calibrated_lodo"
                if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                else "calibrated_lodo_budgeted"
            )
        )
        scored_frames.append(features)
        tuning_rows.append(
            {
                "task": str(target["task"].iloc[0]),
                "dataset": dataset,
                "calibration_objective": calibration_objective,
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "budget_feature": budget_feature,
                "max_budget_seconds": (
                    np.inf
                    if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                    else float(max_budget_seconds)
                ),
                "n_target_units": int(
                    target[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_target_candidates": int(target.shape[0]),
                "n_target_budget_fallback_units": int(
                    target.loc[target["budget_fallback"], ARTIFACT_UNIT_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_training_candidates": int(training.shape[0]),
                "candidate_filter": candidate_filter,
                "n_unfiltered_candidates": (
                    int(n_unfiltered_candidates)
                    if n_unfiltered_candidates is not None
                    else int(candidate_rows.shape[0])
                ),
                "n_filtered_candidates": int(candidate_rows.shape[0]),
                "leave_one_dataset_out": bool(leave_one_dataset_out),
                "fallback_used": bool(fallback_used or prior_fallback_used),
                "n_weight_candidates": int(len(weight_grid)),
                **weights,
            }
        )

    scored = pd.concat(scored_frames, ignore_index=True)
    selector = _selector_rows_from_utility(
        scored,
        method_name=method_name,
        utility=scored["calibrated_utility"],
        details=scored["selector_details"],
    )
    return selector, pd.DataFrame(tuning_rows)


def _validation_confidence_prior_selector(
    candidate_rows: pd.DataFrame,
    *,
    method_name: str,
    weight_grid: list[dict[str, float]],
    leave_one_dataset_out: bool,
    validation_rank_margin: float,
    validation_rank_norm_margin: float | None = None,
    max_budget_seconds: float | None = None,
    budget_feature: str = "fit_seconds",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = _apply_candidate_budget(
        _with_rank_normalized_validation(candidate_rows),
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    selector_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    for dataset in sorted(str(value) for value in ranked["dataset"].unique()):
        target = ranked[ranked["dataset"].astype(str) == dataset].copy()
        if leave_one_dataset_out:
            training = ranked[ranked["dataset"].astype(str) != dataset].copy()
            fallback_used = False
            if training.empty:
                training = ranked.copy()
                fallback_used = True
        else:
            training = ranked.copy()
            fallback_used = False

        weights, calibration_objective = _select_calibration_weights(
            training,
            weight_grid,
            weight_selection="resubstitution",
        )
        features, prior_fallback_used = _calibrated_prior_features(target, training)
        unit_min_rank = features.groupby(ARTIFACT_UNIT_COLUMNS)["val_rank"].transform("min")
        confidence_mask = features["val_rank"].astype(float) <= (
            unit_min_rank.astype(float) + float(validation_rank_margin)
        )
        if validation_rank_norm_margin is not None:
            unit_min_rank_norm = features.groupby(ARTIFACT_UNIT_COLUMNS)[
                "val_rank_norm"
            ].transform("min")
            confidence_mask = confidence_mask | (
                features["val_rank_norm"].astype(float)
                <= unit_min_rank_norm.astype(float) + float(validation_rank_norm_margin)
            )
        eligible = features[confidence_mask].copy()
        if eligible.empty:
            eligible = features.copy()
            fallback_used = True

        unit_sizes = eligible.groupby(ARTIFACT_UNIT_COLUMNS)["method"].size()
        utility = _calibrated_utility(eligible, weights)
        eligible["confidence_prior_utility"] = utility
        eligible["selector_details"] = (
            "utility="
            + utility.map(lambda value: f"{float(value):.6g}")
            + ";validation_rank_margin="
            + f"{float(validation_rank_margin):.6g}"
            + ";validation_rank_norm_margin="
            + (
                "none"
                if validation_rank_norm_margin is None
                else f"{float(validation_rank_norm_margin):.6g}"
            )
            + ";w_val="
            + f"{weights['validation_weight']:.6g}"
            + ";w_task_method="
            + f"{weights['task_method_prior_weight']:.6g}"
            + ";w_task_family="
            + f"{weights['task_family_prior_weight']:.6g}"
            + ";w_global_method="
            + f"{weights['global_method_prior_weight']:.6g}"
            + ";w_runtime="
            + f"{weights['runtime_weight']:.6g}"
            + ";budget_feature="
            + budget_feature
            + ";max_budget_seconds="
            + (
                "inf"
                if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                else f"{float(max_budget_seconds):.6g}"
            )
            + ";scale=validation_confidence_prior"
        )
        selector_frames.append(
            _selector_rows_from_utility(
                eligible,
                method_name=method_name,
                utility=eligible["confidence_prior_utility"],
                details=eligible["selector_details"],
            )
        )
        tuning_rows.append(
            {
                "task": str(target["task"].iloc[0]),
                "dataset": dataset,
                "calibration_objective": calibration_objective,
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": bool(leave_one_dataset_out),
                "fallback_used": bool(fallback_used or prior_fallback_used),
                "n_weight_candidates": int(len(weight_grid)),
                **weights,
                "validation_rank_margin": float(validation_rank_margin),
                "validation_rank_norm_margin": (
                    np.nan
                    if validation_rank_norm_margin is None
                    else float(validation_rank_norm_margin)
                ),
                "confidence_set_mean_size": float(unit_sizes.mean()),
                "confidence_set_min_size": int(unit_sizes.min()),
                "confidence_set_max_size": int(unit_sizes.max()),
                "confidence_budget_feature": budget_feature,
                "confidence_max_budget_seconds": (
                    np.inf
                    if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                    else float(max_budget_seconds)
                ),
                "confidence_budget_fallback_units": int(
                    target.loc[target["budget_fallback"], ARTIFACT_UNIT_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )
    return pd.concat(selector_frames, ignore_index=True), pd.DataFrame(tuning_rows)


def _task_adaptive_selector_from_selectors(
    candidate_rows: pd.DataFrame,
    selector_rows: pd.DataFrame,
    *,
    method_name: str,
    candidate_selector_methods: list[str],
    choice_scope: str,
    leave_one_dataset_out: bool,
    lodo_targets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_selector_methods:
        raise ValueError(f"Selector `{method_name}` requires candidate_selector_methods.")
    if choice_scope not in {"task", "global"}:
        raise ValueError("adaptive choice_scope must be one of: task, global")

    ranked_candidates = _with_rank_normalized_validation(candidate_rows)
    rank_lookup = ranked_candidates[
        [*ARTIFACT_UNIT_COLUMNS, "method", "test_rank_norm"]
    ].rename(
        columns={
            "method": "selected_member",
            "test_rank_norm": "selected_member_test_rank_norm",
        }
    )
    method_order = {name: index for index, name in enumerate(candidate_selector_methods)}
    selector_pool = selector_rows[
        selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    available_methods = set(str(value) for value in selector_pool["method"].unique())
    missing_methods = [name for name in candidate_selector_methods if name not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Selector `{method_name}` references unavailable selector methods: {missing_methods}"
        )

    scored_pool = selector_pool.merge(
        rank_lookup,
        on=[*ARTIFACT_UNIT_COLUMNS, "selected_member"],
        how="left",
    )
    if scored_pool["selected_member_test_rank_norm"].isna().any():
        missing = scored_pool[scored_pool["selected_member_test_rank_norm"].isna()]
        example = missing[[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"]].head(1)
        raise ValueError(
            "Adaptive selector could not map selected members to candidate ranks: "
            f"{example.to_dict(orient='records')}"
        )
    scored_pool["selected_member_total_seconds"] = (
        scored_pool["fit_seconds"].astype(float)
        + scored_pool["predict_seconds"].astype(float)
    )

    selector_outputs: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    targets = (
        candidate_rows[["dataset", "task"]]
        .drop_duplicates()
        .sort_values(["dataset", "task"])
    )
    if lodo_targets is not None:
        targets = targets[targets["dataset"].astype(str).isin(lodo_targets)]
    for _, target_key in targets.iterrows():
        dataset = str(target_key["dataset"])
        task = str(target_key["task"])
        training = scored_pool.copy()
        fallback_used = False
        if leave_one_dataset_out:
            training = training[training["dataset"].astype(str) != dataset]
        if choice_scope == "task":
            scoped = training[training["task"].astype(str) == task]
            if scoped.empty:
                fallback_used = True
            else:
                training = scoped
        if training.empty:
            training = scored_pool.copy()
            fallback_used = True

        objectives = (
            training.groupby("method", as_index=False)
            .agg(
                adaptive_objective=("selected_member_test_rank_norm", "mean"),
                adaptive_mean_total_seconds=("selected_member_total_seconds", "mean"),
                n_training_rows=("selected_member_test_rank_norm", "size"),
            )
            .assign(method_order=lambda frame: frame["method"].map(method_order))
            .sort_values(
                ["adaptive_objective", "adaptive_mean_total_seconds", "method_order"],
                ascending=True,
            )
        )
        selected = objectives.iloc[0]
        selected_method = str(selected["method"])
        target_rows = selector_pool[
            (selector_pool["dataset"].astype(str) == dataset)
            & (selector_pool["task"].astype(str) == task)
            & (selector_pool["method"].astype(str) == selected_method)
        ].copy()
        if target_rows.empty:
            raise ValueError(
                f"Adaptive selector `{method_name}` found no target rows for "
                f"dataset={dataset}, task={task}, selected_method={selected_method}."
            )

        adaptive_suffix = (
            f";adaptive_choice={selected_method}"
            f";adaptive_objective={float(selected['adaptive_objective']):.6g}"
            f";adaptive_choice_scope={choice_scope}"
            f";adaptive_lodo={int(leave_one_dataset_out)}"
        )
        target_rows["method"] = method_name
        target_rows["ensemble_weights"] = (
            target_rows["ensemble_weights"].fillna("").astype(str) + adaptive_suffix
        ).str.lstrip(";")
        selector_outputs.append(target_rows[ARTIFACT_OUTPUT_COLUMNS])

        tuning_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "candidate_selector_methods": ",".join(candidate_selector_methods),
                "selected_selector_method": selected_method,
                "adaptive_objective": float(selected["adaptive_objective"]),
                "adaptive_mean_total_seconds": float(selected["adaptive_mean_total_seconds"]),
                "adaptive_choice_scope": choice_scope,
                "n_candidate_selectors": int(len(candidate_selector_methods)),
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": bool(leave_one_dataset_out),
                "fallback_used": bool(fallback_used),
            }
        )

    selector = pd.concat(selector_outputs, ignore_index=True)
    return selector, pd.DataFrame(tuning_rows)


def _selector_pool_with_member_scores(
    candidate_rows: pd.DataFrame,
    selector_rows: pd.DataFrame,
    *,
    selector_methods: list[str],
    selector_name: str,
) -> pd.DataFrame:
    ranked_candidates = _with_rank_normalized_validation(candidate_rows)
    rank_lookup = ranked_candidates[
        [*ARTIFACT_UNIT_COLUMNS, "method", "test_rank", "test_rank_norm"]
    ].rename(
        columns={
            "method": "selected_member",
            "test_rank": "selected_member_test_rank",
            "test_rank_norm": "selected_member_test_rank_norm",
        }
    )
    selector_pool = selector_rows[
        selector_rows["method"].astype(str).isin(selector_methods)
    ].copy()
    available_methods = set(str(value) for value in selector_pool["method"].unique())
    missing_methods = [name for name in selector_methods if name not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Selector `{selector_name}` references unavailable selector methods: "
            f"{missing_methods}"
        )

    scored_pool = selector_pool.merge(
        rank_lookup,
        on=[*ARTIFACT_UNIT_COLUMNS, "selected_member"],
        how="left",
    )
    if scored_pool["selected_member_test_rank"].isna().any():
        missing = scored_pool[scored_pool["selected_member_test_rank"].isna()]
        example = missing[[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"]].head(1)
        raise ValueError(
            f"Selector `{selector_name}` could not map selected members to candidate ranks: "
            f"{example.to_dict(orient='records')}"
        )
    scored_pool["selected_member_total_seconds"] = (
        scored_pool["fit_seconds"].astype(float)
        + scored_pool["predict_seconds"].astype(float)
    )
    return scored_pool


def _selector_dataset_scores(scored_pool: pd.DataFrame) -> pd.DataFrame:
    return (
        scored_pool.groupby(["dataset", "task", "method"], as_index=False)
        .agg(
            mean_rank=("selected_member_test_rank", "mean"),
            mean_rank_norm=("selected_member_test_rank_norm", "mean"),
            mean_total_seconds=("selected_member_total_seconds", "mean"),
            n_units=("selected_member_test_rank", "size"),
        )
        .sort_values(["dataset", "method"])
    )


def _paired_selector_difference(
    dataset_scores: pd.DataFrame,
    *,
    method: str,
    reference: str,
    value_column: str,
) -> pd.Series:
    left = dataset_scores[dataset_scores["method"].astype(str).eq(method)][
        ["dataset", "task", value_column]
    ].rename(columns={value_column: "method_value"})
    right = dataset_scores[dataset_scores["method"].astype(str).eq(reference)][
        ["dataset", "task", value_column]
    ].rename(columns={value_column: "reference_value"})
    paired = left.merge(right, on=["dataset", "task"], how="inner")
    return paired["method_value"].astype(float) - paired["reference_value"].astype(float)


def _upper_one_sided_ci(values: pd.Series, *, alpha: float) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("inf")
    mean = float(clean.mean())
    if len(clean) <= 1:
        return mean
    sem = float(stats.sem(clean))
    if sem <= 0.0 or not np.isfinite(sem):
        return mean
    return mean + float(stats.t.ppf(1.0 - float(alpha), df=len(clean) - 1)) * sem


def _risk_constrained_selector_from_selectors(
    candidate_rows: pd.DataFrame,
    selector_rows: pd.DataFrame,
    *,
    method_name: str,
    candidate_selector_methods: list[str],
    task_reference_method: str,
    family_reference_method: str,
    fallback_selector_methods: list[str],
    choice_scope: str,
    leave_one_dataset_out: bool,
    noninferiority_margin_rank: float,
    noninferiority_alpha: float,
    min_training_datasets: int,
    require_runtime_gain: bool,
    min_runtime_gain_seconds: float,
    runtime_tradeoff_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_selector_methods:
        raise ValueError(f"Selector `{method_name}` requires candidate_selector_methods.")
    if choice_scope not in {"task", "global"}:
        raise ValueError("risk-constrained choice_scope must be one of: task, global")
    if not fallback_selector_methods:
        fallback_selector_methods = [task_reference_method, family_reference_method]

    required_methods = list(
        dict.fromkeys(
            [
                *candidate_selector_methods,
                task_reference_method,
                family_reference_method,
                *fallback_selector_methods,
            ]
        )
    )
    method_order = {name: index for index, name in enumerate(required_methods)}
    scored_pool = _selector_pool_with_member_scores(
        candidate_rows,
        selector_rows,
        selector_methods=required_methods,
        selector_name=method_name,
    )
    dataset_scores_all = _selector_dataset_scores(scored_pool)
    selector_pool = selector_rows[
        selector_rows["method"].astype(str).isin(required_methods)
    ].copy()

    selector_outputs: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    targets = (
        candidate_rows[["dataset", "task"]]
        .drop_duplicates()
        .sort_values(["dataset", "task"])
    )
    for _, target_key in targets.iterrows():
        dataset = str(target_key["dataset"])
        task = str(target_key["task"])
        training = dataset_scores_all.copy()
        fallback_scope_used = False
        if leave_one_dataset_out:
            training = training[training["dataset"].astype(str) != dataset].copy()
        if choice_scope == "task":
            scoped = training[training["task"].astype(str) == task].copy()
            if scoped.empty:
                fallback_scope_used = True
            else:
                training = scoped
        if training.empty:
            training = dataset_scores_all[
                dataset_scores_all["dataset"].astype(str) != dataset
            ].copy()
            fallback_scope_used = True
        if training.empty:
            training = dataset_scores_all.copy()
            fallback_scope_used = True
        if training.empty:
            raise ValueError(f"Risk-constrained selector `{method_name}` has no training rows.")

        candidate_rows_for_target: list[dict[str, Any]] = []
        for selector_method in candidate_selector_methods:
            method_training = training[training["method"].astype(str).eq(selector_method)]
            if method_training.empty:
                continue
            task_rank_diff = _paired_selector_difference(
                training,
                method=selector_method,
                reference=task_reference_method,
                value_column="mean_rank",
            )
            family_rank_diff = _paired_selector_difference(
                training,
                method=selector_method,
                reference=family_reference_method,
                value_column="mean_rank",
            )
            task_runtime_diff = _paired_selector_difference(
                training,
                method=selector_method,
                reference=task_reference_method,
                value_column="mean_total_seconds",
            )
            n_task_pairs = int(len(task_rank_diff))
            n_family_pairs = int(len(family_rank_diff))
            task_upper = _upper_one_sided_ci(
                task_rank_diff,
                alpha=noninferiority_alpha,
            )
            family_upper = _upper_one_sided_ci(
                family_rank_diff,
                alpha=noninferiority_alpha,
            )
            mean_runtime_diff = (
                float(task_runtime_diff.mean()) if not task_runtime_diff.empty else float("inf")
            )
            enough_pairs = (
                n_task_pairs >= int(min_training_datasets)
                and n_family_pairs >= int(min_training_datasets)
            )
            runtime_gate = (
                mean_runtime_diff < -float(min_runtime_gain_seconds)
                if require_runtime_gain
                else True
            )
            gate_passed = bool(
                enough_pairs
                and task_upper < float(noninferiority_margin_rank)
                and family_upper < float(noninferiority_margin_rank)
                and runtime_gate
            )
            candidate_rows_for_target.append(
                {
                    "method": selector_method,
                    "mean_rank": float(method_training["mean_rank"].mean()),
                    "mean_total_seconds": float(method_training["mean_total_seconds"].mean()),
                    "task_mean_rank_difference": (
                        float(task_rank_diff.mean()) if not task_rank_diff.empty else np.nan
                    ),
                    "family_mean_rank_difference": (
                        float(family_rank_diff.mean())
                        if not family_rank_diff.empty
                        else np.nan
                    ),
                    "task_upper_one_sided_ci": float(task_upper),
                    "family_upper_one_sided_ci": float(family_upper),
                    "task_mean_runtime_difference_seconds": float(mean_runtime_diff),
                    "n_task_pairs": n_task_pairs,
                    "n_family_pairs": n_family_pairs,
                    "gate_passed": gate_passed,
                    "method_order": method_order.get(selector_method, len(method_order)),
                }
            )
        objectives = pd.DataFrame(candidate_rows_for_target)
        if not objectives.empty:
            log_seconds = np.log1p(objectives["mean_total_seconds"].astype(float))
            seconds_span = float(log_seconds.max() - log_seconds.min())
            objectives["risk_runtime_norm"] = (
                (log_seconds - float(log_seconds.min())) / seconds_span
                if seconds_span > 1e-12
                else 0.0
            )
            objectives["risk_objective"] = (
                objectives["mean_rank"].astype(float)
                + float(runtime_tradeoff_weight) * objectives["risk_runtime_norm"].astype(float)
            )
        eligible = (
            objectives[objectives["gate_passed"]].copy()
            if not objectives.empty
            else pd.DataFrame()
        )
        risk_fallback_reason = ""
        if eligible.empty:
            fallback_pool = training[
                training["method"].astype(str).isin(fallback_selector_methods)
            ].copy()
            if fallback_pool.empty:
                raise ValueError(
                    f"Risk-constrained selector `{method_name}` has no fallback rows "
                    f"for dataset={dataset}, task={task}."
                )
            fallback_objectives = (
                fallback_pool.groupby("method", as_index=False)
                .agg(
                    mean_rank=("mean_rank", "mean"),
                    mean_total_seconds=("mean_total_seconds", "mean"),
                )
                .assign(method_order=lambda frame: frame["method"].map(method_order))
                .sort_values(["mean_rank", "mean_total_seconds", "method_order"])
            )
            selected_method = str(fallback_objectives.iloc[0]["method"])
            selected_summary = {
                "risk_objective": float(fallback_objectives.iloc[0]["mean_rank"]),
                "risk_runtime_norm": 0.0,
                "task_upper_one_sided_ci": np.nan,
                "family_upper_one_sided_ci": np.nan,
                "task_mean_rank_difference": np.nan,
                "family_mean_rank_difference": np.nan,
                "task_mean_runtime_difference_seconds": np.nan,
                "gate_passed": False,
            }
            risk_fallback_reason = "no_candidate_passed_risk_gate"
        else:
            selected = eligible.sort_values(
                ["risk_objective", "mean_rank", "mean_total_seconds", "method_order"]
            ).iloc[0]
            selected_method = str(selected["method"])
            selected_summary = {
                "risk_objective": float(selected["risk_objective"]),
                "risk_runtime_norm": float(selected["risk_runtime_norm"]),
                "task_upper_one_sided_ci": float(selected["task_upper_one_sided_ci"]),
                "family_upper_one_sided_ci": float(selected["family_upper_one_sided_ci"]),
                "task_mean_rank_difference": float(selected["task_mean_rank_difference"]),
                "family_mean_rank_difference": float(selected["family_mean_rank_difference"]),
                "task_mean_runtime_difference_seconds": float(
                    selected["task_mean_runtime_difference_seconds"]
                ),
                "gate_passed": True,
            }

        target_rows = selector_pool[
            (selector_pool["dataset"].astype(str) == dataset)
            & (selector_pool["task"].astype(str) == task)
            & (selector_pool["method"].astype(str) == selected_method)
        ].copy()
        if target_rows.empty:
            raise ValueError(
                f"Risk-constrained selector `{method_name}` found no target rows for "
                f"dataset={dataset}, task={task}, selected_method={selected_method}."
            )

        risk_suffix = (
            f";risk_choice={selected_method}"
            f";risk_gate_passed={int(bool(selected_summary['gate_passed']))}"
            f";risk_fallback_reason={risk_fallback_reason}"
            f";risk_choice_scope={choice_scope}"
            f";risk_lodo={int(leave_one_dataset_out)}"
            f";risk_task_upper={float(selected_summary['task_upper_one_sided_ci']):.6g}"
            f";risk_family_upper={float(selected_summary['family_upper_one_sided_ci']):.6g}"
            f";risk_task_runtime_diff={float(selected_summary['task_mean_runtime_difference_seconds']):.6g}"
        )
        target_rows["method"] = method_name
        target_rows["ensemble_weights"] = (
            target_rows["ensemble_weights"].fillna("").astype(str) + risk_suffix
        ).str.lstrip(";")
        selector_outputs.append(target_rows[ARTIFACT_OUTPUT_COLUMNS])

        tuning_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "candidate_selector_methods": ",".join(candidate_selector_methods),
                "selected_selector_method": selected_method,
                "adaptive_objective": float(selected_summary["risk_objective"]),
                "adaptive_mean_total_seconds": float(
                    training.loc[
                        training["method"].astype(str).eq(selected_method),
                        "mean_total_seconds",
                    ].mean()
                ),
                "adaptive_choice_scope": f"risk_constrained_{choice_scope}",
                "n_candidate_selectors": int(len(candidate_selector_methods)),
                "n_training_units": int(training["n_units"].sum()),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": bool(leave_one_dataset_out),
                "fallback_used": bool(fallback_scope_used or risk_fallback_reason),
                "task_reference_method": task_reference_method,
                "family_reference_method": family_reference_method,
                "fallback_selector_methods": ",".join(fallback_selector_methods),
                "risk_noninferiority_margin_rank": float(noninferiority_margin_rank),
                "risk_noninferiority_alpha": float(noninferiority_alpha),
                "risk_min_training_datasets": int(min_training_datasets),
                "risk_require_runtime_gain": bool(require_runtime_gain),
                "risk_min_runtime_gain_seconds": float(min_runtime_gain_seconds),
                "risk_task_upper_one_sided_ci": float(
                    selected_summary["task_upper_one_sided_ci"]
                ),
                "risk_family_upper_one_sided_ci": float(
                    selected_summary["family_upper_one_sided_ci"]
                ),
                "risk_task_mean_rank_difference": float(
                    selected_summary["task_mean_rank_difference"]
                ),
                "risk_family_mean_rank_difference": float(
                    selected_summary["family_mean_rank_difference"]
                ),
                "risk_task_mean_runtime_difference_seconds": float(
                    selected_summary["task_mean_runtime_difference_seconds"]
                ),
                "risk_gate_passed": bool(selected_summary["gate_passed"]),
                "risk_fallback_reason": risk_fallback_reason,
                "risk_objective": float(selected_summary["risk_objective"]),
                "risk_runtime_norm": float(selected_summary["risk_runtime_norm"]),
            }
        )

    selector = pd.concat(selector_outputs, ignore_index=True)
    return selector, pd.DataFrame(tuning_rows)


def _unit_validation_meta(group: pd.DataFrame) -> dict[str, float]:
    values = np.sort(group["val_primary_metric"].astype(float).to_numpy())
    n_values = len(values)
    if n_values <= 1:
        return {
            "val_margin_norm": 0.0,
            "val_entropy_norm": 0.0,
            "val_spread_norm": 0.0,
        }
    best = float(values[0])
    second = float(values[1])
    spread = float(values[-1] - values[0])
    scale = spread if spread > 1e-12 else max(abs(best), 1.0)
    margin = float((second - best) / scale) if scale > 0.0 else 0.0
    scaled = (values - values[0]) / spread if spread > 1e-12 else np.zeros_like(values)
    logits = -scaled
    exp_logits = np.exp(logits - float(np.max(logits)))
    probabilities = exp_logits / float(exp_logits.sum())
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    entropy_norm = entropy / float(np.log(n_values)) if n_values > 1 else 0.0
    spread_norm = spread / max(abs(best), 1.0)
    return {
        "val_margin_norm": margin,
        "val_entropy_norm": entropy_norm,
        "val_spread_norm": spread_norm,
    }


def _safe_feature_name(value: object) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_")


_META_FEATURE_MEMO: dict[tuple[Any, ...], pd.DataFrame] | None = None
# Within one split-aware builder call every guard scores the same selector pool against the
# same candidates. Entries are keyed by the identity of frames that live for the whole call,
# a scored pool is reused only for a selector pool equal to the stored one, and the memo is
# cleared when the call returns.
_GUARD_MEMO: dict[tuple[Any, ...], Any] | None = None
# Tests switch the memo off to check that it changes no row.
_USE_GUARD_MEMO = True


def _scored_selector_pool(
    candidate_rows: pd.DataFrame,
    selector_pool: pd.DataFrame,
    *,
    error_prefix: str,
) -> pd.DataFrame:
    """``selector_pool`` with the test rank and total runtime of each selected member."""
    memo_key = ("scored_pool", id(candidate_rows), int(candidate_rows.shape[0]))
    if _GUARD_MEMO is not None:
        for stored_pool, stored_scored in _GUARD_MEMO.get(memo_key, []):
            if stored_pool.equals(selector_pool):
                return stored_scored.copy()
    lookup_key = ("rank_lookup", id(candidate_rows), int(candidate_rows.shape[0]))
    rank_lookup = _GUARD_MEMO.get(lookup_key) if _GUARD_MEMO is not None else None
    if rank_lookup is None:
        ranked_candidates = _with_rank_normalized_validation(candidate_rows)
        rank_lookup = ranked_candidates[
            [*ARTIFACT_UNIT_COLUMNS, "method", "test_rank_norm"]
        ].rename(
            columns={
                "method": "selected_member",
                "test_rank_norm": "selected_member_test_rank_norm",
            }
        )
        if _GUARD_MEMO is not None:
            _GUARD_MEMO[lookup_key] = rank_lookup
    scored_pool = selector_pool.merge(
        rank_lookup,
        on=[*ARTIFACT_UNIT_COLUMNS, "selected_member"],
        how="left",
    )
    if scored_pool["selected_member_test_rank_norm"].isna().any():
        missing = scored_pool[scored_pool["selected_member_test_rank_norm"].isna()]
        example = missing[[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"]].head(1)
        raise ValueError(f"{error_prefix}: {example.to_dict(orient='records')}")
    scored_pool["selected_member_total_seconds"] = (
        scored_pool["fit_seconds"].astype(float)
        + scored_pool["predict_seconds"].astype(float)
    )
    if _GUARD_MEMO is not None:
        _GUARD_MEMO.setdefault(memo_key, []).append((selector_pool.copy(), scored_pool.copy()))
    return scored_pool


def _dataset_meta_features(
    candidate_rows: pd.DataFrame,
    *,
    dataset_descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Within one split-aware builder call every guard asks for the same features; the
    # memo is keyed by the datasets and row count and cleared when the call returns.
    memo_key = None
    if _META_FEATURE_MEMO is not None:
        memo_key = (
            frozenset(candidate_rows["dataset"].astype(str).unique()),
            int(candidate_rows.shape[0]),
            id(dataset_descriptors),
        )
        if memo_key in _META_FEATURE_MEMO:
            return _META_FEATURE_MEMO[memo_key].copy()
    features = _dataset_meta_features_uncached(
        candidate_rows,
        dataset_descriptors=dataset_descriptors,
    )
    if memo_key is not None:
        _META_FEATURE_MEMO[memo_key] = features.copy()
    return features


def _dataset_meta_features_uncached(
    candidate_rows: pd.DataFrame,
    *,
    dataset_descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ranked = _with_rank_normalized_validation(candidate_rows)
    unit_rows: list[dict[str, Any]] = []
    for unit_key, group in ranked.groupby(ARTIFACT_UNIT_COLUMNS, sort=False):
        unit_values = dict(zip(ARTIFACT_UNIT_COLUMNS, unit_key))
        unit_rows.append({**unit_values, **_unit_validation_meta(group)})
    unit_features = pd.DataFrame(unit_rows)
    unit_summary = (
        unit_features.groupby(["dataset", "task"], as_index=False)
        .agg(
            n_units=("val_margin_norm", "size"),
            val_margin_mean=("val_margin_norm", "mean"),
            val_margin_std=("val_margin_norm", "std"),
            val_entropy_mean=("val_entropy_norm", "mean"),
            val_entropy_std=("val_entropy_norm", "std"),
            val_spread_mean=("val_spread_norm", "mean"),
        )
    )

    rows = ranked.copy()
    rows["log_fit_seconds"] = np.log1p(rows["fit_seconds"].astype(float))
    dataset_summary = (
        rows.groupby(["dataset", "task"], as_index=False)
        .agg(
            n_methods=("method", "nunique"),
            n_folds=("fold", "nunique"),
            n_seeds=("seed", "nunique"),
            log_fit_seconds_mean=("log_fit_seconds", "mean"),
            log_fit_seconds_std=("log_fit_seconds", "std"),
            log_fit_seconds_min=("log_fit_seconds", "min"),
            log_fit_seconds_max=("log_fit_seconds", "max"),
            fit_seconds_median=("fit_seconds", "median"),
        )
    )
    dataset_summary["log_fit_seconds_span"] = (
        dataset_summary["log_fit_seconds_max"] - dataset_summary["log_fit_seconds_min"]
    )
    features = dataset_summary.merge(unit_summary, on=["dataset", "task"], how="left")
    features["val_margin_std"] = features["val_margin_std"].fillna(0.0)
    features["val_entropy_std"] = features["val_entropy_std"].fillna(0.0)
    features["log_fit_seconds_std"] = features["log_fit_seconds_std"].fillna(0.0)
    features["log_fit_seconds_span"] = features["log_fit_seconds_span"].fillna(0.0)

    for task in sorted(str(value) for value in ranked["task"].unique()):
        column = f"task_is_{_safe_feature_name(task)}"
        features[column] = (features["task"].astype(str) == task).astype(float)
    if dataset_descriptors is not None:
        if "dataset" not in dataset_descriptors.columns:
            raise ValueError("Dataset descriptors must contain a `dataset` column.")
        descriptor_rows = dataset_descriptors.copy()
        descriptor_rows["dataset"] = descriptor_rows["dataset"].astype(str)
        features = features.merge(descriptor_rows, on="dataset", how="left")
        descriptor_columns = [
            column for column in descriptor_rows.columns if column != "dataset"
        ]
        missing_descriptor = features[descriptor_columns].isna().all(axis=1)
        if "descriptor_missing" in features.columns:
            features["descriptor_missing"] = missing_descriptor.astype(float)
        else:
            features["descriptor_missing"] = missing_descriptor.astype(float)
    return features


def _default_meta_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"dataset", "task"}
    return [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]


def _meta_feature_distances(
    target: pd.Series,
    training_features: pd.DataFrame,
    feature_columns: list[str],
    feature_groups: list[MetaFeatureGroup] | None = None,
) -> pd.Series:
    train = training_features[feature_columns].astype(float).copy()
    means = train.mean(axis=0).fillna(0.0)
    scales = train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    train = train.fillna(means)
    target_values = target[feature_columns].astype(float).fillna(means)
    normalized_train = (train - means) / scales
    normalized_target = (target_values - means) / scales
    squared_diff = (normalized_train - normalized_target) ** 2
    if feature_groups:
        distance_square = pd.Series(0.0, index=training_features.index)
        for _, columns, weight, normalize_by_size in feature_groups:
            group_square = squared_diff[columns].sum(axis=1)
            if normalize_by_size and columns:
                group_square = group_square / float(len(columns))
            distance_square = distance_square + float(weight) * group_square
        distances = np.sqrt(distance_square.clip(lower=0.0))
    else:
        distances = np.sqrt(squared_diff.sum(axis=1))
    return pd.Series(distances, index=training_features.index)


def _meta_feature_groups_from_config(
    feature_columns: list[str],
    group_specs: list[dict[str, Any]] | None,
) -> list[MetaFeatureGroup] | None:
    if not group_specs:
        return None
    selected = set(feature_columns)
    assigned: set[str] = set()
    groups: list[MetaFeatureGroup] = []
    for index, spec in enumerate(group_specs):
        name = str(spec.get("name", f"group_{index}"))
        configured_columns = [str(value) for value in spec.get("columns", [])]
        unknown_columns = [
            column for column in configured_columns if column not in selected
        ]
        if unknown_columns:
            raise ValueError(
                f"Meta-feature group `{name}` references unselected columns: {unknown_columns}"
            )
        prefixes = [str(value) for value in spec.get("column_prefixes", [])]
        columns = [
            column
            for column in feature_columns
            if column in configured_columns
            or any(column.startswith(prefix) for prefix in prefixes)
        ]
        columns = [column for column in dict.fromkeys(columns) if column not in assigned]
        if not columns:
            continue
        assigned.update(columns)
        groups.append(
            (
                name,
                columns,
                float(spec.get("weight", 1.0)),
                bool(spec.get("normalize_by_size", True)),
            )
        )
    remaining = [column for column in feature_columns if column not in assigned]
    if remaining:
        groups.append(("ungrouped", remaining, 1.0, True))
    if not groups:
        raise ValueError("Meta-feature groups were configured but matched no columns.")
    return groups


def _format_meta_feature_groups(feature_groups: list[MetaFeatureGroup] | None) -> str:
    if not feature_groups:
        return ""
    parts = []
    for name, columns, weight, normalize_by_size in feature_groups:
        suffix = "/mean" if normalize_by_size else "/sum"
        parts.append(f"{name}:n={len(columns)}:w={weight:g}{suffix}")
    return ";".join(parts)


def _select_meta_feature_columns(
    features: pd.DataFrame,
    *,
    feature_columns: list[str] | None,
    feature_column_prefixes: list[str] | None,
    exclude_feature_columns: list[str] | None,
    exclude_feature_column_prefixes: list[str] | None,
) -> list[str]:
    default_columns = _default_meta_feature_columns(features)
    selected = list(feature_columns) if feature_columns else list(default_columns)
    if feature_column_prefixes:
        prefix_columns = [
            column
            for column in default_columns
            if any(column.startswith(prefix) for prefix in feature_column_prefixes)
        ]
        if feature_columns:
            selected.extend(prefix_columns)
        else:
            selected = prefix_columns
    selected = list(dict.fromkeys(selected))
    if exclude_feature_columns:
        excluded = set(exclude_feature_columns)
        selected = [column for column in selected if column not in excluded]
    if exclude_feature_column_prefixes:
        selected = [
            column
            for column in selected
            if not any(
                column.startswith(prefix) for prefix in exclude_feature_column_prefixes
            )
        ]
    if not selected:
        raise ValueError("Meta-feature selector has no feature columns after filtering.")
    return selected


def _meta_feature_adaptive_selector_from_selectors(
    candidate_rows: pd.DataFrame,
    selector_rows: pd.DataFrame,
    *,
    method_name: str,
    candidate_selector_methods: list[str],
    choice_scope: str,
    leave_one_dataset_out: bool,
    feature_columns: list[str] | None = None,
    feature_column_prefixes: list[str] | None = None,
    exclude_feature_columns: list[str] | None = None,
    exclude_feature_column_prefixes: list[str] | None = None,
    feature_groups: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
    top_k: int = 8,
    distance_power: float = 1.0,
    distance_epsilon: float = 1e-6,
    runtime_tradeoff_weight: float = 0.0,
    default_selector_method: str | None = None,
    min_improvement_over_default: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_selector_methods:
        raise ValueError(f"Selector `{method_name}` requires candidate_selector_methods.")
    if choice_scope not in {"task", "global"}:
        raise ValueError("meta-feature adaptive choice_scope must be one of: task, global")
    if top_k <= 0:
        raise ValueError("meta-feature adaptive top_k must be positive.")

    method_order = {name: index for index, name in enumerate(candidate_selector_methods)}
    selector_pool = selector_rows[
        selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    available_methods = set(str(value) for value in selector_pool["method"].unique())
    missing_methods = [name for name in candidate_selector_methods if name not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Selector `{method_name}` references unavailable selector methods: {missing_methods}"
        )

    scored_pool = _scored_selector_pool(
        candidate_rows,
        selector_pool,
        error_prefix="Meta-feature adaptive selector could not map selected members to candidate ranks",
    )

    features = _dataset_meta_features(
        candidate_rows,
        dataset_descriptors=dataset_descriptors,
    )
    selected_feature_columns = _select_meta_feature_columns(
        features,
        feature_columns=feature_columns,
        feature_column_prefixes=feature_column_prefixes,
        exclude_feature_columns=exclude_feature_columns,
        exclude_feature_column_prefixes=exclude_feature_column_prefixes,
    )
    missing_features = [
        column for column in selected_feature_columns if column not in set(features.columns)
    ]
    if missing_features:
        raise ValueError(f"Unknown meta-feature columns: {missing_features}")
    selected_feature_groups = _meta_feature_groups_from_config(
        selected_feature_columns,
        feature_groups,
    )
    formatted_feature_groups = _format_meta_feature_groups(selected_feature_groups)

    selector_outputs: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    for _, target in features.sort_values(["dataset", "task"]).iterrows():
        dataset = str(target["dataset"])
        task = str(target["task"])
        training_features = features.copy()
        fallback_used = False
        if leave_one_dataset_out:
            training_features = training_features[
                training_features["dataset"].astype(str) != dataset
            ].copy()
        if choice_scope == "task":
            scoped = training_features[training_features["task"].astype(str) == task].copy()
            if scoped.empty:
                fallback_used = True
            else:
                training_features = scoped
        if training_features.empty:
            training_features = features[features["dataset"].astype(str) != dataset].copy()
            fallback_used = True
        if training_features.empty:
            training_features = features.copy()
            fallback_used = True

        distances = _meta_feature_distances(
            target,
            training_features,
            selected_feature_columns,
            selected_feature_groups,
        )
        neighbors = training_features.assign(_distance=distances).sort_values(
            ["_distance", "dataset"],
            ascending=True,
        ).head(top_k)
        if float(distance_power) == 0.0:
            weights = np.ones(neighbors.shape[0], dtype=float)
        else:
            weights = 1.0 / np.power(
                neighbors["_distance"].to_numpy(dtype=float) + float(distance_epsilon),
                float(distance_power),
            )
        neighbors = neighbors.assign(_neighbor_weight=weights / float(weights.sum()))

        training = scored_pool[
            scored_pool["dataset"].astype(str).isin(set(neighbors["dataset"].astype(str)))
        ].copy()
        if choice_scope == "task":
            scoped_training = training[training["task"].astype(str) == task].copy()
            if scoped_training.empty:
                fallback_used = True
            else:
                training = scoped_training
        training = training.merge(
            neighbors[["dataset", "_neighbor_weight", "_distance"]],
            on="dataset",
            how="inner",
        )
        if training.empty:
            raise ValueError(
                f"Meta-feature adaptive selector `{method_name}` has no scored neighbors "
                f"for dataset={dataset}, task={task}."
            )

        objective_rows: list[dict[str, Any]] = []
        for selector_method, group in training.groupby("method", sort=False):
            dataset_means = (
                group.groupby("dataset", as_index=False)
                .agg(
                    selected_member_test_rank_norm=("selected_member_test_rank_norm", "mean"),
                    selected_member_total_seconds=("selected_member_total_seconds", "mean"),
                    _neighbor_weight=("_neighbor_weight", "first"),
                )
                .sort_values("dataset")
            )
            weights_for_group = dataset_means["_neighbor_weight"].to_numpy(dtype=float)
            objective_rows.append(
                {
                    "method": str(selector_method),
                    "adaptive_objective": float(
                        np.average(
                            dataset_means["selected_member_test_rank_norm"].to_numpy(
                                dtype=float
                            ),
                            weights=weights_for_group,
                        )
                    ),
                    "adaptive_mean_total_seconds": float(
                        np.average(
                            dataset_means["selected_member_total_seconds"].to_numpy(dtype=float),
                            weights=weights_for_group,
                        )
                    ),
                    "n_training_rows": int(group.shape[0]),
                }
            )
        objectives = pd.DataFrame(objective_rows)
        log_seconds = np.log1p(objectives["adaptive_mean_total_seconds"].astype(float))
        seconds_span = float(log_seconds.max() - log_seconds.min())
        objectives["meta_feature_runtime_norm"] = (
            (log_seconds - float(log_seconds.min())) / seconds_span
            if seconds_span > 1e-12
            else 0.0
        )
        objectives["meta_feature_objective"] = (
            objectives["adaptive_objective"].astype(float)
            + float(runtime_tradeoff_weight) * objectives["meta_feature_runtime_norm"].astype(float)
        )
        objectives["method_order"] = objectives["method"].map(method_order)
        objectives = objectives.sort_values(
            [
                "meta_feature_objective",
                "adaptive_objective",
                "adaptive_mean_total_seconds",
                "method_order",
            ],
            ascending=True,
        )
        selected = objectives.iloc[0]
        if default_selector_method:
            default_rows = objectives[objectives["method"].astype(str) == default_selector_method]
            if not default_rows.empty:
                default_row = default_rows.iloc[0]
                improvement = float(default_row["meta_feature_objective"]) - float(
                    selected["meta_feature_objective"]
                )
                if improvement < float(min_improvement_over_default):
                    selected = default_row
        selected_method = str(selected["method"])
        target_rows = selector_pool[
            (selector_pool["dataset"].astype(str) == dataset)
            & (selector_pool["task"].astype(str) == task)
            & (selector_pool["method"].astype(str) == selected_method)
        ].copy()
        if target_rows.empty:
            raise ValueError(
                f"Meta-feature adaptive selector `{method_name}` found no target rows for "
                f"dataset={dataset}, task={task}, selected_method={selected_method}."
            )

        nearest_datasets = ",".join(str(value) for value in neighbors["dataset"].tolist())
        adaptive_suffix = (
            f";adaptive_choice={selected_method}"
            f";adaptive_objective={float(selected['adaptive_objective']):.6g}"
            f";meta_feature_objective={float(selected['meta_feature_objective']):.6g}"
            f";adaptive_choice_scope=meta_feature_{choice_scope}"
            f";adaptive_lodo={int(leave_one_dataset_out)}"
            f";nearest_datasets={nearest_datasets}"
        )
        target_rows["method"] = method_name
        target_rows["ensemble_weights"] = (
            target_rows["ensemble_weights"].fillna("").astype(str) + adaptive_suffix
        ).str.lstrip(";")
        selector_outputs.append(target_rows[ARTIFACT_OUTPUT_COLUMNS])

        tuning_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "candidate_selector_methods": ",".join(candidate_selector_methods),
                "selected_selector_method": selected_method,
                "adaptive_objective": float(selected["adaptive_objective"]),
                "adaptive_mean_total_seconds": float(selected["adaptive_mean_total_seconds"]),
                "adaptive_choice_scope": f"meta_feature_{choice_scope}",
                "n_candidate_selectors": int(len(candidate_selector_methods)),
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": bool(leave_one_dataset_out),
                "fallback_used": bool(fallback_used),
                "meta_feature_objective": float(selected["meta_feature_objective"]),
                "meta_feature_runtime_norm": float(selected["meta_feature_runtime_norm"]),
                "meta_feature_runtime_tradeoff_weight": float(runtime_tradeoff_weight),
                "meta_feature_columns": ",".join(selected_feature_columns),
                "meta_feature_groups": formatted_feature_groups,
                "nearest_datasets": nearest_datasets,
                "n_neighbor_datasets": int(neighbors.shape[0]),
                "nearest_distance_min": float(neighbors["_distance"].min()),
                "nearest_distance_mean": float(neighbors["_distance"].mean()),
                "default_selector_method": default_selector_method or "",
                "min_improvement_over_default": float(min_improvement_over_default),
            }
        )

    selector = pd.concat(selector_outputs, ignore_index=True)
    return selector, pd.DataFrame(tuning_rows)


def _load_selector_rows_from_result_file(
    path: str | Path,
    candidate_selector_methods: list[str],
) -> pd.DataFrame:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Selector result source does not exist: {result_path}")
    source = pd.read_csv(result_path, low_memory=False)
    missing_columns = set(ARTIFACT_OUTPUT_COLUMNS).difference(source.columns)
    if missing_columns:
        raise ValueError(
            f"Selector result source missing required columns: {sorted(missing_columns)}"
        )
    rows = source[
        (source["method_type"].astype(str) == "selector")
        & (source["method"].astype(str).isin(candidate_selector_methods))
    ].copy()
    available_methods = set(str(value) for value in rows["method"].unique())
    missing_methods = [name for name in candidate_selector_methods if name not in available_methods]
    if missing_methods:
        raise ValueError(
            f"Selector result source `{result_path}` missing selector methods: {missing_methods}"
        )
    return rows[ARTIFACT_OUTPUT_COLUMNS].reset_index(drop=True)


def _targets_equal_history(history_candidates: pd.DataFrame, target_candidates: pd.DataFrame) -> bool:
    if history_candidates is target_candidates:
        return True
    if int(history_candidates.shape[0]) != int(target_candidates.shape[0]):
        return False
    if set(history_candidates["dataset"].astype(str)) != set(
        target_candidates["dataset"].astype(str)
    ):
        return False
    key_columns = [*ARTIFACT_UNIT_COLUMNS, "method"]
    history_keys = set(
        tuple(value)
        for value in history_candidates[key_columns].astype(str).to_numpy(dtype=object)
    )
    target_keys = set(
        tuple(value)
        for value in target_candidates[key_columns].astype(str).to_numpy(dtype=object)
    )
    return history_keys == target_keys


def _empty_tuning_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "task",
            "dataset",
            "runtime_weight",
            "validation_objective",
            "tuning_runtime_weight",
            "n_training_units",
            "n_training_datasets",
            "leave_one_dataset_out",
            "fallback_used",
            "calibration_objective",
            "n_weight_candidates",
            *CALIBRATION_WEIGHT_COLUMNS,
            *BUDGET_TUNING_COLUMNS,
            *FILTER_TUNING_COLUMNS,
            *ADAPTIVE_TUNING_COLUMNS,
            *RISK_TUNING_COLUMNS,
            *CONFIDENCE_TUNING_COLUMNS,
            "selector_method",
        ]
    )


def _calibrated_selector_from_history(
    history_candidate_rows: pd.DataFrame,
    target_candidate_rows: pd.DataFrame,
    *,
    method_name: str,
    weight_grid: list[dict[str, float]],
    weight_selection: str,
    max_budget_seconds: float | None = None,
    budget_feature: str = "fit_seconds",
    candidate_filter: str = "none",
    n_unfiltered_candidates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_weight_selection(weight_selection)
    history_ranked = _apply_candidate_budget(
        _with_rank_normalized_validation(history_candidate_rows),
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    target_ranked = _apply_candidate_budget(
        _with_rank_normalized_validation(target_candidate_rows),
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    if history_ranked.empty:
        raise ValueError(f"Selector `{method_name}` has no outer-training candidates.")
    if target_ranked.empty:
        raise ValueError(f"Selector `{method_name}` has no target candidates.")

    if len(weight_grid) == 1:
        weights = weight_grid[0]
        calibration_objective = float("nan")
    else:
        weights, calibration_objective = _select_calibration_weights(
            history_ranked,
            weight_grid,
            weight_selection=weight_selection,
        )
    max_budget_label = (
        "inf"
        if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
        else f"{float(max_budget_seconds):.6g}"
    )

    selector_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    # The history is the same for every target dataset, so its priors are computed once
    # (row order and values are identical to a per-dataset computation).
    all_features, _ = _calibrated_prior_features(target_ranked, history_ranked)
    all_features.index = target_ranked.index
    # Per-row prior fallback exactly as _calibrated_prior_features flags it for one target:
    # a (task, method), (task, family) or method prior that the history does not contain.
    def _absent_from_history(keys: list[str]) -> np.ndarray:
        known = pd.MultiIndex.from_frame(history_ranked[keys].drop_duplicates())
        return ~pd.MultiIndex.from_frame(target_ranked[keys]).isin(known)

    prior_missing = pd.Series(
        _absent_from_history(["task", "method"])
        | _absent_from_history(["task", "method_family"])
        | _absent_from_history(["method"]),
        index=target_ranked.index,
    )
    n_training_units = int(history_ranked[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0])
    for dataset in sorted(str(value) for value in target_ranked["dataset"].unique()):
        target = target_ranked[target_ranked["dataset"].astype(str) == dataset].copy()
        features = all_features.loc[target.index].copy()
        prior_fallback_used = bool(prior_missing.loc[target.index].any())
        utility = _calibrated_utility(features, weights)
        features["calibrated_utility"] = utility
        features["selector_details"] = (
            "utility="
            + utility.map(lambda value: f"{float(value):.6g}")
            + ";w_val="
            + f"{weights['validation_weight']:.6g}"
            + ";w_task_method="
            + f"{weights['task_method_prior_weight']:.6g}"
            + ";w_task_family="
            + f"{weights['task_family_prior_weight']:.6g}"
            + ";w_global_method="
            + f"{weights['global_method_prior_weight']:.6g}"
            + ";w_runtime="
            + f"{weights['runtime_weight']:.6g}"
            + ";task_method_prior="
            + features["task_method_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";task_family_prior="
            + features["task_family_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";global_method_prior="
            + features["global_method_prior_rank_norm"].map(lambda value: f"{float(value):.6g}")
            + ";budget_feature="
            + budget_feature
            + ";budget_seconds="
            + features["budget_seconds"].map(lambda value: f"{float(value):.6g}")
            + ";max_budget_seconds="
            + max_budget_label
            + ";budget_fallback="
            + features["budget_fallback"].map(lambda value: str(int(bool(value))))
            + ";scale=calibrated_outer_history"
        )
        selector_frames.append(
            _selector_rows_from_utility(
                features,
                method_name=method_name,
                utility=features["calibrated_utility"],
                details=features["selector_details"],
            )
        )
        tuning_rows.append(
            {
                "task": str(target["task"].iloc[0]),
                "dataset": dataset,
                "calibration_objective": calibration_objective,
                "n_training_units": n_training_units,
                "n_training_datasets": int(history_ranked["dataset"].nunique()),
                "budget_feature": budget_feature,
                "max_budget_seconds": (
                    np.inf
                    if max_budget_seconds is None or not np.isfinite(float(max_budget_seconds))
                    else float(max_budget_seconds)
                ),
                "n_target_units": int(
                    target[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_target_candidates": int(target.shape[0]),
                "n_target_budget_fallback_units": int(
                    target.loc[target["budget_fallback"], ARTIFACT_UNIT_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_training_candidates": int(history_ranked.shape[0]),
                "candidate_filter": candidate_filter,
                "n_unfiltered_candidates": (
                    int(n_unfiltered_candidates)
                    if n_unfiltered_candidates is not None
                    else int(target_candidate_rows.shape[0])
                ),
                "n_filtered_candidates": int(target_candidate_rows.shape[0]),
                "leave_one_dataset_out": False,
                "fallback_used": bool(prior_fallback_used),
                "n_weight_candidates": int(len(weight_grid)),
                **weights,
            }
        )
    return pd.concat(selector_frames, ignore_index=True), pd.DataFrame(tuning_rows)


def _task_adaptive_selector_from_history(
    history_candidate_rows: pd.DataFrame,
    history_selector_rows: pd.DataFrame,
    target_candidate_rows: pd.DataFrame,
    target_selector_rows: pd.DataFrame,
    *,
    method_name: str,
    candidate_selector_methods: list[str],
    choice_scope: str,
    leave_one_dataset_out: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_selector_methods:
        raise ValueError(f"Selector `{method_name}` requires candidate_selector_methods.")
    if choice_scope not in {"task", "global"}:
        raise ValueError("adaptive choice_scope must be one of: task, global")

    ranked_candidates = _with_rank_normalized_validation(history_candidate_rows)
    rank_lookup = ranked_candidates[
        [*ARTIFACT_UNIT_COLUMNS, "method", "test_rank_norm"]
    ].rename(
        columns={
            "method": "selected_member",
            "test_rank_norm": "selected_member_test_rank_norm",
        }
    )
    method_order = {name: index for index, name in enumerate(candidate_selector_methods)}
    selector_pool = history_selector_rows[
        history_selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    target_pool = target_selector_rows[
        target_selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    missing_methods = [
        name
        for name in candidate_selector_methods
        if name not in set(selector_pool["method"].astype(str).unique())
    ]
    if missing_methods:
        raise ValueError(
            f"Selector `{method_name}` references unavailable selector methods: {missing_methods}"
        )

    scored_pool = selector_pool.merge(
        rank_lookup,
        on=[*ARTIFACT_UNIT_COLUMNS, "selected_member"],
        how="left",
    )
    if scored_pool["selected_member_test_rank_norm"].isna().any():
        missing = scored_pool[scored_pool["selected_member_test_rank_norm"].isna()]
        example = missing[[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"]].head(1)
        raise ValueError(
            "Adaptive selector could not map selected members to history candidate ranks: "
            f"{example.to_dict(orient='records')}"
        )
    scored_pool["selected_member_total_seconds"] = (
        scored_pool["fit_seconds"].astype(float)
        + scored_pool["predict_seconds"].astype(float)
    )
    history_datasets = set(history_candidate_rows["dataset"].astype(str))

    selector_outputs: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    targets = (
        target_candidate_rows[["dataset", "task"]]
        .drop_duplicates()
        .sort_values(["dataset", "task"])
    )
    for _, target_key in targets.iterrows():
        dataset = str(target_key["dataset"])
        task = str(target_key["task"])
        training = scored_pool.copy()
        fallback_used = False
        if leave_one_dataset_out and dataset in history_datasets:
            training = training[training["dataset"].astype(str) != dataset].copy()
        if choice_scope == "task":
            scoped = training[training["task"].astype(str) == task].copy()
            if scoped.empty:
                fallback_used = True
            else:
                training = scoped
        if training.empty:
            training = scored_pool[
                scored_pool["dataset"].astype(str) != dataset
            ].copy()
            fallback_used = True
        if training.empty:
            training = scored_pool.copy()
            fallback_used = True
        if training.empty:
            raise ValueError(f"Adaptive selector `{method_name}` has no history rows.")

        objectives = (
            training.groupby("method", as_index=False)
            .agg(
                adaptive_objective=("selected_member_test_rank_norm", "mean"),
                adaptive_mean_total_seconds=("selected_member_total_seconds", "mean"),
                n_training_rows=("selected_member_test_rank_norm", "size"),
            )
            .assign(method_order=lambda frame: frame["method"].map(method_order))
            .sort_values(
                ["adaptive_objective", "adaptive_mean_total_seconds", "method_order"],
                ascending=True,
            )
        )
        selected = objectives.iloc[0]
        selected_method = str(selected["method"])
        target_rows = target_pool[
            (target_pool["dataset"].astype(str) == dataset)
            & (target_pool["task"].astype(str) == task)
            & (target_pool["method"].astype(str) == selected_method)
        ].copy()
        if target_rows.empty:
            raise ValueError(
                f"Adaptive selector `{method_name}` found no target rows for "
                f"dataset={dataset}, task={task}, selected_method={selected_method}."
            )

        adaptive_suffix = (
            f";adaptive_choice={selected_method}"
            f";adaptive_objective={float(selected['adaptive_objective']):.6g}"
            f";adaptive_choice_scope={choice_scope}"
            f";adaptive_lodo=0"
            f";nested_history_datasets={int(training['dataset'].nunique())}"
        )
        target_rows["method"] = method_name
        target_rows["ensemble_weights"] = (
            target_rows["ensemble_weights"].fillna("").astype(str) + adaptive_suffix
        ).str.lstrip(";")
        selector_outputs.append(target_rows[ARTIFACT_OUTPUT_COLUMNS])
        tuning_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "candidate_selector_methods": ",".join(candidate_selector_methods),
                "selected_selector_method": selected_method,
                "adaptive_objective": float(selected["adaptive_objective"]),
                "adaptive_mean_total_seconds": float(selected["adaptive_mean_total_seconds"]),
                "adaptive_choice_scope": choice_scope,
                "n_candidate_selectors": int(len(candidate_selector_methods)),
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": False,
                "fallback_used": bool(fallback_used),
            }
        )
    return pd.concat(selector_outputs, ignore_index=True), pd.DataFrame(tuning_rows)


def _meta_feature_adaptive_selector_from_history(
    history_candidate_rows: pd.DataFrame,
    history_selector_rows: pd.DataFrame,
    target_candidate_rows: pd.DataFrame,
    target_selector_rows: pd.DataFrame,
    *,
    method_name: str,
    candidate_selector_methods: list[str],
    choice_scope: str,
    leave_one_dataset_out: bool,
    feature_columns: list[str] | None = None,
    feature_column_prefixes: list[str] | None = None,
    exclude_feature_columns: list[str] | None = None,
    exclude_feature_column_prefixes: list[str] | None = None,
    feature_groups: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
    top_k: int = 8,
    distance_power: float = 1.0,
    distance_epsilon: float = 1e-6,
    runtime_tradeoff_weight: float = 0.0,
    default_selector_method: str | None = None,
    min_improvement_over_default: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate_selector_methods:
        raise ValueError(f"Selector `{method_name}` requires candidate_selector_methods.")
    if choice_scope not in {"task", "global"}:
        raise ValueError("meta-feature adaptive choice_scope must be one of: task, global")
    if top_k <= 0:
        raise ValueError("meta-feature adaptive top_k must be positive.")

    method_order = {name: index for index, name in enumerate(candidate_selector_methods)}
    selector_pool = history_selector_rows[
        history_selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    target_pool = target_selector_rows[
        target_selector_rows["method"].astype(str).isin(candidate_selector_methods)
    ].copy()
    missing_methods = [
        name
        for name in candidate_selector_methods
        if name not in set(selector_pool["method"].astype(str).unique())
    ]
    if missing_methods:
        raise ValueError(
            f"Selector `{method_name}` references unavailable selector methods: {missing_methods}"
        )

    scored_pool = _scored_selector_pool(
        history_candidate_rows,
        selector_pool,
        error_prefix=(
            "Meta-feature adaptive selector could not map selected members to "
            "history candidate ranks"
        ),
    )

    combined_key = (
        "combined_candidates",
        id(history_candidate_rows),
        int(history_candidate_rows.shape[0]),
        id(target_candidate_rows),
        int(target_candidate_rows.shape[0]),
    )
    combined_candidates = _GUARD_MEMO.get(combined_key) if _GUARD_MEMO is not None else None
    if combined_candidates is None:
        combined_candidates = pd.concat(
            [history_candidate_rows, target_candidate_rows],
            ignore_index=True,
        ).drop_duplicates(subset=[*ARTIFACT_UNIT_COLUMNS, "method"], keep="last")
        if _GUARD_MEMO is not None:
            _GUARD_MEMO[combined_key] = combined_candidates
    features = _dataset_meta_features(
        combined_candidates,
        dataset_descriptors=dataset_descriptors,
    )
    selected_feature_columns = _select_meta_feature_columns(
        features,
        feature_columns=feature_columns,
        feature_column_prefixes=feature_column_prefixes,
        exclude_feature_columns=exclude_feature_columns,
        exclude_feature_column_prefixes=exclude_feature_column_prefixes,
    )
    selected_feature_groups = _meta_feature_groups_from_config(
        selected_feature_columns,
        feature_groups,
    )
    formatted_feature_groups = _format_meta_feature_groups(selected_feature_groups)
    history_datasets = set(history_candidate_rows["dataset"].astype(str))
    history_features_all = features[
        features["dataset"].astype(str).isin(history_datasets)
    ].copy()
    target_features = features[
        features["dataset"].astype(str).isin(
            set(target_candidate_rows["dataset"].astype(str))
        )
    ].copy()

    selector_outputs: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    for _, target in target_features.sort_values(["dataset", "task"]).iterrows():
        dataset = str(target["dataset"])
        task = str(target["task"])
        training_features = history_features_all.copy()
        fallback_used = False
        if leave_one_dataset_out and dataset in history_datasets:
            training_features = training_features[
                training_features["dataset"].astype(str) != dataset
            ].copy()
        if choice_scope == "task":
            scoped = training_features[training_features["task"].astype(str) == task].copy()
            if scoped.empty:
                fallback_used = True
            else:
                training_features = scoped
        if training_features.empty:
            training_features = history_features_all[
                history_features_all["dataset"].astype(str) != dataset
            ].copy()
            fallback_used = True
        if training_features.empty:
            training_features = history_features_all.copy()
            fallback_used = True
        if training_features.empty:
            raise ValueError(
                f"Meta-feature adaptive selector `{method_name}` has no history features."
            )

        distances = _meta_feature_distances(
            target,
            training_features,
            selected_feature_columns,
            selected_feature_groups,
        )
        neighbors = training_features.assign(_distance=distances).sort_values(
            ["_distance", "dataset"],
            ascending=True,
        ).head(top_k)
        if float(distance_power) == 0.0:
            weights = np.ones(neighbors.shape[0], dtype=float)
        else:
            weights = 1.0 / np.power(
                neighbors["_distance"].to_numpy(dtype=float) + float(distance_epsilon),
                float(distance_power),
            )
        neighbors = neighbors.assign(_neighbor_weight=weights / float(weights.sum()))

        training = scored_pool[
            scored_pool["dataset"].astype(str).isin(set(neighbors["dataset"].astype(str)))
        ].copy()
        if choice_scope == "task":
            scoped_training = training[training["task"].astype(str) == task].copy()
            if scoped_training.empty:
                fallback_used = True
            else:
                training = scoped_training
        training = training.merge(
            neighbors[["dataset", "_neighbor_weight", "_distance"]],
            on="dataset",
            how="inner",
        )
        if training.empty:
            raise ValueError(
                f"Meta-feature adaptive selector `{method_name}` has no scored neighbors "
                f"for dataset={dataset}, task={task}."
            )

        objective_rows: list[dict[str, Any]] = []
        for selector_method, group in training.groupby("method", sort=False):
            dataset_means = (
                group.groupby("dataset", as_index=False)
                .agg(
                    selected_member_test_rank_norm=("selected_member_test_rank_norm", "mean"),
                    selected_member_total_seconds=("selected_member_total_seconds", "mean"),
                    _neighbor_weight=("_neighbor_weight", "first"),
                )
                .sort_values("dataset")
            )
            weights_for_group = dataset_means["_neighbor_weight"].to_numpy(dtype=float)
            objective_rows.append(
                {
                    "method": str(selector_method),
                    "adaptive_objective": float(
                        np.average(
                            dataset_means[
                                "selected_member_test_rank_norm"
                            ].to_numpy(dtype=float),
                            weights=weights_for_group,
                        )
                    ),
                    "adaptive_mean_total_seconds": float(
                        np.average(
                            dataset_means[
                                "selected_member_total_seconds"
                            ].to_numpy(dtype=float),
                            weights=weights_for_group,
                        )
                    ),
                    "n_training_rows": int(group.shape[0]),
                }
            )
        objectives = pd.DataFrame(objective_rows)
        log_seconds = np.log1p(objectives["adaptive_mean_total_seconds"].astype(float))
        seconds_span = float(log_seconds.max() - log_seconds.min())
        objectives["meta_feature_runtime_norm"] = (
            (log_seconds - float(log_seconds.min())) / seconds_span
            if seconds_span > 1e-12
            else 0.0
        )
        objectives["meta_feature_objective"] = (
            objectives["adaptive_objective"].astype(float)
            + float(runtime_tradeoff_weight)
            * objectives["meta_feature_runtime_norm"].astype(float)
        )
        objectives["method_order"] = objectives["method"].map(method_order)
        objectives = objectives.sort_values(
            [
                "meta_feature_objective",
                "adaptive_objective",
                "adaptive_mean_total_seconds",
                "method_order",
            ],
            ascending=True,
        )
        selected = objectives.iloc[0]
        if default_selector_method:
            default_rows = objectives[
                objectives["method"].astype(str) == default_selector_method
            ]
            if not default_rows.empty:
                default_row = default_rows.iloc[0]
                improvement = float(default_row["meta_feature_objective"]) - float(
                    selected["meta_feature_objective"]
                )
                if improvement < float(min_improvement_over_default):
                    selected = default_row
        selected_method = str(selected["method"])
        target_rows = target_pool[
            (target_pool["dataset"].astype(str) == dataset)
            & (target_pool["task"].astype(str) == task)
            & (target_pool["method"].astype(str) == selected_method)
        ].copy()
        if target_rows.empty:
            raise ValueError(
                f"Meta-feature adaptive selector `{method_name}` found no target rows "
                f"for dataset={dataset}, task={task}, selected_method={selected_method}."
            )

        nearest_datasets = ",".join(str(value) for value in neighbors["dataset"].tolist())
        adaptive_suffix = (
            f";adaptive_choice={selected_method}"
            f";adaptive_objective={float(selected['adaptive_objective']):.6g}"
            f";meta_feature_objective={float(selected['meta_feature_objective']):.6g}"
            f";adaptive_choice_scope=meta_feature_{choice_scope}"
            f";adaptive_lodo=0"
            f";nearest_datasets={nearest_datasets}"
            f";nested_history_datasets={int(training['dataset'].nunique())}"
        )
        target_rows["method"] = method_name
        target_rows["ensemble_weights"] = (
            target_rows["ensemble_weights"].fillna("").astype(str) + adaptive_suffix
        ).str.lstrip(";")
        selector_outputs.append(target_rows[ARTIFACT_OUTPUT_COLUMNS])
        tuning_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "candidate_selector_methods": ",".join(candidate_selector_methods),
                "selected_selector_method": selected_method,
                "adaptive_objective": float(selected["adaptive_objective"]),
                "adaptive_mean_total_seconds": float(selected["adaptive_mean_total_seconds"]),
                "adaptive_choice_scope": f"meta_feature_{choice_scope}",
                "n_candidate_selectors": int(len(candidate_selector_methods)),
                "n_training_units": int(
                    training[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]
                ),
                "n_training_datasets": int(training["dataset"].nunique()),
                "leave_one_dataset_out": False,
                "fallback_used": bool(fallback_used),
                "meta_feature_objective": float(selected["meta_feature_objective"]),
                "meta_feature_runtime_norm": float(selected["meta_feature_runtime_norm"]),
                "meta_feature_runtime_tradeoff_weight": float(runtime_tradeoff_weight),
                "meta_feature_columns": ",".join(selected_feature_columns),
                "meta_feature_groups": formatted_feature_groups,
                "nearest_datasets": nearest_datasets,
                "n_neighbor_datasets": int(neighbors.shape[0]),
                "nearest_distance_min": float(neighbors["_distance"].min()),
                "nearest_distance_mean": float(neighbors["_distance"].mean()),
                "default_selector_method": default_selector_method or "",
                "min_improvement_over_default": float(min_improvement_over_default),
            }
        )
    return pd.concat(selector_outputs, ignore_index=True), pd.DataFrame(tuning_rows)


STANDARD_META_FEATURE_COLUMNS = [
    "descriptor_log_instances",
    "descriptor_log_features",
    "descriptor_n_classes",
    "descriptor_cat_share",
]


def load_standard_meta_features(
    config: dict[str, Any] | None,
    *,
    datasets: list[str],
    repository: Any | None = None,
) -> pd.DataFrame | None:
    """Standard OpenML-style dataset meta-features for the declared datasets.

    Both sources follow the OpenML convention that feature counts include the target
    column and the categorical share counts a nominal target, so the two artifacts use
    the same definition. Columns: ``dataset`` plus ``STANDARD_META_FEATURE_COLUMNS``.
    """
    if not config:
        return None
    kind = str(config.get("kind", ""))
    if kind == "tabarena_metadata":
        frame = pd.read_csv(config.get("local_path", "data/tabarena/tabarena_dataset_metadata.csv"))
        prepared = pd.DataFrame(
            {
                "dataset": frame["dataset_name"].astype(str),
                "descriptor_log_instances": np.log1p(frame["num_instances"].astype(float)),
                "descriptor_log_features": np.log1p(frame["num_features"].astype(float)),
                "descriptor_n_classes": frame["num_classes"].astype(float).fillna(0.0),
                "descriptor_cat_share": frame["percentage_cat_features"].astype(float) / 100.0,
            }
        )
    elif kind == "tabrepo_task_metadata":
        if repository is None:
            raise ValueError("`tabrepo_task_metadata` meta-features need the loaded TabRepo repository.")
        frame = pd.read_csv(config["local_path"]).drop_duplicates(subset=["tid"]).set_index("tid")
        rows = []
        for dataset in datasets:
            tid = int(repository.dataset_to_tid(str(dataset)))
            if tid not in frame.index:
                raise ValueError(f"TabRepo task metadata has no row for dataset {dataset} (tid {tid}).")
            record = frame.loc[tid]
            n_features = float(record["NumberOfFeatures"])
            symbolic = float(record["NumberOfSymbolicFeatures"])
            rows.append(
                {
                    "dataset": str(dataset),
                    "descriptor_log_instances": float(np.log1p(float(record["NumberOfInstances"]))),
                    "descriptor_log_features": float(np.log1p(n_features)),
                    "descriptor_n_classes": float(0.0 if pd.isna(record["NumberOfClasses"]) else record["NumberOfClasses"]),
                    "descriptor_cat_share": float(symbolic / n_features) if n_features > 0 else 0.0,
                }
            )
        prepared = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unknown standard meta-feature kind: {kind!r}")
    prepared = prepared[prepared["dataset"].isin(set(str(value) for value in datasets))]
    missing = sorted(set(str(value) for value in datasets) - set(prepared["dataset"]))
    if missing:
        raise ValueError(f"Standard meta-features missing for datasets: {missing[:10]}")
    if prepared[STANDARD_META_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Standard meta-features contain missing values.")
    return prepared.sort_values("dataset").reset_index(drop=True)


def _budgeted_ranked(
    candidate_rows: pd.DataFrame,
    *,
    max_budget_seconds: float | None,
    budget_feature: str,
) -> pd.DataFrame:
    return _apply_candidate_budget(
        _with_rank_normalized_validation(candidate_rows),
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )


def _optimism_corrected_selector(
    history_candidates: pd.DataFrame,
    target_candidates: pd.DataFrame,
    *,
    method_name: str,
    scope: str,
    max_budget_seconds: float | None,
    budget_feature: str,
    target_is_history: bool,
) -> pd.DataFrame:
    """Validation rank plus a per-method optimism offset estimated on history datasets.

    U(u, m) = r_val(u, m) + o(m), with o(m) the unit-weighted history mean of
    r_test - r_val for method m (per task when ``scope == "task"``). Parameter-free. When
    the targets are the history itself each target dataset is excluded from its own
    offset (leave-one-dataset-out); a method without history gets offset 0.
    """
    if scope not in {"global", "task"}:
        raise ValueError("optimism_corrected_validation scope must be `global` or `task`.")
    history = _budgeted_ranked(
        history_candidates,
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    target = (
        history
        if target_is_history
        else _budgeted_ranked(
            target_candidates,
            max_budget_seconds=max_budget_seconds,
            budget_feature=budget_feature,
        )
    )
    keys = ["method"] if scope == "global" else ["task", "method"]
    history = history.assign(_gap=history["test_rank_norm"] - history["val_rank_norm"])
    by_dataset = history.groupby(["dataset", *keys], as_index=False).agg(
        _gap_sum=("_gap", "sum"), _gap_count=("_gap", "size")
    )
    totals = by_dataset.groupby(keys, as_index=False).agg(
        _total_sum=("_gap_sum", "sum"), _total_count=("_gap_count", "sum")
    )
    scored = target.merge(totals, on=keys, how="left")
    if target_is_history:
        scored = scored.merge(by_dataset, on=["dataset", *keys], how="left")
        own_sum = scored["_gap_sum"].fillna(0.0)
        own_count = scored["_gap_count"].fillna(0.0)
    else:
        own_sum = 0.0
        own_count = 0.0
    remaining = scored["_total_count"].fillna(0.0) - own_count
    offset = np.where(
        remaining > 0,
        (scored["_total_sum"].fillna(0.0) - own_sum) / remaining.where(remaining > 0, 1.0),
        0.0,
    )
    utility = scored["val_rank_norm"].to_numpy(dtype=float) + offset
    details = pd.Series(
        [
            f"utility={float(value):.6g};optimism_offset={float(delta):.6g};scope={scope}"
            f";max_budget_seconds={'inf' if max_budget_seconds is None else f'{float(max_budget_seconds):.6g}'}"
            f";lodo={int(target_is_history)};scale=optimism_corrected_validation"
            for value, delta in zip(utility, offset)
        ],
        index=scored.index,
    )
    return _selector_rows_from_utility(
        scored,
        method_name=method_name,
        utility=pd.Series(utility, index=scored.index),
        details=details,
    )


def _metafeature_knn_selector(
    history_candidates: pd.DataFrame,
    target_candidates: pd.DataFrame,
    *,
    method_name: str,
    descriptors: pd.DataFrame,
    feature_columns: list[str],
    top_k: int,
    validation_weight: float,
    max_budget_seconds: float | None,
    budget_feature: str,
    target_is_history: bool,
) -> pd.DataFrame:
    """Meta-learning prior from the k nearest same-task history datasets.

    Dataset descriptors are standardised on the eligible history datasets. The prior of
    method m is the unit-weighted mean normalised test rank of m over the neighbours'
    units, and U = w_v * r_val + (1 - w_v) * prior. With w_v = 0 this is the classic
    nearest-dataset warm-start choice; the target never contributes to its own prior.
    """
    if descriptors is None:
        raise ValueError(f"`{method_name}` requires standard dataset meta-features.")
    history = _budgeted_ranked(
        history_candidates,
        max_budget_seconds=max_budget_seconds,
        budget_feature=budget_feature,
    )
    target = (
        history
        if target_is_history
        else _budgeted_ranked(
            target_candidates,
            max_budget_seconds=max_budget_seconds,
            budget_feature=budget_feature,
        )
    )
    features = descriptors.set_index("dataset")[feature_columns].astype(float)
    missing = sorted(
        set(history["dataset"].astype(str)).union(target["dataset"].astype(str))
        - set(features.index.astype(str))
    )
    if missing:
        raise ValueError(f"`{method_name}` has no meta-features for datasets: {missing[:10]}")
    sums = history.groupby(["dataset", "method"])["test_rank_norm"].agg(["sum", "size"])
    history_tasks = history.groupby("dataset")["task"].first().astype(str)
    frames: list[pd.DataFrame] = []
    for dataset in sorted(str(value) for value in target["dataset"].unique()):
        target_rows = target[target["dataset"].astype(str) == dataset].copy()
        task = str(target_rows["task"].iloc[0])
        pool = sorted(
            name for name, name_task in history_tasks.items() if name != dataset and name_task == task
        )
        if not pool:
            pool = sorted(name for name in history_tasks.index if name != dataset)
        pool_features = features.loc[pool]
        mean = pool_features.mean(axis=0)
        scale = pool_features.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        distances = np.sqrt(
            (((pool_features - mean) / scale - (features.loc[dataset] - mean) / scale) ** 2).sum(axis=1)
        )
        neighbors = (
            distances.rename("distance")
            .reset_index()
            .sort_values(["distance", "dataset"], kind="stable")["dataset"]
            .head(int(top_k))
            .tolist()
        )
        neighbor_sums = sums.loc[sums.index.get_level_values("dataset").isin(neighbors)]
        per_method = neighbor_sums.groupby(level="method").sum()
        prior = (per_method["sum"] / per_method["size"]).rename("knn_prior")
        fallback = float(per_method["sum"].sum() / per_method["size"].sum())
        target_rows["knn_prior"] = target_rows["method"].map(prior).fillna(fallback).astype(float)
        target_rows["_knn_utility"] = (
            float(validation_weight) * target_rows["val_rank_norm"].astype(float)
            + (1.0 - float(validation_weight)) * target_rows["knn_prior"]
        )
        neighbor_label = ",".join(neighbors)
        target_rows["_knn_details"] = [
            f"utility={float(value):.6g};knn_prior={float(prior_value):.6g};w_val={float(validation_weight):.6g}"
            f";k={int(top_k)};knn_neighbors={neighbor_label};lodo={int(target_is_history)}"
            f";scale=metafeature_knn_prior"
            for value, prior_value in zip(target_rows["_knn_utility"], target_rows["knn_prior"])
        ]
        frames.append(target_rows)
    scored = pd.concat(frames, ignore_index=True)
    return _selector_rows_from_utility(
        scored,
        method_name=method_name,
        utility=scored["_knn_utility"],
        details=scored["_knn_details"],
    )


STATIC_SELECTOR_KINDS = {"validation_rank", "tabrepo_prediction_ensemble"}


def _validation_rank_rows(candidate_rows: pd.DataFrame, selector_config: dict[str, Any]) -> pd.DataFrame:
    method_name = str(selector_config.get("method_name", ""))
    filter_keys = {
        "task_allowlist",
        "method_family_allowlist",
        "method_family_blocklist",
        "method_variant_allowlist",
        "method_name_include_contains",
        "method_name_exclude_contains",
    }
    rows = candidate_rows
    if any(selector_config.get(key) for key in filter_keys):
        rows, _ = _filter_candidate_rows(candidate_rows, selector_config)
    return _validation_rank_selector(rows, method_name=method_name)


def _tabrepo_ensemble_rows(
    candidate_rows: pd.DataFrame,
    selector_config: dict[str, Any],
    *,
    repository: Any,
) -> pd.DataFrame:
    if repository is None:
        raise ValueError("`tabrepo_prediction_ensemble` requires a loaded TabRepo repository.")
    return _tabrepo_prediction_ensemble_selector(
        candidate_rows,
        repository=repository,
        method_name=str(selector_config.get("method_name", "")),
        top_k=int(selector_config.get("top_k", 5)),
        ensemble_size=int(selector_config.get("ensemble_size", 5)),
        runtime_weight=float(selector_config.get("runtime_weight", 0.0)),
        max_budget_seconds=(
            float(selector_config["max_budget_seconds"])
            if selector_config.get("max_budget_seconds") is not None
            else None
        ),
        budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
        backend=str(selector_config.get("backend", "native")),
    )


def build_static_selector_rows(
    candidates: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
    derived_selectors: list[dict[str, Any]] | None,
    ensemble_repository: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Rows of selectors whose choice in a unit depends only on that unit.

    Raw validation, validation rank (optionally filtered) and TabRepo prediction
    ensembles never read history, so they are computed once for the whole artifact and
    sliced by dataset in every split and phase. The Caruana ensemble reseeds its random
    state on every call, so the cached rows equal a per-split recomputation.
    """
    rows = {
        selector_method_name: _raw_validation_selector(
            candidates,
            runtime_weight=runtime_weight,
            selector_method_name=selector_method_name,
        )
    }
    for raw_selector_config in derived_selectors or []:
        selector_config = dict(raw_selector_config)
        kind = str(selector_config["kind"])
        method_name = str(selector_config.get("method_name", ""))
        if kind == "validation_rank":
            rows[method_name] = _validation_rank_rows(candidates, selector_config)
        elif kind == "tabrepo_prediction_ensemble":
            rows[method_name] = _tabrepo_ensemble_rows(
                candidates,
                selector_config,
                repository=ensemble_repository,
            )
    return rows


def _slice_static_rows(
    static_rows: dict[str, pd.DataFrame] | None,
    method_name: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame | None:
    if not static_rows or method_name not in static_rows:
        return None
    frame = static_rows[method_name]
    datasets = set(candidates["dataset"].astype(str))
    return frame[frame["dataset"].astype(str).isin(datasets)].reset_index(drop=True)


def _required_weight_grid(selector_config: dict[str, Any]) -> list[dict[str, float]]:
    grid = selector_config.get("weight_grid")
    if not grid:
        raise ValueError(
            f"Selector `{selector_config.get('method_name')}` must declare its `weight_grid`; "
            "the split-aware benchmark does not fall back to an implicit default grid."
        )
    return _calibration_weight_grid(grid)


def _meta_feature_guard_kwargs(selector_config: dict[str, Any]) -> dict[str, Any]:
    def optional_list(key: str) -> list[str] | None:
        values = selector_config.get(key)
        return [str(value) for value in values] if values else None

    return {
        "choice_scope": str(selector_config.get("choice_scope", "task")),
        "leave_one_dataset_out": bool(selector_config.get("leave_one_dataset_out", True)),
        "feature_columns": optional_list("feature_columns"),
        "feature_column_prefixes": optional_list("feature_column_prefixes"),
        "exclude_feature_columns": optional_list("exclude_feature_columns"),
        "exclude_feature_column_prefixes": optional_list("exclude_feature_column_prefixes"),
        "feature_groups": (
            [dict(value) for value in selector_config.get("feature_groups", [])]
            if selector_config.get("feature_groups")
            else None
        ),
        "top_k": int(selector_config.get("top_k", 8)),
        "distance_power": float(selector_config.get("distance_power", 1.0)),
        "distance_epsilon": float(selector_config.get("distance_epsilon", 1e-6)),
        "runtime_tradeoff_weight": float(selector_config.get("runtime_tradeoff_weight", 0.0)),
        "default_selector_method": (
            str(selector_config["default_selector_method"])
            if selector_config.get("default_selector_method")
            else None
        ),
        "min_improvement_over_default": float(
            selector_config.get("min_improvement_over_default", 0.0)
        ),
    }


def _optional_budget(selector_config: dict[str, Any]) -> float | None:
    value = selector_config.get("max_budget_seconds")
    return float(value) if value is not None else None


def build_split_score_artifact_selector_rows(
    history_candidates: pd.DataFrame,
    target_candidates: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
    derived_selectors: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
    ensemble_repository: Any | None = None,
    precomputed_history_selectors: pd.DataFrame | None = None,
    static_selector_rows: dict[str, pd.DataFrame] | None = None,
    lodo_targets: set[str] | None = None,
    skip_history_kinds: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target selector rows with selector learning restricted to history rows.

    ``precomputed_history_selectors`` are the rows this function returned when called with
    the same history as its targets (the leave-one-dataset-out history rows). Passing them
    to the held-out call avoids rebuilding every history selector a second time; the
    result is identical. ``static_selector_rows`` (see ``build_static_selector_rows``)
    supply the history-independent selectors, sliced by dataset.

    ``lodo_targets`` restricts the leave-one-dataset-out rows of the calibrated and
    task-adaptive selectors to those datasets (every history dataset still enters the
    priors and objectives). ``skip_history_kinds`` names selector kinds whose history rows
    no later selector reads; for them only target rows are built when the targets are not
    the history. Both serve the per-target rebuild of ``inner_source_fit: per_target``.
    """
    global _META_FEATURE_MEMO, _GUARD_MEMO
    if history_candidates.empty:
        raise ValueError("Split-aware selector construction requires history candidates.")
    if target_candidates.empty:
        raise ValueError("Split-aware selector construction requires target candidates.")

    target_is_history = _targets_equal_history(history_candidates, target_candidates)
    if target_is_history:
        precomputed_history_selectors = None

    def cached_history(method_name: str) -> pd.DataFrame | None:
        if precomputed_history_selectors is None:
            return None
        rows = precomputed_history_selectors[
            precomputed_history_selectors["method"].astype(str) == method_name
        ]
        return rows.reset_index(drop=True) if not rows.empty else None

    def static_or(method_name: str, candidates: pd.DataFrame, build: Any) -> pd.DataFrame:
        rows = _slice_static_rows(static_selector_rows, method_name, candidates)
        return rows if rows is not None else build()

    _META_FEATURE_MEMO = {}
    _GUARD_MEMO = {} if _USE_GUARD_MEMO else None
    try:
        history_raw = static_or(
            selector_method_name,
            history_candidates,
            lambda: _raw_validation_selector(
                history_candidates,
                runtime_weight=runtime_weight,
                selector_method_name=selector_method_name,
            ),
        )
        history_selector_frames = [history_raw]
        target_selector_frames = [
            history_raw.copy()
            if target_is_history
            else static_or(
                selector_method_name,
                target_candidates,
                lambda: _raw_validation_selector(
                    target_candidates,
                    runtime_weight=runtime_weight,
                    selector_method_name=selector_method_name,
                ),
            )
        ]
        tuning_frames: list[pd.DataFrame] = []

        for raw_selector_config in derived_selectors or []:
            selector_config = dict(raw_selector_config)
            selector_kind = str(selector_config["kind"])
            method_name = str(selector_config.get("method_name", ""))
            if selector_kind == "include_selector_results" or selector_config.get(
                "selector_results_path"
            ):
                raise ValueError(
                    "Split-aware selector construction does not accept precomputed "
                    "selector_results_path/include_selector_results inputs."
                )
            reused_history = cached_history(method_name)
            target_tuning = pd.DataFrame()

            if selector_kind == "validation_rank":
                history_selector = (
                    reused_history
                    if reused_history is not None
                    else static_or(
                        method_name,
                        history_candidates,
                        lambda: _validation_rank_rows(history_candidates, selector_config),
                    )
                )
                target_selector = (
                    history_selector.copy()
                    if target_is_history
                    else static_or(
                        method_name,
                        target_candidates,
                        lambda: _validation_rank_rows(target_candidates, selector_config),
                    )
                )
            elif selector_kind in {"calibrated", "budgeted_calibrated"}:
                weight_grid = _required_weight_grid(selector_config)
                weight_selection = _validate_weight_selection(
                    str(selector_config.get("weight_selection", "lodo"))
                )
                history_rows, candidate_filter = _filter_candidate_rows(
                    history_candidates,
                    selector_config,
                )
                if reused_history is not None:
                    history_selector, history_tuning = reused_history, pd.DataFrame()
                else:
                    history_selector, history_tuning = _calibrated_selector(
                        history_rows,
                        method_name=method_name,
                        weight_grid=weight_grid,
                        leave_one_dataset_out=bool(
                            selector_config.get("leave_one_dataset_out", True)
                        ),
                        weight_selection=weight_selection,
                        max_budget_seconds=_optional_budget(selector_config),
                        budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
                        candidate_filter=candidate_filter,
                        n_unfiltered_candidates=int(history_candidates.shape[0]),
                        lodo_targets=lodo_targets,
                    )
                if target_is_history:
                    target_selector = history_selector.copy()
                    target_tuning = history_tuning.copy()
                else:
                    target_rows, _ = _filter_candidate_rows(target_candidates, selector_config)
                    target_selector, target_tuning = _calibrated_selector_from_history(
                        history_rows,
                        target_rows,
                        method_name=method_name,
                        weight_grid=weight_grid,
                        weight_selection=weight_selection,
                        max_budget_seconds=_optional_budget(selector_config),
                        budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
                        candidate_filter=candidate_filter,
                        n_unfiltered_candidates=int(target_candidates.shape[0]),
                    )
            elif selector_kind == "optimism_corrected_validation":
                options = {
                    "method_name": method_name,
                    "scope": str(selector_config.get("scope", "global")),
                    "max_budget_seconds": _optional_budget(selector_config),
                    "budget_feature": str(selector_config.get("budget_feature", "fit_seconds")),
                }
                history_selector = (
                    reused_history
                    if reused_history is not None
                    else _optimism_corrected_selector(
                        history_candidates,
                        history_candidates,
                        target_is_history=True,
                        **options,
                    )
                )
                target_selector = (
                    history_selector.copy()
                    if target_is_history
                    else _optimism_corrected_selector(
                        history_candidates,
                        target_candidates,
                        target_is_history=False,
                        **options,
                    )
                )
            elif selector_kind == "metafeature_knn_prior":
                options = {
                    "method_name": method_name,
                    "descriptors": dataset_descriptors,
                    "feature_columns": [
                        str(value)
                        for value in selector_config.get(
                            "feature_columns", STANDARD_META_FEATURE_COLUMNS
                        )
                    ],
                    "top_k": int(selector_config.get("top_k", 5)),
                    "validation_weight": float(selector_config.get("validation_weight", 0.0)),
                    "max_budget_seconds": _optional_budget(selector_config),
                    "budget_feature": str(selector_config.get("budget_feature", "fit_seconds")),
                }
                history_selector = (
                    reused_history
                    if reused_history is not None
                    else _metafeature_knn_selector(
                        history_candidates,
                        history_candidates,
                        target_is_history=True,
                        **options,
                    )
                )
                target_selector = (
                    history_selector.copy()
                    if target_is_history
                    else _metafeature_knn_selector(
                        history_candidates,
                        target_candidates,
                        target_is_history=False,
                        **options,
                    )
                )
            elif selector_kind == "task_adaptive_selector":
                candidate_selector_methods = [
                    str(value) for value in selector_config.get("candidate_selector_methods", [])
                ]
                history_source_rows = pd.concat(history_selector_frames, ignore_index=True)
                target_source_rows = pd.concat(target_selector_frames, ignore_index=True)
                if reused_history is not None:
                    history_selector, history_tuning = reused_history, pd.DataFrame()
                else:
                    history_selector, history_tuning = _task_adaptive_selector_from_selectors(
                        history_candidates,
                        history_source_rows,
                        method_name=method_name,
                        candidate_selector_methods=candidate_selector_methods,
                        choice_scope=str(selector_config.get("choice_scope", "task")),
                        leave_one_dataset_out=bool(
                            selector_config.get("leave_one_dataset_out", True)
                        ),
                        lodo_targets=lodo_targets,
                    )
                if target_is_history:
                    target_selector = history_selector.copy()
                    target_tuning = history_tuning.copy()
                else:
                    target_selector, target_tuning = _task_adaptive_selector_from_history(
                        history_candidates,
                        history_source_rows,
                        target_candidates,
                        target_source_rows,
                        method_name=method_name,
                        candidate_selector_methods=candidate_selector_methods,
                        choice_scope=str(selector_config.get("choice_scope", "task")),
                        leave_one_dataset_out=bool(
                            selector_config.get("leave_one_dataset_out", True)
                        ),
                    )
            elif selector_kind == "meta_feature_adaptive_selector":
                candidate_selector_methods = [
                    str(value) for value in selector_config.get("candidate_selector_methods", [])
                ]
                guard_options = _meta_feature_guard_kwargs(selector_config)
                history_source_rows = pd.concat(history_selector_frames, ignore_index=True)
                target_source_rows = pd.concat(target_selector_frames, ignore_index=True)
                if reused_history is not None:
                    history_selector, history_tuning = reused_history, pd.DataFrame()
                elif selector_kind in skip_history_kinds and not target_is_history:
                    history_selector, history_tuning = None, pd.DataFrame()
                else:
                    history_selector, history_tuning = _meta_feature_adaptive_selector_from_selectors(
                        history_candidates,
                        history_source_rows,
                        method_name=method_name,
                        candidate_selector_methods=candidate_selector_methods,
                        dataset_descriptors=dataset_descriptors,
                        **guard_options,
                    )
                if target_is_history:
                    target_selector = history_selector.copy()
                    target_tuning = history_tuning.copy()
                else:
                    target_selector, target_tuning = _meta_feature_adaptive_selector_from_history(
                        history_candidates,
                        history_source_rows,
                        target_candidates,
                        target_source_rows,
                        method_name=method_name,
                        candidate_selector_methods=candidate_selector_methods,
                        dataset_descriptors=dataset_descriptors,
                        **guard_options,
                    )
            elif selector_kind == "tabrepo_prediction_ensemble":
                history_selector = (
                    reused_history
                    if reused_history is not None
                    else static_or(
                        method_name,
                        history_candidates,
                        lambda: _tabrepo_ensemble_rows(
                            history_candidates,
                            selector_config,
                            repository=ensemble_repository,
                        ),
                    )
                )
                target_selector = (
                    history_selector.copy()
                    if target_is_history
                    else static_or(
                        method_name,
                        target_candidates,
                        lambda: _tabrepo_ensemble_rows(
                            target_candidates,
                            selector_config,
                            repository=ensemble_repository,
                        ),
                    )
                )
            else:
                raise ValueError(f"Unknown split-aware derived selector kind: {selector_kind}")

            if history_selector is not None:
                history_selector_frames.append(history_selector)
            target_selector_frames.append(target_selector)
            if not target_tuning.empty:
                target_tuning = target_tuning.copy()
                target_tuning["selector_method"] = method_name
                tuning_frames.append(target_tuning)
    finally:
        _META_FEATURE_MEMO = None
        _GUARD_MEMO = None

    selector_frame = pd.concat(target_selector_frames, ignore_index=True)
    tuning_result = (
        pd.concat(tuning_frames, ignore_index=True)
        if tuning_frames
        else _empty_tuning_frame()
    )
    return selector_frame[ARTIFACT_OUTPUT_COLUMNS].reset_index(drop=True), tuning_result


def build_score_artifact_result_rows(
    candidate_rows: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
    derived_selectors: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
    ensemble_repository: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = candidate_rows.copy()
    selector_frames = [
        _raw_validation_selector(
            candidate_rows,
            runtime_weight=runtime_weight,
            selector_method_name=selector_method_name,
        )
    ]
    tuning_frames: list[pd.DataFrame] = []
    for selector_config in derived_selectors or []:
        selector_kind = selector_config["kind"]
        if selector_kind == "include_selector_results":
            included = _load_selector_rows_from_result_file(
                selector_config["selector_results_path"],
                [str(value) for value in selector_config.get("selector_methods", [])],
            )
            selector_frames.append(included)
            continue
        method_name = selector_config["method_name"]
        if selector_kind == "validation_rank":
            selector_frames.append(
                _validation_rank_selector(candidate_rows, method_name=method_name)
            )
        elif selector_kind == "task_conditioned":
            selector, tuning = _task_conditioned_selector(
                candidate_rows,
                method_name=method_name,
                runtime_weight_grid=[
                    float(value)
                    for value in selector_config.get(
                        "runtime_weight_grid",
                        [0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
                    )
                ],
                tuning_runtime_weight=float(selector_config.get("tuning_runtime_weight", 0.05)),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind in {"calibrated", "budgeted_calibrated"}:
            selector_candidate_rows, candidate_filter = _filter_candidate_rows(
                candidate_rows,
                selector_config,
            )
            selector, tuning = _calibrated_selector(
                selector_candidate_rows,
                method_name=method_name,
                weight_grid=_calibration_weight_grid(selector_config.get("weight_grid")),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
                # Legacy full-artifact pipeline (results of the earlier versions): unchanged v1 behaviour.
                weight_selection=str(selector_config.get("weight_selection", "resubstitution")),
                max_budget_seconds=(
                    float(selector_config["max_budget_seconds"])
                    if selector_config.get("max_budget_seconds") is not None
                    else None
                ),
                budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
                candidate_filter=candidate_filter,
                n_unfiltered_candidates=int(candidate_rows.shape[0]),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind == "validation_confidence_prior":
            selector_candidate_rows, _ = _filter_candidate_rows(candidate_rows, selector_config)
            selector, tuning = _validation_confidence_prior_selector(
                selector_candidate_rows,
                method_name=method_name,
                weight_grid=_calibration_weight_grid(selector_config.get("weight_grid")),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
                validation_rank_margin=float(
                    selector_config.get("validation_rank_margin", 1.0)
                ),
                validation_rank_norm_margin=(
                    float(selector_config["validation_rank_norm_margin"])
                    if selector_config.get("validation_rank_norm_margin") is not None
                    else None
                ),
                max_budget_seconds=(
                    float(selector_config["max_budget_seconds"])
                    if selector_config.get("max_budget_seconds") is not None
                    else None
                ),
                budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind == "task_adaptive_selector":
            candidate_selector_methods = [
                str(value) for value in selector_config.get("candidate_selector_methods", [])
            ]
            source_selector_rows = pd.concat(selector_frames, ignore_index=True)
            loaded_source_rows = pd.DataFrame(columns=ARTIFACT_OUTPUT_COLUMNS)
            if selector_config.get("selector_results_path"):
                loaded_source_rows = _load_selector_rows_from_result_file(
                    selector_config["selector_results_path"],
                    candidate_selector_methods,
                )
                source_selector_rows = pd.concat(
                    [source_selector_rows, loaded_source_rows],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"],
                    keep="last",
                )
                if bool(selector_config.get("include_source_selectors", False)):
                    existing_methods = set(
                        str(value)
                        for frame in selector_frames
                        for value in frame["method"].unique()
                    )
                    new_source_rows = loaded_source_rows[
                        ~loaded_source_rows["method"].astype(str).isin(existing_methods)
                    ]
                    if not new_source_rows.empty:
                        selector_frames.append(new_source_rows)
            selector, tuning = _task_adaptive_selector_from_selectors(
                candidate_rows,
                source_selector_rows,
                method_name=method_name,
                candidate_selector_methods=candidate_selector_methods,
                choice_scope=str(selector_config.get("choice_scope", "task")),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind == "risk_constrained_selector":
            candidate_selector_methods = [
                str(value) for value in selector_config.get("candidate_selector_methods", [])
            ]
            task_reference_method = str(
                selector_config.get(
                    "task_reference_method",
                    "meta_selector_task_adaptive_budgeted",
                )
            )
            family_reference_method = str(
                selector_config.get(
                    "family_reference_method",
                    "meta_selector_family_prior_budgeted",
                )
            )
            fallback_selector_methods = [
                str(value)
                for value in selector_config.get(
                    "fallback_selector_methods",
                    [task_reference_method, family_reference_method],
                )
            ]
            source_methods = list(
                dict.fromkeys(
                    [
                        *candidate_selector_methods,
                        task_reference_method,
                        family_reference_method,
                        *fallback_selector_methods,
                    ]
                )
            )
            source_selector_rows = pd.concat(selector_frames, ignore_index=True)
            loaded_source_rows = pd.DataFrame(columns=ARTIFACT_OUTPUT_COLUMNS)
            if selector_config.get("selector_results_path"):
                loaded_source_rows = _load_selector_rows_from_result_file(
                    selector_config["selector_results_path"],
                    source_methods,
                )
                source_selector_rows = pd.concat(
                    [source_selector_rows, loaded_source_rows],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"],
                    keep="last",
                )
                if bool(selector_config.get("include_source_selectors", False)):
                    existing_methods = set(
                        str(value)
                        for frame in selector_frames
                        for value in frame["method"].unique()
                    )
                    new_source_rows = loaded_source_rows[
                        ~loaded_source_rows["method"].astype(str).isin(existing_methods)
                    ]
                    if not new_source_rows.empty:
                        selector_frames.append(new_source_rows)
            selector, tuning = _risk_constrained_selector_from_selectors(
                candidate_rows,
                source_selector_rows,
                method_name=method_name,
                candidate_selector_methods=candidate_selector_methods,
                task_reference_method=task_reference_method,
                family_reference_method=family_reference_method,
                fallback_selector_methods=fallback_selector_methods,
                choice_scope=str(selector_config.get("choice_scope", "task")),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
                noninferiority_margin_rank=float(
                    selector_config.get("noninferiority_margin_rank", 0.5)
                ),
                noninferiority_alpha=float(selector_config.get("noninferiority_alpha", 0.05)),
                min_training_datasets=int(selector_config.get("min_training_datasets", 5)),
                require_runtime_gain=bool(selector_config.get("require_runtime_gain", True)),
                min_runtime_gain_seconds=float(
                    selector_config.get("min_runtime_gain_seconds", 0.0)
                ),
                runtime_tradeoff_weight=float(
                    selector_config.get("runtime_tradeoff_weight", 0.0)
                ),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind == "meta_feature_adaptive_selector":
            candidate_selector_methods = [
                str(value) for value in selector_config.get("candidate_selector_methods", [])
            ]
            source_selector_rows = pd.concat(selector_frames, ignore_index=True)
            loaded_source_rows = pd.DataFrame(columns=ARTIFACT_OUTPUT_COLUMNS)
            if selector_config.get("selector_results_path"):
                loaded_source_rows = _load_selector_rows_from_result_file(
                    selector_config["selector_results_path"],
                    candidate_selector_methods,
                )
                source_selector_rows = pd.concat(
                    [source_selector_rows, loaded_source_rows],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=[*ARTIFACT_UNIT_COLUMNS, "method", "selected_member"],
                    keep="last",
                )
                if bool(selector_config.get("include_source_selectors", False)):
                    existing_methods = set(
                        str(value)
                        for frame in selector_frames
                        for value in frame["method"].unique()
                    )
                    new_source_rows = loaded_source_rows[
                        ~loaded_source_rows["method"].astype(str).isin(existing_methods)
                    ]
                    if not new_source_rows.empty:
                        selector_frames.append(new_source_rows)
            selector, tuning = _meta_feature_adaptive_selector_from_selectors(
                candidate_rows,
                source_selector_rows,
                method_name=method_name,
                candidate_selector_methods=candidate_selector_methods,
                choice_scope=str(selector_config.get("choice_scope", "task")),
                leave_one_dataset_out=bool(selector_config.get("leave_one_dataset_out", True)),
                feature_columns=(
                    [str(value) for value in selector_config.get("feature_columns", [])]
                    if selector_config.get("feature_columns")
                    else None
                ),
                feature_column_prefixes=(
                    [
                        str(value)
                        for value in selector_config.get("feature_column_prefixes", [])
                    ]
                    if selector_config.get("feature_column_prefixes")
                    else None
                ),
                exclude_feature_columns=(
                    [
                        str(value)
                        for value in selector_config.get("exclude_feature_columns", [])
                    ]
                    if selector_config.get("exclude_feature_columns")
                    else None
                ),
                exclude_feature_column_prefixes=(
                    [
                        str(value)
                        for value in selector_config.get(
                            "exclude_feature_column_prefixes", []
                        )
                    ]
                    if selector_config.get("exclude_feature_column_prefixes")
                    else None
                ),
                feature_groups=(
                    [
                        dict(value)
                        for value in selector_config.get("feature_groups", [])
                    ]
                    if selector_config.get("feature_groups")
                    else None
                ),
                dataset_descriptors=dataset_descriptors,
                top_k=int(selector_config.get("top_k", 8)),
                distance_power=float(selector_config.get("distance_power", 1.0)),
                distance_epsilon=float(selector_config.get("distance_epsilon", 1e-6)),
                runtime_tradeoff_weight=float(
                    selector_config.get("runtime_tradeoff_weight", 0.0)
                ),
                default_selector_method=(
                    str(selector_config["default_selector_method"])
                    if selector_config.get("default_selector_method")
                    else None
                ),
                min_improvement_over_default=float(
                    selector_config.get("min_improvement_over_default", 0.0)
                ),
            )
            selector_frames.append(selector)
            tuning["selector_method"] = method_name
            tuning_frames.append(tuning)
        elif selector_kind == "tabrepo_prediction_ensemble":
            if ensemble_repository is None:
                raise ValueError(
                    "`tabrepo_prediction_ensemble` requires a loaded TabRepo repository."
                )
            selector = _tabrepo_prediction_ensemble_selector(
                candidate_rows,
                repository=ensemble_repository,
                method_name=method_name,
                top_k=int(selector_config.get("top_k", 5)),
                ensemble_size=int(selector_config.get("ensemble_size", 5)),
                runtime_weight=float(selector_config.get("runtime_weight", 0.0)),
                max_budget_seconds=(
                    float(selector_config["max_budget_seconds"])
                    if selector_config.get("max_budget_seconds") is not None
                    else None
                ),
                budget_feature=str(selector_config.get("budget_feature", "fit_seconds")),
                backend=str(selector_config.get("backend", "native")),
            )
            selector_frames.append(selector)
        else:
            raise ValueError(f"Unknown derived selector kind: {selector_kind}")

    selector_frame = pd.concat(selector_frames, ignore_index=True)
    result = pd.concat([candidate_rows, selector_frame], ignore_index=True)
    tuning_result = (
        pd.concat(tuning_frames, ignore_index=True)
        if tuning_frames
        else pd.DataFrame(
            columns=[
                "task",
                "dataset",
                "runtime_weight",
                "validation_objective",
                "tuning_runtime_weight",
                "n_training_units",
                "n_training_datasets",
                "leave_one_dataset_out",
                "fallback_used",
                "calibration_objective",
                "n_weight_candidates",
                *CALIBRATION_WEIGHT_COLUMNS,
                *BUDGET_TUNING_COLUMNS,
                *FILTER_TUNING_COLUMNS,
                *ADAPTIVE_TUNING_COLUMNS,
                *RISK_TUNING_COLUMNS,
                *CONFIDENCE_TUNING_COLUMNS,
                "selector_method",
            ]
        )
    )
    return result, tuning_result


def build_tabarena_result_rows(
    frame: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
    derived_selectors: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_score_artifact_result_rows(
        _candidate_rows_from_tabarena(frame),
        runtime_weight=runtime_weight,
        selector_method_name=selector_method_name,
        derived_selectors=derived_selectors,
        dataset_descriptors=dataset_descriptors,
    )


def build_tabrepo_result_rows(
    frame: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
    derived_selectors: list[dict[str, Any]] | None = None,
    dataset_descriptors: pd.DataFrame | None = None,
    ensemble_repository: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_score_artifact_result_rows(
        _candidate_rows_from_tabrepo(frame),
        runtime_weight=runtime_weight,
        selector_method_name=selector_method_name,
        derived_selectors=derived_selectors,
        dataset_descriptors=dataset_descriptors,
        ensemble_repository=ensemble_repository,
    )


def tabarena_to_result_rows(
    frame: pd.DataFrame,
    *,
    runtime_weight: float,
    selector_method_name: str,
) -> pd.DataFrame:
    result, _ = build_tabarena_result_rows(
        frame,
        runtime_weight=runtime_weight,
        selector_method_name=selector_method_name,
    )
    return result


def write_artifact_metadata(
    frame: pd.DataFrame,
    result: pd.DataFrame,
    output_dir: Path,
    artifact_config: dict[str, Any],
    config_path: str | Path,
    dataset_descriptors: pd.DataFrame | None = None,
) -> None:
    meta = (
        frame.groupby(["dataset", "problem_type", "metric"], as_index=False)
        .agg(n_folds=("fold", "nunique"), n_methods=("method", "nunique"))
        .sort_values(["dataset", "problem_type", "metric"])
    )
    meta.to_csv(output_dir / "dataset_meta_features.csv", index=False)

    metadata = {
        "artifact_kind": artifact_config.get("kind", "tabarena_results"),
        "artifact_url": artifact_config.get(
            "url",
            "" if artifact_config.get("kind") == "tabrepo_repository" else TABARENA_RESULTS_URL,
        ),
        "artifact_local_path": artifact_config.get(
            "local_path",
            "" if artifact_config.get("kind") == "tabrepo_repository" else "data/tabarena/df_results.csv",
        ),
        "artifact_context_name": artifact_config.get("context_name", ""),
        "artifact_cache_dir": artifact_config.get("cache_dir", ""),
        "config_path": str(config_path),
        "n_source_rows": int(frame.shape[0]),
        "n_result_rows": int(result.shape[0]),
        "n_datasets": int(frame["dataset"].nunique()),
        "n_candidate_methods": int(frame["method"].nunique()),
        "n_folds_min": int(frame.groupby("dataset")["fold"].nunique().min()),
        "n_folds_max": int(frame.groupby("dataset")["fold"].nunique().max()),
        "n_selector_methods": int(result[result["method_type"] == "selector"]["method"].nunique()),
        "n_descriptor_datasets": (
            int(dataset_descriptors["dataset"].nunique())
            if dataset_descriptors is not None
            else 0
        ),
        "n_descriptor_features": (
            int(len([column for column in dataset_descriptors.columns if column != "dataset"]))
            if dataset_descriptors is not None
            else 0
        ),
        "generated_at_epoch": time.time(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if dataset_descriptors is not None:
        dataset_descriptors.to_csv(output_dir / "dataset_descriptor_features.csv", index=False)


def _selector_configs_require_tabrepo_predictions(config: dict[str, Any]) -> bool:
    return any(
        dict(selector).get("kind") == "tabrepo_prediction_ensemble"
        for selector in config.get("selector", {}).get("derived_selectors", [])
    )


def run_artifact_experiment(config: dict[str, Any], config_path: str | Path) -> Path:
    experiment = config["experiment"]
    artifact_config = dict(config["artifact_source"])
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_kind = str(artifact_config.get("kind", "tabarena_results"))
    repository = None
    if artifact_kind == "tabarena_results":
        frame = load_tabarena_results(artifact_config)
    elif artifact_kind == "tabrepo_repository":
        if _selector_configs_require_tabrepo_predictions(config):
            artifact_config["load_predictions"] = True
        repository = load_tabrepo_repository(artifact_config)
        frame = load_tabrepo_results(artifact_config, repository=repository)
    else:
        raise ValueError(f"Unknown artifact source kind: {artifact_kind}")

    dataset_descriptors = load_dataset_descriptors(config.get("dataset_descriptors"))
    selector_config = config.get("selector", {})
    selector_method_name = selector_config.get("selector_method_name", "meta_selector_single")
    if artifact_kind == "tabrepo_repository":
        result, tuning = build_tabrepo_result_rows(
            frame,
            runtime_weight=float(selector_config.get("runtime_weight", 0.0)),
            selector_method_name=selector_method_name,
            derived_selectors=selector_config.get("derived_selectors", []),
            dataset_descriptors=dataset_descriptors,
            ensemble_repository=repository,
        )
    else:
        result, tuning = build_tabarena_result_rows(
            frame,
            runtime_weight=float(selector_config.get("runtime_weight", 0.0)),
            selector_method_name=selector_method_name,
            derived_selectors=selector_config.get("derived_selectors", []),
            dataset_descriptors=dataset_descriptors,
        )
    result_path = output_dir / "results.csv"
    result.to_csv(result_path, index=False)
    if not tuning.empty:
        tuning.to_csv(output_dir / "selector_tuning.csv", index=False)
    write_artifact_metadata(
        frame,
        result,
        output_dir,
        artifact_config,
        config_path,
        dataset_descriptors=dataset_descriptors,
    )
    return result_path
