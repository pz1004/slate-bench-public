"""The prespecified fresh-validation experiment (docs/fresh_validation_protocol.md).

Each task's rows are split four ways per repetition: training rows; construction-validation
rows, which early stopping, tuning and ensemble weights use; an equally large set of fresh
selection-validation rows, which nothing uses before selection; and test rows. Every learner
family yields three frozen candidates (default, tuned, tuned + ensemble). Validation-best then
chooses among the same candidates once on the reused construction-validation scores and once
on the fresh scores. Test labels are read only to score the frozen candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

VARIANTS = ["default", "tuned", "tuned + ensemble"]
SPLITS = ["train", "construction", "selection", "test"]
PROBABILITY_FLOOR = 1e-15


# ----------------------------------------------------------------------------- seeds


def derived_seed(master: int, *keys: int) -> int:
    """A 31-bit seed determined by the master seed and integer keys."""
    state = np.random.SeedSequence([int(master), *[int(key) for key in keys]]).generate_state(1)[0]
    return int(state) % (2**31 - 1)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Task:
    dataset_id: int
    name: str
    path: Path
    target: str
    drop: tuple[str, ...] = ()


def load_task(task: Task) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Features, integer labels and class names; rows without a label are dropped."""
    frame = pd.read_parquet(task.path)
    if task.target not in frame.columns:
        raise ValueError(f"{task.name}: target {task.target!r} is not a column")
    labels = frame[task.target]
    keep = labels.notna().to_numpy()
    columns = [column for column in frame.columns if column != task.target and column not in set(task.drop)]
    features = frame.loc[keep, columns].reset_index(drop=True)
    classes, y = np.unique(labels[keep].astype(str).to_numpy(), return_inverse=True)
    return features, y.astype(np.int64), [str(value) for value in classes]


def feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric and categorical columns: numeric and boolean dtypes are numeric, all others categorical."""
    numeric, categorical = [], []
    for column in features.columns:
        dtype = features[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def capped_rows(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """At most ``cap`` row indices, stratified by class, in their original order."""
    indices = np.arange(len(y))
    if len(y) <= cap:
        return indices
    kept, _ = train_test_split(indices, train_size=int(cap), stratify=y, random_state=seed)
    return np.sort(kept)


def four_way_split(y: np.ndarray, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Stratified training, construction-validation, selection-validation and test indices."""
    total = sum(float(fractions[name]) for name in SPLITS)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"Split fractions sum to {total}, not 1")
    if not math.isclose(float(fractions["construction"]), float(fractions["selection"]), abs_tol=1e-12):
        raise ValueError("The construction and selection validation sets must be equally large")
    indices = np.arange(len(y))
    rest, test = train_test_split(indices, test_size=float(fractions["test"]), stratify=y, random_state=seed)
    rest_share = 1.0 - float(fractions["test"])
    train, validation = train_test_split(
        rest, train_size=float(fractions["train"]) / rest_share, stratify=y[rest], random_state=seed + 1
    )
    construction, selection = train_test_split(validation, test_size=0.5, stratify=y[validation], random_state=seed + 2)
    return {
        "train": np.sort(train),
        "construction": np.sort(construction),
        "selection": np.sort(selection),
        "test": np.sort(test),
    }


@dataclass
class Encoded:
    """One unit's feature matrices, every encoder fitted on the training rows only."""

    trees: dict[str, np.ndarray]
    trees_imputed: dict[str, np.ndarray]
    dense: dict[str, np.ndarray]
    categorical_mask: np.ndarray


def encode(features: pd.DataFrame, parts: dict[str, np.ndarray], *, max_onehot: int = 20, max_native: int = 250) -> Encoded:
    numeric, categorical = feature_types(features)
    frame = features.copy()
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__missing__").astype(object)
    train = frame.iloc[parts["train"]]

    trees = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    trees.fit(train)
    tree_matrices = {name: np.asarray(trees.transform(frame.iloc[rows]), dtype=float) for name, rows in parts.items()}
    cardinality = [len(values) for values in trees.named_transformers_["categorical"].categories_] if categorical else []
    mask = np.array([False] * len(numeric) + [count <= max_native for count in cardinality], dtype=bool)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(tree_matrices["train"])
    imputed = {name: imputer.transform(matrix) for name, matrix in tree_matrices.items()}

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=max_onehot, sparse_output=False),
                categorical,
            )
        )
    dense = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    dense.fit(train)
    dense_matrices = {name: np.asarray(dense.transform(frame.iloc[rows]), dtype=float) for name, rows in parts.items()}
    return Encoded(trees=tree_matrices, trees_imputed=imputed, dense=dense_matrices, categorical_mask=mask)


# ----------------------------------------------------------------------------- configurations


def sample_configurations(space: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` random configurations; each entry of ``space`` is [kind, ...]."""
    rng = np.random.default_rng(seed)
    configurations = []
    for _ in range(int(n)):
        config: dict[str, Any] = {}
        for name, spec in space.items():
            kind = spec[0]
            if kind == "choice":
                options = spec[1]
                value = options[int(rng.integers(len(options)))]
                config[name] = tuple(value) if isinstance(value, list) else value
            elif kind == "uniform":
                config[name] = float(rng.uniform(float(spec[1]), float(spec[2])))
            elif kind == "loguniform":
                config[name] = float(math.exp(rng.uniform(math.log(float(spec[1])), math.log(float(spec[2])))))
            else:
                raise ValueError(f"Unknown parameter kind {kind!r}")
        configurations.append(config)
    return configurations


def family_configurations(family_spec: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    """The default first, then the random configurations."""
    default = {key: tuple(value) if isinstance(value, list) else value for key, value in dict(family_spec["default"]).items()}
    return [default, *sample_configurations(dict(family_spec["space"]), n, seed)]


# ----------------------------------------------------------------------------- models


def log_loss(y: np.ndarray, proba: np.ndarray) -> float:
    """Mean negative log-likelihood with probabilities floored at 1e-15 and renormalized."""
    clipped = np.clip(proba, PROBABILITY_FLOOR, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(y)), y])))


def _aligned(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


@dataclass
class Fit:
    predictions: dict[str, np.ndarray]
    fit_seconds: float
    info: dict[str, Any]
    predict_seconds: float = 0.0


def _fit_mlp(params: dict[str, Any], X: dict[str, np.ndarray], y: dict[str, np.ndarray], n_classes: int, seed: int, schedule: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """An MLP early-stopped on construction-validation log loss, restored to its best epoch."""
    model = MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        batch_size=int(params["batch_size"]),
        solver="adam",
        random_state=seed,
    )
    classes = np.arange(n_classes)
    best_loss, best_state, best_epoch, waited = math.inf, None, 0, 0
    for epoch in range(1, int(schedule["max_epochs"]) + 1):
        model.partial_fit(X["train"], y["train"], classes=classes)
        loss = log_loss(y["construction"], _aligned(model, X["construction"], n_classes))
        if loss < best_loss:
            best_loss, best_epoch, waited = loss, epoch, 0
            best_state = (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_))
        else:
            waited += 1
            if waited >= int(schedule["patience"]):
                break
    if best_state is not None:
        model.coefs_, model.intercepts_ = best_state
    return model, {"best_epoch": best_epoch}


def fit_configuration(
    family: str,
    params: dict[str, Any],
    data: Encoded,
    y: dict[str, np.ndarray],
    n_classes: int,
    seed: int,
    schedule: dict[str, Any],
) -> Fit:
    """Fit one configuration on the training rows and predict every split.

    Only the construction-validation labels may shape the fit (early stopping); the
    selection-validation and test labels are never passed in.
    """
    labels = {"train": y["train"], "construction": y["construction"]}
    info: dict[str, Any] = {}
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        if family == "hist_gradient_boosting":
            X = data.trees
            model = HistGradientBoostingClassifier(
                learning_rate=float(params["learning_rate"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                l2_regularization=float(params["l2_regularization"]),
                max_features=float(params["max_features"]),
                max_iter=int(schedule["hgb_max_iter"]),
                early_stopping=True,
                n_iter_no_change=int(schedule["hgb_patience"]),
                scoring="loss",
                categorical_features=data.categorical_mask if data.categorical_mask.any() else None,
                random_state=seed,
            )
            model.fit(X["train"], labels["train"], X_val=X["construction"], y_val=labels["construction"])
            info["iterations"] = int(model.n_iter_)
        elif family in {"random_forest", "extra_trees"}:
            X = data.trees_imputed
            estimator = RandomForestClassifier if family == "random_forest" else ExtraTreesClassifier
            max_features = params["max_features"]
            model = estimator(
                n_estimators=int(schedule["forest_trees"]),
                max_features=max_features if isinstance(max_features, str) else float(max_features),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
                criterion=str(params["criterion"]),
                bootstrap=family == "random_forest",
                n_jobs=1,
                random_state=seed,
            )
            model.fit(X["train"], labels["train"])
        elif family == "logistic_regression":
            X = data.dense
            model = LogisticRegression(
                C=float(params["C"]),
                class_weight=params["class_weight"],
                max_iter=int(schedule["lr_max_iter"]),
            )
            model.fit(X["train"], labels["train"])
        elif family == "knn":
            X = data.dense
            model = KNeighborsClassifier(
                n_neighbors=min(int(params["n_neighbors"]), len(labels["train"])),
                weights=str(params["weights"]),
                p=int(params["p"]),
            )
            model.fit(X["train"], labels["train"])
        elif family == "mlp":
            X = data.dense
            model, info = _fit_mlp(params, X, labels, n_classes, seed, schedule)
        else:
            raise ValueError(f"Unknown family {family!r}")
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = {name: _aligned(model, X[name], n_classes) for name in ["construction", "selection", "test"]}
    predict_seconds = time.perf_counter() - started
    for name, proba in predictions.items():
        if not np.all(np.isfinite(proba)):
            raise FloatingPointError(f"{family}: non-finite probabilities on the {name} rows")
    return Fit(predictions=predictions, fit_seconds=fit_seconds, info=info, predict_seconds=predict_seconds)


# ----------------------------------------------------------------------------- candidates


def greedy_ensemble(probabilities: Sequence[np.ndarray], y: np.ndarray, steps: int) -> np.ndarray:
    """Greedy ensemble selection with replacement (Caruana et al., 2004) on one validation set.

    Each step adds the member whose inclusion gives the lowest log loss of the averaged
    probabilities; exact ties go to the earlier member. Returns the member weights.
    """
    if not probabilities:
        raise ValueError("No members to ensemble")
    counts = np.zeros(len(probabilities), dtype=int)
    running = np.zeros_like(probabilities[0], dtype=float)
    for step in range(1, int(steps) + 1):
        losses = [log_loss(y, (running + member) / step) for member in probabilities]
        best = int(np.argmin(losses))
        running = running + probabilities[best]
        counts[best] += 1
    return counts / counts.sum()


def family_candidates(
    family: str,
    configurations: list[dict[str, Any]],
    data: Encoded,
    y: dict[str, np.ndarray],
    n_classes: int,
    *,
    seeds: list[int],
    schedule: dict[str, Any],
    ensemble_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-configuration records and the family's three frozen candidates.

    A configuration that raises or gives non-finite probabilities is dropped from the search;
    if the default fails, the family yields no candidates.
    """
    fits: dict[int, Fit] = {}
    records = []
    for index, params in enumerate(configurations):
        record: dict[str, Any] = {"family": family, "configuration": index, "params": json.dumps(params, sort_keys=True, default=str)}
        try:
            fit = fit_configuration(family, params, data, y, n_classes, seeds[index], schedule)
        except Exception as error:  # noqa: BLE001 - every failure is recorded, none is silent
            record.update({"failed": True, "error": f"{type(error).__name__}: {error}"[:300]})
            records.append(record)
            continue
        fits[index] = fit
        record.update(
            {
                "failed": False,
                "error": "",
                "fit_seconds": fit.fit_seconds,
                "predict_seconds": fit.predict_seconds,
                **{f"{name}_loss": log_loss(y[name], fit.predictions[name]) for name in ["construction", "selection", "test"]},
                **{f"info_{key}": value for key, value in fit.info.items()},
            }
        )
        records.append(record)
    config_frame = pd.DataFrame(records)
    if 0 not in fits:
        return config_frame, pd.DataFrame()

    ok = config_frame[~config_frame["failed"]].copy()
    tuned_index = int(ok.sort_values(["construction_loss", "configuration"], kind="stable").iloc[0]["configuration"])
    members = sorted(fits)
    weights = greedy_ensemble([fits[index].predictions["construction"] for index in members], y["construction"], ensemble_steps)
    ensemble = {
        name: sum(weight * fits[index].predictions[name] for index, weight in zip(members, weights) if weight > 0)
        for name in ["construction", "selection", "test"]
    }
    used = [index for index, weight in zip(members, weights) if weight > 0]

    def candidate(variant: str, predictions: dict[str, np.ndarray], fit_seconds: float, detail: str) -> dict[str, Any]:
        return {
            "family": family,
            "variant": variant,
            "candidate": f"{family} ({variant})",
            "fit_seconds": float(fit_seconds),
            "detail": detail,
            **{f"{name}_loss": log_loss(y[name], predictions[name]) for name in ["construction", "selection", "test"]},
        }

    rows = [
        candidate("default", fits[0].predictions, fits[0].fit_seconds, "configuration=0"),
        candidate("tuned", fits[tuned_index].predictions, fits[tuned_index].fit_seconds, f"configuration={tuned_index}"),
        candidate(
            "tuned + ensemble",
            ensemble,
            sum(fits[index].fit_seconds for index in used),
            ";".join(f"{index}:{weight:.4f}" for index, weight in zip(members, weights) if weight > 0),
        ),
    ]
    return config_frame, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- selection rules


def candidate_order(families: Sequence[str]) -> dict[str, int]:
    """The fixed candidate order: families in the configured order, variants as in VARIANTS."""
    return {f"{family} ({variant})": index for index, (family, variant) in enumerate((f, v) for f in families for v in VARIANTS)}


def lexicographic_choice(units: pd.DataFrame, score: str, unit_columns: Sequence[str]) -> pd.DataFrame:
    """Per unit, the lowest ``score``, then the lowest fit time, then the earlier candidate."""
    ordered = units.sort_values([*unit_columns, score, "fit_seconds", "order"], kind="stable")
    return ordered.groupby(list(unit_columns), sort=False).head(1)


def rule_choices(candidates: pd.DataFrame, unit_columns: Sequence[str]) -> pd.DataFrame:
    """The prespecified rules' choices: one row per unit and rule, with the chosen candidate."""
    rules = []
    reused = lexicographic_choice(candidates, "construction_loss", unit_columns).assign(rule="validation_best_reused")
    fresh = lexicographic_choice(candidates, "selection_loss", unit_columns).assign(rule="validation_best_fresh")
    defaults = candidates[candidates["variant"].eq("default")]
    best_default = lexicographic_choice(defaults, "construction_loss", unit_columns)
    rules += [reused, fresh, best_default.assign(rule="validation_best_defaults")]
    deployed = []
    for _, row in best_default.iterrows():
        mask = np.ones(len(candidates), dtype=bool)
        for column in unit_columns:
            mask &= candidates[column].to_numpy() == row[column]
        same_family = candidates[mask & candidates["family"].eq(row["family"]).to_numpy()]
        for variant in ["tuned + ensemble", "tuned", "default"]:
            match = same_family[same_family["variant"].eq(variant)]
            if not match.empty:
                deployed.append(match.iloc[0])
                break
    rules.append(pd.DataFrame(deployed).assign(rule="defaults_rule"))
    oracle = lexicographic_choice(candidates, "test_loss", unit_columns).assign(rule="oracle")
    rules.append(oracle)
    return pd.concat(rules, ignore_index=True)
