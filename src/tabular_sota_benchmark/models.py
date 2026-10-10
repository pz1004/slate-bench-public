from __future__ import annotations

import importlib.util
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


TREE_BASELINES = {
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
}


def is_model_available(name: str) -> bool:
    if name in {
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }:
        return True
    module_name = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }.get(name)
    if module_name is None:
        return False
    return importlib.util.find_spec(module_name) is not None


def make_model(name: str, task: str, seed: int, n_jobs: int = 1) -> Any:
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")

    is_classification = task == "classification"

    if name == "decision_tree":
        estimator = (
            DecisionTreeClassifier(random_state=seed, min_samples_leaf=2)
            if is_classification
            else DecisionTreeRegressor(random_state=seed, min_samples_leaf=2)
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "random_forest":
        estimator = (
            RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
            if is_classification
            else RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "extra_trees":
        estimator = (
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
            if is_classification
            else ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "hist_gradient_boosting":
        estimator = (
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.06,
                random_state=seed,
                early_stopping=True,
            )
            if is_classification
            else HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.06,
                random_state=seed,
                early_stopping=True,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        estimator = (
            XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
            )
            if is_classification
            else XGBRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        estimator = (
            LGBMClassifier(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            )
            if is_classification
            else LGBMRegressor(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        estimator = (
            CatBoostClassifier(
                iterations=180,
                depth=6,
                learning_rate=0.05,
                loss_function="MultiClass" if is_classification else "RMSE",
                random_seed=seed,
                thread_count=n_jobs,
                verbose=False,
                allow_writing_files=False,
            )
            if is_classification
            else CatBoostRegressor(
                iterations=180,
                depth=6,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=seed,
                thread_count=n_jobs,
                verbose=False,
                allow_writing_files=False,
            )
        )
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    raise ValueError(f"Unknown model name: {name}")
