from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class TabularDataset:
    name: str
    task: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def classes(self) -> list[int] | None:
        if self.task != "classification":
            return None
        return sorted(int(value) for value in self.y.unique())


def _frame_from_sklearn(bundle: object, name: str, task: str) -> TabularDataset:
    X = pd.DataFrame(bundle.data, columns=list(bundle.feature_names))
    y = pd.Series(bundle.target, name="target")
    return TabularDataset(name=name, task=task, X=X, y=y)


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == values.notna().sum():
            encoded[column] = numeric.astype(float)
            continue
        codes, _ = pd.factorize(values.astype("string"), sort=True)
        codes = codes.astype(float)
        codes[codes < 0] = np.nan
        encoded[column] = codes
    return encoded


def _tabarena_metadata() -> pd.DataFrame:
    path = Path("data/tabarena/tabarena_dataset_metadata.csv")
    if not path.exists():
        raise FileNotFoundError(
            "TabArena metadata is required for local retraining subset: "
            "`data/tabarena/tabarena_dataset_metadata.csv`"
        )
    return pd.read_csv(path)


def _tabarena_parquet_path(dataset_id: int) -> Path:
    return (
        Path("data/tabarena/openml_parquet")
        / f"{dataset_id // 10000:04d}"
        / str(dataset_id)
        / f"dataset_{dataset_id}.pq"
    )


def _load_tabarena_openml_dataset(dataset_name: str) -> TabularDataset:
    metadata = _tabarena_metadata()
    matches = metadata[metadata["dataset_name"].astype(str) == dataset_name]
    if matches.empty:
        raise ValueError(f"Unknown TabArena dataset: {dataset_name}")
    row = matches.iloc[0]
    dataset_id = int(row["dataset_id"])
    target_name = str(row["target_feature"])
    parquet_path = _tabarena_parquet_path(dataset_id)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Cached OpenML parquet file missing for {dataset_name}: {parquet_path}"
        )
    frame = pd.read_parquet(parquet_path)
    if target_name not in frame.columns:
        raise ValueError(
            f"Target column `{target_name}` missing from cached dataset `{dataset_name}`."
        )
    y_raw = frame[target_name]
    X = _encode_features(frame.drop(columns=[target_name]))
    problem_type = str(row["problem_type"])
    if problem_type == "regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna()
        return TabularDataset(
            name=dataset_name,
            task="regression",
            X=X.loc[valid].reset_index(drop=True),
            y=pd.Series(y.loc[valid].to_numpy(dtype=float), name="target"),
        )
    encoder = LabelEncoder()
    valid = y_raw.notna()
    y_values = encoder.fit_transform(y_raw.loc[valid].astype(str))
    return TabularDataset(
        name=dataset_name,
        task="classification",
        X=X.loc[valid].reset_index(drop=True),
        y=pd.Series(y_values.astype(int), name="target"),
    )


def load_builtin_dataset(name: str) -> TabularDataset:
    if name.startswith("tabarena:"):
        return _load_tabarena_openml_dataset(name.split(":", 1)[1])
    if name == "breast_cancer":
        return _frame_from_sklearn(load_breast_cancer(), name, "classification")
    if name == "wine":
        return _frame_from_sklearn(load_wine(), name, "classification")
    if name == "diabetes":
        return _frame_from_sklearn(load_diabetes(), name, "regression")
    raise ValueError(f"Unknown built-in dataset: {name}")
