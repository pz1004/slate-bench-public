from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_sota_benchmark.data import TabularDataset


def compute_meta_features(dataset: TabularDataset) -> dict[str, float | int | str]:
    X = dataset.X
    y = dataset.y
    missing_fraction = float(X.isna().to_numpy().mean())
    numeric_fraction = float(
        sum(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes) / max(X.shape[1], 1)
    )
    features: dict[str, float | int | str] = {
        "dataset": dataset.name,
        "task": dataset.task,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_fraction": missing_fraction,
        "numeric_fraction": numeric_fraction,
    }
    if dataset.task == "classification":
        class_counts = y.value_counts(normalize=True)
        features["n_classes"] = int(class_counts.shape[0])
        features["class_entropy"] = float(
            -(class_counts * np.log(class_counts + 1e-12)).sum()
        )
        features["minority_class_fraction"] = float(class_counts.min())
    else:
        features["target_std"] = float(y.std())
        features["target_skew"] = float(y.skew())
    return features

