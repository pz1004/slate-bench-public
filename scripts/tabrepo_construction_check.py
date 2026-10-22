"""Check how the analysed TabRepo slice built its validation and test predictions.

TabRepo's paper describes every model as bagged: B fold models fitted on non-overlapping
cross-validation splits, out-of-fold predictions as validation predictions and the mean of
the B models as test predictions. TabArena's paper reports instead that TabRepo fitted
RandomForest and ExtraTrees once, with 300 trees, and used their out-of-bag predictions for
validation. For binary classification the stored prediction vectors tell the two apart:

* a single forest of 300 trees with pure leaves predicts multiples of 1/300 on test, an
  average of 8 such forests multiples of 1/2400;
* out-of-bag predictions average a different number of trees for every row (about 300/e), so
  they are rarely multiples of 1/300, whereas out-of-fold predictions of a 300-tree forest
  always are;
* a nearest-neighbour model with uniform weights predicts multiples of 1/k for k neighbours
  from one model, and multiples of 1/(8k) from a bag of eight. TabRepo's KNeighbors_c1 sets no
  neighbour count; its validation predictions turn out to be multiples of 1/5 and its test
  predictions multiples of 1/6, which the check reports for k = 5 and k + 1 = 6.

The check reads the default configuration (``_c1``) of RandomForest, ExtraTrees and KNN, and of
CatBoost and LightGBM as controls whose predictions lie on no such lattice, on every binary
dataset of the slice and all three folds. Leaves that are not pure move a forest's predictions
off the lattice, so shares below one are expected on some datasets. Predictions of exactly 0
or 1 lie on every lattice and are left out of the shares.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:  # pragma: no cover - run from another directory
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark import artifacts  # noqa: E402

CONFIGS = {
    "RandomForest_c1_BAG_L1": "rf",
    "ExtraTrees_c1_BAG_L1": "xt",
    "KNeighbors_c1_BAG_L1": "knn",
    "CatBoost_c1_BAG_L1": "cat",
    "LightGBM_c1_BAG_L1": "gbm",
}
TREES = 300
FOLD_MODELS = 8
KNN_NEIGHBOURS = 5  # the lattice of KNeighbors_c1's validation predictions (test: k + 1)
TOLERANCE = 1e-6
MAX_DENOMINATOR = 2400


def interior(values: np.ndarray) -> np.ndarray:
    """Predictions strictly between 0 and 1; 0 and 1 lie on every lattice."""
    values = np.asarray(values, dtype=float)
    return values[(values > TOLERANCE) & (values < 1.0 - TOLERANCE)]


def lattice_share(values: np.ndarray, denominator: int) -> float:
    """Share of interior predictions within TOLERANCE of a multiple of 1/denominator."""
    values = interior(values)
    if values.size == 0:
        return float("nan")
    scaled = values * float(denominator)
    return float(np.mean(np.abs(scaled - np.round(scaled)) < TOLERANCE * float(denominator)))


def reduced_denominators(values: np.ndarray, max_denominator: int = MAX_DENOMINATOR) -> np.ndarray:
    """Smallest q <= max_denominator with q * value within tolerance of an integer; 0 if none."""
    values = np.asarray(values, dtype=float)
    result = np.zeros(values.shape[0], dtype=np.int64)
    pending = np.ones(values.shape[0], dtype=bool)
    for q in range(1, int(max_denominator) + 1):
        if not pending.any():
            break
        scaled = values[pending] * q
        hit = np.abs(scaled - np.round(scaled)) < TOLERANCE * q
        index = np.flatnonzero(pending)[hit]
        result[index] = q
        pending[index] = False
    return result


def task_row(dataset: str, fold: int, config: str, val: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    key = CONFIGS[config]
    denominators = reduced_denominators(interior(val))
    row: dict[str, Any] = {
        "dataset": dataset,
        "fold": int(fold),
        "config": config,
        "family": key,
        "n_val": int(val.size),
        "n_test": int(test.size),
        "n_val_interior": int(interior(val).size),
        "n_test_interior": int(interior(test).size),
        "test_share_1_300": lattice_share(test, TREES),
        "test_share_1_2400": lattice_share(test, TREES * FOLD_MODELS),
        "val_share_1_300": lattice_share(val, TREES),
        "val_median_denominator": float(np.median(denominators[denominators > 0])) if (denominators > 0).any() else float("nan"),
        "val_share_no_denominator": float(np.mean(denominators == 0)) if denominators.size else float("nan"),
    }
    if key == "knn":
        k = KNN_NEIGHBOURS
        row["val_share_1_k"] = lattice_share(val, k)
        row["test_share_1_k"] = lattice_share(test, k)
        row["test_share_1_k1"] = lattice_share(test, k + 1)
        # A bag of eight would put test predictions on 1/(8k) or 1/(8(k+1)) and off 1/k, 1/(k+1).
        row["test_share_bag"] = max(lattice_share(test, FOLD_MODELS * k), lattice_share(test, FOLD_MODELS * (k + 1)))
    return row


def summarise(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, frame in tasks.groupby("config", sort=False):
        row = {
            "config": config,
            "family": CONFIGS[config],
            "n_tasks": int(frame.shape[0]),
            "n_datasets": int(frame["dataset"].nunique()),
            "n_tasks_with_interior": int(frame["test_share_1_300"].notna().sum()),
            "tasks_test_on_1_300": float(np.mean(frame["test_share_1_300"].dropna() >= 0.99)),
            "median_test_share_1_300": float(frame["test_share_1_300"].median()),
            "tasks_val_on_1_300": float(np.mean(frame["val_share_1_300"].dropna() >= 0.99)),
            "median_val_share_1_300": float(frame["val_share_1_300"].median()),
            "median_val_denominator": float(frame["val_median_denominator"].median()),
        }
        if "val_share_1_k" in frame.columns and frame["val_share_1_k"].notna().any():
            knn = frame.dropna(subset=["val_share_1_k", "test_share_1_k"])
            row["tasks_val_on_1_k"] = float(np.mean(knn["val_share_1_k"] >= 0.99))
            row["tasks_test_on_1_k_or_1_k1"] = float(np.mean((knn["test_share_1_k"] >= 0.99) | (knn["test_share_1_k1"] >= 0.99)))
            row["tasks_test_bag_only"] = float(
                np.mean((knn["test_share_bag"] >= 0.99) & (knn["test_share_1_k"] < 0.99) & (knn["test_share_1_k1"] < 0.99))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = dict(config["artifacts"][0]["artifact_source"])
    source["load_predictions"] = True
    allowlist = [line.strip() for line in Path(source["dataset_allowlist_file"]).read_text().splitlines() if line.strip() and not line.startswith("#")]
    repo = artifacts.load_tabrepo_repository(source)
    available = set(repo.configs())
    missing = [name for name in CONFIGS if name not in available]
    if missing:
        raise ValueError(f"Configurations not in the slice: {missing}")
    binary = [name for name in sorted(allowlist) if str(repo.dataset_info(name)["problem_type"]) == "binary"]
    if args.limit:
        binary = binary[: int(args.limit)]
    rows = []
    for dataset in binary:
        for fold in repo.folds:
            for name in CONFIGS:
                val = np.asarray(repo.predict_val(dataset, fold, name), dtype=float).ravel()
                test = np.asarray(repo.predict_test(dataset, fold, name), dtype=float).ravel()
                rows.append(task_row(dataset, int(fold), name, val, test))
    tasks = pd.DataFrame(rows)
    summary = summarise(tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output_dir / "construction_by_task.csv", index=False)
    summary.to_csv(output_dir / "construction_summary.csv", index=False)
    metadata = {
        "config": str(args.config),
        "context": source.get("context_name"),
        "n_binary_datasets": len(binary),
        "folds": [int(fold) for fold in repo.folds],
        "configs": CONFIGS,
        "trees": TREES,
        "fold_models": FOLD_MODELS,
        "knn_neighbours": KNN_NEIGHBOURS,
        "tolerance": TOLERANCE,
        "lattice_threshold": 0.99,
        "seconds": round(time.time() - started, 1),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/score_level_tabrepo_nested.yaml"), help="read only")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output/v2/tabrepo_construction"))
    parser.add_argument("--limit", type=int, default=0, help="first N binary datasets only (for a quick look)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
