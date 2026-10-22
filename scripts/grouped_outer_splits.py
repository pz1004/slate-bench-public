"""Rerun a nested audit on outer splits that keep groups of related datasets together.

The 100 outer splits of the TabRepo audit are drawn per dataset, so one variant of a source
(configs/tabrepo_source_groups.csv) can be held out while another trains the policies. This
script draws group-aware splits instead: R repeats of a 4-fold ``StratifiedGroupKFold``
over the datasets, stratified by task type and with every group in one fold, so each split
holds out about a quarter of the datasets and each dataset is held out once per repeat. It
then runs the unchanged benchmark (scripts/score_level_policy_benchmark.py, with its split
generator replaced by these splits) and writes the benchmark's standard output layout, which
scripts/score_level_review.py and scripts/score_level_policy_pool_ablation.py read unchanged.

Only the outer splits change. The training-side estimates stay leave-one-dataset-out, so a
training dataset's own siblings still train the policies scored on it, and the fold sizes
vary a little with the group sizes. ``grouped_splits.json`` records the grouping, the seed
and every split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_grouped", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
# The benchmark's worker pool pickles its split function by module name.
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)

FOLDS = 4


def read_groups(path: Path, column: str) -> dict[str, str]:
    """Dataset -> group id from the source-group file; listed datasets with an empty value
    and unlisted datasets are singletons."""
    frame = pd.read_csv(path, comment="#", dtype=str).fillna("")
    return {row["dataset"]: row[column] for _, row in frame.iterrows() if row[column]}


def grouped_splits(
    dataset_task: pd.DataFrame, groups: dict[str, str], *, repeats: int, seed: int
) -> list[tuple[int, list[str], list[str], str]]:
    datasets = sorted(dataset_task["dataset"].astype(str))
    tasks = dataset_task.set_index("dataset")["task"].astype(str).reindex(datasets)
    labels = np.array([groups.get(name, f"single:{name}") for name in datasets], dtype=object)
    splits = []
    for repeat in range(int(repeats)):
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=int(seed) + repeat)
        for fold, (train_index, heldout_index) in enumerate(folds.split(datasets, tasks.to_numpy(), labels)):
            train = [datasets[i] for i in train_index]
            heldout = [datasets[i] for i in heldout_index]
            for task in tasks.unique():
                if not set(tasks[tasks.eq(task)].index) & set(train):
                    raise ValueError(f"Split {repeat}/{fold} leaves task {task} without training datasets.")
            train_groups = {labels[i] for i in train_index}
            if any(labels[i] in train_groups for i in heldout_index):
                raise ValueError("A group is split across training and held-out datasets.")
            splits.append((repeat * FOLDS + fold, train, heldout, "group_task_stratified"))
    return splits


def run(args: argparse.Namespace) -> None:
    groups = read_groups(Path(args.groups), str(args.group_column))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {}

    def split_datasets(candidate_rows: pd.DataFrame, *, n_splits: int, heldout_fraction: float, seed: int):
        dataset_task = bench._task_by_dataset(candidate_rows)
        kept = {name: group for name, group in groups.items() if name in set(dataset_task["dataset"].astype(str))}
        splits = grouped_splits(dataset_task, kept, repeats=int(args.repeats), seed=int(args.split_seed))
        if len(splits) != int(n_splits):
            raise ValueError(f"{len(splits)} grouped splits, but the run expects {n_splits}.")
        record.update(
            {
                "groups_file": str(args.groups),
                "group_column": str(args.group_column),
                "split_seed": int(args.split_seed),
                "folds": FOLDS,
                "repeats": int(args.repeats),
                "n_groups_used": len(set(kept.values())),
                "n_grouped_datasets": len(kept),
                "splits": [
                    {"split_id": split_id, "n_train": len(train), "n_heldout": len(heldout), "heldout": heldout}
                    for split_id, train, heldout, _ in splits
                ],
            }
        )
        (output_dir / "grouped_splits.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return splits

    bench._split_datasets = split_datasets
    bench_args = argparse.Namespace(
        config=Path(args.config),
        output_dir=output_dir,
        splits=int(args.repeats) * FOLDS,
        heldout_fraction=None,
        seed=None,
        jobs=int(args.jobs),
        resume=bool(args.resume),
        command=sys.argv,
    )
    bench.run(bench_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="a nested benchmark config (read only)")
    parser.add_argument("--groups", type=Path, default=Path("configs/tabrepo_source_groups.csv"))
    parser.add_argument("--group-column", default="group", choices=["group", "documented_group"])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=20260621)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
