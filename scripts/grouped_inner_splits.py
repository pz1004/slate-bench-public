"""Rerun the grouped TabRepo audit with source groups excluded at every inner stage.

scripts/grouped_outer_splits.py keeps each group of related datasets
(configs/tabrepo_source_groups.csv) on one side of every outer split, but its training-side
estimates still leave out one dataset at a time, so a training dataset's own relatives train
the rules scored on it. This script runs the same 24 outer splits (same groups, repeats and
seed) and also leaves out each training dataset's whole group wherever the benchmark leaves out
that dataset: in the priors, in the leave-one-out weight choice of the scoring rules, in the
source rows that the adaptive policies read, in the task-adaptive objective, in the guards'
neighbours and therefore in the nested choice. A held-out dataset's group is never on the
training side, so the held-out rows need no change beyond the weight choice, which also leaves
out whole groups.

The frozen pipeline files are not edited. The driver patches, in its own process and the
workers it forks, three places where the excluded dataset is chosen:

* ``artifacts._factorize_columns`` returns group codes when it is asked for dataset codes. It
  is called with ``["dataset"]`` only as the exclusion key of the leave-one-out priors and
  weight choice (``_loo_prior_features``) and of the fast leave-one-dataset-out rows
  (``_calibrated_selector_fast_lodo``). The fast path is therefore forced for every
  leave-one-out calibrated selector, its ``lodo_targets`` are translated to group labels, and
  its rows are filtered back to the requested datasets and restored to the dataset order of
  the unpatched code.
* ``artifacts._task_adaptive_selector_from_selectors`` and
  ``score_level_policy_benchmark._per_target_adaptive_rows`` are rebuilt from their own source
  with the lines that exclude one dataset replaced by lines that exclude its group
  (``GROUP_EDITS``); every other line is the frozen code.

With every dataset its own group (``--inner-group-column none``) the patched code reproduces
the outer-grouped run exactly, which tests/test_grouped_inner.py checks on a synthetic artifact
and the README's check on split 0 of the real run. The training-side rows of the reference
rules (offset-corrected validation, meta-feature kNN) keep their own leave-one-dataset-out
exclusion; they feed no policy, no train-selected rule and no reported result.
"""

from __future__ import annotations

import __future__
import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_OUTER_PATH = SCRIPTS / "grouped_outer_splits.py"
_OUTER_SPEC = importlib.util.spec_from_file_location("grouped_outer_splits_for_inner", _OUTER_PATH)
assert _OUTER_SPEC is not None and _OUTER_SPEC.loader is not None
outer = importlib.util.module_from_spec(_OUTER_SPEC)
sys.modules[_OUTER_SPEC.name] = outer
_OUTER_SPEC.loader.exec_module(outer)

bench = outer.bench
from tabular_sota_benchmark import artifacts  # noqa: E402

# Lines replaced in the frozen source, with the number of times each must occur.
GROUP_EDITS: dict[str, list[tuple[str, str]]] = {
    "_task_adaptive_selector_from_selectors": [
        (
            'training = training[training["dataset"].astype(str) != dataset]',
            'training = training[~training["dataset"].astype(str).isin(_group_members(dataset))]',
        ),
    ],
    "_per_target_adaptive_rows": [
        (
            'mask = train_candidates["dataset"].astype(str).eq(dataset)',
            'mask = train_candidates["dataset"].astype(str).isin(_group_members(dataset))',
        ),
        (
            "target = train_candidates[mask]",
            'target = train_candidates[train_candidates["dataset"].astype(str).eq(dataset)]',
        ),
        (
            "and name != dataset}",
            "and name not in _group_members(dataset)}",
        ),
    ],
}

_LABELS: dict[str, str] = {}
_MEMBERS: dict[str, frozenset[str]] = {}
_ORIGINALS: dict[tuple[Any, str], Any] = {}


def group_label(dataset: str) -> str:
    """Namespaced group label: group names can equal dataset names (boston, splice, madelon)."""
    name = str(dataset)
    return _LABELS.get(name, f"single:{name}")


def group_members(dataset: str) -> frozenset[str]:
    name = str(dataset)
    return _MEMBERS.get(name, frozenset({name}))


def set_groups(groups: dict[str, str]) -> None:
    """Dataset -> group id; datasets not listed are singletons."""
    _LABELS.clear()
    _MEMBERS.clear()
    by_group: dict[str, set[str]] = {}
    for dataset, group in groups.items():
        if group:
            _LABELS[str(dataset)] = f"group:{group}"
            by_group.setdefault(str(group), set()).add(str(dataset))
    for members in by_group.values():
        frozen = frozenset(members)
        for dataset in members:
            _MEMBERS[dataset] = frozen


def _factorize_columns_grouped(rows: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, Any]:
    if list(columns) != ["dataset"]:
        return _ORIGINALS[(artifacts, "_factorize_columns")](rows, columns)
    values = rows["dataset"].astype(str).map(group_label)
    codes, uniques = pd.factorize(values, sort=False)
    return codes.astype(np.int64, copy=False), uniques


def _fast_lodo_grouped(ranked: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = kwargs.pop("lodo_targets", None)
    targets = None if requested is None else {group_label(name) for name in requested}
    selector, tuning = _ORIGINALS[(artifacts, "_calibrated_selector_fast_lodo")](
        ranked, lodo_targets=targets, **kwargs
    )
    names = selector["dataset"].astype(str)
    if requested is not None:
        keep = names.isin({str(name) for name in requested}).to_numpy()
        selector, names = selector[keep], names[keep]
    # The unpatched code concatenates its rows dataset by dataset in sorted order.
    position = {name: index for index, name in enumerate(sorted(set(names)))}
    order = np.argsort(names.map(position).to_numpy(), kind="stable")
    return selector.iloc[order].reset_index(drop=True), tuning


def _rebuild(module: Any, name: str, edits: list[tuple[str, str]]) -> Callable[..., Any]:
    source = inspect.getsource(getattr(module, name))
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f"{name}: expected one occurrence of {old!r}, found {source.count(old)}.")
        source = source.replace(old, new)
    namespace = module.__dict__
    code = compile(
        source,
        f"<grouped {module.__name__}.{name}>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    scope: dict[str, Any] = {}
    exec(code, namespace, scope)  # noqa: S102 - the source is the frozen function, edited above
    return scope[name]


def _set(module: Any, name: str, value: Any) -> None:
    key = (module, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(module, name, None)
    setattr(module, name, value)


def apply_patches(groups: dict[str, str]) -> None:
    """Patch the benchmark for group-aware inner estimates; ``restore_patches`` undoes it."""
    set_groups(groups)
    if _ORIGINALS:
        return
    _set(artifacts, "_group_members", group_members)
    _set(bench, "_group_members", group_members)
    _set(artifacts, "_factorize_columns", _factorize_columns_grouped)
    _set(artifacts, "_calibrated_selector_fast_lodo", _fast_lodo_grouped)
    # The slow path leaves out one dataset by name, so every leave-one-out calibrated
    # selector takes the fast path, which keys its exclusion on _factorize_columns.
    _set(artifacts, "_calibrated_selector", functools.partial(artifacts._calibrated_selector, fast_path_min_rows=0))
    rebuilt_adaptive = _rebuild(artifacts, "_task_adaptive_selector_from_selectors", GROUP_EDITS["_task_adaptive_selector_from_selectors"])
    _set(artifacts, "_task_adaptive_selector_from_selectors", rebuilt_adaptive)
    rebuilt_rows = _rebuild(bench, "_per_target_adaptive_rows", GROUP_EDITS["_per_target_adaptive_rows"])
    _set(bench, "_per_target_adaptive_rows", rebuilt_rows)


def restore_patches() -> None:
    for (module, name), value in reversed(list(_ORIGINALS.items())):
        if value is None:
            delattr(module, name)
        else:
            setattr(module, name, value)
    _ORIGINALS.clear()
    set_groups({})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> None:
    outer_groups = outer.read_groups(Path(args.groups), str(args.group_column))
    inner_groups = {} if args.inner_group_column == "none" else outer.read_groups(Path(args.groups), str(args.inner_group_column))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {}

    def split_datasets(candidate_rows: pd.DataFrame, *, n_splits: int, heldout_fraction: float, seed: int):
        dataset_task = bench._task_by_dataset(candidate_rows)
        present = set(dataset_task["dataset"].astype(str))
        kept = {name: group for name, group in outer_groups.items() if name in present}
        splits = outer.grouped_splits(dataset_task, kept, repeats=int(args.repeats), seed=int(args.split_seed))
        if len(splits) != int(n_splits):
            raise ValueError(f"{len(splits)} grouped splits, but the run expects {n_splits}.")
        inner_kept = {name: group for name, group in inner_groups.items() if name in present}
        record.update(
            {
                "groups_file": str(args.groups),
                "groups_file_sha256": _sha256(Path(args.groups)),
                "group_column": str(args.group_column),
                "inner_group_column": str(args.inner_group_column),
                "split_seed": int(args.split_seed),
                "folds": outer.FOLDS,
                "repeats": int(args.repeats),
                "n_groups_used": len(set(kept.values())),
                "n_grouped_datasets": len(kept),
                "n_inner_groups_used": len(set(inner_kept.values())),
                "n_inner_grouped_datasets": len(inner_kept),
                "patched": {
                    "artifacts._factorize_columns": "group codes for ['dataset']",
                    "artifacts._calibrated_selector_fast_lodo": "lodo_targets as group labels; rows filtered and reordered",
                    "artifacts._calibrated_selector": "fast_path_min_rows=0",
                    **{name: [new for _, new in edits] for name, edits in GROUP_EDITS.items()},
                },
                "splits": [
                    {"split_id": split_id, "n_train": len(train), "n_heldout": len(heldout), "heldout": heldout}
                    for split_id, train, heldout, _ in splits
                ],
            }
        )
        (output_dir / "grouped_splits.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return splits

    apply_patches(inner_groups)
    bench._split_datasets = split_datasets
    bench_args = argparse.Namespace(
        config=Path(args.config),
        output_dir=output_dir,
        splits=int(args.repeats) * outer.FOLDS,
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
    parser.add_argument("--group-column", default="group", choices=["group", "documented_group"], help="groups of the outer splits")
    parser.add_argument(
        "--inner-group-column",
        default="group",
        choices=["group", "documented_group", "none"],
        help="groups left out of the inner estimates; 'none' leaves out one dataset at a time, as the outer-grouped run does",
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=20260621)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
