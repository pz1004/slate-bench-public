from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "procedure_level_jackknife.py"
SPEC = importlib.util.spec_from_file_location("procedure_level_jackknife", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
jackknife = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jackknife)


def test_cyclic_design_holds_out_every_kept_dataset_once_per_cycle() -> None:
    names = [f"b{index:02d}" for index in range(30)] + [f"m{index}" for index in range(8)] + [f"r{index:02d}" for index in range(13)]
    tasks = pd.Series(["binary"] * 30 + ["multiclass"] * 8 + ["regression"] * 13, index=names)
    keys = pd.Series(np.random.default_rng(0).uniform(size=len(names)), index=names)
    for subset_id in range(5):
        kept = jackknife.subset_datasets(tasks, n_delete=10, rng=np.random.default_rng(subset_id))
        counts = pd.Series(0, index=kept)
        for fold in range(4):
            train, heldout = jackknife.cyclic_split(kept, tasks, keys, fold=fold, n_folds=4)
            assert sorted(train + heldout) == sorted(kept)
            counts[heldout] += 1
            # Every task keeps training datasets.
            assert set(tasks[train].unique()) == set(tasks[kept].unique())
        assert (counts == 1).all()
    # The fold of a kept dataset does not depend on which datasets were deleted.
    full = jackknife.cyclic_split(names, tasks, keys, fold=1, n_folds=4)[1]
    kept = jackknife.subset_datasets(tasks, n_delete=10, rng=np.random.default_rng(7))
    assert set(jackknife.cyclic_split(kept, tasks, keys, fold=1, n_folds=4)[1]) == set(full) & set(kept)
