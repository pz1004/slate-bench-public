"""The TabRepo frozen-fit check (docs/fresh_validation_protocol.md, second part).

One stratified 4-fold repeat of grouped outer splits, run by the unedited fully grouped driver
(``scripts/grouped_inner_splits.py --repeats 1 --split-seed 20261022``), fits every procedure
on three quarters of TabRepo's source groups, with whole groups left out of every inner
estimate. The declared fold, fold 0, is the frozen-fit evaluation: the procedures fitted on its
training groups are held fixed and scored on its held-out groups, which no fitted quantity
saw, so ordinary tests over the held-out datasets target the frozen-fit estimand for that
fit. The other three folds are computed because the driver runs whole repeats; each is
reported as its own evaluation and never pooled with fold 0.

Validation-best is the literal rule here, as in every TabRepo contrast
(scripts/literal_validation_best.py). For each pool the declared contrasts are nested
selection minus the train-selected rules, validation-best and the family-prior rule, tested
with the dataset bootstrap, the paired t-test and a bootstrap over the held-out source groups,
each Holm-adjusted within the pool's family.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import holm  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


literal = _load("literal_validation_best_frozen_fit", "literal_validation_best.py")
groups_module = _load("tabrepo_source_groups_frozen_fit", "tabrepo_source_groups.py")

NESTED = "nested_policy_selected"
FAMILY_PRIOR = "meta_selector_family_prior_budgeted"
FAMILIES = {
    "ensemble": [
        "train_selected_fixed_reference__ensemble_inclusive",
        "train_selected_fixed_reference__score_only",
        literal.VB,
        FAMILY_PRIOR,
    ],
    "score_only": ["train_selected_fixed_reference__score_only", literal.VB, FAMILY_PRIOR],
}
DECLARED_FOLD = 0


_LOADED: dict[str, dict[str, Any]] = {}


def _nested(run_dir: Path) -> dict[str, Any]:
    key = str(Path(run_dir).resolve())
    if key not in _LOADED:
        _LOADED[key] = literal.review.load_nested_output(Path(run_dir))
    return _LOADED[key]


def fold_review(run_dir: Path, review_config: Path, units_path: Path, split_id: int, references: list[str]) -> dict[str, Any]:
    """The frozen review code on one fold, with the literal validation-best reference."""
    nested = _nested(run_dir)
    config = yaml.safe_load(review_config.read_text(encoding="utf-8")) or {}
    rows = literal._literal_rows(units_path)
    mapping = {literal.VB: literal.VB_RUNTIME}
    units = nested["units"]
    units = units[units["split_id"].astype(int).eq(split_id)].copy()
    units["method"] = literal._rename(units["method"], mapping)
    units["original_method"] = literal._rename(units["original_method"].fillna(""), mapping)
    training = nested["training"]
    training = training[training["split_id"].astype(int).eq(split_id)].copy()
    training["method"] = literal._rename(training["method"], mapping)
    splits = nested["splits"]
    splits = splits[splits["split_id"].astype(int).eq(split_id)].copy()
    splits["selected_policy"] = literal._rename(splits["selected_policy"], mapping)
    base = units[units["phase"].eq("heldout") & units["method"].eq(literal.VB_RUNTIME) & ~units["role"].astype(str).eq("alias")]
    reference = literal._with_values(base, rows)
    reference["method"], reference["role"], reference["original_method"] = literal.VB, "reference", ""
    derived = pd.concat([units, reference[units.columns]], ignore_index=True)
    pools = {
        str(name): [literal.VB_RUNTIME if str(member) == literal.VB else str(member) for member in members]
        for name, members in config.get("fixed_reference_pools", {}).items()
    }
    config = {**config, "families": {"frozen_fit": {"left": NESTED, "references": references}}}
    result = literal.run_review(derived, training, splits, config, pools=pools)
    result["selected_policy"] = str(splits["selected_policy"].iloc[0])
    return result


def cluster_tests(result: dict[str, Any], groups: dict[str, str], *, seed: int, resamples: int) -> pd.DataFrame:
    means = result["dataset_mean_scores"]
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret", aggfunc="first")
    rows = []
    for _, contrast in result["contrasts"].iterrows():
        difference = (regret[contrast["left"]] - regret[contrast["right"]]).dropna()
        low, high, p, clusters = groups_module.cluster_interval(difference, groups, seed=seed, resamples=resamples)
        rows.append({"left": contrast["left"], "right": contrast["right"], "cluster_ci_low": low, "cluster_ci_high": high,
                     "cluster_p": p, "n_clusters": clusters})
    table = pd.DataFrame(rows)
    table["cluster_holm_p"] = holm(table["cluster_p"].tolist())
    return table


def run(args: argparse.Namespace) -> None:
    groups = groups_module.read_groups(Path(args.groups), "group")
    pools = {"ensemble": (Path(args.run), Path(args.review_config)), "score_only": (Path(args.score_only_run), Path(args.score_only_review_config))}
    metadata = json.loads((Path(args.run) / "metadata.json").read_text(encoding="utf-8"))
    splits_record = json.loads((Path(args.run) / "grouped_splits.json").read_text(encoding="utf-8"))
    frames, heldout, choices = [], [], []
    for pool, (run_dir, review_config) in pools.items():
        config = yaml.safe_load(review_config.read_text(encoding="utf-8")) or {}
        bootstrap = dict(config.get("bootstrap", {}))
        for split in splits_record["splits"]:
            split_id = int(split["split_id"])
            result = fold_review(run_dir, review_config, Path(args.units), split_id, FAMILIES[pool])
            table = result["contrasts"].merge(
                cluster_tests(result, groups, seed=int(bootstrap.get("seed", 20260608)), resamples=int(bootstrap.get("resamples", 10_000))),
                on=["left", "right"],
            )
            table.insert(0, "pool", pool)
            table.insert(1, "fold", split_id)
            table.insert(2, "declared", split_id == DECLARED_FOLD)
            frames.append(table)
            choices.append({"pool": pool, "fold": split_id, "nested_choice": result["selected_policy"],
                            **{f"train_selected_{row['pool']}": row["selected_reference_method"]
                               for _, row in result["train_selected_reference_choices"].iterrows()}})
            if pool == "ensemble":
                for name in split["heldout"]:
                    heldout.append({"fold": split_id, "dataset": name, "group": groups.get(name, name)})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = pd.concat(frames, ignore_index=True)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    pd.DataFrame(heldout).to_csv(output_dir / "heldout_datasets.csv", index=False)
    pd.DataFrame(choices).to_csv(output_dir / "fitted_choices.csv", index=False)
    driver = SCRIPTS / "grouped_inner_splits.py"
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run": str(args.run),
                "score_only_run": str(args.score_only_run),
                "declared_fold": DECLARED_FOLD,
                "split_seed": splits_record.get("split_seed"),
                "repeats": splits_record.get("repeats"),
                "driver_sha256_now": hashlib.sha256(driver.read_bytes()).hexdigest(),
                "run_config_sha256": metadata.get("config_sha256"),
                "families": FAMILIES,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    declared = contrasts[contrasts["declared"]]
    columns = ["pool", "right", "mean_difference", "ci_low", "ci_high", "bootstrap_holm_p", "ttest_holm_p", "cluster_holm_p"]
    print(declared[columns].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the fully grouped run of one repeat (ensemble-inclusive pool)")
    parser.add_argument("--score-only-run", type=Path, required=True, help="its score-only pool (score_level_policy_pool_ablation.py)")
    parser.add_argument("--review-config", type=Path, default=Path("configs/review_tabrepo.yaml"))
    parser.add_argument("--score-only-review-config", type=Path, default=Path("configs/review_tabrepo_score_only.yaml"))
    parser.add_argument("--units", type=Path, default=Path("analysis-output/v2/literal_vb/tabrepo_units.parquet"))
    parser.add_argument("--groups", type=Path, default=Path("configs/tabrepo_source_groups.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
