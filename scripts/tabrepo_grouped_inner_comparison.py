"""Inner grouping on TabRepo: the group-aware rerun against the outer-grouped run, split by split.

scripts/grouped_outer_splits.py keeps each group of related datasets on one side of every
outer split but leaves one dataset at a time out of the training-side estimates;
scripts/grouped_inner_splits.py reruns the same 24 outer splits with whole groups left out of
the training-side estimates as well. Because the two runs share their outer splits, the change
of a contrast between them is paired by dataset and carries no split-design noise: for every
central contrast this script takes, per dataset, the inner-grouped run's held-out difference
minus the outer-grouped run's, and reports the mean change with a percentile interval and the
test that inverts it, from the review's shared dataset-resample matrix and, as a sensitivity,
from a bootstrap over the groups (configs/tabrepo_source_groups.csv). p-values are
Holm-adjusted over the contrasts of a pool. Nothing is refitted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import TIE_TOLERANCE, ArtifactResampler, bootstrap_mean_p, holm  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(file))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review = _load("score_level_review_inner_comparison", "score_level_review.py")
comparison = _load("tabrepo_grouped_comparison_inner", "tabrepo_grouped_comparison.py")
source_groups = _load("tabrepo_source_groups_inner", "tabrepo_source_groups.py")
NESTED = comparison.NESTED
REFERENCES = comparison.REFERENCES
SEED, RESAMPLES = 20260608, 10_000


def differences(directory: Path, right: str) -> pd.Series:
    """Per dataset, nested selection minus ``right`` (mean held-out normalized rank regret)."""
    means = review.dataset_means(pd.read_parquet(directory / "dataset_policy_scores.parquet"))
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret", aggfunc="first")
    return (regret[NESTED] - regret[right]).dropna()


def paired_change(outer: pd.Series, inner: pd.Series, groups: dict[str, str]) -> dict[str, float]:
    if set(outer.index) != set(inner.index):
        raise SystemExit("The two runs do not cover the same datasets.")
    change = (inner - outer.reindex(inner.index)).astype(float)
    resampler = ArtifactResampler(change.index, seed=SEED, n_resamples=RESAMPLES)
    boot = resampler.align(change)[resampler.indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    moved = bool((change.abs() > TIE_TOLERANCE).any())
    c_low, c_high, c_p, clusters = source_groups.cluster_interval(change, groups, seed=SEED, resamples=RESAMPLES)
    return {
        "outer_grouped": float(outer.mean()),
        "inner_grouped": float(inner.mean()),
        "change": float(change.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p": bootstrap_mean_p(boot) if moved else 1.0,
        "cluster_ci_low": c_low,
        "cluster_ci_high": c_high,
        "cluster_p": c_p if moved else 1.0,
        "n_clusters": clusters,
        "n_datasets_changed": int((change.abs() > TIE_TOLERANCE).sum()),
    }


def compare(outer_dirs: dict[str, Path], inner_dirs: dict[str, Path], groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for panel, references in REFERENCES.items():
        for right in references:
            result = paired_change(differences(outer_dirs[panel], right), differences(inner_dirs[panel], right), groups)
            rows.append({"panel": panel, "right": right, **result})
    table = pd.DataFrame(rows)
    for column in ["p", "cluster_p"]:
        table[f"{column}_holm"] = np.nan
        for _, index in table.groupby("panel", sort=False).groups.items():
            table.loc[index, f"{column}_holm"] = holm(table.loc[index, column].tolist())
    return table


def run(args: argparse.Namespace) -> None:
    outer = {spec.partition("=")[0]: Path(spec.partition("=")[2]) for spec in args.outer}
    inner = {spec.partition("=")[0]: Path(spec.partition("=")[2]) for spec in args.inner}
    groups = source_groups.read_groups(Path(args.groups), "group")
    # The same review on both sides must give no change at all.
    check = compare(outer, outer, groups)
    if not (np.allclose(check["change"], 0.0, atol=1e-15) and (check["p"] == 1.0).all()):
        raise SystemExit("A run compared with itself does not give zero changes.")
    table = compare(outer, inner, groups)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "grouped_inner_vs_outer.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"outer": {k: str(v) for k, v in outer.items()}, "inner": {k: str(v) for k, v in inner.items()},
                    "groups": str(args.groups), "seed": SEED, "resamples": RESAMPLES, "command": sys.argv}, indent=2),
        encoding="utf-8",
    )
    print(table.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outer", action="append", required=True, help="pool=review directory of the outer-grouped run")
    parser.add_argument("--inner", action="append", required=True, help="pool=review directory of the inner-grouped run")
    parser.add_argument("--groups", type=Path, default=Path("configs/tabrepo_source_groups.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
