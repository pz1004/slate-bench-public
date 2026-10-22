"""Cluster bootstrap of a review's contrasts over groups of related datasets.

TabRepo's 200 datasets include groups of variants of one source (configs/tabrepo_source_groups.csv),
which the review's dataset bootstrap resamples as if they were independent. This script
resamples whole groups instead. For every contrast of every family of a review it draws groups
with replacement, an ``ArtifactResampler`` over group ids with the review's seed and resample
count, and takes the mean of the dataset-level differences in the drawn groups, so the point
estimate stays the review's dataset-level mean. The p-value inverts the percentile interval,
as the review's bootstrap test does, and is Holm-adjusted within the review's family. Before
anything else, the same code with every dataset its own group must reproduce the review's
intervals and p-values exactly.

Two groupings are run: ``group`` (all related pairs, including those resting on identical
shapes or background knowledge) and ``documented_group`` (only groups resting on repository
records, OpenML descriptions or names). For each review's outer splits the script also
reports how many held-out datasets have a related dataset on the training side.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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

from tabular_sota_benchmark.inference import TIE_TOLERANCE, ArtifactResampler, bootstrap_mean_p, holm  # noqa: E402

REVIEW_PATH = Path(__file__).resolve().with_name("score_level_review.py")
_SPEC = importlib.util.spec_from_file_location("score_level_review_source_groups", REVIEW_PATH)
assert _SPEC is not None and _SPEC.loader is not None
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)

GROUPINGS = ["group", "documented_group"]


def read_groups(path: Path, column: str) -> dict[str, str]:
    frame = pd.read_csv(path, comment="#", dtype=str).fillna("")
    return {row["dataset"]: row[column] for _, row in frame.iterrows() if row[column]}


def cluster_interval(
    differences: pd.Series, groups: dict[str, str], *, seed: int, resamples: int
) -> tuple[float, float, float, int]:
    """Percentile interval and inverted-test p-value of the mean, resampling whole groups."""
    values = differences.astype(float)
    labels = pd.Index([groups.get(str(name), str(name)) for name in values.index])
    sums = values.groupby(labels).sum()
    sizes = values.groupby(labels).size().astype(float)
    resampler = ArtifactResampler(sums.index, seed=seed, n_resamples=resamples)
    boot = resampler.align(sums)[resampler.indices].sum(axis=1) / resampler.align(sizes)[resampler.indices].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    nonzero = bool((values.abs() > TIE_TOLERANCE).any())
    return float(low), float(high), (bootstrap_mean_p(boot) if nonzero else 1.0), int(sums.shape[0])


def contrasts(means: pd.DataFrame, config: dict, groups: dict[str, str]) -> pd.DataFrame:
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret", aggfunc="first")
    bootstrap = dict(config.get("bootstrap", {}))
    seed, resamples = int(bootstrap.get("seed", 20260608)), int(bootstrap.get("resamples", 10_000))
    rows = []
    for family, spec in dict(config.get("families", {})).items():
        left = str(spec["left"])
        for right in spec["references"]:
            diff = (regret[left] - regret[str(right)]).dropna()
            low, high, p, clusters = cluster_interval(diff, groups, seed=seed, resamples=resamples)
            rows.append(
                {
                    "family": family,
                    "left": left,
                    "right": str(right),
                    "mean_difference": float(diff.mean()),
                    "cluster_ci_low": low,
                    "cluster_ci_high": high,
                    "cluster_p": p,
                    "n_clusters": clusters,
                }
            )
    table = pd.DataFrame(rows)
    table["cluster_holm_p"] = np.nan
    for _, index in table.groupby("family", sort=False).groups.items():
        table.loc[index, "cluster_holm_p"] = holm(table.loc[index, "cluster_p"].tolist())
    return table


def sibling_share(scores: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Per outer split, the share of held-out datasets with a related dataset in training."""
    members: dict[str, set[str]] = {}
    for name, group in groups.items():
        members.setdefault(group, set()).add(name)
    everything = set(scores["dataset"].astype(str))
    rows = []
    for split_id, frame in scores.groupby("split_id"):
        heldout = set(frame["dataset"].astype(str))
        train = everything - heldout
        hit = [name for name in heldout if name in groups and (members[groups[name]] - {name}) & train]
        rows.append({"split_id": int(split_id), "n_heldout": len(heldout), "with_training_sibling": len(hit)})
    out = pd.DataFrame(rows)
    out["share"] = out["with_training_sibling"] / out["n_heldout"]
    return out


def run(args: argparse.Namespace) -> None:
    frames, shares, panels = [], [], {}
    for spec in args.review:
        panel, _, directory = spec.partition("=")
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        config = yaml.safe_load(Path(metadata["config"]).read_text(encoding="utf-8")) or {}
        scores = pd.read_parquet(directory / "dataset_policy_scores.parquet")
        means = review.dataset_means(scores)
        stored = pd.read_csv(directory / "contrasts.csv").set_index(["family", "left", "right"])

        # Every dataset its own group: the review's dataset bootstrap, exactly.
        check = contrasts(means, config, {}).set_index(["family", "left", "right"])
        for ours, theirs in [("cluster_ci_low", "ci_low"), ("cluster_ci_high", "ci_high"), ("cluster_p", "bootstrap_p"), ("cluster_holm_p", "bootstrap_holm_p")]:
            if not np.allclose(check[ours].to_numpy(), stored.loc[check.index, theirs].to_numpy(), rtol=0, atol=1e-12):
                raise SystemExit(f"{panel}: singleton groups do not reproduce the review's `{theirs}`")

        held = scores[["split_id", "dataset"]].drop_duplicates()
        for grouping in GROUPINGS:
            groups = read_groups(Path(args.groups), grouping)
            table = contrasts(means, config, groups).set_index(["family", "left", "right"])
            table = table.join(stored[["ci_low", "ci_high", "bootstrap_p", "bootstrap_holm_p", "se_bootstrap"]])
            frames.append(table.reset_index().assign(panel=panel, grouping=grouping))
            shares.append(sibling_share(held, groups).assign(panel=panel, grouping=grouping))
        panels[panel] = {"review": str(directory), "config": str(metadata["config"])}

    out = pd.concat(frames, ignore_index=True)
    out["cluster_width_ratio"] = (out["cluster_ci_high"] - out["cluster_ci_low"]) / (out["ci_high"] - out["ci_low"])
    share = pd.concat(shares, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "cluster_contrasts.csv", index=False)
    share.to_csv(output_dir / "sibling_share.csv", index=False)
    clusters = {g: int(out.loc[out["grouping"].eq(g), "n_clusters"].max()) for g in GROUPINGS}
    (output_dir / "metadata.json").write_text(
        json.dumps({"groups": str(args.groups), "panels": panels, "n_clusters": clusters, "command": sys.argv}, indent=2),
        encoding="utf-8",
    )
    flips = out[(out["bootstrap_holm_p"] < 0.05) != (out["cluster_holm_p"] < 0.05)]
    print(f"clusters: {clusters}; verdicts that change: {len(flips)}", flush=True)
    print(share.groupby(["panel", "grouping"])["share"].agg(["mean", "min", "max"]).round(3).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", action="append", required=True, help="panel=review directory; repeatable")
    parser.add_argument("--groups", type=Path, default=Path("configs/tabrepo_source_groups.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
