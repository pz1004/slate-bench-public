"""Rerun the TabRepo contrasts without the datasets that TabArena also contains.

The selector components were developed on TabArena, so TabRepo tests their rule
structure outside that development. Sixteen TabRepo datasets share a name with a
TabArena dataset (the pooled diagnostic's overlap). Two more, wine-quality-red and
wine-quality-white, together make up TabArena's wine_quality. This script reruns every
contrast family of each TabRepo review on the remaining datasets, from the stored
dataset-level scores. It uses the review's own contrast code and its resampler,
restricted to the kept datasets. Nothing is refitted: the nested choices were made on
all training datasets, so this removes the shared datasets from the evaluation only.

Before excluding anything, the full-sample rerun must reproduce the review's
``contrasts.csv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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

from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402

REVIEW_PATH = Path(__file__).resolve().with_name("score_level_review.py")
_SPEC = importlib.util.spec_from_file_location("score_level_review_overlap", REVIEW_PATH)
assert _SPEC is not None and _SPEC.loader is not None
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)

SALT = "exclude_tabarena_overlap"
# TabRepo datasets that are parts of a TabArena dataset under another name.
RELATED = ["wine-quality-red", "wine-quality-white"]
KEY_COLUMNS = ["family", "left", "right"]


def rerun(
    means: pd.DataFrame, config: dict, resampler: ArtifactResampler
) -> pd.DataFrame:
    return review.contrast_families(
        means,
        resampler,
        families=dict(config.get("families", {})),
        margins=[float(value) for value in config.get("margins", [0.005, 0.010, 0.015, 0.020])],
        rank_scale=float(config["rank_scale"]),
    )


def check_reproduces(rerun_full: pd.DataFrame, stored: pd.DataFrame, panel: str) -> None:
    """The full-sample rerun must equal the stored review, row for row."""
    left = rerun_full.set_index(KEY_COLUMNS).sort_index()
    right = stored.set_index(KEY_COLUMNS).sort_index()
    if not left.index.equals(right.index):
        raise SystemExit(f"{panel}: the rerun contrasts are not the stored ones")
    numeric = [column for column in right.columns if pd.api.types.is_numeric_dtype(right[column])]
    for column in numeric:
        a = left[column].to_numpy(dtype=float)
        b = right[column].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise SystemExit(f"{panel}: the full-sample rerun does not reproduce `{column}` of contrasts.csv")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    overlap = sorted(pd.read_csv(args.overlap)["dataset"].astype(str))
    excluded = sorted(set(overlap) | set(args.related))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    panels = {}
    fixed_rows: dict[str, pd.DataFrame] = {}
    for spec in args.review:
        panel, _, directory = spec.partition("=")
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        config = yaml.safe_load(Path(metadata["config"]).read_text(encoding="utf-8")) or {}
        # The review's own dataset means, from its stored split-level scores; the CSV copy
        # loses the last bits, which can untie near-equal Wilcoxon ranks.
        means = review.dataset_means(pd.read_parquet(directory / "dataset_policy_scores.parquet"))
        datasets = sorted(means["dataset"].astype(str).unique())
        absent = [name for name in excluded if name not in datasets]
        if absent:
            raise SystemExit(f"{panel}: datasets to exclude are not in the review: {absent}")
        bootstrap = dict(config.get("bootstrap", {}))
        resampler = ArtifactResampler(
            datasets,
            seed=int(bootstrap.get("seed", 20260608)),
            n_resamples=int(bootstrap.get("resamples", 10_000)),
        )
        full = rerun(means, config, resampler)
        check_reproduces(full, pd.read_csv(directory / "contrasts.csv"), panel)
        kept = [name for name in datasets if name not in set(excluded)]
        subset = resampler.subset(kept, salt=SALT)
        result = rerun(means[means["dataset"].astype(str).isin(kept)], config, subset)
        merged = result.merge(
            full[KEY_COLUMNS + ["mean_difference", "ci_low", "ci_high"]].rename(
                columns={
                    "mean_difference": "full_mean_difference",
                    "ci_low": "full_ci_low",
                    "ci_high": "full_ci_high",
                }
            ),
            on=KEY_COLUMNS,
            how="left",
        )
        merged.insert(0, "panel", panel)
        frames.append(merged)
        # Rows outside the nested choice are the same in both panels; keep them to check.
        fixed_rows[panel] = means[~means["method"].eq("nested_policy_selected")]
        panels[panel] = {
            "review": str(directory),
            "config": str(metadata["config"]),
            "n_datasets": len(datasets),
            "n_kept": len(kept),
            "derived_seed": int(subset.seed),
        }
    if len(fixed_rows) > 1:
        shared = None
        for panel, frame in fixed_rows.items():
            keyed = frame.set_index(["dataset", "method"])["mean_normalized_rank_regret"]
            shared = keyed if shared is None else shared
            common = shared.index.intersection(keyed.index)
            if not np.allclose(shared.loc[common].to_numpy(), keyed.loc[common].to_numpy(), atol=1e-12):
                raise SystemExit(f"{panel}: rows outside the nested choice differ between the panels")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_dir / "overlap_contrasts.csv", index=False)
    metadata = {
        "command": sys.argv,
        "overlap_file": str(args.overlap),
        "excluded": excluded,
        "n_excluded": len(excluded),
        "n_overlap_by_name": len(overlap),
        "related": list(args.related),
        "salt": SALT,
        "panels": panels,
        "wall_seconds": time.time() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_dir}: {len(excluded)} datasets excluded, {len(out)} contrasts", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overlap", type=Path, required=True, help="CSV with a `dataset` column (exact-name overlap)")
    parser.add_argument("--related", nargs="*", default=RELATED, help="further TabRepo datasets to exclude")
    parser.add_argument(
        "--review", action="append", required=True, help="PANEL=DIR of a TabRepo review (repeatable)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
