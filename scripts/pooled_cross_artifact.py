"""Pooled cross-artifact diagnostic: nested selection against each artifact's own
train-selected fixed reference.

Stacks the dataset-level contrasts of the TabArena run (51 datasets) and the TabRepo
score-only ablation (200 datasets), each against the train-selected reference built from
the same declared fixed pool. The reported scope keeps only datasets the two artifacts do
not share: the TabRepo datasets that share a name with a TabArena dataset are dropped, and so
are the two TabRepo datasets that together make up a TabArena dataset under another name,
keeping the TabArena contrast. Related datasets within TabRepo stay, as in every TabRepo
analysis. Dropping only the shared names, and the naive stack, are kept as sensitivities.
The normalized regret scales differ (one rank position is 1/43 on TabArena and about
1/1529 on TabRepo), so the pooled mean is a direction-consistency diagnostic, and its
minimum detectable effect is reported on the normalized-regret scale only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import ArtifactResampler, paired_contrast  # noqa: E402

NESTED = "nested_policy_selected"
# TabRepo datasets that together make up TabArena's wine_quality under other names
# (scripts/tabrepo_overlap_sensitivity.py excludes the same two).
RELATED = ["wine-quality-red", "wine-quality-white"]


def paired_frame(review_dir: Path, reference: str, artifact: str) -> pd.DataFrame:
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    wide = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    frame = pd.DataFrame({"nested": wide[NESTED], "reference": wide[reference]}).dropna()
    frame["artifact"] = artifact
    return frame.reset_index()


def run(args: argparse.Namespace) -> None:
    tabarena = paired_frame(Path(args.tabarena_review), args.tabarena_reference, "TabArena")
    tabrepo = paired_frame(Path(args.tabrepo_review), args.tabrepo_reference, "TabRepo score-only")
    overlap = sorted(set(tabarena["dataset"]) & set(tabrepo["dataset"]))
    missing = sorted(set(RELATED) - set(tabrepo["dataset"]))
    if missing:
        raise SystemExit(f"related TabRepo datasets not found: {missing}")
    excluded = sorted(set(overlap) | set(RELATED))
    scopes = {
        "TabArena": tabarena,
        "TabRepo score-only": tabrepo,
        "Pooled, no shared datasets": pd.concat(
            [tabarena, tabrepo[~tabrepo["dataset"].isin(excluded)]], ignore_index=True
        ),
        "Pooled, distinct datasets": pd.concat(
            [tabarena, tabrepo[~tabrepo["dataset"].isin(overlap)]], ignore_index=True
        ),
        "Pooled, naive stack": pd.concat([tabarena, tabrepo], ignore_index=True),
    }
    rows = []
    for scope, frame in scopes.items():
        keys = frame["artifact"].str.replace(" ", "_") + "::" + frame["dataset"].astype(str)
        left = pd.Series(frame["nested"].to_numpy(), index=keys)
        right = pd.Series(frame["reference"].to_numpy(), index=keys)
        resampler = ArtifactResampler(keys, seed=int(args.bootstrap_seed))
        result = paired_contrast(left, right, resampler)
        rows.append({"scope": scope, **result})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "pooled_contrasts.csv", index=False)
    pd.DataFrame({"dataset": overlap}).to_csv(output_dir / "overlapping_datasets.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_overlapping_datasets": len(overlap),
                "related_datasets": RELATED,
                "excluded_tabrepo_datasets": excluded,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(overlap)} datasets in both artifacts; wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tabarena-review", type=Path, required=True)
    parser.add_argument("--tabrepo-review", type=Path, required=True)
    parser.add_argument("--tabarena-reference", default="train_selected_fixed_reference__declared_fixed")
    parser.add_argument("--tabrepo-reference", default="train_selected_fixed_reference__score_only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
