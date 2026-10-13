"""Find the TabArena result rows that are copies, and write the artifact without them.

TabArena fills the runs a method could not complete with another method's results. Its
dataset metadata marks two such cases: ``can_run_tabpfnv2`` and ``can_run_tabicl`` are
False on the datasets where TabPFNv2 or TabICL did not run, and every row of that method
on those datasets is a copy. Other copies are not marked: a row of any method other than
RandomForest whose validation and test errors both equal those of a RandomForest row of
the same evaluation unit is treated as a copy too.

The script writes the list of copied rows, a summary by method, and the results file
without them, which ``configs/score_level_tabarena_no_imputed.yaml`` reruns the main
TabArena audit on. The original file is not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

UNIT = ["dataset", "fold", "seed"]
FLAGS = {"TABPFNV2": "can_run_tabpfnv2", "TABICL": "can_run_tabicl"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family(method: str) -> str:
    """The learner name before the parenthesized variant, e.g. TABPFNV2 (tuned) -> TABPFNV2."""
    return method.split(" (", 1)[0]


def copied_rows(results: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Every copied row of ``results`` with the rule that marks it."""
    frame = results.copy()
    frame["family"] = frame["method"].map(family)
    flagged = pd.Series(False, index=frame.index)
    for name, column in FLAGS.items():
        cannot_run = set(metadata.loc[metadata[column].eq(False), "dataset_name"])
        flagged |= frame["family"].eq(name) & frame["dataset"].isin(cannot_run)
    rf = frame.loc[frame["family"].eq("RF"), [*UNIT, "metric_error", "metric_error_val"]].drop_duplicates()
    rf["_rf"] = True
    match = frame.merge(rf, on=[*UNIT, "metric_error", "metric_error_val"], how="left")["_rf"]
    copy = match.fillna(False).astype(bool).to_numpy() & frame["family"].ne("RF").to_numpy()
    rule = pd.Series("", index=frame.index)
    rule[flagged] = "metadata flag"
    rule[~flagged & copy] = "copy of a RandomForest row"
    out = frame.loc[flagged | copy, [*UNIT, "method", "family"]].copy()
    out["rule"] = rule[flagged | copy]
    return out.sort_values([*UNIT, "method"]).reset_index(drop=True)


def summary(copies: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    by_method = (
        copies.groupby(["method", "rule"])
        .agg(rows=("dataset", "size"), datasets=("dataset", "nunique"))
        .reset_index()
    )
    n_units = results[UNIT].drop_duplicates().shape[0]
    units_hit = copies[UNIT].drop_duplicates().shape[0]
    total = pd.DataFrame(
        [
            {
                "method": "all",
                "rule": "any",
                "rows": int(copies.shape[0]),
                "datasets": int(copies["dataset"].nunique()),
                "rows_total": int(results.shape[0]),
                "units": int(units_hit),
                "units_total": int(n_units),
                "datasets_total": int(results["dataset"].nunique()),
                "flagged_rows": int(copies["rule"].eq("metadata flag").sum()),
                "copy_rows": int(copies["rule"].eq("copy of a RandomForest row").sum()),
            }
        ]
    )
    return pd.concat([by_method, total], ignore_index=True)


def run(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    metadata_path = Path(args.metadata)
    results = pd.read_csv(results_path)
    metadata = pd.read_csv(metadata_path)
    copies = copied_rows(results, metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copies.to_csv(output_dir / "copied_rows.csv", index=False)
    summary(copies, results).to_csv(output_dir / "copied_summary.csv", index=False)
    keyed = results.merge(copies[[*UNIT, "method"]].assign(_copy=True), on=[*UNIT, "method"], how="left")
    kept = results.loc[~keyed["_copy"].fillna(False).astype(bool).to_numpy()]
    filtered = output_dir / "df_results_no_imputed.csv"
    kept.to_csv(filtered, index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "results": str(results_path),
                "results_sha256": _sha256(results_path),
                "metadata": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "rows": int(results.shape[0]),
                "copied_rows": int(copies.shape[0]),
                "kept_rows": int(kept.shape[0]),
                "filtered_sha256": _sha256(filtered),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{copies.shape[0]} copied rows of {results.shape[0]}; wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="data/tabarena/df_results.csv")
    parser.add_argument("--metadata", default="data/tabarena/tabarena_dataset_metadata.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
