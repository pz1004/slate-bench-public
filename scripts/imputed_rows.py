"""Find the TabArena result rows that were imputed, and write the artifact without them.

TabArena imputes the results of a model on datasets where it is not applicable with those
of RandomForest (default), to penalize incomplete coverage on its leaderboard (Erickson et
al. 2025, App. A.1). The results file does not mark these rows. They carry a signature: the
prediction time of the unit's RandomForest (default) row, exactly, with a fit time that is a
multiple of RandomForest's (1x for defaults, 51x or 201x for tuned variants). A row of any
method other than RandomForest with that prediction time is treated as imputed. The dataset
metadata marks the datasets where TabPFNv2 or TabICL is not applicable (``can_run_tabpfnv2``
and ``can_run_tabicl`` False); every row of those methods there is imputed too, and all of
them carry the signature. Imputed rows repeat RandomForest (default)'s errors exactly or
within about half a percent. Equal errors alone are not a signature: a degenerate predictor
that outputs the class prior can tie a RandomForest row by chance.

The script writes the list of imputed rows, a summary by method, and the results file
without them, which ``configs/score_level_tabarena_no_imputed.yaml`` reruns the main
TabArena audit on. With ``--mask metadata`` it removes only the rows that the metadata
documents (``df_results_no_flagged.csv``, for ``configs/score_level_tabarena_no_flagged.yaml``)
and keeps the rows found by their timing signature alone. The original file is not changed.
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
RF_DEFAULT = "RF (default)"


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
    """Every imputed row of ``results`` with the rule that marks it, and how far its errors
    are from the unit's RandomForest (default) row."""
    frame = results.copy()
    frame["family"] = frame["method"].map(family)
    flagged = pd.Series(False, index=frame.index)
    for name, column in FLAGS.items():
        cannot_run = set(metadata.loc[metadata[column].eq(False), "dataset_name"])
        flagged |= frame["family"].eq(name) & frame["dataset"].isin(cannot_run)
    rf = frame.loc[frame["method"].eq(RF_DEFAULT), [*UNIT, "time_infer_s", "metric_error", "metric_error_val"]]
    rf = rf.rename(columns={"time_infer_s": "_rf_infer", "metric_error": "_rf_test", "metric_error_val": "_rf_val"})
    joined = frame.merge(rf, on=UNIT, how="left")
    joined.index = frame.index
    timing = joined["time_infer_s"].eq(joined["_rf_infer"]) & frame["family"].ne("RF")
    rule = pd.Series("", index=frame.index)
    rule[timing] = "RandomForest timing"
    rule[flagged] = "metadata flag"
    marked = flagged | timing
    out = frame.loc[marked, [*UNIT, "method", "family"]].copy()
    out["rule"] = rule[marked]
    out["timing"] = timing[marked].to_numpy()
    gap = pd.concat(
        [
            (joined["metric_error"] - joined["_rf_test"]).abs() / joined["_rf_test"].abs(),
            (joined["metric_error_val"] - joined["_rf_val"]).abs() / joined["_rf_val"].abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["relative_error_gap"] = gap[marked].to_numpy()
    return out.sort_values([*UNIT, "method"]).reset_index(drop=True)


def summary(copies: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    by_method = (
        copies.groupby(["method", "rule"])
        .agg(rows=("dataset", "size"), datasets=("dataset", "nunique"), max_relative_error_gap=("relative_error_gap", "max"))
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
                "timing_rows": int(copies["rule"].eq("RandomForest timing").sum()),
                "flagged_without_timing": int((copies["rule"].eq("metadata flag") & ~copies["timing"]).sum()),
                "exact_error_rows": int(copies["relative_error_gap"].eq(0.0).sum()),
                "max_relative_error_gap": float(copies["relative_error_gap"].max()),
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
    removed = copies if args.mask == "all" else copies[copies["rule"].eq("metadata flag")]
    keyed = results.merge(removed[[*UNIT, "method"]].assign(_copy=True), on=[*UNIT, "method"], how="left")
    kept = results.loc[~keyed["_copy"].fillna(False).astype(bool).to_numpy()]
    filtered = output_dir / ("df_results_no_imputed.csv" if args.mask == "all" else "df_results_no_flagged.csv")
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
                "mask": str(args.mask),
                "removed_rows": int(removed.shape[0]),
                "kept_rows": int(kept.shape[0]),
                "filtered_sha256": _sha256(filtered),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{copies.shape[0]} imputed rows of {results.shape[0]}, {removed.shape[0]} removed; wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="data/tabarena/df_results.csv")
    parser.add_argument("--metadata", default="data/tabarena/tabarena_dataset_metadata.csv")
    parser.add_argument("--mask", choices=["all", "metadata"], default="all",
                        help="remove every imputed row, or only those the metadata documents")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
