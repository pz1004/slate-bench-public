from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

import score_level_policy_benchmark as nested  # noqa: E402


DEFAULT_EXCLUDED_SUBSTRINGS = ["tabrepo_ensemble"]
DEFAULT_REFERENCES = [
    nested.TASK_ADAPTIVE_METHOD,
    "meta_selector_meta_feature_guard_005",
    nested.FAMILY_PRIOR_METHOD,
    nested.VALIDATION_ONLY_METHOD,
    "meta_selector_single",
]


def _method_excluded(methods: pd.Series, excluded_substrings: list[str]) -> pd.Series:
    mask = pd.Series(False, index=methods.index)
    values = methods.astype(str)
    for substring in excluded_substrings:
        mask |= values.str.contains(str(substring), regex=False)
    return mask


def _eligible_policy_methods(
    training_scores: pd.DataFrame,
    *,
    artifact: str,
    excluded_substrings: list[str],
) -> list[str]:
    rows = training_scores[training_scores["artifact"].astype(str).eq(str(artifact))].copy()
    if "role" in rows.columns:
        # v2 outputs mark reference rows, which were never selectable.
        rows = rows[rows["role"].astype(str).eq("pool")]
    rows = rows[
        ~_method_excluded(rows["method"], excluded_substrings)
        & ~rows["method"].astype(str).isin(["oracle", nested.NESTED_SELECTED_METHOD])
    ].copy()
    if rows.empty:
        raise ValueError(f"No eligible policy methods remain for artifact `{artifact}`.")
    method_order = (
        rows.groupby("method", as_index=False)
        .agg(method_order=("method_order", "min"))
        .sort_values(["method_order", "method"])
    )
    return method_order["method"].astype(str).tolist()


def derive_pool_ablation(
    *,
    nested_output: Path,
    output_dir: Path,
    excluded_substrings: list[str],
    references: list[str],
    command: str,
) -> dict[str, Any]:
    training_scores = pd.read_csv(nested_output / "training_policy_scores.csv")
    heldout_scores = pd.read_csv(nested_output / "heldout_policy_scores.csv")
    split_results = pd.read_csv(nested_output / "split_results.csv")

    selected_split_rows: list[dict[str, Any]] = []
    selected_heldout_rows: list[pd.Series] = []
    eligible_heldout_frames: list[pd.DataFrame] = []
    eligible_training_frames: list[pd.DataFrame] = []
    artifact_policy_methods: dict[str, list[str]] = {}
    artifact_reference_methods: dict[str, list[str]] = {}
    selections: list[dict[str, Any]] = []

    for artifact, artifact_splits in split_results.groupby("artifact", sort=False):
        policy_methods = _eligible_policy_methods(
            training_scores,
            artifact=str(artifact),
            excluded_substrings=excluded_substrings,
        )
        artifact_policy_methods[str(artifact)] = policy_methods

        reference_methods = (
            sorted(
                training_scores.loc[
                    training_scores["artifact"].astype(str).eq(str(artifact))
                    & training_scores["role"].astype(str).eq("reference"),
                    "method",
                ].astype(str).unique()
            )
            if "role" in training_scores.columns
            else []
        )
        artifact_reference_methods[str(artifact)] = reference_methods
        artifact_training = training_scores[
            training_scores["artifact"].astype(str).eq(str(artifact))
            & training_scores["method"].astype(str).isin(policy_methods)
        ].copy()
        artifact_heldout = heldout_scores[
            heldout_scores["artifact"].astype(str).eq(str(artifact))
        ].copy()
        eligible_training_frames.append(
            training_scores[
                training_scores["artifact"].astype(str).eq(str(artifact))
                & training_scores["method"].astype(str).isin([*policy_methods, *reference_methods])
            ].copy()
        )
        eligible_heldout_frames.append(
            artifact_heldout[
                artifact_heldout["method"].astype(str).isin([*policy_methods, *reference_methods, "oracle"])
            ].copy()
        )

        for _, split_row in artifact_splits.sort_values("split_id").iterrows():
            split_id = int(split_row["split_id"])
            train_candidates = artifact_training[artifact_training["split_id"].astype(int).eq(split_id)]
            selected = nested._choose_policy(train_candidates, policy_methods)
            selected_method = str(selected["method"])
            heldout_match = artifact_heldout[
                artifact_heldout["split_id"].astype(int).eq(split_id)
                & artifact_heldout["method"].astype(str).eq(selected_method)
            ]
            if heldout_match.empty:
                raise ValueError(
                    f"Held-out scores are missing selected policy `{selected_method}` "
                    f"for artifact `{artifact}`, split {split_id}."
                )
            heldout_row = heldout_match.iloc[0].copy()
            train_row = selected.copy()

            selections.append({"artifact": str(artifact), "split_id": split_id, "selected_policy": selected_method})
            selected_heldout = heldout_row.copy()
            selected_heldout["method"] = nested.NESTED_SELECTED_METHOD
            selected_heldout["method_order"] = -1
            selected_heldout["original_method"] = selected_method
            if "role" in selected_heldout.index:
                selected_heldout["role"] = "alias"
            selected_heldout_rows.append(selected_heldout)

            row = split_row.to_dict()
            row["selected_policy"] = selected_method
            row["train_mean_rank"] = float(train_row["mean_rank"])
            row["train_mean_total_seconds"] = float(train_row["mean_total_seconds"])
            row["train_mean_normalized_rank_regret"] = float(train_row["mean_normalized_rank_regret"])
            row["heldout_mean_rank"] = float(heldout_row["mean_rank"])
            row["heldout_tied_best_rate"] = float(heldout_row["tied_best_rate"])
            row["heldout_mean_total_seconds"] = float(heldout_row["mean_total_seconds"])
            row["heldout_mean_rank_regret"] = float(heldout_row["mean_rank_regret"])
            row["heldout_mean_normalized_rank_regret"] = float(
                heldout_row["mean_normalized_rank_regret"]
            )
            row["heldout_n_units"] = int(heldout_row["n_units"])
            selected_split_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_training_scores = pd.concat(eligible_training_frames, ignore_index=True)
    output_heldout_scores = pd.concat(
        [*eligible_heldout_frames, pd.DataFrame(selected_heldout_rows)],
        ignore_index=True,
    ).sort_values(["artifact", "split_id", "method_order", "method"])
    output_split_results = pd.DataFrame(selected_split_rows).sort_values(["artifact", "split_id"])
    summary = nested._summary(output_heldout_scores)
    selection_counts = nested._selection_counts(output_split_results)
    paired = nested._paired_tests(output_heldout_scores, references)

    unit_path = nested_output / "unit_scores.parquet"
    if unit_path.exists():
        units = pd.read_parquet(unit_path)
        keep = set().union(*artifact_policy_methods.values(), *artifact_reference_methods.values(), {"oracle"})
        kept = units[units["method"].astype(str).isin(keep) & ~units["role"].astype(str).eq("alias")]
        chosen = pd.DataFrame(selections)
        alias = units[units["phase"].eq("heldout") & ~units["role"].astype(str).eq("alias")].merge(
            chosen,
            left_on=["artifact", "split_id", "method"],
            right_on=["artifact", "split_id", "selected_policy"],
        )
        alias["original_method"] = alias["method"]
        alias["method"] = nested.NESTED_SELECTED_METHOD
        alias["role"] = "alias"
        alias = alias[units.columns]
        pd.concat([kept, alias], ignore_index=True).to_parquet(
            output_dir / "unit_scores.parquet", index=False
        )
    output_split_results.to_csv(output_dir / "split_results.csv", index=False)
    output_training_scores.to_csv(output_dir / "training_policy_scores.csv", index=False)
    output_heldout_scores.to_csv(output_dir / "heldout_policy_scores.csv", index=False)
    selection_counts.to_csv(output_dir / "policy_selection_counts.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_tests.csv", index=False)

    metadata = {
        "source_nested_output": str(nested_output),
        "output_dir": str(output_dir),
        "excluded_method_substrings": list(excluded_substrings),
        "references": list(references),
        "command": command,
        "generated_at_epoch": time.time(),
        "artifacts": [
            {
                "name": artifact,
                "n_policy_methods": len(methods),
                "policy_methods": list(methods),
                "reference_methods": list(artifact_reference_methods.get(artifact, [])),
            }
            for artifact, methods in artifact_policy_methods.items()
        ],
    }
    source_metadata_path = nested_output / "metadata.json"
    if source_metadata_path.exists():
        source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        metadata["source_metadata"] = {
            key: source.get(key) for key in ["git", "rank_rule", "seed", "n_splits", "config", "config_sha256"]
        }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    nested._write_figures(
        output_dir,
        split_results=output_split_results,
        heldout_scores=output_heldout_scores,
        summary=summary,
    )
    nested._write_report(
        output_dir,
        split_results=output_split_results,
        summary=summary,
        selection_counts=selection_counts,
        paired=paired,
        command=command,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive nested policy-pool ablations from completed score-level split scores."
    )
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-method-substrings",
        nargs="+",
        default=DEFAULT_EXCLUDED_SUBSTRINGS,
    )
    parser.add_argument("--references", nargs="+", default=DEFAULT_REFERENCES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    derive_pool_ablation(
        nested_output=args.nested_output,
        output_dir=args.output_dir,
        excluded_substrings=[str(value) for value in args.exclude_method_substrings],
        references=[str(value) for value in args.references],
        command=" ".join(str(value) for value in sys.argv),
    )
    print(f"Wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
