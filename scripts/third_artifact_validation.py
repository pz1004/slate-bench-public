from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import ArtifactResampler, holm, paired_contrast
from tabular_sota_benchmark.markdown import dataframe_to_markdown
from tabular_sota_benchmark.ranking import rank_against_candidates as shared_rank_against_candidates


UNIT_COLUMNS = ["dataset", "fold", "seed", "task", "primary_metric_name"]

METHOD_LABELS = {
    "meta_selector_local_validation": "Validation-best single model",
    "meta_selector_local_ensemble": "Local soft ensemble",
}


def _present_unit_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in UNIT_COLUMNS if column in frame.columns]


def _holm_adjust(values: list[float]) -> list[float]:
    return holm(values)


def rank_against_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Candidate-relative ranks under the shared tolerance-consistent rule."""
    frame = results.copy()
    frame["total_seconds"] = (
        frame["fit_seconds"].astype(float) + frame["predict_seconds"].astype(float)
    )
    ranked = shared_rank_against_candidates(frame, unit_columns=_present_unit_columns(frame))
    return ranked.rename(
        columns={"tied_best": "tied_best_candidate", "rank_regret": "candidate_rank_regret"}
    )


def dataset_method_metrics(ranked: pd.DataFrame) -> pd.DataFrame:
    return (
        ranked.groupby(["dataset", "task", "method", "method_type"], as_index=False)
        .agg(
            mean_rank=("candidate_rank", "mean"),
            mean_primary_metric=("primary_metric", "mean"),
            mean_val_primary_metric=("val_primary_metric", "mean"),
            tied_best_candidate_rate=("tied_best_candidate", "mean"),
            mean_total_seconds=("total_seconds", "mean"),
            mean_normalized_rank_regret=("normalized_rank_regret", "mean"),
            n_units=("candidate_rank", "size"),
            n_candidates=("n_candidates", "mean"),
        )
        .sort_values(["method_type", "method", "dataset"])
    )


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, clean.size, size=(int(n_bootstrap), clean.size))
    boot = clean[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def method_summary(
    dataset_metrics: pd.DataFrame,
    methods: list[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "mean_rank",
        "mean_primary_metric",
        "mean_val_primary_metric",
        "mean_normalized_rank_regret",
        "tied_best_candidate_rate",
        "mean_total_seconds",
    ]
    rows: list[dict[str, object]] = []
    for method_index, (method, group) in enumerate(
        dataset_metrics[dataset_metrics["method"].isin(methods)].groupby("method", sort=False)
    ):
        row: dict[str, object] = {
            "method": method,
            "label": METHOD_LABELS.get(str(method), str(method)),
            "method_type": str(group["method_type"].iloc[0]),
            "n_datasets": int(group["dataset"].nunique()),
            "n_units": int(group["n_units"].sum()),
        }
        resampler = ArtifactResampler(group["dataset"].unique(), seed=int(seed), n_resamples=int(n_bootstrap))
        for metric in metrics:
            estimate, low, high = resampler.mean_ci(group.set_index("dataset")[metric])
            row[metric] = estimate
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_primary_metric", "mean_normalized_rank_regret", "method"]
    )


def paired_contrasts(
    dataset_metrics: pd.DataFrame,
    *,
    method: str,
    reference: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = dataset_metrics[dataset_metrics["method"].eq(method)][
        ["dataset", "mean_rank", "mean_primary_metric", "mean_normalized_rank_regret"]
    ].rename(
        columns={
            "mean_rank": "method_mean_rank",
            "mean_primary_metric": "method_mean_primary_metric",
            "mean_normalized_rank_regret": "method_mean_normalized_rank_regret",
        }
    )
    right = dataset_metrics[dataset_metrics["method"].eq(reference)][
        ["dataset", "mean_rank", "mean_primary_metric", "mean_normalized_rank_regret"]
    ].rename(
        columns={
            "mean_rank": "reference_mean_rank",
            "mean_primary_metric": "reference_mean_primary_metric",
            "mean_normalized_rank_regret": "reference_mean_normalized_rank_regret",
        }
    )
    paired = left.merge(right, on="dataset", how="inner").sort_values("dataset")
    paired["rank_difference"] = paired["method_mean_rank"] - paired["reference_mean_rank"]
    paired["normalized_rank_regret_difference"] = (
        paired["method_mean_normalized_rank_regret"]
        - paired["reference_mean_normalized_rank_regret"]
    )
    paired["primary_metric_difference"] = (
        paired["method_mean_primary_metric"] - paired["reference_mean_primary_metric"]
    )

    endpoint_map = {
        "normalized_rank_regret": "normalized_rank_regret_difference",
        "primary_metric": "primary_metric_difference",
    }
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    mean_p_values: list[float] = []
    resampler = ArtifactResampler(paired["dataset"].unique(), seed=int(seed), n_resamples=int(n_bootstrap))
    indexed = paired.set_index("dataset")
    for endpoint, column in endpoint_map.items():
        diff = paired[column].to_numpy(dtype=float)
        contrast = paired_contrast(indexed[column], indexed[column] * 0.0, resampler)
        wilcoxon_p = float(contrast["wilcoxon_p"])
        wins = int(contrast["wins"])
        losses = int(contrast["losses"])
        ties = int(contrast["ties"])
        sign_p = float(contrast["sign_test_p"])
        low, high = float(contrast["ci_low"]), float(contrast["ci_high"])
        p_values.append(wilcoxon_p)
        mean_p_values.append(float(contrast["bootstrap_p"]))
        rows.append(
            {
                "method": method,
                "reference_method": reference,
                "method_label": METHOD_LABELS.get(method, method),
                "reference_label": METHOD_LABELS.get(reference, reference),
                "endpoint": endpoint,
                "n_datasets": int(diff.size),
                "mean_method_minus_reference": float(np.nanmean(diff)),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_p_value": float(contrast["bootstrap_p"]),
                "wilcoxon_p_value": wilcoxon_p,
                "sign_test_p_value": sign_p,
                "wins_method_lower": wins,
                "ties": ties,
                "losses_method_higher": losses,
            }
        )
    adjusted = _holm_adjust(p_values)
    for row, adjusted_value, mean_adjusted in zip(rows, adjusted, _holm_adjust(mean_p_values)):
        row["holm_p_value"] = adjusted_value
        row["bootstrap_holm_p_value"] = mean_adjusted
    return pd.DataFrame(rows), paired


def run_validation(
    *,
    input_path: Path,
    output_dir: Path,
    method: str,
    reference: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    results = pd.read_csv(input_path)
    ranked = rank_against_candidates(results)
    dataset_metrics = dataset_method_metrics(ranked)
    methods = [method, reference]
    summary = method_summary(
        dataset_metrics,
        methods,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paired, per_dataset = paired_contrasts(
        dataset_metrics,
        method=method,
        reference=reference,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_dir / "ranked_rows.csv", index=False)
    dataset_metrics.to_csv(output_dir / "dataset_method_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_contrasts.csv", index=False)
    per_dataset.to_csv(output_dir / "per_dataset_contrasts.csv", index=False)
    _write_markdown(
        output_dir,
        input_path=input_path,
        method_summary_frame=summary,
        paired_frame=paired,
        per_dataset_frame=per_dataset,
    )
    return {
        "ranked": ranked,
        "dataset_metrics": dataset_metrics,
        "summary": summary,
        "paired": paired,
        "per_dataset": per_dataset,
    }


def _write_markdown(
    output_dir: Path,
    *,
    input_path: Path,
    method_summary_frame: pd.DataFrame,
    paired_frame: pd.DataFrame,
    per_dataset_frame: pd.DataFrame,
) -> None:
    lines = [
        "# Third-Artifact Offline Validation",
        "",
        f"Input artifact: `{input_path}`.",
        "",
        "This is an offline validation artifact, not a real-world deployment. It uses",
        "the compact local retraining table as a third audit surface and keeps",
        "candidate-relative rank regret separate from metric-scale prediction gains.",
        "",
        "## Method Summary",
        "",
        dataframe_to_markdown(method_summary_frame),
        "",
        "## Paired Dataset Contrasts",
        "",
        dataframe_to_markdown(paired_frame),
        "",
        "## Per-Dataset Differences",
        "",
        dataframe_to_markdown(per_dataset_frame),
        "",
    ]
    (output_dir / "analysis-summary.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-clustered offline validation on a compact third artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis-output/v2/tabarena_ict_retraining_extended/results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output/third_artifact_validation"),
    )
    parser.add_argument("--method", default="meta_selector_local_ensemble")
    parser.add_argument("--reference", default="meta_selector_local_validation")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input artifact: {args.input}")
    run_validation(
        input_path=args.input,
        output_dir=args.output_dir,
        method=str(args.method),
        reference=str(args.reference),
        n_bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
