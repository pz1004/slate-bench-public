"""Validation-best as a literal lexicographic rule, the reference of every TabRepo contrast.

The pool's first row, ``meta_selector_single``, selects the candidate with the lowest
validation error plus ``10^-6 * log(1 + fit seconds)``. On TabRepo that additive term
changes the choice in a few units, so the row is not the literal rule (lowest validation
error, then lowest fit time, then candidate order). This script makes the literal rule the
reference without touching the frozen pipeline:

``units``
    reloads an artifact's candidates and computes, per unit, the runtime-term choice, which
    must reproduce the stored rows exactly, and the literal choice, both ranked by the
    benchmark's own rule.
``review``
    rebuilds one run's review in memory with the frozen review code: the pool's row is
    renamed ``meta_selector_single_runtime`` and keeps every training-side choice, and the
    literal rule is added as the reference row ``meta_selector_single``. It checks that
    nothing else moves apart from Holm adjustment within families with a validation-best
    contrast.
``substitute``
    checks what putting the literal rule inside the pool would change: it re-chooses nested
    selection and the train-selected rules from training regrets computed on one summation
    path, and rebuilds the reviews.
``tabarena-ties``
    checks TabArena's validation-best restricted to one provenance class and the defaults
    rule under the literal tie convention.
"""

from __future__ import annotations

import argparse
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

from tabular_sota_benchmark.artifacts import ARTIFACT_UNIT_COLUMNS, _method_family, _method_variant  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402


def _load(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("score_level_policy_benchmark_literal_vb", "score_level_policy_benchmark.py")
review = _load("score_level_review_literal_vb", "score_level_review.py")

VB = "meta_selector_single"
VB_RUNTIME = "meta_selector_single_runtime"
NESTED = "nested_policy_selected"
UNIT = list(ARTIFACT_UNIT_COLUMNS)
VALUES = [
    "selected_member",
    "primary_metric",
    "val_primary_metric",
    "candidate_rank",
    "rank_regret",
    "normalized_rank_regret",
    "tied_best",
    "total_seconds",
    "n_candidates",
    "candidate_min_error",
    "candidate_median_error",
    "candidate_max_error",
]
FLOATS = [column for column in VALUES if column not in {"selected_member", "n_candidates"}]


# ----------------------------------------------------------------------------- units


def load_candidates(config_path: Path) -> tuple[pd.DataFrame, float, str]:
    """The artifact's candidate rows as the tie-rule check loads them (no ensembles, no vectors)."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    bench.set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config.get("selector", {}))
    runtime_weight = float(selector.get("runtime_weight", 0.0))
    selector.pop("reference_selectors", None)
    selector["policy_grid"] = {**dict(selector.get("policy_grid", {})), "tabrepo_ensemble_sizes": []}
    artifact_config["selector"] = selector
    artifact_config["artifact_source"] = {**dict(artifact_config["artifact_source"]), "load_predictions": False}
    artifact = bench._load_artifact(artifact_config, seed=int(benchmark.get("seed", 20260604)))
    return artifact["candidates"], runtime_weight, str(artifact["name"])


def choose(frame: pd.DataFrame, *, rule: str, runtime_weight: float) -> pd.DataFrame:
    """One candidate per unit: ``runtime`` as the selector does, ``literal`` lexicographically."""
    frame = frame.reset_index(drop=True).copy()
    frame["_order"] = np.arange(len(frame))
    if rule == "runtime":
        frame["_score"] = frame["val_primary_metric"].astype(float) + runtime_weight * np.log1p(
            frame["fit_seconds"].astype(float)
        )
        # idxmin keeps the first minimum in candidate order, as np.argmin in the selector does.
        chosen = frame.loc[frame.groupby(UNIT, sort=False)["_score"].idxmin()]
    elif rule == "literal":
        chosen = (
            frame.sort_values([*UNIT, "val_primary_metric", "fit_seconds", "_order"], kind="stable")
            .groupby(UNIT, sort=False)
            .head(1)
        )
    else:
        raise ValueError(rule)
    return chosen.drop(columns=[column for column in ["_order", "_score"] if column in chosen.columns])


def ranked_choices(candidates: pd.DataFrame, *, runtime_weight: float) -> pd.DataFrame:
    """Both rules' choices, ranked against the unit's candidates by the benchmark's rule."""
    selectors = []
    for rule in ["runtime", "literal"]:
        rows = choose(candidates, rule=rule, runtime_weight=runtime_weight).copy()
        rows["selected_member"] = rows["method"]
        rows["method"] = f"__validation_best_{rule}"
        rows["method_type"] = "selector"
        selectors.append(rows)
    ranked = bench._rank_against_candidates(pd.concat([candidates, *selectors], ignore_index=True))
    rows = ranked[ranked["method"].astype(str).str.startswith("__validation_best_")].copy()
    rows["rule"] = rows["method"].str.replace("__validation_best_", "", regex=False)
    return rows[[*UNIT, "rule", *VALUES]].reset_index(drop=True)


def _stored_rows(units: pd.DataFrame, method: str, phase: str) -> pd.DataFrame:
    rows = units[units["method"].astype(str).eq(method) & units["phase"].eq(phase) & ~units["role"].astype(str).eq("alias")]
    distinct = rows.drop_duplicates([*UNIT, *VALUES])
    if distinct[UNIT].duplicated().any():
        raise ValueError(f"Stored {method} rows ({phase}) differ between splits for the same unit.")
    return distinct[[*UNIT, *VALUES]].reset_index(drop=True)


def _same(left: pd.DataFrame, right: pd.DataFrame, *, what: str) -> None:
    merged = left.merge(right, on=UNIT, how="outer", suffixes=("_l", "_r"), indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{what}: the unit sets differ.")
    for column in VALUES:
        a, b = merged[f"{column}_l"], merged[f"{column}_r"]
        if column in FLOATS:
            equal = np.isclose(a.astype(float), b.astype(float), rtol=0.0, atol=0.0, equal_nan=True)
        else:
            equal = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        if not bool(np.all(equal)):
            raise ValueError(f"{what}: column {column} differs in {int((~np.asarray(equal)).sum())} units.")


def command_units(args: argparse.Namespace) -> None:
    candidates, runtime_weight, artifact = load_candidates(Path(args.benchmark_config))
    rows = ranked_choices(candidates, runtime_weight=runtime_weight)
    stored = pd.read_parquet(Path(args.nested_output) / "unit_scores.parquet", filters=[("method", "==", VB)])
    runtime = rows[rows["rule"].eq("runtime")].drop(columns=["rule"])
    literal = rows[rows["rule"].eq("literal")].drop(columns=["rule"])
    for phase in ["heldout", "train"]:
        _same(runtime, _stored_rows(stored, VB, phase), what=f"{artifact} runtime-term rule against the stored {phase} rows")
    merged = runtime.merge(literal, on=UNIT, suffixes=("_runtime", "_literal"), validate="one_to_one")
    changed = merged[merged["selected_member_runtime"].astype(str) != merged["selected_member_literal"].astype(str)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    summary = {
        "artifact": artifact,
        "runtime_weight": runtime_weight,
        "n_units": int(len(runtime)),
        "units_with_other_choice": int(len(changed)),
        "units_with_other_test_error": int(
            (changed["primary_metric_runtime"].astype(float) != changed["primary_metric_literal"].astype(float)).sum()
        ),
        "units_with_other_rank_regret": int(
            (changed["normalized_rank_regret_runtime"] != changed["normalized_rank_regret_literal"]).sum()
        ),
        "changed_units": changed[[*UNIT, "selected_member_runtime", "selected_member_literal"]].astype(str).to_dict("records"),
        "stored_rows_reproduced": True,
        "benchmark_config": str(args.benchmark_config),
        "nested_output": str(args.nested_output),
        "command": sys.argv,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "changed_units"}), flush=True)


# ----------------------------------------------------------------------------- reviews


def _literal_rows(units_path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(units_path)
    return rows[rows["rule"].eq("literal")].drop(columns=["rule"]).reset_index(drop=True)


def _with_values(frame: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with its VALUES columns taken from ``values`` by unit."""
    keyed = frame.drop(columns=VALUES).reset_index(drop=True)
    merged = keyed.merge(values[[*UNIT, *VALUES]], on=UNIT, how="left", validate="many_to_one")
    if merged[VALUES].isna().any().any():
        raise ValueError("Some units have no literal validation-best row.")
    out = frame.reset_index(drop=True).copy()
    for column in VALUES:
        out[column] = merged[column].astype(frame[column].dtype).to_numpy()
    return out


def _rename(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return values.astype(str).replace(mapping)


def run_review(
    units: pd.DataFrame,
    training: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, Any],
    *,
    pools: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """The steps of ``score_level_review.run`` on frames in memory."""
    scores = review.dataset_split_scores(units)
    choice_frames = [review.train_selected_choices(training, pool_name=name, members=members) for name, members in pools.items()]
    choices = pd.concat(choice_frames, ignore_index=True) if choice_frames else pd.DataFrame()
    if not choices.empty:
        scores = review.append_train_selected(scores, choices)
    means = review.dataset_means(scores)
    bootstrap = dict(config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        means["dataset"].unique(),
        seed=int(bootstrap.get("seed", 20260608)),
        n_resamples=int(bootstrap.get("resamples", 10_000)),
    )
    summary = review.clustered_summary(means, resampler, single_best=config.get("single_best"))
    contrasts = review.contrast_families(
        means,
        resampler,
        families=dict(config.get("families", {})),
        margins=[float(value) for value in config.get("margins", [0.005, 0.010, 0.015, 0.020])],
        rank_scale=float(config["rank_scale"]),
    )
    strata = review.task_stratified_summary(
        means, resampler, methods=[str(value) for value in config.get("task_strata_methods", [])]
    )
    selection_counts = (
        splits.groupby("selected_policy").size().rename("n_splits").reset_index()
        .sort_values(["n_splits", "selected_policy"], ascending=[False, True])
    )
    choice_counts = (
        choices.groupby(["pool", "selected_reference_method"]).size().rename("n_splits").reset_index()
        if not choices.empty
        else pd.DataFrame()
    )
    return {
        "dataset_policy_scores": scores,
        "dataset_mean_scores": means,
        "clustered_summary": summary,
        "contrasts": contrasts,
        "task_stratified_summary": strata,
        "policy_selection_counts": selection_counts,
        "train_selected_reference_choices": choices,
        "train_selected_reference_counts": choice_counts,
        "_resampler": resampler,
    }


def write_review(result: dict[str, Any], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["dataset_policy_scores"].to_parquet(output_dir / "dataset_policy_scores.parquet", index=False)
    for name in [
        "dataset_mean_scores",
        "clustered_summary",
        "contrasts",
        "task_stratified_summary",
        "policy_selection_counts",
        "train_selected_reference_choices",
        "train_selected_reference_counts",
    ]:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)
    resampler = result["_resampler"]
    metadata = {**metadata, "n_datasets": resampler.n, "bootstrap_seed": resampler.seed, "bootstrap_resamples": resampler.n_resamples}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


HOLM_COLUMNS_SUFFIX = "holm_p"


def verify_against_original(result: dict[str, Any], original_dir: Path) -> dict[str, Any]:
    """Everything but the literal reference row equals the original review, except Holm
    adjustments in families that hold a validation-best contrast."""
    report: dict[str, Any] = {}
    means = result["dataset_mean_scores"]
    original = _read(original_dir / "dataset_mean_scores.csv")
    numeric = [column for column in original.columns if column not in {"artifact", "dataset", "task", "method"}]
    new = means.copy()
    new["method"] = _rename(new["method"], {VB: "__literal__", VB_RUNTIME: VB})
    merged = original.merge(new, on=["artifact", "dataset", "task", "method"], how="outer", suffixes=("_o", "_n"), indicator=True)
    extra = merged[merged["_merge"].ne("both")]
    if set(extra["method"]) - {"__literal__"} or not extra["_merge"].eq("right_only").all():
        raise ValueError(f"Rows differ between the literal and the original review: {extra[['method', '_merge']].drop_duplicates().to_dict('records')}")
    both = merged[merged["_merge"].eq("both")]
    worst = 0.0
    for column in numeric:
        gap = (both[f"{column}_o"].astype(float) - both[f"{column}_n"].astype(float)).abs().max()
        worst = max(worst, float(0.0 if pd.isna(gap) else gap))
    if worst > 1e-12:
        raise ValueError(f"Dataset means move by {worst:.3g} outside the reference row.")
    report["max_mean_gap_outside_reference"] = worst

    choices = result["train_selected_reference_choices"].copy()
    choices["selected_reference_method"] = _rename(choices["selected_reference_method"], {VB_RUNTIME: VB})
    original_choices = _read(original_dir / "train_selected_reference_choices.csv")
    key = ["artifact", "split_id", "pool"]
    joined = original_choices.merge(choices, on=key, suffixes=("_o", "_n"), validate="one_to_one")
    if len(joined) != len(original_choices) or (joined["selected_reference_method_o"] != joined["selected_reference_method_n"]).any():
        raise ValueError("Train-selected choices differ from the original review.")
    report["train_selected_choices_identical"] = True

    contrasts = result["contrasts"]
    original_contrasts = _read(original_dir / "contrasts.csv")
    key = ["family", "left", "right"]
    joined = original_contrasts.merge(contrasts, on=key, suffixes=("_o", "_n"), validate="one_to_one")
    if len(joined) != len(original_contrasts):
        raise ValueError("The contrast families differ from the original review.")
    vb_families = set(joined.loc[joined["left"].eq(VB) | joined["right"].eq(VB), "family"])
    # Display labels come from the current label registry, which was reformatted after some
    # reviews were written; they are not results.
    shared = [
        column[:-2]
        for column in joined.columns
        if column.endswith("_o") and f"{column[:-2]}_n" in joined.columns and not column[:-2].endswith("_label")
    ]
    moved = []
    for _, row in joined.iterrows():
        involves = row["left"] == VB or row["right"] == VB
        for column in shared:
            a, b = row[f"{column}_o"], row[f"{column}_n"]
            if isinstance(a, (float, int, np.floating, np.integer)) and isinstance(b, (float, int, np.floating, np.integer)):
                equal = (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) <= 1e-12
            else:
                equal = str(a) == str(b)
            if equal:
                continue
            if involves:
                continue
            if column.endswith(HOLM_COLUMNS_SUFFIX) and row["family"] in vb_families:
                moved.append({"family": row["family"], "left": row["left"], "right": row["right"], "column": column, "original": a, "literal": b})
                continue
            raise ValueError(f"{row['family']} {row['left']} - {row['right']}: {column} moved ({a} -> {b}).")
    report["families_with_validation_best"] = sorted(vb_families)
    report["holm_adjustments_moved"] = moved
    report["validation_best_contrasts"] = int((joined["left"].eq(VB) | joined["right"].eq(VB)).sum())
    return report


def command_review(args: argparse.Namespace) -> None:
    nested = review.load_nested_output(Path(args.nested_output))
    config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    literal = _literal_rows(Path(args.units))
    mapping = {VB: VB_RUNTIME}
    units = nested["units"].copy()
    units["method"] = _rename(units["method"], mapping)
    units["original_method"] = _rename(units["original_method"].fillna(""), mapping)
    training = nested["training"].copy()
    training["method"] = _rename(training["method"], mapping)
    splits = nested["splits"].copy()
    splits["selected_policy"] = _rename(splits["selected_policy"], mapping)

    base = units[units["phase"].eq("heldout") & units["method"].eq(VB_RUNTIME) & ~units["role"].astype(str).eq("alias")]
    reference = _with_values(base, literal)
    reference["method"] = VB
    reference["role"] = "reference"
    reference["original_method"] = ""
    derived = pd.concat([units, reference[units.columns]], ignore_index=True)

    pools = {
        str(name): [VB_RUNTIME if str(member) == VB else str(member) for member in members]
        for name, members in config.get("fixed_reference_pools", {}).items()
    }
    result = run_review(derived, training, splits, config, pools=pools)
    report = verify_against_original(result, Path(args.original_review))
    write_review(
        result,
        Path(args.output_dir),
        {
            "config": str(args.review_config),
            "nested_output": str(args.nested_output),
            "original_review": str(args.original_review),
            "literal_units": str(args.units),
            "reference_row": VB,
            "pool_row": VB_RUNTIME,
            "verification": report,
            "command": sys.argv,
        },
    )
    print(json.dumps({"output": str(args.output_dir), **{k: v for k, v in report.items() if k != "holm_adjustments_moved"}, "holm_moved": len(report["holm_adjustments_moved"])}), flush=True)


# ----------------------------------------------------------------------------- substitution


def _training_means(train_rows: pd.DataFrame) -> pd.DataFrame:
    """Unit-weighted training means per split and policy, one summation path for every policy."""
    return (
        train_rows.groupby(["artifact", "split_id", "method"], as_index=False, sort=True)
        .agg(
            mean_normalized_rank_regret=("normalized_rank_regret", "mean"),
            mean_total_seconds=("total_seconds", "mean"),
        )
    )


def _nested_choices(means: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    rows = []
    for (artifact, split_id), group in means.groupby(["artifact", "split_id"], sort=True):
        chosen = bench._choose_policy(group, pool)
        winner = group[group["method"].isin(pool)]
        ties = winner[winner["mean_normalized_rank_regret"].eq(chosen["mean_normalized_rank_regret"])]["method"].tolist()
        rows.append({"artifact": artifact, "split_id": int(split_id), "selected_policy": str(chosen["method"]), "tied_on_regret": ";".join(sorted(ties)) if len(ties) > 1 else ""})
    return pd.DataFrame(rows)


def _alias_rows(units: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    heldout = units[units["phase"].eq("heldout") & ~units["role"].astype(str).eq("alias")]
    alias = heldout.merge(choices[["artifact", "split_id", "selected_policy"]], left_on=["artifact", "split_id", "method"], right_on=["artifact", "split_id", "selected_policy"])
    alias["original_method"] = alias["method"]
    alias["method"] = NESTED
    alias["role"] = "alias"
    return alias[units.columns]


def command_substitute(args: argparse.Namespace) -> None:
    """Re-choose nested selection and the train-selected rules with the literal rule in the pool."""
    nested_dir = Path(args.nested_output)
    units = pd.read_parquet(nested_dir / "unit_scores.parquet")
    metadata = json.loads((nested_dir / "metadata.json").read_text(encoding="utf-8"))
    pool = [str(method) for method in metadata["artifacts"][0]["policy_methods"]]
    if args.score_only:
        pool = [method for method in pool if "tabrepo_ensemble" not in method]
        units = units[~units["method"].astype(str).str.contains("tabrepo_ensemble", regex=False) | units["role"].astype(str).eq("alias")]
    stored_splits = pd.read_csv(Path(args.stored_splits) if args.stored_splits else nested_dir / "split_results.csv")
    config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    pools = {str(name): [str(member) for member in members] for name, members in config.get("fixed_reference_pools", {}).items()}
    literal = _literal_rows(Path(args.units))

    pool_rows = units[units["phase"].eq("train") & units["role"].astype(str).eq("pool") & units["method"].isin(set(pool) | {m for ms in pools.values() for m in ms})]
    runtime_means = _training_means(pool_rows)
    vb_mask = pool_rows["method"].eq(VB)
    substituted = pd.concat([pool_rows[~vb_mask], _with_values(pool_rows[vb_mask], literal)], ignore_index=True)
    literal_means = _training_means(substituted)

    runtime_nested = _nested_choices(runtime_means, pool)
    check = runtime_nested.merge(stored_splits[["artifact", "split_id", "selected_policy"]], on=["artifact", "split_id"], suffixes=("", "_stored"))
    if (check["selected_policy"] != check["selected_policy_stored"]).any():
        raise ValueError("Recomputed training means do not reproduce the stored nested choices.")
    literal_nested = _nested_choices(literal_means, pool)
    nested_table = runtime_nested.merge(literal_nested, on=["artifact", "split_id"], suffixes=("_runtime", "_literal"))

    pool_tables = []
    stored_choices = _read(Path(args.original_review) / "train_selected_reference_choices.csv")
    for name, members in pools.items():
        for label, means in [("runtime", runtime_means), ("literal", literal_means)]:
            frame = review.train_selected_choices(means, pool_name=name, members=members)
            frame = frame.rename(columns={"selected_reference_method": f"selected_{label}"})[["artifact", "split_id", "pool", f"selected_{label}"]]
            pool_tables.append((name, label, frame))
    fixed = []
    for name in pools:
        runtime = next(frame for pool_name, label, frame in pool_tables if pool_name == name and label == "runtime")
        literal_choice = next(frame for pool_name, label, frame in pool_tables if pool_name == name and label == "literal")
        joined = runtime.merge(literal_choice, on=["artifact", "split_id", "pool"])
        stored = stored_choices[stored_choices["pool"].eq(name)]
        verify = joined.merge(stored, on=["artifact", "split_id", "pool"])
        if (verify["selected_runtime"] != verify["selected_reference_method"]).any():
            raise ValueError(f"Recomputed training means do not reproduce the train-selected choices of pool {name}.")
        fixed.append(joined)
    fixed_table = pd.concat(fixed, ignore_index=True)

    # The substituted review: the literal rule is the pool row and the reference.
    all_units = units[~units["role"].astype(str).eq("alias")]
    vb_all = all_units["method"].eq(VB)
    derived = pd.concat([all_units[~vb_all], _with_values(all_units[vb_all], literal)], ignore_index=True)
    derived = pd.concat([derived, _alias_rows(derived, literal_nested)], ignore_index=True)
    splits = stored_splits.drop(columns=["selected_policy"]).merge(literal_nested[["artifact", "split_id", "selected_policy"]], on=["artifact", "split_id"])
    result = run_review(derived, literal_means, splits, config, pools=pools)
    output_dir = Path(args.output_dir)
    write_review(result, output_dir, {"nested_output": str(nested_dir), "score_only": bool(args.score_only), "command": sys.argv})
    nested_table.to_csv(output_dir / "nested_choices.csv", index=False)
    fixed_table.to_csv(output_dir / "train_selected_choices_compared.csv", index=False)

    reference = _read(Path(args.literal_review) / "contrasts.csv")
    compared = reference.merge(result["contrasts"], on=["family", "left", "right"], suffixes=("_reference", "_substituted"), validate="one_to_one")
    for test in ["bootstrap", "ttest"]:
        column = f"{test}_holm_p"
        if f"{column}_reference" in compared.columns:
            compared[f"{test}_rejects_reference"] = compared[f"{column}_reference"] < 0.05
            compared[f"{test}_rejects_substituted"] = compared[f"{column}_substituted"] < 0.05
            compared[f"{test}_changed"] = compared[f"{test}_rejects_reference"] != compared[f"{test}_rejects_substituted"]
    compared.to_csv(output_dir / "contrasts_compared.csv", index=False)
    changed_cols = [column for column in compared.columns if column.endswith("_changed")]
    summary = {
        "nested_flips": int((nested_table["selected_policy_runtime"] != nested_table["selected_policy_literal"]).sum()),
        "nested_flip_splits": nested_table.loc[nested_table["selected_policy_runtime"] != nested_table["selected_policy_literal"], "split_id"].astype(int).tolist(),
        "nested_regret_ties_literal": nested_table.loc[nested_table["tied_on_regret_literal"].ne(""), "split_id"].astype(int).tolist(),
        "train_selected_flips": {
            name: fixed_table.loc[fixed_table["pool"].eq(name) & (fixed_table["selected_runtime"] != fixed_table["selected_literal"]), "split_id"].astype(int).tolist()
            for name in pools
        },
        "test_results_changed": int(compared[changed_cols].any(axis=1).sum()) if changed_cols else None,
        "max_abs_change_in_mean_difference": float((compared["mean_difference_reference"] - compared["mean_difference_substituted"]).abs().max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


# ----------------------------------------------------------------------------- TabArena


def command_tabarena_ties(args: argparse.Namespace) -> None:
    """Restricted validation-best and the defaults rule's family choice under the literal tie rule."""
    candidates, _, _ = load_candidates(Path(args.benchmark_config))
    frame = candidates[candidates["method_type"].astype(str).eq("candidate")].reset_index(drop=True).copy()
    frame["_order"] = np.arange(len(frame))
    frame["variant"] = frame["method"].map(_method_variant)
    frame["family"] = frame["method"].map(_method_family)
    rows = []
    for variant in ["default", "tuned", "tuned + ensemble"]:
        subset = frame[frame["variant"].eq(variant)]
        # As implemented: lowest validation error, exact ties to the earlier candidate.
        implemented = subset.loc[subset.groupby(UNIT, sort=False)["val_primary_metric"].idxmin()]
        literal = subset.sort_values([*UNIT, "val_primary_metric", "fit_seconds", "_order"], kind="stable").groupby(UNIT, sort=False).head(1)
        joined = implemented[[*UNIT, "method", "family"]].merge(literal[[*UNIT, "method", "family"]], on=UNIT, suffixes=("_implemented", "_literal"))
        rows.append(
            {
                "rule": f"validation-best among {variant}",
                "units": int(len(joined)),
                "units_with_other_choice": int((joined["method_implemented"] != joined["method_literal"]).sum()),
                "units_with_other_family": int((joined["family_implemented"] != joined["family_literal"]).sum()),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "tabarena_tie_convention.csv", index=False)
    print(table.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    units = sub.add_parser("units")
    units.add_argument("--benchmark-config", type=Path, required=True)
    units.add_argument("--nested-output", type=Path, required=True)
    units.add_argument("--output", type=Path, required=True)
    rev = sub.add_parser("review")
    rev.add_argument("--nested-output", type=Path, required=True)
    rev.add_argument("--review-config", type=Path, required=True)
    rev.add_argument("--units", type=Path, required=True)
    rev.add_argument("--original-review", type=Path, required=True)
    rev.add_argument("--output-dir", type=Path, required=True)
    subst = sub.add_parser("substitute")
    subst.add_argument("--nested-output", type=Path, required=True, help="the run whose training rows hold every pool policy")
    subst.add_argument("--review-config", type=Path, required=True)
    subst.add_argument("--units", type=Path, required=True)
    subst.add_argument("--original-review", type=Path, required=True)
    subst.add_argument("--literal-review", type=Path, required=True)
    subst.add_argument("--stored-splits", type=Path, help="split_results.csv of the pool's stored nested choice")
    subst.add_argument("--score-only", action="store_true")
    subst.add_argument("--output-dir", type=Path, required=True)
    ties = sub.add_parser("tabarena-ties")
    ties.add_argument("--benchmark-config", type=Path, required=True)
    ties.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {
        "units": command_units,
        "review": command_review,
        "substitute": command_substitute,
        "tabarena-ties": command_tabarena_ties,
    }[arguments.command](arguments)
