from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "tabrepo_overlap_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("tabrepo_overlap_sensitivity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
overlap_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overlap_module
SPEC.loader.exec_module(overlap_module)

from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402

METHODS = ["nested_policy_selected", "meta_selector_single", "meta_selector_family_prior_budgeted"]
CONFIG = {
    "rank_scale": 100,
    "bootstrap": {"seed": 11, "resamples": 200},
    "margins": [0.01],
    "families": {
        "primary": {"left": "nested_policy_selected", "references": ["meta_selector_single"]},
        "mechanism": {"left": "meta_selector_single", "references": ["meta_selector_family_prior_budgeted"]},
    },
}


def _review(directory: Path, *, nested_shift: float, seed: int = 0) -> None:
    """A stored review: split-level scores, its contrasts and its metadata."""
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in [f"d{number:02d}" for number in range(30)]:
        base = {"meta_selector_single": rng.uniform(0.1, 0.3), "meta_selector_family_prior_budgeted": rng.uniform(0.1, 0.3)}
        base["nested_policy_selected"] = base["meta_selector_single"] + nested_shift + rng.normal(0, 0.01)
        for split in range(2):
            for method in METHODS:
                rows.append({
                    "artifact": "TabRepo", "split_id": split, "dataset": dataset, "task": "binary", "method": method,
                    "mean_rank": 1.0, "mean_normalized_rank_regret": base[method], "mean_rank_regret": 0.0,
                    "tied_best_rate": 0.0, "mean_total_seconds": 1.0, "mean_maxmin_error_regret": 0.0,
                    "mean_median_error_regret": 0.0, "beats_best_rate": 0.0, "n_units": 3,
                })
    directory.mkdir(parents=True)
    scores = pd.DataFrame(rows)
    scores.to_parquet(directory / "dataset_policy_scores.parquet", index=False)
    config_path = directory / "review.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    means = overlap_module.review.dataset_means(scores)
    resampler = ArtifactResampler(sorted(means["dataset"].unique()), seed=11, n_resamples=200)
    overlap_module.rerun(means, CONFIG, resampler).to_csv(directory / "contrasts.csv", index=False)
    (directory / "metadata.json").write_text(json.dumps({"config": str(config_path)}), encoding="utf-8")


def _args(tmp_path: Path, names: list[str], *, related: list[str] | None = None) -> Namespace:
    overlap = tmp_path / "overlap.csv"
    pd.DataFrame({"dataset": names}).to_csv(overlap, index=False)
    return Namespace(
        overlap=overlap,
        related=related or [],
        review=[f"ensemble={tmp_path / 'ensemble'}", f"score_only={tmp_path / 'score_only'}"],
        output_dir=tmp_path / "out",
    )


def test_excluded_datasets_are_dropped_and_the_full_sample_reproduces(tmp_path: Path) -> None:
    _review(tmp_path / "ensemble", nested_shift=-0.02)
    _review(tmp_path / "score_only", nested_shift=0.0)
    overlap_module.run(_args(tmp_path, ["d01", "d02"], related=["d03"]))
    out = pd.read_csv(tmp_path / "out" / "overlap_contrasts.csv")
    assert set(out["n_datasets"]) == {27}
    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert metadata["excluded"] == ["d01", "d02", "d03"]
    assert metadata["panels"]["score_only"]["derived_seed"] != 11
    stored = pd.read_csv(tmp_path / "ensemble" / "contrasts.csv")
    row = out[(out["panel"] == "ensemble") & (out["family"] == "primary")].iloc[0]
    assert row["full_mean_difference"] == pytest.approx(stored.loc[stored["family"] == "primary", "mean_difference"].iloc[0])


def test_unknown_overlap_name_is_refused(tmp_path: Path) -> None:
    _review(tmp_path / "ensemble", nested_shift=-0.02)
    _review(tmp_path / "score_only", nested_shift=0.0)
    with pytest.raises(SystemExit, match="not in the review"):
        overlap_module.run(_args(tmp_path, ["d01", "elsewhere"]))


def test_stored_contrasts_that_do_not_reproduce_are_refused(tmp_path: Path) -> None:
    _review(tmp_path / "ensemble", nested_shift=-0.02)
    _review(tmp_path / "score_only", nested_shift=0.0)
    path = tmp_path / "ensemble" / "contrasts.csv"
    stored = pd.read_csv(path)
    stored.loc[0, "mean_difference"] += 0.01
    stored.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="does not reproduce"):
        overlap_module.run(_args(tmp_path, ["d01"]))


def test_panels_must_share_the_rows_outside_the_nested_choice(tmp_path: Path) -> None:
    _review(tmp_path / "ensemble", nested_shift=-0.02, seed=0)
    _review(tmp_path / "score_only", nested_shift=0.0, seed=1)
    with pytest.raises(SystemExit, match="differ between the panels"):
        overlap_module.run(_args(tmp_path, ["d01"]))
