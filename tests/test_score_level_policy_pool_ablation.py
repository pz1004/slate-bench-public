from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_level_policy_pool_ablation.py"
SPEC = importlib.util.spec_from_file_location("score_level_policy_pool_ablation", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def _score_row(
    *,
    split_id: int,
    method: str,
    method_order: int,
    regret: float,
    rank: float | None = None,
    seconds: float = 1.0,
    phase: str = "heldout",
) -> dict[str, object]:
    mean_rank = rank if rank is not None else 1.0 + regret * 100.0
    return {
        "artifact": "Toy",
        "split_id": split_id,
        "phase": phase,
        "method": method,
        "method_order": method_order,
        "mean_rank": mean_rank,
        "tied_best_rate": 0.0,
        "mean_total_seconds": seconds,
        "mean_rank_regret": mean_rank - 1.0,
        "mean_normalized_rank_regret": regret,
        "n_units": 3,
        "n_datasets": 1,
    }


def test_pool_ablation_reselects_after_excluding_methods(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "ablation"
    source.mkdir()

    training = pd.DataFrame(
        [
            _score_row(split_id=0, method="score_a", method_order=0, regret=0.20, phase="train"),
            _score_row(split_id=0, method="score_b", method_order=1, regret=0.10, phase="train"),
            _score_row(split_id=0, method="tabrepo_ensemble_k3", method_order=2, regret=0.01, phase="train"),
            _score_row(split_id=1, method="score_a", method_order=0, regret=0.05, phase="train"),
            _score_row(split_id=1, method="score_b", method_order=1, regret=0.08, phase="train"),
            _score_row(split_id=1, method="tabrepo_ensemble_k3", method_order=2, regret=0.01, phase="train"),
        ]
    )
    heldout = pd.DataFrame(
        [
            _score_row(split_id=0, method="oracle", method_order=-2, regret=0.00),
            _score_row(split_id=0, method="nested_policy_selected", method_order=-1, regret=0.01),
            _score_row(split_id=0, method="score_a", method_order=0, regret=0.30),
            _score_row(split_id=0, method="score_b", method_order=1, regret=0.12),
            _score_row(split_id=0, method="tabrepo_ensemble_k3", method_order=2, regret=0.02),
            _score_row(split_id=1, method="oracle", method_order=-2, regret=0.00),
            _score_row(split_id=1, method="nested_policy_selected", method_order=-1, regret=0.01),
            _score_row(split_id=1, method="score_a", method_order=0, regret=0.09),
            _score_row(split_id=1, method="score_b", method_order=1, regret=0.07),
            _score_row(split_id=1, method="tabrepo_ensemble_k3", method_order=2, regret=0.02),
        ]
    )
    split_results = pd.DataFrame(
        [
            {
                "artifact": "Toy",
                "split_id": 0,
                "split_mode": "global",
                "selected_policy": "tabrepo_ensemble_k3",
                "n_train_datasets": 2,
                "n_heldout_datasets": 1,
                "train_mean_rank": 2.0,
                "train_mean_total_seconds": 1.0,
                "train_mean_normalized_rank_regret": 0.01,
                "heldout_mean_rank": 3.0,
                "heldout_tied_best_rate": 0.0,
                "heldout_mean_total_seconds": 1.0,
                "heldout_mean_rank_regret": 2.0,
                "heldout_mean_normalized_rank_regret": 0.02,
                "heldout_n_units": 3,
            },
            {
                "artifact": "Toy",
                "split_id": 1,
                "split_mode": "global",
                "selected_policy": "tabrepo_ensemble_k3",
                "n_train_datasets": 2,
                "n_heldout_datasets": 1,
                "train_mean_rank": 2.0,
                "train_mean_total_seconds": 1.0,
                "train_mean_normalized_rank_regret": 0.01,
                "heldout_mean_rank": 3.0,
                "heldout_tied_best_rate": 0.0,
                "heldout_mean_total_seconds": 1.0,
                "heldout_mean_rank_regret": 2.0,
                "heldout_mean_normalized_rank_regret": 0.02,
                "heldout_n_units": 3,
            },
        ]
    )
    training.to_csv(source / "training_policy_scores.csv", index=False)
    heldout.to_csv(source / "heldout_policy_scores.csv", index=False)
    split_results.to_csv(source / "split_results.csv", index=False)

    ablation.derive_pool_ablation(
        nested_output=source,
        output_dir=output,
        excluded_substrings=["tabrepo_ensemble"],
        references=["score_a", "score_b"],
        command="test",
    )

    selected = pd.read_csv(output / "split_results.csv").sort_values("split_id")
    summary = pd.read_csv(output / "summary.csv")
    heldout_ablation = pd.read_csv(output / "heldout_policy_scores.csv")

    assert selected["selected_policy"].tolist() == ["score_b", "score_a"]
    assert "tabrepo_ensemble_k3" not in set(heldout_ablation["method"])
    nested_row = summary[summary["method"].eq("nested_policy_selected")].iloc[0]
    assert nested_row["mean_normalized_rank_regret"] == 0.105
