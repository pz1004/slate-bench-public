from __future__ import annotations

import pandas as pd

from tabular_sota_benchmark.verify import verify_step_3


def _write_config(tmp_path, selector_name: str) -> str:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "d1",
                "task": "classification",
                "primary_metric_name": "log_loss",
                "fold": 0,
                "seed": 0,
                "method": "candidate_a",
                "method_type": "candidate",
                "primary_metric": 0.4,
            },
            {
                "dataset": "d1",
                "task": "classification",
                "primary_metric_name": "log_loss",
                "fold": 0,
                "seed": 0,
                "method": selector_name,
                "method_type": "selector",
                "primary_metric": 0.4,
            },
        ]
    ).to_csv(output_dir / "results.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: selector_presence_test",
                f"  output_dir: {output_dir}",
                "verification:",
                "  min_datasets: 1",
                "  min_candidate_models: 1",
                "  min_comparison_units: 1",
                "  require_selector: true",
                "  selector_must_beat_median_rank: false",
                f"  selector_must_be_present: {selector_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(config_path)


def test_verify_step_3_accepts_required_selector_method(tmp_path) -> None:
    config_path = _write_config(tmp_path, "meta_selector_required")

    result = verify_step_3(config_path)

    assert result.passed


def test_verify_step_3_rejects_missing_required_selector_method(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, "meta_selector_required")
    config_text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        config_text.replace("meta_selector_required", "meta_selector_missing"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = verify_step_3(config_path)

    assert not result.passed
    assert "required selector methods missing" in result.reason
