from __future__ import annotations

from pathlib import Path

from tabular_sota_benchmark.failures import append_failure


def test_append_failure_writes_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "failures.md"
    append_failure(
        step="Step X",
        baseline="baseline",
        verification="check",
        reason="reason",
        cause_analysis="cause",
        corrective_action="fix",
        restart_point="Step X",
        path=path,
    )
    text = path.read_text(encoding="utf-8")
    assert "Step X" in text
    assert "baseline" in text
    assert "cause" in text

