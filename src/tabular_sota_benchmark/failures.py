from __future__ import annotations

from datetime import datetime
from pathlib import Path


HEADER = """# Failures

No failures logged yet.

| Execution date | Step | Baseline | Verification | Reason for failure | Cause analysis | Corrective action | Restart point |
|---|---|---|---|---|---|---|---|
"""


def _escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def ensure_failure_file(path: str | Path = "failures.md") -> Path:
    failure_path = Path(path)
    if not failure_path.exists() or not failure_path.read_text(encoding="utf-8").strip():
        failure_path.write_text(HEADER, encoding="utf-8")
    return failure_path


def append_failure(
    *,
    step: str,
    baseline: str,
    verification: str,
    reason: str,
    cause_analysis: str,
    corrective_action: str,
    restart_point: str,
    path: str | Path = "failures.md",
) -> None:
    failure_path = ensure_failure_file(path)
    text = failure_path.read_text(encoding="utf-8")
    text = text.replace("No failures logged yet.\n\n", "Failures are logged below.\n\n")
    execution_date = datetime.now().astimezone().isoformat(timespec="seconds")
    row = "| " + " | ".join(
        _escape_cell(cell)
        for cell in [
            execution_date,
            step,
            baseline,
            verification,
            reason,
            cause_analysis,
            corrective_action,
            restart_point,
        ]
    ) + " |\n"
    failure_path.write_text(text.rstrip() + "\n" + row, encoding="utf-8")

