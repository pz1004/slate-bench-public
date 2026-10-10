from __future__ import annotations

import pandas as pd


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])

