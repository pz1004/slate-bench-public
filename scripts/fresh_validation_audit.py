"""Materialize the 2026-10-25 source audit and inspect the original downloaded tasks.

Eligibility decisions are reviewed inputs, never inferred from model scores. This script
reads all 181 complete cached descriptions/schemas and all 46 original data files, but
does not read experiment outcomes or train models. Only aggregate/schema evidence is
written; individual records remain private. See docs/fresh_validation_eligibility_audit.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    from scripts._path_setup import ensure_src_path
ensure_src_path()
from tabular_sota_benchmark.fresh_validation import Task, load_task

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(root: Path = ROOT) -> None:
    review_path = root / "configs/fresh_validation_eligibility_audit.csv"
    original_path = root / "configs/fresh_validation_source_check.csv"
    tasks_path = root / "configs/fresh_validation_tasks.csv"
    review = pd.read_csv(review_path, keep_default_na=False)
    original = pd.read_csv(original_path, comment="#", keep_default_na=False)
    tasks = pd.read_csv(tasks_path, comment="#", keep_default_na=False).set_index("dataset_id")
    if review.dataset_id.duplicated().any() or set(review.dataset_id) != set(original.did):
        raise ValueError("Review must cover each original screening ID exactly once")
    if not review.status.isin(["eligible", "ineligible", "unresolved"]).all():
        raise ValueError("Unknown review status")
    if not review.set_index("dataset_id").original_decision.sort_index().equals(original.set_index("did").decision.sort_index().rename("original_decision")):
        raise ValueError("Review must preserve original decisions")
    out = root / "analysis-output/v2/fresh_validation/audit_2026-10-25"
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    for row in review.itertuples(index=False):
        folder = root / f"data/fresh_validation/catalogue/metadata/{row.dataset_id}"
        desc_path, schema_path = folder / "description.json", folder / "features.json"
        desc = json.loads(desc_path.read_text())["data_set_description"]
        schema = json.loads(schema_path.read_text())["data_features"]["feature"] if schema_path.exists() else []
        item = {"dataset_id": row.dataset_id, "status": row.status,
                "description_sha256": digest(desc_path),
                "features_sha256": digest(schema_path) if schema_path.exists() else None,
                "description_path": str(desc_path.relative_to(root)),
                "features_path": str(schema_path.relative_to(root)),
                "metadata_target": desc.get("default_target_attribute"),
                "metadata_features": [f["name"] for f in schema],
                "metadata_types": {f["name"]: f["data_type"] for f in schema},
                "source_citations": row.source_citations.split(";"),
                "source_citation_field": desc.get("citation"),
                "description_characters_reviewed": len(str(desc.get("description", ""))),
                "criterion": row.criterion, "rationale": row.rationale}
        if row.dataset_id in tasks.index:
            spec = tasks.loc[row.dataset_id]
            task = Task(dataset_id=int(row.dataset_id), name=spec["name"], path=root / spec.path,
                        target=spec.target, drop=tuple(x for x in spec["drop"].split(";") if x))
            features, labels, classes = load_task(task)
            item.update(data_sha256=digest(task.path), actual_target=task.target, dropped_features=list(task.drop),
                        retained_features=list(features.columns), retained_types={k: str(v) for k,v in features.dtypes.items()},
                        n_rows=len(features), n_features=len(features.columns), classes=classes,
                        class_counts=np.bincount(labels).tolist(),
                        duplicate_predictor_rows=int(features.duplicated().sum()),
                        duplicate_labelled_rows=int(features.assign(_audit_label=labels).duplicated().sum()))
            # Descriptive diagnostics, not an automatic rule equating association with leakage.
            checks = {46527: "2019_price_var", 46541: "five-level_classification", 46361: "risk_score",
                      43892: "income96", 40666: "molecule_name", 46372: "spid", 46519: "credit_score"}
            if row.dataset_id in checks:
                column = checks[row.dataset_id]
                paired = pd.DataFrame({"feature": features[column], "target": np.array(classes)[labels]})
                item["inspected_mapping"] = {"feature": column,
                    "distinct_values": int(features[column].nunique()),
                    "values_with_multiple_targets": int(paired.groupby("feature", observed=True).target.nunique().gt(1).sum())}
            if not (2 <= len(classes) <= 10 and min(item["class_counts"]) >= 100 and 1 <= len(features.columns) <= 500):
                raise ValueError(f"Automatic eligibility failed for {row.dataset_id}")
        evidence.append(item)
    evidence_path = out / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    cohort = review[review.dataset_id.isin(tasks.index)].copy()
    if len(cohort) != len(tasks):
        raise ValueError("Trained-task audit incomplete")
    cohort["audit_sha256"] = digest(review_path)
    cohort["evidence_sha256"] = digest(evidence_path)
    cohort.to_csv(root / "configs/fresh_validation_cohort_2026-10-25.csv", index=False)
    paths = [review_path, original_path, tasks_path, root / "configs/fresh_validation_excluded.csv",
             root / "configs/fresh_validation_experiment.yaml", evidence_path, Path(__file__),
             out / "protocol_before_correction.md"]
    summary = {"audit_date": "2026-10-25", "n_screened": len(review), "n_trained": len(cohort),
               "screening_status": review.status.value_counts().to_dict(),
               "trained_status": cohort.status.value_counts().to_dict(),
               "original_rationale_errors": review.loc[review.original_rationale_error.eq("yes"), "dataset_id"].tolist(),
               "inputs_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
               "decision_basis": "source documentation and feature/target inspection; no model outcomes",
               "new_training": False, "population_is_census": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    materialize()
