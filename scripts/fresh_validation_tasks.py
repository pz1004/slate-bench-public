"""Select the fresh-validation experiment's tasks from OpenML (docs/fresh_validation_protocol.md).

``catalogue``
    downloads OpenML's list of active datasets with their metadata qualities, records the
    file's SHA-256, and applies the metadata-only eligibility rule.
``sheet``
    puts the eligible datasets in a seeded random order and, for the first ones, downloads
    their descriptions and feature lists (metadata, no data) into a sheet for the source check.
``select``
    walks the order with the completed source check and freezes the first candidates.
``download``
    after the task list is frozen and published, downloads each candidate's data file and runs
    the prespecified automatic checks; the first tasks that pass are the experiment's tasks.

Downloaded files are data, never code: each goes into its own new directory and is read only
as JSON or parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

API = "https://www.openml.org/api/v1/json"
USER_AGENT = "slate-bench fresh-validation task selection (research; contact via github.com/pz1004/slate-bench-public)"
ROOT = Path(__file__).resolve().parents[1]

# Generator families and other artificial data, by name (case-insensitive).
SYNTHETIC_NAMES = [
    r"^bng\(", r"^fri_c\d", r"gametes", r"^autouniv", r"randomrbf", r"^sea\(", r"^sea_", r"^led", r"hyperplane",
    r"agrawal", r"stagger", r"waveform", r"^friedman", r"^madelon", r"^monks", r"^parity", r"^xor", r"^mv$",
    r"^2dplanes", r"^puma", r"^simulated", r"^synthetic", r"^artificial", r"^hill-valley", r"^ringnorm",
    r"^twonorm", r"^threenorm", r"^chscase", r"^fried",
]
# Families that are not independent tabular tasks, by name: machine-made subsamples of other
# OpenML datasets ("<name>_seed_<k>_nrows_<n>_..."), and exchange-rate time series.
DERIVED_NAMES = {r"_seed_\d+_nrows_\d+": "derived subsample", r"^forex_": "time-series family"}
QUALITIES = [
    "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "MinorityClassSize", "MajorityClassSize",
    "NumberOfMissingValues", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures", "MaxNominalAttDistinctValues",
]


def normalized(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url: str, *, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in {404, 412}:
                raise
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- exclusions


EXCLUDED = ROOT / "configs" / "fresh_validation_excluded.csv"


def command_excluded(args: argparse.Namespace) -> None:
    """Write the committed list of TabArena-v0.1's 51 and TabRepo's 244 OpenML datasets."""
    rows = []
    tabarena = pd.read_csv(ROOT / "data" / "tabarena" / "tabarena_dataset_metadata.csv")
    for _, row in tabarena.iterrows():
        rows.append({"source": "TabArena-v0.1", "did": int(row["dataset_id"]), "name": row["dataset_name"], "openml_name": row["openml_dataset_name"]})
    tabrepo = pd.read_csv(ROOT / ".deps" / "tabrepo" / "data" / "metadata" / "task_metadata_244.csv")
    for _, row in tabrepo.iterrows():
        rows.append({"source": "TabRepo-244", "did": int(row["did"]), "name": row["name"], "openml_name": row["name"]})
    with EXCLUDED.open("w", encoding="utf-8") as handle:
        handle.write("# OpenML datasets of TabArena-v0.1 (data/tabarena/tabarena_dataset_metadata.csv) and of TabRepo's\n")
        handle.write("# 244-dataset context (its task_metadata_244.csv); the fresh-validation experiment excludes them,\n")
        handle.write("# and datasets from the same sources (docs/fresh_validation_protocol.md).\n")
        pd.DataFrame(rows).to_csv(handle, index=False)
    print(f"Wrote {len(rows)} rows to {EXCLUDED}", flush=True)


def _excluded() -> pd.DataFrame:
    return pd.read_csv(EXCLUDED, comment="#")


def excluded_names() -> dict[str, str]:
    """Normalized names of TabArena-v0.1's 51 and TabRepo's 244 datasets, with their source."""
    names: dict[str, str] = {}
    for _, row in _excluded().iterrows():
        for value in [row["name"], row["openml_name"]]:
            if isinstance(value, str) and value:
                names.setdefault(normalized(value), str(row["source"]))
    return names


def excluded_ids() -> dict[int, str]:
    ids: dict[int, str] = {}
    for _, row in _excluded().iterrows():
        ids.setdefault(int(row["did"]), str(row["source"]))
    return ids


def eligible(catalogue: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    """The metadata-only eligibility rule; every exclusion keeps its reason."""
    frame = catalogue.copy()
    reasons = pd.Series("", index=frame.index, dtype=object)

    def mark(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask & reasons.eq("")] = reason

    mark(~frame["status"].eq("active"), "not active")
    mark(frame["format"].fillna("").str.lower().str.contains("sparse"), "sparse format")
    classes = frame["NumberOfClasses"]
    mark(~classes.between(int(rule["min_classes"]), int(rule["max_classes"])), "classes")
    mark(~frame["NumberOfInstances"].between(int(rule["min_rows"]), int(rule["max_rows"])), "rows")
    mark(
        frame["NumberOfFeatures"].isna() | frame["NumberOfFeatures"].sub(1).gt(int(rule["max_features"])) | frame["NumberOfFeatures"].lt(2),
        "features",
    )
    mark(frame["MinorityClassSize"].lt(int(rule["min_class_rows"])) | frame["MinorityClassSize"].isna(), "smallest class")
    pattern = re.compile("|".join(SYNTHETIC_NAMES), flags=re.IGNORECASE)
    mark(frame["name"].map(lambda name: bool(pattern.search(str(name)))), "synthetic generator")
    for derived, reason in DERIVED_NAMES.items():
        compiled = re.compile(derived, flags=re.IGNORECASE)
        mark(frame["name"].map(lambda name, compiled=compiled: bool(compiled.search(str(name)))), reason)
    names = excluded_names()
    mark(frame["name"].map(lambda name: normalized(name) in names), "TabArena or TabRepo name")
    ids = excluded_ids()
    mark(frame["did"].map(lambda did: int(did) in ids), "TabArena or TabRepo id")
    frame["exclusion"] = reasons
    # One version per name: the latest version among the otherwise eligible datasets.
    candidates = frame[frame["exclusion"].eq("")].copy()
    candidates["_name"] = candidates["name"].map(normalized)
    latest = candidates.sort_values(["_name", "version", "did"]).groupby("_name").tail(1).index
    frame.loc[candidates.index.difference(latest), "exclusion"] = "earlier version of an eligible name"
    return frame


def command_catalogue(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_dir) / "openml_data_list_active.json"
    if not raw.exists():
        datasets: list[dict[str, Any]] = []
        offset, page = 0, 5000
        while True:
            try:
                payload = json.loads(fetch(f"{API}/data/list/status/active/limit/{page}/offset/{offset}"))
            except urllib.error.HTTPError as error:
                if error.code == 412:  # "No results": past the last page
                    break
                raise
            batch = payload["data"]["dataset"]
            datasets.extend(batch)
            if len(batch) < page:
                break
            offset += page
        raw.write_text(json.dumps({"retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": datasets}), encoding="utf-8")
    payload = json.loads(raw.read_text(encoding="utf-8"))
    rows = []
    for item in payload["datasets"]:
        qualities = {entry["name"]: entry["value"] for entry in item.get("quality", [])}
        rows.append(
            {
                "did": int(item["did"]),
                "name": str(item["name"]),
                "version": int(item["version"]),
                "status": str(item["status"]),
                "format": str(item.get("format", "")),
                **{name: pd.to_numeric(qualities.get(name), errors="coerce") for name in QUALITIES},
            }
        )
    catalogue = pd.DataFrame(rows).sort_values("did").reset_index(drop=True)
    rule = {"min_classes": 2, "max_classes": 10, "min_rows": 1000, "max_rows": 100000, "max_features": 500, "min_class_rows": 100}
    table = eligible(catalogue, rule)
    table.to_csv(out / "catalogue_eligibility.csv", index=False)
    summary = {
        "catalogue_file": str(raw),
        "catalogue_sha256": sha256(raw),
        "retrieved_utc": payload.get("retrieved_utc"),
        "n_active_datasets": int(len(catalogue)),
        "rule": rule,
        "synthetic_name_patterns": SYNTHETIC_NAMES,
        "derived_name_patterns": DERIVED_NAMES,
        "n_eligible": int(table["exclusion"].eq("").sum()),
        "exclusions": table["exclusion"].replace("", "eligible").value_counts().to_dict(),
    }
    (out / "catalogue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith("_patterns")}, indent=2), flush=True)


# ----------------------------------------------------------------------------- source-check sheet


def _description(did: int, directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """The dataset's description and feature list; the feature list is None if OpenML cannot serve it."""
    target = directory / str(did)
    target.mkdir(parents=True, exist_ok=True)
    description_path, features_path = target / "description.json", target / "features.json"
    if not description_path.exists():
        description_path.write_bytes(fetch(f"{API}/data/{did}"))
    if not features_path.exists():
        try:
            features_path.write_bytes(fetch(f"{API}/data/features/{did}"))
        except urllib.error.HTTPError as error:
            (target / "features_unavailable.txt").write_text(f"HTTP {error.code}\n", encoding="utf-8")
    description = json.loads(description_path.read_text(encoding="utf-8"))["data_set_description"]
    if not features_path.exists():
        return description, None
    features = json.loads(features_path.read_text(encoding="utf-8"))["data_features"]["feature"]
    return description, features


def _as_list(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def command_sheet(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    table = pd.read_csv(out / "catalogue_eligibility.csv")
    pool = table[table["exclusion"].isna() | table["exclusion"].eq("")].sort_values("did")
    order = np.random.default_rng(int(args.seed)).permutation(pool["did"].to_numpy())
    ordered = pool.set_index("did").loc[order].reset_index()
    ordered.insert(0, "order", np.arange(1, len(ordered) + 1))
    ordered.to_csv(out / "eligible_in_random_order.csv", index=False)
    rows = []
    for _, row in ordered.head(int(args.first)).iterrows():
        description, features = _description(int(row["did"]), Path(args.raw_dir) / "metadata")
        target = str(description.get("default_target_attribute") or "")
        available = features is not None
        features = features or []
        types = pd.Series([feature["data_type"] for feature in features], dtype=object)
        target_type = next((feature["data_type"] for feature in features if feature["name"] == target), "") if available else "unavailable"
        text = re.sub(r"\s+", " ", str(description.get("description") or "")).strip()
        rows.append(
            {
                "order": int(row["order"]),
                "did": int(row["did"]),
                "name": row["name"],
                "version": int(row["version"]),
                "rows": int(row["NumberOfInstances"]),
                "features": int(row["NumberOfFeatures"]) - 1,
                "classes": int(row["NumberOfClasses"]),
                "smallest_class": int(row["MinorityClassSize"]),
                "default_target": target,
                "target_type": target_type,
                "row_id": ";".join(_as_list(description.get("row_id_attribute"))),
                "ignore": ";".join(_as_list(description.get("ignore_attribute"))),
                "string_features": int((types == "string").sum()) if available else -1,
                "date_features": int((types == "date").sum()) if available else -1,
                "licence": str(description.get("licence") or ""),
                "original_data_url": str(description.get("original_data_url") or ""),
                "collection_date": str(description.get("collection_date") or ""),
                "upload_date": str(description.get("upload_date") or ""),
                "tags": ";".join(_as_list(description.get("tag"))),
                "parquet_url": str(description.get("parquet_url") or ""),
                "citation": re.sub(r"\s+", " ", str(description.get("citation") or ""))[:300],
                "description": text[: int(args.characters)],
            }
        )
    sheet = pd.DataFrame(rows)
    sheet.to_csv(out / "source_check_sheet.csv", index=False)
    print(f"Wrote {len(sheet)} rows to {out / 'source_check_sheet.csv'}", flush=True)


# ----------------------------------------------------------------------------- selection


def command_select(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    sheet = pd.read_csv(out / "source_check_sheet.csv", dtype={"row_id": str, "ignore": str}).fillna("")
    check = pd.read_csv(args.source_check, comment="#", dtype=str).fillna("")
    merged = sheet.merge(check[["did", "decision", "reason", "evidence", "note"]].astype({"did": int}), on="did", how="left", validate="one_to_one")
    merged["decision"] = merged["decision"].fillna("")
    if not merged["decision"].isin(["include", "exclude", ""]).all():
        raise ValueError("Decisions must be include or exclude")
    included = merged[merged["decision"].eq("include")].sort_values("order").head(int(args.candidates))
    if len(included) < int(args.candidates):
        raise ValueError(f"Only {len(included)} included datasets; extend the sheet")
    last = int(included["order"].max())
    if merged.loc[merged["order"].le(last), "decision"].eq("").any():
        raise ValueError("The source check leaves datasets undecided before the last selected one")
    summary = json.loads((out / "catalogue_summary.json").read_text(encoding="utf-8"))
    tasks = pd.DataFrame(
        {
            "order": included["order"].astype(int),
            "dataset_id": included["did"].astype(int),
            "name": included["name"],
            "version": included["version"].astype(int),
            "rows": included["rows"].astype(int),
            "features": included["features"].astype(int),
            "classes": included["classes"].astype(int),
            "smallest_class": included["smallest_class"].astype(int),
            "licence": included["licence"],
            "target": included["default_target"],
            "drop": [";".join(item for item in [row_id, ignore] if item) for row_id, ignore in zip(included["row_id"], included["ignore"])],
            "parquet_url": included["parquet_url"],
            "path": [f"data/fresh_validation/openml/{did}/dataset_{did}.pq" for did in included["did"].astype(int)],
            "use": "pending",
        }
    )
    target = Path(args.tasks)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("# Candidate tasks of the fresh-validation experiment (docs/fresh_validation_protocol.md), in the\n")
        handle.write(f"# seeded random order of the eligible OpenML datasets (catalogue SHA-256 {summary['catalogue_sha256']},\n")
        handle.write(f"# retrieved {summary['retrieved_utc']}), after the source check: every included dataset. `use` turns\n")
        handle.write("# to yes for each that passes the automatic checks after download, and to no with the reason otherwise.\n")
        tasks.to_csv(handle, index=False)
    print(f"Froze {len(tasks)} candidate tasks in {target}", flush=True)


# ----------------------------------------------------------------------------- download and checks


def automatic_checks(path: Path, target: str, drop: list[str], rule: dict[str, Any]) -> str:
    """'' if the file passes the prespecified checks, else the first failed check."""
    try:
        table = pq.read_table(path)
    except Exception as error:  # noqa: BLE001 - an unreadable file fails the check
        return f"unreadable: {type(error).__name__}"
    frame = table.to_pandas()
    if target not in frame.columns:
        return "default target missing"
    missing_drop = [column for column in drop if column not in frame.columns]
    if missing_drop:
        return f"attributes to drop missing: {missing_drop}"
    labels = frame[target].dropna().astype(str)
    counts = labels.value_counts()
    features = [column for column in frame.columns if column != target and column not in set(drop)]
    if not int(rule["min_classes"]) <= len(counts) <= int(rule["max_classes"]):
        return f"{len(counts)} classes"
    if int(counts.min()) < int(rule["min_class_rows"]):
        return f"smallest class has {int(counts.min())} rows"
    if not 1 <= len(features) <= int(rule["max_features"]):
        return f"{len(features)} features"
    return ""


def command_download(args: argparse.Namespace) -> None:
    tasks_path = Path(args.tasks)
    header = [line for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    tasks = pd.read_csv(tasks_path, comment="#", dtype=str).fillna("")
    rule = {"min_classes": 2, "max_classes": 10, "max_features": 500, "min_class_rows": 100}
    log = []
    used = 0
    for index, row in tasks.sort_values("order", key=lambda column: column.astype(int)).iterrows():
        did = int(row["dataset_id"])
        path = ROOT / row["path"]
        if used >= int(args.tasks_needed):
            tasks.loc[index, "use"] = "no (not needed)"
            continue
        if not path.exists():
            if path.parent.exists() and any(path.parent.iterdir()):
                raise FileExistsError(f"{path.parent} is not empty")
            path.parent.mkdir(parents=True, exist_ok=True)
            url = row["parquet_url"] or f"https://data.openml.org/datasets/{did // 10000:04d}/{did:04d}/dataset_{did}.pq"
            path.write_bytes(fetch(url))
        failure = automatic_checks(path, row["target"], [item for item in row["drop"].split(";") if item], rule)
        log.append({"dataset_id": did, "name": row["name"], "bytes": path.stat().st_size, "sha256": sha256(path), "check": failure or "pass"})
        if failure:
            tasks.loc[index, "use"] = f"no ({failure})"
        else:
            tasks.loc[index, "use"] = "yes"
            used += 1
    with tasks_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        tasks.to_csv(handle, index=False)
    pd.DataFrame(log).to_csv(Path(args.log), index=False)
    print(f"{used} tasks pass the checks; log in {args.log}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("excluded")
    catalogue = sub.add_parser("catalogue")
    catalogue.add_argument("--raw-dir", type=Path, default=Path("data/fresh_validation/catalogue"))
    catalogue.add_argument("--output-dir", type=Path, default=Path("analysis-output/v2/fresh_validation_tasks"))
    sheet = sub.add_parser("sheet")
    sheet.add_argument("--raw-dir", type=Path, default=Path("data/fresh_validation/catalogue"))
    sheet.add_argument("--output-dir", type=Path, default=Path("analysis-output/v2/fresh_validation_tasks"))
    sheet.add_argument("--seed", type=int, default=20261022)
    sheet.add_argument("--first", type=int, default=150, help="how many datasets of the random order to describe")
    sheet.add_argument("--characters", type=int, default=600, help="description characters kept in the sheet")
    select = sub.add_parser("select")
    select.add_argument("--output-dir", type=Path, default=Path("analysis-output/v2/fresh_validation_tasks"))
    select.add_argument("--source-check", type=Path, default=Path("configs/fresh_validation_source_check.csv"))
    select.add_argument("--candidates", type=int, default=46, help="every dataset the source check includes")
    select.add_argument("--tasks", type=Path, default=Path("configs/fresh_validation_tasks.csv"))
    download = sub.add_parser("download")
    download.add_argument("--tasks", type=Path, default=Path("configs/fresh_validation_tasks.csv"))
    download.add_argument("--tasks-needed", type=int, default=1000, help="use every candidate that passes the checks")
    download.add_argument("--log", type=Path, default=Path("analysis-output/v2/fresh_validation_tasks/downloads.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {
        "excluded": command_excluded,
        "catalogue": command_catalogue,
        "sheet": command_sheet,
        "select": command_select,
        "download": command_download,
    }[
        arguments.command
    ](arguments)
