"""Build, verify, or reproduce the local 2026-10-26 correction package.

The build reads only an explicit allowlist. Verification uses included snapshots,
not Git or raw task rows. Reproduction reads frozen candidate scores and runs no
training, source audit, downloads, or subprocesses.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import socket
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "validation-reuse-correction-2026-10-26"
RUN = Path("analysis-output/v2/fresh_validation")
AUDIT = RUN / "audit_2026-10-25"
ALLOWLIST = Path("configs/validation_revision_package_2026-10-26.txt")
COHORTS = ("original", "three-exclusion", "eligible", "eligible-unresolved")
DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "pyarrow",
                "joblib", "threadpoolctl", "python-dateutil", "six")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe package path: {value}")
    return path


def allowed_paths(root: Path) -> list[Path]:
    names = [line.strip() for line in (root / ALLOWLIST).read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate allowlist entries")
    paths = [safe_relative(name) for name in names]
    for path in paths:
        if path.parts[0] in {"paper", ".git", ".deps"} or "__pycache__" in path.parts:
            raise ValueError(f"Forbidden package input: {path}")
        if path.parts[0] == "data" and not (str(path).startswith("data/fresh_validation/catalogue/metadata/")
                                            and path.name in {"description.json", "features.json"}):
            raise ValueError(f"Raw data or unrelated cache is not allowed: {path}")
        source = root / path
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing or linked allowlist input: {path}")
    return paths


def cohort_ids(root: Path) -> dict[str, list[int]]:
    manifest = rows(root / "configs/fresh_validation_cohort_2026-10-25.csv")
    all_ids = {int(row["dataset_id"]) for row in manifest}
    return {
        "original": sorted(all_ids),
        "three-exclusion": sorted(all_ids - {46527, 40666, 46864}),
        "eligible": sorted(int(row["dataset_id"]) for row in manifest if row["status"] == "eligible"),
        "eligible-unresolved": sorted(int(row["dataset_id"]) for row in manifest if row["status"] != "ineligible"),
    }


def export_memberships(root: Path, destination: Path) -> None:
    register = {int(row["dataset_id"]): row for row in rows(root / "configs/fresh_validation_cohort_2026-10-25.csv")}
    destination.mkdir(parents=True, exist_ok=True)
    for selection, ids in cohort_ids(root).items():
        with (destination / f"{selection}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset_id", "name", "status"])
            writer.writeheader()
            writer.writerows({key: register[did][key] for key in writer.fieldnames} for did in ids)


def check_evidence(root: Path) -> dict:
    """Check cross-file links independently of the archive's inventory hashes."""
    audit = rows(root / "configs/fresh_validation_eligibility_audit.csv")
    historical = rows(root / "configs/fresh_validation_source_check.csv")
    evidence = read_json(root / AUDIT / "evidence.json")
    ids = {int(row["dataset_id"]) for row in audit}
    if len(audit) != 181 or len(ids) != 181 or ids != {int(row["did"]) for row in historical}:
        raise ValueError("Incomplete 181-decision register")
    if ids != {int(item["dataset_id"]) for item in evidence} or len(evidence) != 181:
        raise ValueError("Incomplete inspected evidence")
    missing_schemas = []
    for item in evidence:
        for field in ("description", "features"):
            expected = item[f"{field}_sha256"]
            path = root / safe_relative(item[f"{field}_path"])
            if expected is None:
                if path.exists():
                    raise ValueError(f"Undocumented new evidence: {path}")
                missing_schemas.append(item["dataset_id"])
            elif not path.is_file() or digest(path) != expected:
                raise ValueError(f"Evidence hash mismatch: {path}")
    manifest = rows(root / "configs/fresh_validation_cohort_2026-10-25.csv")
    if any(row["audit_sha256"] != digest(root / "configs/fresh_validation_eligibility_audit.csv")
           or row["evidence_sha256"] != digest(root / AUDIT / "evidence.json") for row in manifest):
        raise ValueError("Stale cohort manifest")
    selections = cohort_ids(root)
    if {key: len(value) for key, value in selections.items()} != dict(zip(COHORTS, (46, 43, 11, 34))):
        raise ValueError("Unexpected cohort sizes")
    if {item["dataset_id"] for item in evidence if "actual_target" in item} != set(selections["original"]):
        raise ValueError("Missing inspected feature/target records")
    for selection, expected in selections.items():
        dec = read_json(root / RUN / f"analysis_2026-10-25_{selection}/decisions.json")
        if dec["cohort"]["selected_dataset_ids"] != expected:
            raise ValueError(f"Stored membership mismatch: {selection}")
        for key, path in [("candidates", RUN / "candidates.parquet"),
                          ("config", Path("configs/fresh_validation_experiment.yaml")),
                          ("analysis_code", Path("scripts/fresh_validation_analysis.py"))]:
            if dec["input_sha256"][key] != digest(root / path):
                raise ValueError(f"Stored analysis hash mismatch: {selection}/{key}")
        if dec["cohort"]["manifest_sha256"] != digest(root / "configs/fresh_validation_cohort_2026-10-25.csv"):
            raise ValueError("Stored manifest hash mismatch")
        if selection != "original" and (dec["analysis_status"] != "post hoc eligibility correction"
                or any(d.get("confirmatory") is not False for d in dec["decisions"].values())):
            raise ValueError("Corrected analysis lost post hoc status")
    metadata = read_json(root / RUN / "metadata.json")
    training = {"config": Path("configs/fresh_validation_experiment.yaml"),
                "tasks": Path("configs/fresh_validation_tasks.csv"),
                "module": Path("src/tabular_sota_benchmark/fresh_validation.py"),
                "runner": Path("scripts/fresh_validation_experiment.py"),
                "protocol": AUDIT / "protocol_at_training.md"}
    for key, path in training.items():
        if digest(root / path) != metadata["sha256"][key]:
            raise ValueError(f"Training snapshot hash mismatch: {path}")
    # The old verification record may mention other artifacts outside this package.
    # Check its preserved experiment files, excluding today's changed asset builder.
    record = read_json(root / AUDIT / "verification.json")
    checked = 0
    for name, expected in record["input_and_output_sha256"].items():
        path = root / name
        if path.is_file() and name != "scripts/build_paper_assets.py":
            if digest(path) != expected:
                raise ValueError(f"October 25 record changed: {name}")
            checked += 1
    return {"screened": len(audit), "inspected_trained_tasks": 46,
            "cohorts": {k: len(v) for k, v in selections.items()},
            "missing_feature_schemas": missing_schemas, "historical_hashes_checked": checked}


def build(root: Path, output: Path) -> Path:
    paths = allowed_paths(root)
    check_evidence(root)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{NAME}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="validation-package-") as temp:
        stage = Path(temp) / NAME
        stage.mkdir()
        for path in paths:
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, target)
        export_memberships(root, stage / "cohorts")
        versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
        (stage / "requirements-reproduction.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n")
        source_index, gaps = [], []
        for item in read_json(root / AUDIT / "evidence.json"):
            desc = read_json(root / item["description_path"])["data_set_description"]
            source_index.append({"dataset_id": item["dataset_id"], "source_urls": item["source_citations"],
                                 "description_path": item["description_path"], "features_path": item["features_path"],
                                 "license_as_recorded": desc.get("licence", desc.get("license")),
                                 "citation_as_recorded": desc.get("citation"), "url_as_recorded": desc.get("url"),
                                 "primary_pages": "Cited URLs retained; no separate primary-page snapshot included."})
            if item["status"] == "unresolved" or item["features_sha256"] is None:
                gaps.append({"dataset_id": item["dataset_id"], "status": item["status"],
                             "missing_features_snapshot": item["features_sha256"] is None,
                             "documentation_limit": item["rationale"], "source_urls": item["source_citations"]})
        write_json(stage / "source_index.json", source_index)
        write_json(stage / "missing_documentation.json", gaps)
        shutil.copyfile(root / "docs/validation_revision_package.md", stage / "README.md")
        shutil.copyfile(root / "docs/validation_revision_notice.md", stage / "NOTICE.md")
        write_json(stage / "packaging_record.json", {
            "date": "2026-10-26", "name": NAME, "publication_status": "local only; not yet published",
            "evidence_date": "2026-10-25", "new_training": False, "new_source_audit": False,
            "python": platform.python_version(), "platform": platform.platform(), "dependencies": versions,
            "allowlist": str(ALLOWLIST), "scope": "fresh-validation correction and stored-score reproduction",
            "limitations": ["Raw data inspection is documented, not repeated by this package.",
                            "Primary web pages are cited, not all captured.",
                            "Supporting artifact checks remain in the October 25 record; they are not rerun here."]})
        files = {str(path.relative_to(stage)): digest(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "manifest.json", {"format": 1, "name": NAME, "sha256": files})
        verify(stage)
        # Fixed headers make a rebuild with identical inputs byte-identical.
        with archive.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        info = tar.gettarinfo(str(path), arcname=f"{NAME}/{path.relative_to(stage)}")
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        info.mode = 0o644
                        with path.open("rb") as content:
                            tar.addfile(info, content)
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    print(f"{archive}\nSHA-256 {checksum}")
    return archive


def verify(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    expected = manifest["sha256"]
    for name, checksum in expected.items():
        path = root / safe_relative(name)
        if not path.is_file() or path.is_symlink() or digest(path) != checksum:
            raise ValueError(f"Package hash failure: {name}")
    # Reject additions to evidence/code; runtime outputs have dedicated directories.
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
              and "__pycache__" not in path.parts and path.relative_to(root).parts[0] not in {"reproduced", "generated-fresh"}}
    if actual != set(expected) | {"manifest.json"}:
        raise ValueError(f"Package inventory mismatch: {sorted(actual ^ (set(expected) | {'manifest.json'}))}")
    allowlisted = set(str(p) for p in allowed_paths(root))
    if not allowlisted <= set(expected):
        raise ValueError("Package is missing allowlisted files")
    details = check_evidence(root)
    for selection, ids in cohort_ids(root).items():
        exported = rows(root / "cohorts" / f"{selection}.csv")
        original = {int(row["dataset_id"]): row for row in rows(root / "configs/fresh_validation_cohort_2026-10-25.csv")}
        if [int(row["dataset_id"]) for row in exported] != ids or any(
                any(row[key] != original[int(row["dataset_id"])][key] for key in ("name", "status")) for row in exported):
            raise ValueError(f"Exported membership mismatch: {selection}")
    return {"files_verified": len(expected), **details}


@contextlib.contextmanager
def offline():
    """Fail any accidental Python network access, including in future dependencies."""
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled during stored-score reproduction")
    old_socket, old_connect, old_lookup = socket.socket, socket.create_connection, socket.getaddrinfo
    socket.socket = socket.create_connection = socket.getaddrinfo = blocked
    try:
        yield
    finally:
        socket.socket, socket.create_connection, socket.getaddrinfo = old_socket, old_connect, old_lookup


def reproduce(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents and output.relative_to(root).parts[0] != "reproduced":
        raise ValueError("Use reproduced/ or an output directory outside the package; preserve stored results")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Reproduction output directory must be empty")
    integrity = verify(root) if (root / "manifest.json").exists() else check_evidence(root)
    # Require the code being executed to belong to the root being verified.
    if ROOT.resolve() != root:
        raise ValueError("Run the script inside the extracted package")
    sys.path.insert(0, str(root / "scripts"))
    from fresh_validation_analysis import run
    import pandas as pd
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    with contextlib.chdir(root), offline():
        for selection in COHORTS:
            destination = output / selection
            with contextlib.redirect_stdout(io.StringIO()):
                run(argparse.Namespace(config=root / "configs/fresh_validation_experiment.yaml", run_dir=root / RUN,
                    output_dir=destination, cohort_manifest=Path("configs/fresh_validation_cohort_2026-10-25.csv"),
                    cohort_selection=selection))
            preserved = root / RUN / f"analysis_2026-10-25_{selection}"
            for path in preserved.glob("*.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(destination / path.name), check_exact=True)
                if selection == "original":
                    pd.testing.assert_frame_equal(pd.read_csv(root / RUN / "analysis" / path.name),
                                                  pd.read_csv(destination / path.name), check_exact=True)
            expected, actual = read_json(preserved / "decisions.json"), read_json(destination / "decisions.json")
            if {k: v for k, v in expected.items() if k != "command"} != {k: v for k, v in actual.items() if k != "command"}:
                raise ValueError(f"Decision mismatch: {selection}")
            if selection == "original":
                old = read_json(root / RUN / "analysis/decisions.json")
                if any(actual[k] != v for k, v in old.items() if k != "command"):
                    raise ValueError("Original decisions changed")
            results[selection] = {"n_datasets": actual["n_datasets"], "decisions": actual["decisions"],
                                  "all_seven_csvs_exact": True, "decision_fields_exact_except_command": True}
    c1 = results["three-exclusion"]["decisions"]["C1"]
    if [round(c1["mean_difference"], 5), *[round(v, 5) for v in c1["ci_95"]]] != [.03365, .01477, .05527]:
        raise ValueError("Three-exclusion result changed")
    report = {"date": "2026-10-26", "network_disabled": True, "raw_task_data_used": False,
              "training_performed": False, "integrity": integrity, "cohorts": results,
              "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES}}
    write_json(output / "verification.json", report)
    print("Four cohorts reproduced exactly (all endpoints, intervals, tests, choices and status); no training or network.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "verify", "reproduce"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(ROOT, args.output or ROOT / "dist")
    elif args.action == "verify":
        print(json.dumps(verify(ROOT), indent=2))
    else:
        reproduce(ROOT, args.output or ROOT / "reproduced")


if __name__ == "__main__":
    main()
