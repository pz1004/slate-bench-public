"""The archive must be inspectable and replayable without the originating checkout."""
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import validation_revision_package as package


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("archive")
    archive = package.build(package.ROOT, directory)
    checksum = package.digest(archive)
    assert package.digest(package.build(package.ROOT, directory)) == checksum
    with tarfile.open(archive) as handle:
        handle.extractall(directory, filter="data")
    return directory / package.NAME


def test_package_completeness_and_exports(built_package):
    result = package.verify(built_package)
    assert result["screened"] == 181
    assert result["missing_feature_schemas"] == [46520]
    assert result["cohorts"] == dict(zip(package.COHORTS, (46, 43, 11, 34)))
    assert not (built_package / "paper").exists()
    assert not (built_package / ".git").exists()
    assert not (built_package / "data/fresh_validation/openml").exists()
    assert len(package.read_json(built_package / "missing_documentation.json")) == 42
    for name, ids in package.cohort_ids(built_package).items():
        assert [int(row["dataset_id"]) for row in package.rows(built_package / "cohorts" / f"{name}.csv")] == ids


@pytest.mark.parametrize("failure", ["modified", "missing", "extra", "membership"])
def test_package_rejects_hash_and_inventory_failures(built_package, tmp_path, failure):
    root = tmp_path / "copy"
    shutil.copytree(built_package, root)
    path = root / "cohorts/eligible.csv"
    if failure == "missing":
        path.unlink()
    elif failure == "extra":
        (root / "unexpected.txt").write_text("not in the allowlist")
    else:
        path.write_text(path.read_text().replace("1462,", "1463,"))
        if failure == "membership":
            manifest = package.read_json(root / "manifest.json")
            manifest["sha256"]["cohorts/eligible.csv"] = package.digest(path)
            package.write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="hash failure|inventory mismatch|membership mismatch"):
        package.verify(root)


def test_allowlist_rejects_raw_data(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / package.ALLOWLIST).write_text("data/fresh_validation/openml/records.parquet\n")
    with pytest.raises(ValueError, match="Raw data"):
        package.allowed_paths(tmp_path)


def test_extracted_archive_reproduces_and_builds_partial_assets(built_package, tmp_path):
    # A fresh interpreter imports code from the extracted archive only. The
    # reproduction command disables networking and contains no Git/raw-data read.
    command = [sys.executable, "-I", str(built_package / "scripts/validation_revision_package.py"),
               "reproduce", "--output", str(tmp_path / "replay")]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = package.read_json(tmp_path / "replay/verification.json")
    assert report["network_disabled"] and not report["raw_task_data_used"]
    assert all(row["all_seven_csvs_exact"] for row in report["cohorts"].values())
    generated = tmp_path / "assets"
    result = subprocess.run([sys.executable, "-I", str(built_package / "scripts/build_paper_assets.py"),
                             "--out", str(built_package / "analysis-output/v2"),
                             "--fresh-validation-only", "--generated", str(generated)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    numbers = (generated / "numbers.tex").read_text()
    assert r"\defres{fvc.c1.rankpos}{$+$0.37}" in numbers
    assert r"\defres{fvc.c1.rankci}{[$-$0.34, $+$1.22]}" in numbers
    table = (generated / "tables/fresh_validation_cohort_tasks.tex").read_text()
    assert r"}\label{tab:cohort_tasks}\\" in table
    assert "ta.primary" not in numbers


def test_partial_builder_preserves_full_asset_directory(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import build_paper_assets as builder
    monkeypatch.setattr(builder, "__file__", str(tmp_path / "scripts/build_paper_assets.py"))
    full = tmp_path / "paper/generated"
    full.mkdir(parents=True)
    (full / "numbers.tex").write_text("preserve this full build")
    before = package.digest(full / "numbers.tex")
    with pytest.raises(ValueError, match="separate"):
        builder.run(Namespace(fresh_validation_only=True, generated=full))
    assert package.digest(full / "numbers.tex") == before
    separate = tmp_path / "separate"
    separate.mkdir()
    (separate / "numbers.tex").write_text("a full build")
    with pytest.raises(ValueError, match="empty directory"):
        builder.run(Namespace(fresh_validation_only=True, generated=separate))


def test_offline_guard_rejects_network():
    import socket
    with package.offline(), pytest.raises(RuntimeError, match="Network access"):
        socket.create_connection(("example.com", 443))
