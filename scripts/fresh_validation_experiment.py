"""Run the fresh-validation experiment (docs/fresh_validation_protocol.md).

Every (task, repetition, learner family) is one job and is written as its own part, so an
interrupted run resumes where it stopped. The task list gives each task's parquet file,
target and the attributes to drop; the configuration file fixes everything else.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark import fresh_validation as fv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_tasks(path: Path) -> list[fv.Task]:
    frame = pd.read_csv(path, comment="#", dtype=str).fillna("")
    if "use" in frame.columns:
        frame = frame[frame["use"].str.lower().isin(["yes", "true", "1"])]
    tasks = []
    for _, row in frame.iterrows():
        drop = tuple(item for item in str(row.get("drop", "")).split(";") if item)
        tasks.append(
            fv.Task(
                dataset_id=int(row["dataset_id"]),
                name=str(row["name"]),
                path=(ROOT / str(row["path"])).resolve(),
                target=str(row["target"]),
                drop=drop,
            )
        )
    return tasks


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    task: fv.Task = job["task"]
    config: dict[str, Any] = job["config"]
    part = Path(job["part"])
    started = time.time()
    master = int(config["seed"])
    features, y, classes = fv.load_task(task)
    rows = fv.capped_rows(y, int(config["row_cap"]), fv.derived_seed(master, task.dataset_id, 0))
    features = features.iloc[rows].reset_index(drop=True)
    y = y[rows]
    parts = fv.four_way_split(y, dict(config["splits"]), fv.derived_seed(master, task.dataset_id, 1, job["repetition"]))
    data = fv.encode(features, parts, max_onehot=int(config["max_onehot_categories"]))
    labels = {name: y[index] for name, index in parts.items()}
    configurations = job["configurations"]
    seeds = [
        fv.derived_seed(master, task.dataset_id, 2, job["repetition"], job["family_index"], index)
        for index in range(len(configurations))
    ]
    config_frame, candidates = fv.family_candidates(
        job["family"],
        configurations,
        data,
        labels,
        len(classes),
        seeds=seeds,
        schedule=dict(config["schedule"]),
        ensemble_steps=int(config["ensemble_steps"]),
    )
    context = {
        "dataset_id": task.dataset_id,
        "dataset": task.name,
        "repetition": int(job["repetition"]),
        "n_rows": int(len(y)),
        "n_features": int(features.shape[1]),
        "n_classes": int(len(classes)),
        **{f"n_{name}": int(len(index)) for name, index in parts.items()},
    }
    part.parent.mkdir(parents=True, exist_ok=True)
    # Candidates last: a part counts as done once its candidates file exists.
    for kind, frame in [("configurations", config_frame), ("candidates", candidates)]:
        frame = frame.assign(**context)[[*context, *frame.columns]]
        target = part.with_name(f"{part.name}_{kind}.parquet")
        temporary = target.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    return {"part": part.name, "seconds": time.time() - started, "failed_configurations": int(config_frame["failed"].sum())}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort outside a repository
        return ""


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = read_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: int(args.limit)]
    repetitions = int(args.repetitions or config["repetitions"])
    families = list(config["families"])
    master = int(config["seed"])
    configurations = {
        family: fv.family_configurations(
            dict(config["families"][family]),
            int(config["n_random_configurations"]),
            fv.derived_seed(master, 3, index),
        )
        for index, family in enumerate(families)
    }
    output_dir = Path(args.output_dir)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configurations_searched.json").write_text(
        json.dumps({family: configs for family, configs in configurations.items()}, indent=2, default=str), encoding="utf-8"
    )
    jobs = []
    for task in tasks:
        for repetition in range(repetitions):
            for index, family in enumerate(families):
                part = parts_dir / f"{task.dataset_id}_r{repetition}_{family}"
                if args.resume and part.with_name(f"{part.name}_candidates.parquet").exists():
                    continue
                jobs.append(
                    {
                        "task": task,
                        "config": config,
                        "repetition": repetition,
                        "family": family,
                        "family_index": index,
                        "configurations": configurations[family],
                        "part": str(part),
                    }
                )
    # Longest families first, so the pool does not end on one slow job.
    weight = {family: int(config["families"][family].get("cost_weight", 1)) for family in families}
    jobs.sort(key=lambda job: -weight[job["family"]])
    started = time.time()
    log = []
    print(f"{len(jobs)} jobs on {args.jobs} workers", flush=True)
    with multiprocessing.get_context("fork").Pool(int(args.jobs), maxtasksperchild=20) as pool:
        for done, result in enumerate(pool.imap_unordered(run_job, jobs), start=1):
            log.append(result)
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} parts, {time.time() - started:.0f} s", flush=True)

    if log:
        jobs_path = output_dir / "jobs.csv"
        previous = pd.read_csv(jobs_path) if jobs_path.exists() else pd.DataFrame()
        pd.concat([previous, pd.DataFrame(log)], ignore_index=True).to_csv(jobs_path, index=False)
    candidate_parts = sorted(parts_dir.glob("*_candidates.parquet"))
    config_parts = sorted(parts_dir.glob("*_configurations.parquet"))
    candidates = pd.concat([pd.read_parquet(path) for path in candidate_parts], ignore_index=True)
    searched = pd.concat([pd.read_parquet(path) for path in config_parts], ignore_index=True)
    candidates.to_parquet(output_dir / "candidates.parquet", index=False)
    searched.to_parquet(output_dir / "configurations.parquet", index=False)
    hashes = {
        "config": fv.file_sha256(config_path),
        "tasks": fv.file_sha256(Path(args.tasks)),
        "module": fv.file_sha256(ROOT / "src" / "tabular_sota_benchmark" / "fresh_validation.py"),
        "runner": fv.file_sha256(Path(__file__)),
    }
    if args.protocol:
        hashes["protocol"] = fv.file_sha256(Path(args.protocol))
    metadata = {
        "config": str(config_path),
        "tasks": str(args.tasks),
        "protocol": str(args.protocol or ""),
        "sha256": hashes,
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short", "--untracked-files=no"),
        "repetitions": repetitions,
        "n_tasks": len(tasks),
        "families": families,
        "n_parts": len(candidate_parts),
        "wall_seconds_this_invocation": time.time() - started,
        "job_seconds_this_invocation": float(sum(item["seconds"] for item in log)),
        "failed_configurations": int(searched["failed"].sum()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir}: {len(candidates)} candidate rows, {len(searched)} configuration rows", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/fresh_validation_experiment.yaml"))
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="the protocol whose hash the run records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, help="override the configured number (pilot only)")
    parser.add_argument("--limit", type=int, help="use only the first tasks of the list (pilot only)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
