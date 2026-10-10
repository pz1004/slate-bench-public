"""Recompress the nested runs' per-unit score files losslessly with zstd.

Each nested run writes ``unit_scores.parquet`` (one row per split, unit and method) at
over 200 MB. Rewritten with zstd at level 19 it takes about 40 MB, which keeps every file
within GitHub's limits. A file is replaced only after the rewritten table is checked to
equal the original, schema metadata included. Files already stored with zstd are skipped,
so the script can be rerun after any nested run.

    .venv/bin/python scripts/compress_unit_scores.py                 # every published run under analysis-output/v2
    .venv/bin/python scripts/compress_unit_scores.py path/to/unit_scores.parquet
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

NAME = "unit_scores.parquet"
LEVEL = 19
# Resume caches and smoke runs are not published.
SKIP_DIRS = {"split_parts", "smoke"}


def is_zstd(path: Path) -> bool:
    metadata = pq.ParquetFile(path).metadata
    return all(
        metadata.row_group(group).column(column).compression == "ZSTD"
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def compress(path: Path) -> tuple[Path, int, int] | None:
    """Rewrite one file with zstd; return its size before and after, or None if skipped."""
    if is_zstd(path):
        return None
    table = pq.read_table(path)
    target = path.with_name(f"{path.name}.zstd.tmp")
    pq.write_table(table, target, compression="zstd", compression_level=LEVEL)
    if not pq.read_table(target).equals(table, check_metadata=True):
        target.unlink()
        raise RuntimeError(f"Recompressed table differs from its source: {path}")
    before = path.stat().st_size
    os.replace(target, path)
    return path, before, path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help=f"files to recompress (default: every {NAME} under --out)")
    parser.add_argument("--out", type=Path, default=Path("analysis-output/v2"))
    parser.add_argument("--jobs", type=int, default=6, help="files recompressed in parallel")
    args = parser.parse_args()
    paths = args.paths or sorted(path for path in args.out.rglob(NAME) if not SKIP_DIRS & set(path.parts))
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(paths)))) as pool:
        for path, result in zip(paths, pool.map(compress, paths)):
            if result is None:
                print(f"  already zstd: {path}")
            else:
                _, before, after = result
                print(f"  zstd {path}: {before / 2**20:.1f} MB -> {after / 2**20:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
