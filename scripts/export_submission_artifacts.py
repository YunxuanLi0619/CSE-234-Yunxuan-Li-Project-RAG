#!/usr/bin/env python3
"""Export Project 1 artifacts in the Gradescope submission format.

The latest guideline expects a single outer archive named `team_<team_id>.zip`
with exact top-level filenames:

    main.py
    output.json
    report.pdf
    your_golden_qa.json
    requirements.txt
    repo.zip

The outer archive also includes `src/*.py` as helper modules because `main.py`
imports them directly. The inner `repo.zip` is a scrubbed snapshot of this repo:
no `.git`, `.env`, pycache, `.DS_Store`, or nested zip archives.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
DEFAULT_TEAM_ID = 40


def _metric(entry: dict, key: str):
    value = entry.get(key, {})
    if isinstance(value, dict):
        return value.get("value", "")
    return value


def export_rapidfire_csvs() -> None:
    results_path = LOGS / "rapidfire_project1_results.json"
    if not results_path.exists():
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))

    rows = []
    for key, value in sorted(data.items(), key=lambda kv: int(kv[0])):
        metrics = value[1]
        embedding = _metric(metrics, "embedding_cfg") or {}
        search = _metric(metrics, "search_cfg") or {}
        reranker = _metric(metrics, "reranker_cfg") or {}
        model_config = _metric(metrics, "model_config") or {}
        rows.append(
            {
                "run_key": key,
                "run_id": _metric(metrics, "run_id"),
                "model_name": _metric(metrics, "model_name"),
                "chunk_size": _metric(metrics, "chunk_size"),
                "chunk_overlap": _metric(metrics, "chunk_overlap"),
                "embedding_model": embedding.get("model_name", ""),
                "vector_store": (_metric(metrics, "vector_store_cfg") or {}).get("type", ""),
                "search_type": search.get("type", ""),
                "search_k": search.get("k", ""),
                "fetch_k": search.get("fetch_k", ""),
                "lambda_mult": search.get("lambda_mult", ""),
                "reranker_model": reranker.get("model_name", "none") if reranker else "none",
                "reranker_top_n": reranker.get("top_n", "") if reranker else "",
                "temperature": model_config.get("temperature", ""),
                "max_tokens": model_config.get("max_tokens", ""),
                "samples_processed": _metric(metrics, "Samples Processed"),
                "processing_time": _metric(metrics, "Processing Time"),
                "source_precision_at_5": _metric(metrics, "Source Precision@5"),
                "source_recall_at_5": _metric(metrics, "Source Recall@5"),
                "source_f1_at_5": _metric(metrics, "Source F1@5"),
                "source_hit_rate": _metric(metrics, "Source Hit Rate"),
                "reference_token_recall": _metric(metrics, "Reference Token Recall"),
            }
        )

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for filename in ("rapidfire_project1_metrics.csv", "rapidfire_project1_runs_info.csv"):
        with (LOGS / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def copy_required_artifacts() -> None:
    copies = [(LOGS / "validation_output.json", ROOT / "output.json")]
    for src, dst in copies:
        if not src.exists():
            raise FileNotFoundError(f"Required artifact missing: {src}")
        shutil.copy2(src, dst)

    for required in (ROOT / "report.pdf", ROOT / "your_golden_qa.json"):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")


def should_skip_repo_path(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}:
        return True
    if path.name in {".DS_Store", ".env", "api-key.txt", "repo.zip"}:
        return True
    if path.name.startswith("team_") and path.suffix == ".zip":
        return True
    if path.suffix in {".pyc", ".pyo", ".zip"}:
        return True
    return False


def iter_repo_files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not should_skip_repo_path(path):
            yield path


def add_file(zf: zipfile.ZipFile, path: Path, arcname: str | None = None) -> None:
    archive_name = arcname or path.relative_to(ROOT).as_posix()
    zf.write(path, archive_name)


def build_repo_zip(destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in iter_repo_files():
            add_file(zf, path)


def build_outer_zip(team_id: int, repo_zip: Path) -> Path:
    outer_zip = ROOT / f"team_{team_id}.zip"
    required = [
        ROOT / "main.py",
        ROOT / "output.json",
        ROOT / "report.pdf",
        ROOT / "your_golden_qa.json",
        ROOT / "requirements.txt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Cannot package missing required file: {path}")

    with zipfile.ZipFile(outer_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in required:
            add_file(zf, path, path.name)
        for path in sorted((ROOT / "src").glob("*.py")):
            add_file(zf, path)
        zf.write(repo_zip, "repo.zip")
    return outer_zip


def verify_outer_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    required = {"main.py", "output.json", "report.pdf", "your_golden_qa.json", "requirements.txt", "repo.zip"}
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"Outer zip missing required entries: {missing}")
    nested_zips = [name for name in names if name.endswith(".zip")]
    if nested_zips != ["repo.zip"]:
        raise RuntimeError(f"Outer zip must contain exactly one nested repo.zip, got {nested_zips}")
    if any("\\" in name for name in names):
        raise RuntimeError("Outer zip contains a Windows-style backslash path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team_<team_id>.zip for Project 1")
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    args = parser.parse_args()

    export_rapidfire_csvs()
    copy_required_artifacts()

    tmp_dir = Path("/private/tmp") / f"project1_team_{args.team_id}_submission"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    repo_zip = tmp_dir / "repo.zip"
    build_repo_zip(repo_zip)
    outer_zip = build_outer_zip(args.team_id, repo_zip)
    verify_outer_zip(outer_zip)
    print(f"Wrote {outer_zip}")


if __name__ == "__main__":
    main()
