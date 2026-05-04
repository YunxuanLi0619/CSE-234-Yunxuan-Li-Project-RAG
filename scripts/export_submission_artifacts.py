#!/usr/bin/env python3
"""Export small submission convenience artifacts from existing run outputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def _metric(entry: dict, key: str):
    value = entry.get(key, {})
    if isinstance(value, dict):
        return value.get("value", "")
    return value


def export_rapidfire_csvs() -> None:
    results_path = LOGS / "rapidfire_project1_results.json"
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

    fieldnames = list(rows[0].keys())
    for filename in ("rapidfire_project1_metrics.csv", "rapidfire_project1_runs_info.csv"):
        with (LOGS / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def copy_validation_output() -> None:
    shutil.copy2(LOGS / "validation_output.json", ROOT / "validation_output.json")


def main() -> None:
    export_rapidfire_csvs()
    copy_validation_output()
    print("Exported RapidFire CSVs and top-level validation_output.json")


if __name__ == "__main__":
    main()
