#!/usr/bin/env python3
"""Run local retrieval-only config sweeps.

This script is for fast iteration and report tables. It does not replace the
required RapidFire AI experimentation, but it produces the same released
retrieval metrics and logs the exact knobs tried.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Metrics"))

from project1_eval import f1_at_k, precision_at_k, recall_at_k, to_spans  # type: ignore
from src.rag import RAGPipeline


def load_configs(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_config(config: Dict[str, Any], validation: List[Dict[str, Any]], docs: str | None) -> Dict[str, Any]:
    pipeline = RAGPipeline(
        docs,
        chunk_lines=int(config.get("chunk_lines", 54)),
        overlap_lines=int(config.get("chunk_overlap", 14)),
    )
    top_k = int(config.get("top_k", 5))
    candidate_pool = int(config.get("candidate_pool", 36))
    context_token_budget = int(config.get("context_token_budget", 1700))

    rows = []
    for item in validation:
        _, chunks = pipeline.context_for_question(
            item["question"],
            top_k=top_k,
            candidate_pool=candidate_pool,
            token_budget=context_token_budget,
        )
        retrieved = to_spans([chunk.source() for chunk in chunks])
        gt = to_spans(item["source_evidence"])
        rows.append(
            {
                "question_id": int(item["question_id"]),
                "F1@5": f1_at_k(retrieved, gt, 5),
                "Precision@5": precision_at_k(retrieved, gt, 5),
                "Recall@5": recall_at_k(retrieved, gt, 5),
            }
        )

    n = len(rows) or 1
    mean_f1 = sum(r["F1@5"] for r in rows) / n
    mean_p = sum(r["Precision@5"] for r in rows) / n
    mean_r = sum(r["Recall@5"] for r in rows) / n
    return {
        "config": config,
        "summary": {
            "F1@5": mean_f1,
            "Precision@5": mean_p,
            "Recall@5": mean_r,
            "Retrieval Score": (mean_f1 + mean_p + mean_r) / 3.0,
        },
        "per_question": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default=str(PROJECT_ROOT / "validation-set-golden-qa-pairs.json"))
    parser.add_argument("--configs", default=str(ROOT / "configs" / "experiment_configs.json"))
    parser.add_argument("--docs", default=None)
    parser.add_argument("--output", default=str(ROOT / "logs" / "retrieval_experiments.json"))
    args = parser.parse_args()

    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    configs = load_configs(Path(args.configs))
    results = [evaluate_config(config, validation, args.docs) for config in configs]
    results.sort(key=lambda r: r["summary"]["Retrieval Score"], reverse=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for rank, result in enumerate(results, start=1):
        summary = result["summary"]
        print(
            f"{rank}. {result['config'].get('name')} "
            f"score={summary['Retrieval Score']:.4f} "
            f"F1={summary['F1@5']:.4f} P={summary['Precision@5']:.4f} R={summary['Recall@5']:.4f}"
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
