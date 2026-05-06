#!/usr/bin/env python3
"""Final Project 1 CLI.

Usage:
    python3 main.py --input input.json --output output.json

The input JSON may be the released validation set with extra keys or the hidden
test-set schema containing only question_id and question. The output always
matches the project-required schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.llm_client import LLMConfig, discover_api_key, discover_base_url, generate_answer, get_config_value
from src.rag import RAGPipeline


def load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("questions", "data", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects")

    questions: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed question entry: {item!r}")
        if "question_id" not in item or "question" not in item:
            raise ValueError(f"Each entry must contain question_id and question: {item!r}")
        questions.append({"question_id": int(item["question_id"]), "question": str(item["question"])})
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Line-aware RAG QA over RapidFire AI docs")
    parser.add_argument("--input", required=True, help="Input JSON test set")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--docs", default=None, help="Path to sourcedocs directory; auto-detected by default")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum retrieved chunks/sources per question")
    parser.add_argument("--candidate-pool", type=int, default=36, help="BM25 candidates considered before diversity filtering")
    parser.add_argument("--chunk-lines", type=int, default=54, help="Line-window chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=24, help="Line overlap between adjacent chunks")
    parser.add_argument("--context-token-budget", type=int, default=1700, help="Budget for serialized retrieved_context")
    parser.add_argument("--model", default=None, help="Generator model; defaults to env GENERATOR_MODEL/TRITONAI_MODEL/OPENAI_MODEL/api-llama-4-scout")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; TritonAI auto-detected when ~/api-key.txt exists")
    parser.add_argument("--api-key", default=None, help="API key; defaults to env or ~/api-key.txt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=90, help="Per-request API timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="API retry count before using fallback or failing")
    parser.add_argument("--fail-on-llm-error", action="store_true", help="Stop instead of writing fallback answers when generation API fails")
    parser.add_argument("--no-llm", action="store_true", help="Disable API calls and use extractive fallback")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    questions = load_questions(input_path)
    pipeline = RAGPipeline(
        args.docs,
        chunk_lines=args.chunk_lines,
        overlap_lines=args.chunk_overlap,
    )

    model = args.model
    if model is None:
        import os

        model = get_config_value("GENERATOR_MODEL", "TRITONAI_MODEL", "OPENAI_MODEL") or "api-llama-4-scout"

    llm_config = LLMConfig(
        model=model,
        api_key=discover_api_key(args.api_key),
        base_url=discover_base_url(args.base_url),
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.request_timeout,
        retries=args.retries,
        disabled=args.no_llm,
        fail_on_error=args.fail_on_llm_error,
    )

    outputs: List[Dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        qid = item["question_id"]
        question = item["question"]
        context, used_chunks = pipeline.context_for_question(
            question,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            token_budget=args.context_token_budget,
        )
        answer = generate_answer(question, context, llm_config)
        outputs.append(
            {
                "question_id": qid,
                "answer": answer,
                "retrieved_context": context,
                "sources": [chunk.source() for chunk in used_chunks],
            }
        )
        if not args.quiet:
            print(f"[{idx}/{len(questions)}] answered question_id={qid}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"Wrote {len(outputs)} answers to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
