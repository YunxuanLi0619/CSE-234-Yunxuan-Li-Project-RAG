#!/usr/bin/env python3
"""Run the class-required RapidFire AI multi-config RAG experiments.

This script is separate from `main.py` on purpose:

* `main.py` is the deterministic final pipeline the graders execute.
* this file is the RapidFire AI experimentation workflow required for
  Learning Outcomes 4/5 and the "Experimentation and Evaluation" component.

It uses RapidFire AI OSS `Experiment.run_evals()` with a set of explicit RAG
configs that vary chunking, embedding model, search strategy, reranking, and
prompt style. The resulting RapidFire logs/metrics should be committed under
`logs/` and summarized in the project report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
LOG_DIR = ROOT / "logs"


def _load_dotenv() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_env_value(*names: str) -> str | None:
    dotenv = _load_dotenv()
    for name in names:
        value = os.environ.get(name) or dotenv.get(name)
        if value:
            return value
    key_file = Path.home() / "api-key.txt"
    if "OPENAI_API_KEY" in names and key_file.exists():
        text = key_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return None


def require_experiment_deps() -> Dict[str, Any]:
    try:
        from rapidfireai import Experiment
        from rapidfireai.automl import RFGridSearch, RFLangChainRagSpec, RFOpenAIAPIModelConfig
        from datasets import Dataset
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise SystemExit(
            "Missing RapidFire/LangChain experimentation dependencies.\n"
            "On DataHub run:\n"
            "  pip install rapidfireai datasets langchain langchain-community "
            "langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "Experiment": Experiment,
        "RFGridSearch": RFGridSearch,
        "RFLangChainRagSpec": RFLangChainRagSpec,
        "RFOpenAIAPIModelConfig": RFOpenAIAPIModelConfig,
        "Dataset": Dataset,
        "DirectoryLoader": DirectoryLoader,
        "TextLoader": TextLoader,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    }


def compact_tokens(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", text)]


def unique_sources_from_evidence(evidence: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted({str(item["file"]) for item in evidence if "file" in item})


def load_dataset(validation_path: Path, limit: int | None, dataset_cls: Any) -> Any:
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    if limit:
        raw = raw[:limit]
    data = {
        "question_id": [int(x["question_id"]) for x in raw],
        "question": [str(x["question"]) for x in raw],
        "reference_answer": [str(x.get("reference_answer", "")) for x in raw],
        "source_files_json": [json.dumps(unique_sources_from_evidence(x.get("source_evidence", []))) for x in raw],
        "source_evidence_json": [json.dumps(x.get("source_evidence", [])) for x in raw],
    }
    return dataset_cls.from_dict(data)


def preprocess_concise(batch: Dict[str, list], rag: Any, prompt_manager: Any) -> Dict[str, list]:
    all_context = rag.get_context(batch_queries=batch["question"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    retrieved_sources = [
        [Path(str(doc.metadata.get("source", doc.metadata.get("file", "")))).name for doc in docs]
        for docs in all_context
    ]
    prompts = [
        [
            {
                "role": "system",
                "content": (
                    "Answer technical documentation questions using only the retrieved context. "
                    "Be precise and concise."
                ),
            },
            {
                "role": "user",
                "content": f"Retrieved context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
            },
        ]
        for question, context in zip(batch["question"], serialized_context)
    ]
    return {
        **batch,
        "prompts": prompts,
        "retrieved_sources": retrieved_sources,
        "retrieved_context": serialized_context,
    }


def preprocess_structured(batch: Dict[str, list], rag: Any, prompt_manager: Any) -> Dict[str, list]:
    all_context = rag.get_context(batch_queries=batch["question"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    retrieved_sources = [
        [Path(str(doc.metadata.get("source", doc.metadata.get("file", "")))).name for doc in docs]
        for docs in all_context
    ]
    prompts = [
        [
            {
                "role": "system",
                "content": (
                    "You are a documentation QA assistant. Use only the provided context. "
                    "Cover relevant conditions, parameter names, defaults, and distinctions. "
                    "Do not invent unsupported details."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{question}\n\n"
                    "Write a complete but compact answer grounded in the context."
                ),
            },
        ]
        for question, context in zip(batch["question"], serialized_context)
    ]
    return {
        **batch,
        "prompts": prompts,
        "retrieved_sources": retrieved_sources,
        "retrieved_context": serialized_context,
    }


def postprocess_outputs(batch: Dict[str, list]) -> Dict[str, list]:
    batch["expected_sources"] = [json.loads(x) for x in batch["source_files_json"]]
    return batch


def compute_metrics(batch: Dict[str, list]) -> Dict[str, Dict[str, Any]]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    source_hits = 0
    answer_recalls: List[float] = []

    generated = batch.get("generated_text", [""] * len(batch["question"]))
    for retrieved, expected, answer, reference in zip(
        batch["retrieved_sources"],
        batch["expected_sources"],
        generated,
        batch["reference_answer"],
    ):
        retrieved_unique = []
        seen = set()
        for src in retrieved:
            if src and src not in seen:
                retrieved_unique.append(src)
                seen.add(src)

        pred_set = set(retrieved_unique[:5])
        expected_set = set(expected)
        overlap = len(pred_set & expected_set)
        precision = overlap / len(pred_set) if pred_set else 0.0
        recall = overlap / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        source_hits += int(overlap > 0)

        answer_tokens = set(compact_tokens(str(answer)))
        reference_tokens = {tok for tok in compact_tokens(str(reference)) if len(tok) > 2}
        answer_recalls.append(len(answer_tokens & reference_tokens) / len(reference_tokens) if reference_tokens else 0.0)

    total = len(batch["question"]) or 1
    return {
        "Total": {"value": total},
        "Source Precision@5": {"value": sum(precisions) / total},
        "Source Recall@5": {"value": sum(recalls) / total},
        "Source F1@5": {"value": sum(f1s) / total},
        "Source Hit Rate": {"value": source_hits / total},
        "Reference Token Recall": {"value": sum(answer_recalls) / total},
    }


def accumulate_metrics(aggregated_metrics: Dict[str, list]) -> Dict[str, Dict[str, Any]]:
    totals = [m["value"] for m in aggregated_metrics["Total"]]
    total = sum(totals) or 1
    algebraic = [
        "Source Precision@5",
        "Source Recall@5",
        "Source F1@5",
        "Source Hit Rate",
        "Reference Token Recall",
    ]
    out: Dict[str, Dict[str, Any]] = {"Total": {"value": total, "is_distributive": True}}
    for metric in algebraic:
        out[metric] = {
            "value": sum(m["value"] * n for m, n in zip(aggregated_metrics[metric], totals)) / total,
            "is_algebraic": True,
            "value_range": (0, 1),
        }
    return out


def load_configs(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    configs = json.loads(path.read_text(encoding="utf-8"))
    return configs[:limit] if limit else configs


def document_template(doc: Any) -> str:
    source = Path(str(doc.metadata.get("source", ""))).name
    return f"[Source: {source}]\n{doc.page_content}"


def make_rag(cfg: Dict[str, Any], deps: Dict[str, Any], *, include_reranker: bool) -> Any:
    RFLangChainRagSpec = deps["RFLangChainRagSpec"]
    DirectoryLoader = deps["DirectoryLoader"]
    TextLoader = deps["TextLoader"]
    RecursiveCharacterTextSplitter = deps["RecursiveCharacterTextSplitter"]
    HuggingFaceEmbeddings = deps["HuggingFaceEmbeddings"]

    search_cfg = {
        "type": cfg["search_type"],
        "k": int(cfg["top_k"]),
    }
    if cfg["search_type"] == "mmr":
        search_cfg.update(
            {
                "fetch_k": int(cfg.get("fetch_k", 24)),
                "lambda_mult": float(cfg.get("lambda_mult", 0.4)),
            }
        )

    reranker_cfg = None
    if include_reranker and cfg.get("reranker") == "cross_encoder":
        try:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        except ImportError as exc:
            raise SystemExit(
                "Cross-encoder reranker config was requested but langchain_classic is unavailable. "
                "Install the RapidFire eval dependencies or rerun with --skip-rerankers."
            ) from exc
        reranker_cfg = {
            "class": CrossEncoderReranker,
            "model_name": cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2"),
            "model_kwargs": {"device": "cpu"},
            "top_n": int(cfg.get("reranker_top_n", 5)),
        }

    return RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(ROOT / "sourcedocs"),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=int(cfg["chunk_size"]),
            chunk_overlap=int(cfg["chunk_overlap"]),
        ),
        embedding_cfg={
            "class": HuggingFaceEmbeddings,
            "model_name": cfg["embedding_model"],
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
        },
        vector_store_cfg={"type": "faiss"},
        search_cfg=search_cfg,
        reranker_cfg=reranker_cfg,
        enable_gpu_search=False,
        document_template=document_template,
    )


def make_config(
    cfg: Dict[str, Any],
    deps: Dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    include_reranker: bool,
    rpm_limit: int,
    tpm_limit: int,
) -> Dict[str, Any]:
    RFOpenAIAPIModelConfig = deps["RFOpenAIAPIModelConfig"]
    rag = make_rag(cfg, deps, include_reranker=include_reranker)
    preprocess_fn: Callable[[Dict[str, list], Any, Any], Dict[str, list]]
    preprocess_fn = preprocess_structured if cfg.get("prompt_style") == "structured" else preprocess_concise
    generator = RFOpenAIAPIModelConfig(
        client_config={
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 4,
            "timeout": 90,
        },
        model_config={
            "model": cfg.get("generator_model", "api-llama-4-scout"),
            "temperature": float(cfg.get("temperature", 0.0)),
            "max_tokens": int(cfg.get("max_tokens", 300)),
        },
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        rag=rag,
        prompt_manager=None,
    )
    return {
        "name": cfg["name"],
        "openai_config": generator,
        "batch_size": 16,
        "preprocess_fn": preprocess_fn,
        "postprocess_fn": postprocess_outputs,
        "compute_metrics_fn": compute_metrics,
        "accumulate_metrics_fn": accumulate_metrics,
        "online_strategy_kwargs": {
            "strategy_name": "normal",
            "confidence_level": 0.95,
            "use_fpc": True,
        },
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def create_experiment(experiment_cls: Any, name: str, path: Path) -> Any:
    try:
        return experiment_cls(experiment_name=name, mode="evals", experiment_path=str(path))
    except TypeError:
        return experiment_cls(experiment_name=name, mode="eval", experiments_path=str(path))


def run_evals(experiment: Any, config_group: Any, dataset: Any, args: argparse.Namespace) -> Any:
    kwargs = {"dataset": dataset, "num_shards": args.num_shards, "num_actors": args.num_actors, "seed": args.seed}
    try:
        return experiment.run_evals(config_group=config_group, **kwargs)
    except TypeError:
        return experiment.run_evals(configs=config_group, **kwargs)


def write_plan(configs: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plan = {
        "purpose": "RapidFire AI OSS multi-config experiments for Project 1 Learning Outcomes 4/5.",
        "official_workflow": [
            "rapidfireai init --evals",
            "rapidfireai start",
            "python3 scripts/run_rapidfire_experiments.py",
        ],
        "num_configs": len(configs),
        "configs": configs,
        "notes": [
            "These configs vary chunking, embedding model, retrieval search type, reranking, and prompt style.",
            "The final main.py remains deterministic; RapidFire artifacts are for methodology, report, and interview evidence.",
        ],
    }
    (LOG_DIR / "rapidfire_experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote {LOG_DIR / 'rapidfire_experiment_plan.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RapidFire AI multi-config RAG experiments for Project 1")
    parser.add_argument("--validation", default=str(PROJECT_ROOT / "validation-set-golden-qa-pairs.json"))
    parser.add_argument("--configs", default=str(ROOT / "configs" / "rapidfire_experiment_configs.json"))
    parser.add_argument("--experiment-name", default="project1-rapidfire-rag-contexteng")
    parser.add_argument("--experiment-path", default=str(ROOT / "experiments" / "rapidfire_project1"))
    parser.add_argument("--validation-limit", type=int, default=24, help="Use a smaller subset while iterating to control API cost")
    parser.add_argument("--config-limit", type=int, default=None, help="Run only the first N configs for a smoke test")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--rpm-limit", type=int, default=80)
    parser.add_argument("--tpm-limit", type=int, default=500000)
    parser.add_argument("--skip-rerankers", action="store_true", help="Disable cross-encoder reranker configs if dependency/runtime is too heavy")
    parser.add_argument("--dry-run", action="store_true", help="Only write the experiment plan; do not import RapidFire or call APIs")
    parser.add_argument("--keep-open", action="store_true", help="Do not call experiment.end() after run_evals")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = load_configs(Path(args.configs), args.config_limit)
    write_plan(configs, args)

    if args.dry_run:
        for i, cfg in enumerate(configs, start=1):
            print(
                f"{i}. {cfg['name']} | chunk={cfg['chunk_size']}/{cfg['chunk_overlap']} "
                f"embed={cfg['embedding_model']} search={cfg['search_type']} "
                f"rerank={cfg['reranker']} prompt={cfg['prompt_style']}"
            )
        return 0

    api_key = args.api_key or get_env_value("OPENAI_API_KEY", "GENERATOR_API_KEY", "TRITONAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set OPENAI_API_KEY or create .env with OPENAI_API_KEY=...")
    base_url = (args.base_url or get_env_value("GENERATOR_BASE_URL", "OPENAI_BASE_URL") or "https://tritonai-api.ucsd.edu").rstrip("/")

    deps = require_experiment_deps()
    dataset = load_dataset(Path(args.validation), args.validation_limit, deps["Dataset"])
    rf_configs = [
        make_config(
            cfg,
            deps,
            api_key=api_key,
            base_url=base_url,
            include_reranker=not args.skip_rerankers,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
        )
        for cfg in configs
    ]
    try:
        config_group = deps["RFGridSearch"](configs=rf_configs)
    except TypeError:
        config_group = deps["RFGridSearch"](rf_configs)
    experiment = create_experiment(deps["Experiment"], args.experiment_name, Path(args.experiment_path))

    try:
        results = run_evals(experiment, config_group, dataset, args)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "rapidfire_project1_results.json").write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")

        try:
            runs_info = experiment.get_runs_info()
            runs_info.to_csv(LOG_DIR / "rapidfire_project1_runs_info.csv", index=False)
        except Exception as exc:
            print(f"[warn] could not save get_runs_info(): {exc}", file=sys.stderr)

        try:
            results_df = experiment.get_results()
            results_df.to_csv(LOG_DIR / "rapidfire_project1_metrics.csv", index=False)
        except Exception as exc:
            print(f"[warn] could not save get_results(): {exc}", file=sys.stderr)

        try:
            log_file = experiment.get_log_file_path()
            if log_file and Path(log_file).exists():
                shutil.copy2(log_file, LOG_DIR / "rapidfire_project1_rapidfire.log")
        except Exception as exc:
            print(f"[warn] could not copy RapidFire log file: {exc}", file=sys.stderr)
    finally:
        if not args.keep_open:
            try:
                experiment.end()
            except Exception as exc:
                print(f"[warn] could not end experiment cleanly: {exc}", file=sys.stderr)

    print(f"Wrote RapidFire artifacts under {LOG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
