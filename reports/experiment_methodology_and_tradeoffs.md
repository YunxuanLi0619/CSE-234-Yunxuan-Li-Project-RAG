# Project 1 Experimentation Methodology and Engineering Tradeoffs

This note completes the project requirements tied to:

- Learning Outcome 4: use RapidFire AI OSS for systematic multi-config experiments.
- Learning Outcome 5: develop engineering judgment about retrieval/context tradeoffs.
- High-Level Description Component 3: experimentation and evaluation.

## What the Project Requires

The project statement requires using RapidFire AI OSS to compare multiple RAG configurations,
not just hand-tune one final pipeline. The final `main.py` remains a deterministic CLI for grading,
while RapidFire AI is used to produce experiment logs, metrics, and evidence of systematic
configuration search.

The RapidFire documentation supports this workflow through:

- `Experiment(mode="evals")` and `run_evals()` for multi-config evaluation.
- `RFLangChainRagSpec` for data loading, chunking, embedding, indexing, retrieval, reranking,
  and context serialization.
- `RFGridSearch` / `List` style config groups for running multiple RAG settings side by side.
- User-provided `preprocess_fn`, `postprocess_fn`, `compute_metrics_fn`, and
  `accumulate_metrics_fn` for task-specific evaluation.

Relevant references:

- https://oss-docs.rapidfire.ai/en/latest/walkthroughrag.html
- https://oss-docs.rapidfire.ai/en/latest/ragcontexteng.html
- https://oss-docs.rapidfire.ai/en/latest/experiment.html
- https://oss-docs.rapidfire.ai/en/latest/configs.html

## RapidFire AI Experiment Plan

The executable RapidFire workflow is in:

```text
scripts/run_rapidfire_experiments.py
```

The 8 planned RapidFire configurations are in:

```text
configs/rapidfire_experiment_configs.json
```

The configs vary the major knobs required by the project:

| Family | Knobs varied |
|---|---|
| Chunking | `chunk_size`, `chunk_overlap` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` |
| Retrieval | similarity search vs. MMR, `top_k`, `fetch_k`, `lambda_mult` |
| Reranking | no reranker vs. cross-encoder reranker |
| Prompting | concise prompt vs. structured grounded-answer prompt |
| Generation | TritonAI `api-llama-4-scout`, temperature 0.0, max output 300 |

Run plan:

```bash
cd "/home/yul386/Project1 2/project1_solution"

export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2-)"

rapidfireai init --evals
rapidfireai start

python3 scripts/run_rapidfire_experiments.py \
  --validation ../validation-set-golden-qa-pairs.json \
  --validation-limit 24 \
  --num-shards 4 \
  --num-actors 4
```

During cost-controlled iteration, use `--validation-limit 24`. For the final report run, remove
that flag to run on the full released validation set. If cross-encoder setup is too slow on CPU,
run a smoke test with:

```bash
python3 scripts/run_rapidfire_experiments.py --dry-run
python3 scripts/run_rapidfire_experiments.py --config-limit 2 --validation-limit 6 --skip-rerankers
```

RapidFire artifacts expected under `logs/`:

- `rapidfire_experiment_plan.json`
- `rapidfire_project1_results.json`
- `rapidfire_project1_runs_info.csv`
- `rapidfire_project1_metrics.csv`
- `rapidfire_project1_rapidfire.log`

## Local Mirror Results Used to Choose the Final Pipeline

Before the full RapidFire run, I ran a fast local mirror sweep over 8 BM25 configurations using
the same released validation set and line-span evaluator. This produced:

| Rank | Config | F1@5 | Precision@5 | Recall@5 | Retrieval Score |
|---:|---|---:|---:|---:|---:|
| 1 | bm25_lines54_overlap24_top5 | 0.7292 | 0.6315 | 0.9611 | 0.7739 |
| 2 | bm25_lines96_overlap24_top5 | 0.7324 | 0.6667 | 0.8944 | 0.7645 |
| 3 | bm25_lines80_overlap20_top5 | 0.6953 | 0.6074 | 0.8944 | 0.7324 |
| 4 | bm25_lines66_overlap18_top5 | 0.6746 | 0.5796 | 0.9167 | 0.7236 |
| 5 | bm25_lines54_overlap14_top4 | 0.6568 | 0.5444 | 0.9278 | 0.7097 |
| 6 | bm25_lines54_overlap14_top5 | 0.6531 | 0.5407 | 0.9278 | 0.7072 |
| 7 | bm25_lines42_overlap10_top5 | 0.6449 | 0.5304 | 0.9389 | 0.7047 |
| 8 | bm25_lines36_overlap12_top5 | 0.6330 | 0.5037 | 0.9667 | 0.7011 |

Final validation output with TritonAI `api-llama-4-scout` reached:

| Metric group | Metric | Value |
|---|---|---:|
| Retrieval | F1@5 | 0.7292 |
| Retrieval | Precision@5 | 0.6315 |
| Retrieval | Recall@5 | 0.9611 |
| Retrieval | Retrieval Score | 0.7739 |
| Generation | Correctness | 0.93 |
| Generation | Faithfulness | 0.96 |
| Generation | Completeness | 4.36 / 5 |
| Generation | Partial Generation Score | 0.9200 |

## Engineering Tradeoffs and Judgment

1. Recall is more important than precision for this documentation QA task, up to the point where
   irrelevant context starts harming generation. The best final retrieval config has very high
   Recall@5 (0.9611), which means most gold evidence is seen by the generator.

2. Very small chunks improved recall slightly but hurt precision and F1. For example, the
   36-line setup reached Recall@5 of 0.9667 but had lower Precision@5 of 0.5037 and lower
   Retrieval Score of 0.7011. This suggests that short chunks fragment evidence and add noisy
   neighboring matches.

3. Larger chunks improved precision but can lose evidence coverage. The 96-line setup had the
   highest Precision@5 among the local configs (0.6667), but recall dropped to 0.8944. This
   is risky under the 2,000-token context budget because fewer distinct source regions can fit.

4. The selected 54-line / 24-overlap / top-5 configuration is the best balance. It kept recall
   very high while preserving enough precision to avoid flooding the prompt with unrelated
   documentation.

5. Structured prompt design and strict context grounding improved generation stability. After
   switching the generation and judge model to `api-llama-4-scout`, local LLM-as-judge scoring
   improved to Correctness 0.93, Faithfulness 0.96, and Partial Generation Score 0.9200.

6. RapidFire AI is used for broader side-by-side experimentation, while `main.py` is locked to
   the best deterministic configuration for grading. This separation matches the project statement:
   graders should run the final pipeline, while the report/logs show the interactive and systematic
   experimentation process.

## Final Decision

The final submitted pipeline keeps the dependency-light BM25 implementation because it produced
strong validation retrieval, exact source line spans, fast deterministic execution, and no dependence
on model downloads during hidden-test grading. RapidFire AI experiments are used to document and
compare the broader design space involving dense embeddings, FAISS, MMR, cross-encoder reranking,
and prompt styles.
