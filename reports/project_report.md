# Team 40 Project Report: RAG and Context Engineering

## a. Data Creation Methodology

I created 22 hand-labeled golden Q&A pairs over the RapidFire AI documentation corpus in `golden_qa_pairs.json`. Each example follows the released validation schema: `question_id`, `question`, `reference_answer`, and `source_evidence` with exact filename and line-span annotations. The examples were designed to cover the required variety of user intents: factual lookup, conceptual explanation, procedural setup, and comparative questions.

The data creation process started by scanning the RST documentation pages for high-value API and workflow concepts: config group generation, experiment lifecycle, RAG specification, vector stores, search configuration, prompt management, dashboards, IC Ops, and troubleshooting. For each candidate question, I verified the answer against the source file and recorded a narrow evidence span rather than a whole page. I then checked that the question could not be answered reliably without consulting the corpus and that the reference answer captured important conditions, defaults, and distinctions.

I also used the released validation set to calibrate question style and source annotation granularity, but the golden file is separate from the released validation output and is not used to hardcode any validation answers. The final golden set has 22 examples, above the required minimum of 20. I did not add extra custom judge metrics beyond the provided retrieval metrics and LLM-as-judge metrics; instead, I focused on making the evidence spans precise enough to support robust retrieval evaluation.

## b. Final Pipeline Architecture

The final automated pipeline is implemented in `main.py` and `src/`. It is intentionally dependency-light so the hidden-test grader can run it deterministically with only the standard library plus the optional OpenAI client. The pipeline reads a JSON test set, retrieves documentation context, generates answers with a TritonAI/OpenAI-compatible chat model when configured, and writes the required output schema with `question_id`, `answer`, `retrieved_context`, and `sources`.

The corpus loader automatically locates `sourcedocs/` and loads `.rst`, `.txt`, and `.md` files. The selected chunking strategy uses overlapping line windows with `chunk_lines=54` and `chunk_overlap=24`. This preserves exact source line spans, which matters because the evaluator scores overlap between retrieved source spans and gold evidence spans. The retriever is a custom BM25 lexical retriever rather than a dense vector retriever. It tokenizes API identifiers, camel case, numeric terms, and conservative stems; it also adds boosts for exact API names, code identifiers, phrases, headings, and filename matches. I selected this approach because the documentation corpus contains many precise API names, and lexical matching produced stronger line-span retrieval than the dense RapidFire experiments on the released validation set.

The final retrieval settings are `top_k=5`, `candidate_pool=48`, and a context token budget of 1700 for serialized retrieved chunks. The output contexts were checked to stay below the 2000-token project budget after allowing room for prompt and question text; the largest retrieved context in the current validation output is about 1692 tokens. The context serializer includes labels such as `[Source: ragspecs.rst lines 103-156]`, making the evidence transparent to both the generator and the faithfulness judge.

The generation layer is in `src/llm_client.py`. It uses an OpenAI-compatible chat-completions endpoint, so it works with TritonAI by setting `OPENAI_API_KEY`, `GENERATOR_BASE_URL`, and `GENERATOR_MODEL`. The prompt instructs the model to answer only from retrieved context, be concise and complete, and avoid unsupported claims. If the API is unavailable, the code has a deterministic extractive fallback for debugging, but the submitted validation output was checked to contain no fallback warning text.

## c. Experimentation Methodology

I used two complementary experiment tracks. First, I ran a fast local BM25 sweep over eight line-aware retrieval configurations. This directly matched the final pipeline implementation and used the released span-overlap retrieval evaluator. Second, I used RapidFire AI OSS with `RFLangChainRagSpec` and `run_evals()` to compare eight dense RAG configurations side by side. Those experiments varied chunking, embedding model, search strategy, reranking, prompting, and generation settings.

The RapidFire workflow is implemented in `scripts/run_rapidfire_experiments.py`. It builds configs from `configs/rapidfire_experiment_configs.json`, loads the validation data, constructs a `Dataset`, defines preprocess/postprocess/metric functions, wraps the generator with `RFOpenAIAPIModelConfig`, and launches all configs through `RFGridSearch`/RapidFire evals. The RAG spec uses `DirectoryLoader`, `RecursiveCharacterTextSplitter`, HuggingFace CPU embeddings, FAISS vector store, similarity or MMR search, optional cross-encoder reranking, and either a concise or structured grounded-answer prompt.

RapidFire config table:

| Run | Chunk | Embedding | Search | Reranker | Prompt |
|---:|---:|---|---|---|---|
| 1 | 256/32 | MiniLM-L6-v2 | similarity k=5 | none | concise |
| 2 | 256/32 | MiniLM-L6-v2 | MMR k=5 fetch=24 lambda=0.35 | none | structured |
| 3 | 512/64 | MiniLM-L6-v2 | similarity k=7 | cross-encoder top_n=5 | concise |
| 4 | 512/64 | MiniLM-L6-v2 | MMR k=7 fetch=32 lambda=0.35 | cross-encoder top_n=5 | structured |
| 5 | 256/32 | BGE-small | similarity k=5 | none | structured |
| 6 | 384/64 | BGE-small | MMR k=5 fetch=24 lambda=0.40 | none | concise |
| 7 | 512/64 | BGE-small | similarity k=7 | cross-encoder top_n=5 | structured |
| 8 | 512/96 | BGE-small | MMR k=7 fetch=32 lambda=0.35 | cross-encoder top_n=5 | concise |

The local BM25 configs varied `chunk_lines`, `chunk_overlap`, `top_k`, and `candidate_pool`. These runs were useful because they optimized the exact deterministic retriever used by `main.py`, while RapidFire explored the broader design space requested by the project.

## d. Results and Analysis

The best final pipeline config was `bm25_lines54_overlap24_top5`: `chunk_lines=54`, `chunk_overlap=24`, `top_k=5`, and `candidate_pool=48`. It achieved the highest overall retrieval score among the local configs:

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

The main tradeoff was chunk size versus recall and precision. Very small chunks produced high recall but fragmented evidence and lowered precision. Very large chunks improved precision but lost some evidence coverage and consumed more of the context budget. The 54-line/24-overlap setting was the best balance: it kept Recall@5 very high while limiting irrelevant context enough for stable generation.

In the RapidFire dense retrieval experiments, the strongest source F1 was Run 5, `rf_bge_256_similarity_no_rerank_structured`, with Source F1@5 = 0.5363, Precision@5 = 0.3792, Recall@5 = 0.9792, and Hit Rate = 1.0000. Run 7 had the best reference-token recall for generation, but its source F1 was slightly lower. BGE embeddings generally outperformed MiniLM on recall, while MMR and cross-encoder reranking did not consistently improve the source metrics in this small documentation corpus. One surprising outcome was that the simpler BGE similarity setup without reranking beat several more complex reranked configs.

The final validation output from the selected deterministic pipeline achieved Retrieval Score = 0.7739. The released LLM judge metrics on the current output were Correctness pass rate = 0.9556, Faithfulness pass rate = 0.9778, Completeness = 4.6667/5, and Partial Generation Score = 0.9556. If I had more time or budget, I would try a hybrid retriever combining BM25 with BGE similarity and then apply a lightweight reranker only to ambiguous cases. That would preserve the lexical strength on API-name questions while adding semantic recall for paraphrased conceptual questions.

## e. AI Tools Statement

I used AI assistants, including Codex/ChatGPT-style tools, to help inspect the codebase, draft implementation ideas, generate boilerplate, debug scripts, summarize experiment outputs, and organize this report. I reviewed the generated code and artifacts, ran validation checks, and made final design choices based on the measured retrieval and generation results. No released validation answers or hidden-test-specific logic are hardcoded in the final pipeline.
