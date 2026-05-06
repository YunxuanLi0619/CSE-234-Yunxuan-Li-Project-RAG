# CSE/DSC 234 Project 1 RAG Solution

This folder contains a complete, dependency-light implementation of the required final pipeline.

## Final CLI

Run from this folder:

```bash
python3 main.py --input ../validation-set-golden-qa-pairs.json --output logs/validation_output.json
```

The script also accepts the hidden-test schema because it only requires `question_id` and `question`.

Output entries contain:

- `question_id`
- `answer`
- `retrieved_context`
- `sources` as `{"file": "...", "lines": [start, end]}`

## LLM Configuration

The pipeline retrieves context locally, then uses an OpenAI-compatible chat-completions endpoint if configured.
If no API key is available, it falls back to a deterministic extractive answer so the CLI still runs end to end.

Useful environment variables:

```bash
export OPENAI_API_KEY="$(cat ~/api-key.txt)"
export GENERATOR_BASE_URL="https://tritonai-api.ucsd.edu"
export GENERATOR_MODEL="api-llama-4-scout"
```

You can also pass `--api-key`, `--base-url`, and `--model` directly.

Or create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
# edit .env and paste your TritonAI key
```

Do not commit API keys. `.env` is ignored by git; keep it local to DataHub.

For retrieval-only smoke tests:

```bash
python3 main.py --input ../validation-set-golden-qa-pairs.json --output logs/validation_output.json --no-llm
```

For a final generation run, fail loudly instead of silently writing fallback answers:

```bash
python3 main.py \
  --input ../validation-set-golden-qa-pairs.json \
  --output logs/final_validation_output.json \
  --max-output-tokens 300 \
  --retries 8 \
  --fail-on-llm-error

grep -n "Generation API was unavailable" logs/final_validation_output.json
```

The `grep` command should print nothing.

## Retrieval Evaluation

```bash
bash scripts/run_validation.sh
```

or manually:

```bash
python3 ../Metrics/evaluate_retrieval.py \
  --output logs/validation_output.json \
  --validation ../validation-set-golden-qa-pairs.json \
  --k 5 > logs/retrieval_report.json
```

## Submission Artifacts

Team 40 submission-ready artifacts are included in this folder:

- `golden_qa_pairs.json`: 22 hand-labeled golden Q&A pairs.
- `validation_output.json`: output JSON from the final pipeline on the released validation set.
- `project_report.pdf`: final 4-page LaTeX project report with the five required sections.
- `reports/project_report.tex`: LaTeX source for the final report.
- `logs/`: retrieval, generation judge, and RapidFire AI experiment logs/metrics.

Regenerate the convenience artifacts after a new run with:

```bash
python3 scripts/export_submission_artifacts.py
```

## Local Config Sweep

Fast local sweep over the 8 configurations in `configs/experiment_configs.json`:

```bash
python3 scripts/run_retrieval_experiments.py
```

This writes `logs/retrieval_experiments.json`, useful for the report table. For the class-required RapidFire AI workflow, use these same knobs in RapidFire `run_evals` experiments and keep the RapidFire logs under `logs/`.

## RapidFire AI Multi-Config Experiments

The project requires a RapidFire AI OSS experimentation workflow in addition to the deterministic final `main.py`.
The runnable RapidFire experiment script is:

```bash
python3 scripts/run_rapidfire_experiments.py --dry-run
```

The 8 RapidFire configs are listed in:

```text
configs/rapidfire_experiment_configs.json
```

On DataHub, after `rapidfireai init --evals` and `rapidfireai start`, run:

```bash
python3 scripts/run_rapidfire_experiments.py \
  --validation ../validation-set-golden-qa-pairs.json \
  --validation-limit 24 \
  --num-shards 4 \
  --num-actors 4
```

Remove `--validation-limit 24` for the final full validation run. This writes RapidFire artifacts under `logs/`.
The report-ready tradeoff analysis is in:

```text
reports/experiment_methodology_and_tradeoffs.md
reports/rapidfire_config_comparison.md
```

## Design

- The corpus loader reads `../sourcedocs/*.rst`.
- Chunks are overlapping line windows, preserving exact source line spans.
- Retrieval uses BM25 with boosts for API names, code identifiers, headings, phrases, and file names.
- Context serialization includes source file and line labels and stays under the configured budget.
- The code never hardcodes released validation answers or question IDs.
