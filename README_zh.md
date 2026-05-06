# CSE/DSC 234 Project 1 RAG 方案说明

这个文件夹包含本项目最终 pipeline 的实现代码。核心目标是：给定一个问题 JSON 文件，自动检索 `sourcedocs/` 文档，生成答案，并输出项目要求的 JSON 格式。

## 最终运行命令

在 `project1_solution` 目录下运行：

```bash
python3 main.py --input ../validation-set-golden-qa-pairs.json --output logs/validation_output.json
```

`main.py` 也可以处理 hidden test 的输入格式，因为它只依赖每条样本里的：

- `question_id`
- `question`

输出文件中每条样本包含：

- `question_id`
- `answer`
- `retrieved_context`
- `sources`，格式为 `{"file": "...", "lines": [start, end]}`

## TritonAI / LLM 配置

本 pipeline 会先在本地完成文档检索，然后调用 OpenAI-compatible chat completion API 生成答案。UCSD TritonAI API 是 OpenAI-compatible 的，所以可以直接使用。

在 DataHub / VSCode SSH terminal 中设置：

```bash
export OPENAI_API_KEY="$(cat ~/api-key.txt)"
export GENERATOR_BASE_URL="https://tritonai-api.ucsd.edu"
export GENERATOR_MODEL="api-gpt-oss-120b"
```

也可以直接传命令行参数：

```bash
python3 main.py \
  --input ../validation-set-golden-qa-pairs.json \
  --output logs/validation_output.json \
  --api-key "$OPENAI_API_KEY" \
  --base-url "https://tritonai-api.ucsd.edu" \
  --model "api-gpt-oss-120b"
```

也可以使用本地 `.env` 文件：

```bash
cp .env.example .env
# 然后编辑 .env，把你的 TritonAI key 填进去
```

`.env` 已经加入 `.gitignore`，不会被 git 提交。不要把 API key 写进代码或提交到 GitHub。推荐把 key 放在 DataHub 的 `~/api-key.txt`，或者只在当前 shell session / `.env` 里保存。

## 测试 TritonAI API 是否可用

先运行：

```bash
python3 scripts/test_triton_api.py
```

如果 API 配置正确，它会调用 `api-gpt-oss-120b` 并打印模型回复。

## 无 API 的 Smoke Test

如果只是想检查代码、检索、输出格式是否正常，可以禁用 LLM：

```bash
python3 main.py \
  --input ../validation-set-golden-qa-pairs.json \
  --output logs/validation_output.json \
  --no-llm
```

`--no-llm` 会使用本地 extractive fallback 生成答案。这个模式适合调试，但正式提交前应该用 TritonAI API 重新生成答案。

## Retrieval 评估

一键运行：

```bash
bash scripts/run_validation.sh
```

或者手动运行：

```bash
python3 ../Metrics/evaluate_retrieval.py \
  --output logs/validation_output.json \
  --validation ../validation-set-golden-qa-pairs.json \
  --k 5 > logs/retrieval_report.json
```

## Generation 评估

正式调用 TritonAI 生成 `logs/final_validation_output.json` 后，运行：

```bash
python3 ../Metrics/run_judge.py \
  --output logs/final_validation_output.json \
  --validation ../validation-set-golden-qa-pairs.json \
  --base-url "$GENERATOR_BASE_URL" \
  --model "$GENERATOR_MODEL" > logs/final_judge_report.json
```

该脚本会用 LLM-as-judge 评估：

- Correctness
- Faithfulness
- Completeness

## 本地配置实验

快速跑 8 个 retrieval 配置：

```bash
python3 scripts/run_retrieval_experiments.py
```

结果会写入：

```text
logs/retrieval_experiments.json
```

这个文件可以用于 report 的实验表格。课程要求还需要使用 RapidFire AI 的 `run_evals` 进行实验，并把 RapidFire 产生的 logs 和 metrics 放进 `logs/`。

## RapidFire AI 多配置实验

为了补齐 project statement 里的 Learning Outcomes 4/5 和 High-Level Description 第 3 部分，
本文件夹新增了 RapidFire AI OSS 的正式实验脚本：

```bash
python3 scripts/run_rapidfire_experiments.py --dry-run
```

8 组 RapidFire 配置写在：

```text
configs/rapidfire_experiment_configs.json
```

这些配置系统性对比：

- 分块大小和重叠长度
- CPU embedding 模型
- similarity search 和 MMR
- 是否使用 cross-encoder reranker
- concise prompt 和 structured prompt
- TritonAI `api-llama-4-scout` 生成设置

在 DataHub 上启动 RapidFire 后运行：

```bash
rapidfireai init --evals
rapidfireai start

python3 scripts/run_rapidfire_experiments.py \
  --validation ../validation-set-golden-qa-pairs.json \
  --validation-limit 24 \
  --num-shards 4 \
  --num-actors 4
```

`--validation-limit 24` 用于控制 API 成本，最终 report 前可以去掉该参数跑完整 released validation set。
RapidFire 结果会写到 `logs/`，实验设计和 trade-off 分析稿在：

```text
reports/experiment_methodology_and_tradeoffs.md
reports/rapidfire_config_comparison.md
```

## 代码设计

- `sourcedocs/` 是知识库。
- `main.py` 是最终提交要求的 CLI 入口。
- `src/rag.py` 负责 line-aware chunking、BM25 检索、context 拼接和 source line span 输出。
- `src/llm_client.py` 负责调用 TritonAI/OpenAI-compatible API。
- `golden_qa_pairs.json` 是我们自己创建的 golden Q&A 数据。
- `configs/experiment_configs.json` 包含至少 8 个实验配置。

## 正式提交前检查

正式提交前建议确认：

```bash
python3 main.py \
  --input ../validation-set-golden-qa-pairs.json \
  --output logs/final_validation_output.json \
  --max-output-tokens 300 \
  --retries 8 \
  --fail-on-llm-error

grep -n "Generation API was unavailable" logs/final_validation_output.json
```

如果第一条命令中途停止，说明 API 没有稳定生成完所有答案，重新运行即可；加上
`--fail-on-llm-error` 的目的就是避免把失败题悄悄写成 fallback。第二条 `grep`
没有输出时，才说明最终 JSON 里没有 fallback。

还需要确认：

- `main.py` 可以从 repo 根目录直接运行。
- `README.md` 存在。
- `golden_qa_pairs.json` 至少 20 条。
- `logs/` 中包含 RapidFire AI 实验日志和 metrics。
- `logs/final_validation_output.json` 是最终 pipeline 在 released validation set 上生成的输出。
- project report PDF 已经写好。
