# QuizBench (medARC) — Multi‑LLM Medical MCQ Generation & Evaluation

> **Scope:** This document explains how to (1) generate very hard 5‑option medical MCQs with one LLM, (2) evaluate multiple LLMs on each quiz across `num_quizzes` rounds, and (3) aggregate accuracy as **mean ± SEM** into a CSV.

> **Safety:** All content is **NOT FOR CLINICAL USE**. This is a benchmarking pipeline, not medical guidance.

---

## Quickstart (one command)

```bash
# at repo root
bash v1_scripts/eval_quizbench_triplet.sh
````

The script:

* installs QuizBench deps,
* generates `NUM_QUIZZES` quizzes with `GENERATOR_MODEL`,
* evaluates **gpt‑4o**, **gemini‑1.5‑pro‑latest**, **claude‑3‑sonnet‑20240229** on each quiz,
* writes a CSV summary at `eval_results/quizbench/quizbench_summary.csv`.

You can override defaults via env vars (see **Configuration** below).

---

## Requirements

* Python 3.10+ recommended
* API keys in environment (do **not** hardcode):

  * `OPENAI_API_KEY` (for gpt‑4o / gpt‑4o‑mini / gpt‑4)
  * `ANTHROPIC_API_KEY` (for claude‑3‑sonnet‑20240229 / claude‑3‑opus‑20240229)
  * `GOOGLE_API_KEY` (for gemini‑1.5‑pro‑latest / gemini‑1.5‑flash‑latest)

Install QuizBench requirements (idempotent):

```bash
python3 -m pip install -r quizbench/requirements_quizbench.txt
```

---

## Supported model name strings

These strings must match what `quizbench/clients.py` routes:

* **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4`
* **Anthropic**: `claude-3-sonnet-20240229`, `claude-3-opus-20240229`
* **Google**: `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`

> To add more providers (e.g., Grok), add a branch in `quizbench/clients.py` that authenticates and returns a string response.

---

## Directory outputs

* **Generated quizzes**: `data_medARCv1/quizbench_quizzes/quiz_*.jsonl`
* **Per‑quiz evaluation runs**: `eval_results/quizbench/runs/<quiz_id>/`

  * `<model>_result.json` : per‑item predictions & raw outputs
  * `<model>_summary.json` : `{acc, total_corr, total_wrong, n_items}`
  * `manifest.json` : quiz metadata
* **Aggregate CSV**: `eval_results/quizbench/quizbench_summary.csv`

  * columns: `model, num_quizzes, mean_accuracy, sem, mean_pm_sem, mean_accuracy_percent, sem_percent`

---

## End‑to‑end (manual, step‑by‑step)

> **All commands are run from repo root** (`/Users/.../medARC`).

### 1) Set environment

```bash
export OPENAI_API_KEY=sk-...        # required for generator + GPT tests
export GOOGLE_API_KEY=AIza...       # required for Gemini tests
export ANTHROPIC_API_KEY=sk-ant-... # required for Claude tests
```

### 2) Install QuizBench deps

```bash
python3 -m pip install -r quizbench/requirements_quizbench.txt
```

### 3) Generate one quiz (sanity check)

```bash
QUIZ_PATH=$(python3 quizbench/generate_quiz.py \
  --generator_model gpt-4o \
  --num_questions 3)
echo "Quiz wrote to: ${QUIZ_PATH}"
```

### 4) Evaluate that quiz on multiple LLMs

```bash
python3 quizbench/eval_quiz.py \
  --quiz_file "${QUIZ_PATH}" \
  --test_models_csv "gpt-4o,gemini-1.5-pro-latest,claude-3-sonnet-20240229" \
  --out_dir eval_results/quizbench/runs
```

### 5) Orchestrate N quizzes × M models (recommended)

```bash
python3 quizbench/run_quizbench.py \
  --generator_model gpt-4o \
  --test_models_csv "gpt-4o,gemini-1.5-pro-latest,claude-3-sonnet-20240229" \
  --num_quizzes 10 \
  --num_questions 10 \
  --seed0 123 \
  --quizzes_dir data_medARCv1/quizbench_quizzes \
  --runs_root eval_results/quizbench/runs
```

### 6) Aggregate mean ± SEM

```bash
python3 quizbench/aggregate_results.py \
  --runs_root eval_results/quizbench/runs \
  --out_csv eval_results/quizbench/quizbench_summary.csv
```

---

## Configuration (env overrides for the bash script)

The script `v1_scripts/eval_quizbench_triplet.sh` accepts these environment variables:

* `GENERATOR_MODEL` (default: `gpt-4o`)
* `TEST_MODELS_CSV` (default: `gpt-4o,gemini-1.5-pro-latest,claude-3-sonnet-20240229`)
* `NUM_QUIZZES` (default: `10`)
* `NUM_QUESTIONS` (default: `10`)
* `SEED0` (default: `123`)
* `QUIZZES_DIR` (default: `data_medARCv1/quizbench_quizzes`)
* `RUNS_ROOT` (default: `eval_results/quizbench/runs`)
* `SUMMARY_CSV` (default: `eval_results/quizbench/quizbench_summary.csv`)

Example:

```bash
GENERATOR_MODEL=gpt-4o \
TEST_MODELS_CSV="gpt-4o,gemini-1.5-pro-latest,claude-3-sonnet-20240229" \
NUM_QUIZZES=5 NUM_QUESTIONS=8 \
bash v1_scripts/eval_quizbench_triplet.sh
```

---

## Determinism and scoring

* Generation uses `temperature=0.2` (hard MCQs); evaluation uses `temperature=0.0`.
* Each round seeds the generator with `seed = SEED0 + k` to diversify quizzes.
* Accuracy = (# correct / # questions) per quiz per model.
* Final CSV reports **mean ± SEM across quizzes** per model.


---
**Quizbench Script Usage examples**

* Default triplet (OpenAI + Gemini + Claude), 10×10:

```bash
bash v1_scripts/eval_quizbench_triplet.sh
```

* Smaller, faster smoke test:

```bash
NUM_QUIZZES=2 NUM_QUESTIONS=5 bash v1_scripts/eval_quizbench_triplet.sh
```

---

## Troubleshooting

* **Missing API key**: set `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`.
* **Model not wired**: add its provider branch to `quizbench/clients.py`.
* **Rate limits**: lower `NUM_QUIZZES`/`NUM_QUESTIONS` or remove one test model.
* **CSV empty**: ensure runs exist under `eval_results/quizbench/runs/quiz_*`.

---
