#!/usr/bin/env bash
set -euo pipefail

# Generator run directories (each should contain a quizbench_manifest*.json)
RUN_DIRS=(
  "eval_results/quizbench/runs/gpt-5.1-2025-11-13"
  "eval_results/quizbench/runs/gemini-3-pro-preview"
  "eval_results/quizbench/runs/claude-opus-4-5-20251101"
  "eval_results/quizbench/runs/grok-4-0709"
  "eval_results/quizbench/runs/deepseek-v3.2"
  "eval_results/quizbench/runs/kimi-k2-thinking"
)

# Evaluation models to use for topic classification
EVAL_MODELS=(
  # "gpt-4o"
  "claude-3-7-sonnet-20250219"
  # "gemini-3-pro-preview"
)

# Limit how many quizzes/questions to process (0 means all)
MAX_QUIZZES=0
MAX_QUESTIONS=0

# Optional overrides
TOPICS_FILE=""        # e.g., "custom_topics.txt"
MAX_TOKENS=512        # per-call token cap for classification
TEMPERATURE=0.0       # keep deterministic
OVERWRITE=0           # set to 1 to replace existing topics files
USE_OPENROUTER=0      # set to 1 when routing DeepSeek/Kimi via OpenRouter

for MODEL in "${EVAL_MODELS[@]}"; do
  echo "[INFO] === Topic model: ${MODEL} ==="
  for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "[INFO]   run_dir: ${RUN_DIR}"

    cmd=(uv run python -m quizbench.categorize_quiz_topics
      --eval_model "${MODEL}"
      --run_dir "${RUN_DIR}"
      --max_tokens "${MAX_TOKENS}"
      --temperature "${TEMPERATURE}"
    )

    if [[ "${MAX_QUIZZES}" -gt 0 ]]; then
      cmd+=(--max_quizzes "${MAX_QUIZZES}")
    fi
    if [[ "${MAX_QUESTIONS}" -gt 0 ]]; then
      cmd+=(--max_questions "${MAX_QUESTIONS}")
    fi
    if [[ -n "${TOPICS_FILE}" ]]; then
      cmd+=(--topics_file "${TOPICS_FILE}")
    fi
    if [[ "${OVERWRITE}" -eq 1 ]]; then
      cmd+=(--overwrite)
    fi
    if [[ "${USE_OPENROUTER}" -eq 1 ]]; then
      cmd+=(--use_openrouter)
    fi

    echo "[CMD] ${cmd[*]}"
    "${cmd[@]}"
  done
done
