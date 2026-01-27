#!/usr/bin/env bash
set -euo pipefail

# Directories containing quizbench_manifest*.json + per-quiz run dirs
GEN_RUNS=(
  # "eval_results/quizbench/runs/gpt-5.1-2025-11-13"
  # "eval_results/quizbench/runs/gemini-3-pro-preview"
  "eval_results/quizbench/runs/claude-opus-4-5-20251101"
  # "eval_results/quizbench/runs/grok-4-0709"
  # "eval_results/quizbench/runs/deepseek-v3.2"
  # "eval_results/quizbench/runs/kimi-k2-thinking"
)

# Quiz-taking (eval) models to run on every generator's quizzes
EVAL_MODELS=(
  "gpt-5.1-2025-11-13"
  # "gemini-3-pro-preview"
  # "claude-opus-4-5-20251101"
  # "grok-4-0709"
  # "kimi-k2-thinking"
  # "deepseek-v3.2"
)
# Lighter models
#  "gemini-2.5-flash-lite", "gpt-5-nano-2025-08-07", "claude-3-7-sonnet-20250219"

for MODEL in "${EVAL_MODELS[@]}"; do
  echo "[INFO] === Eval model: ${MODEL} ==="
  for RUNS_ROOT in "${GEN_RUNS[@]}"; do
    echo "[INFO]   runs_root: ${RUNS_ROOT}"
    uv run python -m quizbench.run_eval \
      --runs_root "${RUNS_ROOT}" \
      --eval_model "${MODEL}" \
      --use_batch_api
  done
done

