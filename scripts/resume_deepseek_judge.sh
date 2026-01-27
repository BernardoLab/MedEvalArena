#!/usr/bin/env bash
set -euo pipefail

# Resume only the quizzes that are still missing
# deepseek-v3.2 judge outputs.

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v3.2}"
MAX_TOKENS="${MAX_TOKENS:-16000}"

run_judge_subset() {
  local runs_root="$1"
  local quiz_ids_csv="$2"

  if [[ -z "${quiz_ids_csv}" ]]; then
    echo "[INFO] No pending quizzes for ${runs_root}"
    return 0
  fi

  echo "[INFO] Resuming judge=${JUDGE_MODEL} for runs_root=${runs_root}"
  echo "[INFO]   quiz_ids: ${quiz_ids_csv}"

  uv run python -m quizbench.run_eval_judge \
    --runs_root "${runs_root}" \
    --judge_model "${JUDGE_MODEL}" \
    --only_quiz_ids_csv "${quiz_ids_csv}" \
    --max_tokens "${MAX_TOKENS}"
}

# gpt-5.1-2025-11-13 generator: seeds 126, 127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/gpt-5.1-2025-11-13" \
  "20251129T045001943Z_gpt-5.1-2025-11-13_seed126,20251129T045001943Z_gpt-5.1-2025-11-13_seed127"

# grok-4-0709 generator: seeds 126, 127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/grok-4-0709" \
  "20251201T041413890Z_grok-4-0709_seed126,20251201T041413890Z_grok-4-0709_seed127"

# gemini-3-pro-preview generator: seeds 126, 127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/gemini-3-pro-preview" \
  "20251129T125426901Z_gemini-3-pro-preview_seed126,20251129T125426901Z_gemini-3-pro-preview_seed127"

# kimi-k2-thinking generator: seeds 125, 126, 127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/kimi-k2-thinking" \
  "20251205T080918850Z_kimi-k2-thinking_seed125,20251205T080918850Z_kimi-k2-thinking_seed126,20251205T080918850Z_kimi-k2-thinking_seed127"
