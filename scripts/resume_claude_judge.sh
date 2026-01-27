#!/usr/bin/env bash
set -euo pipefail

# Resume only the quizzes that are still missing
# claude-opus-4-5-20251101 judge outputs for the
# eval_judge_round_robin.sh generators.

JUDGE_MODEL="${JUDGE_MODEL:-claude-opus-4-5-20251101}"
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

# gpt-5.1-2025-11-13 generator: seeds 124–127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/gpt-5.1-2025-11-13" \
  "20251129T045001943Z_gpt-5.1-2025-11-13_seed124,20251129T045001943Z_gpt-5.1-2025-11-13_seed125,20251129T045001943Z_gpt-5.1-2025-11-13_seed126,20251129T045001943Z_gpt-5.1-2025-11-13_seed127"

# gemini-3-pro-preview generator: seeds 124–127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/gemini-3-pro-preview" \
  "20251129T125426901Z_gemini-3-pro-preview_seed124,20251129T125426901Z_gemini-3-pro-preview_seed125,20251129T125426901Z_gemini-3-pro-preview_seed126,20251129T125426901Z_gemini-3-pro-preview_seed127"

# claude-opus-4-5-20251101 generator: seeds 124–127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/claude-opus-4-5-20251101" \
  "20251129T135352000Z_claude-opus-4-5-20251101_seed124,20251129T135352000Z_claude-opus-4-5-20251101_seed125,20251129T135352000Z_claude-opus-4-5-20251101_seed126,20251129T135352000Z_claude-opus-4-5-20251101_seed127"

# deepseek-v3.2 generator: seeds 124–127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/deepseek-v3.2" \
  "20251205T082731900Z_deepseek-v3.2_seed124,20251205T084024622Z_deepseek-v3.2_seed125,20251205T084024622Z_deepseek-v3.2_seed126,20251205T084024622Z_deepseek-v3.2_seed127"

# kimi-k2-thinking generator: seeds 124–127 unfinished
run_judge_subset \
  "eval_results/quizbench/runs/kimi-k2-thinking" \
  "20251205T080918850Z_kimi-k2-thinking_seed124,20251205T080918850Z_kimi-k2-thinking_seed125,20251205T080918850Z_kimi-k2-thinking_seed126,20251205T080918850Z_kimi-k2-thinking_seed127"

