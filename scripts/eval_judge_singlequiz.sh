#!/usr/bin/env bash

QUIZ="quizzes/quizzes_Jan2026/20251218T014545383Z_claude-opus-4-5-20251101_seed131.jsonl"
OUT="eval_results/quizbench/quizzes_Jan2026/runs/claude-opus-4-5-20251101"

for J in gemini-3-pro-preview claude-opus-4-5-20251101 grok-4-0709 deepseek-v3.2; do
  EXTRA=()
  [[ "$J" == deepseek-* || "$J" == kimi-* ]] && EXTRA+=(--use_openrouter)
  uv run python -m quizbench.eval_quiz_judge --quiz_file "$QUIZ" --judge_model "$J" --out_dir "$OUT" --max_tokens 16000 "${EXTRA[@]}"
done
