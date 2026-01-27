#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: $0 <quiz_tag>"
  echo "Example: $0 Jan2026"
  exit 0
fi

if [ -z "${1:-}" ]; then
  echo "Error: missing required quiz_tag."
  echo "Usage: $0 <quiz_tag>"
  echo "Example: $0 Jan2026"
  exit 2
fi

QUIZ_TAG="$1"
if [[ "$QUIZ_TAG" == quizzes_* ]]; then
  QUIZ_COLLECTION="$QUIZ_TAG"
  QUIZ_TAG="${QUIZ_TAG#quizzes_}"
else
  QUIZ_COLLECTION="quizzes_${QUIZ_TAG}"
fi

if [[ ! "$QUIZ_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Error: invalid quiz_tag '$QUIZ_TAG' (allowed: letters/digits/._-; must start with letter/digit)."
  exit 2
fi

echo "[INFO] quiz_collection='$QUIZ_COLLECTION' (quiz_tag='$QUIZ_TAG')"

# Step 1: generate questions:

for phase in plan generate; do
  plan_args=()
  if [ "$phase" = "plan" ]; then
    plan_args=(--plan_targets_only)
  fi

  # Grok
  uv run quizbench/run_batch_gen_quiz.py \
        --config configs/config_grok.yaml \
        --quiz_collection "$QUIZ_COLLECTION" \
        --quiz_batch_tag "$QUIZ_TAG" \
        --use_target_batches \
        "${plan_args[@]}"

  # Deepseek example
  uv run quizbench/run_batch_gen_quiz.py \
        --config configs/config_deepseek3.2.yaml \
        --quiz_collection "$QUIZ_COLLECTION" \
        --quiz_batch_tag "$QUIZ_TAG" \
        --use_target_batches \
        "${plan_args[@]}"

  # Kimi
  uv run quizbench/run_batch_gen_quiz.py \
        --config configs/config_kimi-k2-thinking.yaml \
        --quiz_collection "$QUIZ_COLLECTION" \
        --quiz_batch_tag "$QUIZ_TAG" \
        --use_target_batches \
        "${plan_args[@]}"

  # Anthropic
  uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_anthropic.yaml \
      --quiz_collection "$QUIZ_COLLECTION" \
      --quiz_batch_tag "$QUIZ_TAG" \
      --use_target_batches \
      "${plan_args[@]}"

  # OpenAI
  uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_gpt.yaml \
      --quiz_collection "$QUIZ_COLLECTION" \
      --quiz_batch_tag "$QUIZ_TAG" \
      --use_target_batches \
      "${plan_args[@]}"

  # Gemini
  uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_gemini.yaml \
      --quiz_collection "$QUIZ_COLLECTION" \
      --quiz_batch_tag "$QUIZ_TAG" \
      --use_target_batches \
      "${plan_args[@]}"

  if [ "$phase" = "plan" ]; then
    printf "Plan-only complete. Proceed with generation? [y/N] "
    read -r reply
    case "$reply" in
      [yY]|[yY][eE][sS]) ;;
      *) echo "Aborting before generation."; exit 0 ;;
    esac
  fi
done
