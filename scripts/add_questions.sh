#!/usr/bin/env bash

# New question re-balancing pipeline:

# Step 1: generate questions:

# Grok
uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_grok.yaml \
      --quiz_collection quizzes_Jan2026 \
      --use_target_batches \
      --min_medical_score \
      --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/grok-4-0709/accumulated_mapped_topics.json \
      --plan_targets_only

# Deepseek example (uncomment to use)
uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_deepseek3.2.yaml \
      --quiz_collection quizzes_Jan2026 \
      --use_target_batches \
      --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/deepseek-v3.2/accumulated_mapped_topics.json \
      --plan_targets_only 

uv run quizbench/run_batch_gen_quiz.py \
      --config configs/config_kimi-k2-thinking.yaml \
      --quiz_collection quizzes_Jan2026 \
      --use_target_batches \
      --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/kimi-k2-thinking/accumulated_mapped_topics.json \
      --plan_targets_only 

# Anthropic example
# uv run quizbench/run_batch_gen_quiz.py \
#     --config configs/config_anthropic.yaml \
#     --quiz_collection quizzes_Jan2026 \
#     --quizzes_dir quizzes/quizzes_Jan2026 \
#     --use_target_batches \
#     --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/claude-opus-4-5-20251101/accumulated_mapped_topics.json \
#     --topic_map data/topic_to_abms.yaml \
#     --plan_targets_only 

# OpenAI example
# uv run quizbench/run_batch_gen_quiz.py \
#     --config configs/config_gpt.yaml \
#     --quiz_collection quizzes_Jan2026 \
#     --use_target_batches \
#     --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/claude-opus-4-5-20251101/accumulated_mapped_topics.json \
#     --topic_map data/topic_to_abms.yaml \
#     --plan_targets_only 

# Gemini example
uv run quizbench/run_batch_gen_quiz.py \
    --config configs/config_gemini.yaml \
    --quiz_collection quizzes_Jan2026 \
    --use_target_batches \
    --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/claude-opus-4-5-20251101/accumulated_mapped_topics.json \
    --topic_map data/topic_to_abms.yaml \
    --plan_targets_only

# (Optional) Use the existing consolidator quizbench/refresh_quizbench_manifest.py. (it deterministically merges all quizbench_manifest*.json, dedupes by quiz_id, sorts by quiz_id, and backs up
  # the output file before overwriting).
  # - Preview (no writes):
  #   python3 quizbench/refresh_quizbench_manifest.py --runs_root eval_results/quizbench/quizzes_Jan2026/runs/kimi-k2-thinking/kimi-k2-dryrun --quiz_batch_tag Jan2026 --dry_run
  # - Write consolidated quizbench_manifest_Jan2026.json (with backup):
  #   python3 quizbench/refresh_quizbench_manifest.py --runs_root eval_results/quizbench/quizzes_Jan2026/runs/kimi-k2-thinking/kimi-k2-dryrun --quiz_batch_tag Jan2026

# Step 2a: Round robin LLM-as-judge
./scripts/eval_judge_round_robin.sh Jan2026

# Step 2b: Or for select quizzes on one judge where quiz_id_1 does not contain .json suffix
uv run python -m quizbench.run_eval_judge \
    --runs_root eval_results/quizbench/quizzes_Jan2026/runs/<generator_dir> \
    --manifest_path quizbench_manifest_Jan2026*.json \
    --judge_model <judge_model> \
    --only_quiz_ids_csv "quiz_id_1,quiz_id_2,quiz_id_3"

# Step 2c: Or for select quizzes on select judges where quiz_id_1 does not contain .json suffix KIMI example
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/kimi-k2-thinking"   ONLY_QUIZ_IDS_CSV="20251225T234023630Z_kimi-k2-thinking_seed133"   QUIZ_BATCH_TAG=Jan2026   bash scripts/eval_judge_subset.sh

# GPT example
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/gpt-5.1-2025-11-13"   ONLY_QUIZ_IDS_CSV="20251223T225810101Z_gpt-5.1-2025-11-13_seed128,20251223T225810101Z_gpt-5.1-2025-11-13_seed129,20251223T225810101Z_gpt-5.1-2025-11-13_seed130,20251223T225810101Z_gpt-5.1-2025-11-13_seed131"   QUIZ_BATCH_TAG=Jan2026   bash scripts/eval_judge_subset.sh

# Gemini example
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/gemini-3-pro-preview"   ONLY_QUIZ_IDS_CSV="20251223T232137723Z_gemini-3-pro-preview_seed128,20251223T232137723Z_gemini-3-pro-preview_seed129,20251223T232137723Z_gemini-3-pro-preview_seed130"   QUIZ_BATCH_TAG=Jan2026   bash scripts/eval_judge_subset.sh

# Deepseek example
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/deepseek-v3.2"   ONLY_QUIZ_IDS_CSV="20251225T200227925Z_deepseek-v3.2_seed134"   QUIZ_BATCH_TAG=Jan2026   bash scripts/eval_judge_subset.sh

# Grok
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/grok-4-0709"   ONLY_QUIZ_IDS_CSV="20251225T231536524Z_grok-4-0709_seed133"   QUIZ_BATCH_TAG=Jan2026   bash scripts/eval_judge_subset.sh

# Step 2d: Or select quiz with one judge model
RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/gemini-3-pro-preview" \
  ONLY_QUIZ_IDS_CSV="20251223T232137723Z_gemini-3-pro-preview_seed128,20251223T232137723Z_gemini-3-pro-preview_seed129,20251223T232137723Z_gemini-3-pro-preview_seed130" \
  QUIZ_BATCH_TAG=Jan2026 \
  JUDGE_MODELS_CSV="kimi-k2-thinking" \
  bash scripts/eval_judge_subset.sh

RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/gemini-3-pro-preview"   ONLY_QUIZ_IDS_CSV="20251223T232137723Z_gemini-3-pro-preview_seed128,20251223T232137723Z_gemini-3-pro-preview_seed129,20251223T232137723Z_gemini-3-pro-preview_seed130"   QUIZ_BATCH_TAG=Jan2026   JUDGE_MODELS_CSV="kimi-k2-thinking"   bash scripts/eval_judge_subset.sh

# Step 3 (Optional): Check questions
QUIZ_BATCH_TAG=Jan2026 bash scripts/aggregate_filtered_results.sh

# Step 4: Consolidate
uv run quizbench/refresh_quizbench_manifest.py --runs_root eval_results/quizbench/quizzes_Jan2026/runs --quiz_batch_tag Jan2026

# Step 5: Rewrite accumulated_topic_distribution.json
uv run quizbench/apply_topic_mapping.py \
    --runs_root eval_results/quizbench/quizzes_Jan2026/runs \
    --targets_csv data/ABMS_specialties.csv \
    --operation accumulate \
    --quizzes_dir quizzes/quizzes_Jan2026 \
    --topic_map data/topic_to_abms.yaml




###########


# Dry-run feasibility: 
uv run quizbench/build_abms_valid_subset.py --subset_tag ABMS20260101 --source_runs_roots_csv \
    eval_results/quizbench/quizzes_Jan2026/runs --dry_run

# Build artifacts: 
uv run quizbench/build_abms_valid_subset.py --subset_tag ABMS20260101 --source_runs_roots_csv eval_results/quizbench/quizzes_Jan2026/runs

# Eval: 
SUBSET_TAG=ABMS20260101 bash scripts/eval_round_robin_subset.sh




###########
# Generate website
OUT_CSV_BY_MODEL=tmp/agg_majority_by_model_ABMS20260101.csv bash scripts/aggregate_filtered_results.sh ABMS20260101

uv run quizbench/generate_leaderboard_site.py --csv-by-model "/tmp/agg_majority_by_model_ABMS20260101.csv"

