#!/usr/bin/env bash

uv run quizbench/run_batch_gen_quiz.py --config configs/config_grok.yaml

uv run quizbench/run_batch_gen_quiz.py --config configs/config_anthropic.yaml


uv run python -m quizbench.run_eval \
    --runs_root eval_results/quizbench/runs/openai/ \
    --eval_model gpt-5-nano-2025-08-07 \
    --only_generator_models_csv gpt-5.1-2025-11-13


uv run python -m quizbench.run_eval \
    --runs_root eval_results/quizbench/runs/gemini/ \
    --eval_model gpt-5-nano-2025-08-07 \
    --only_generator_models_csv gemini-3-pro-preview


uv run python -m quizbench.run_eval_judge \
    --runs_root eval_results/quizbench/runs/openai/ \
    --judge_model gpt-4o

# Per-quiz judge example
uv run python -m quizbench.eval_quiz_judge \
    --quiz_file quizzes/20251129T003834777Z_gpt-5-nano-2025-08-07_seed123.jsonl \
    --judge_model gpt-4o
    
