#!/usr/bin/env bash

# Statistical analyses 



# Pairwise validity heatmap
QUIZ_BATCH_TAG=Jan2026 bash scripts/aggregate_filtered_results.sh | tee agg_majority_Jan2026.log
Rscript r_analysis/validity_by_generator_heatmap.R --agg_log agg_majority_Jan2026.log --quiz_batch_tag Jan2026


# Sensitivity analysis
uv run quizbench/medical_score_sensitivity.py --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs --use_selection_reports


# Inter-judge reliability
uv run quizbench/inter_judge_reliability.py


# Evaluate topic difficulty (Fig 5)
Rscript r_analysis/topic_difficulty_only.R \
	--runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
	--outdir R_results/topic_difficulty_ABMS20260101 \
	--boot_reps 5000 \
	--perm_reps 5000 \
	--seed 2026




# Bayesian analysis
Rscript r_analysis/eb_topic_brms_mainmodels.R --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs --reuse_fit --fit_rds R_results/eb_topic_brms_20260109T013531/brms_fit.rds


Rscript r_analysis/mixed_effects_analysis.R \
    --input eval_results/quizbench/quizzes_ABMS20260101/judge_item_level.csv \
    --outdir R_results/quizzes_ABMS20260101/judge_glmm \
    --adjust holm




