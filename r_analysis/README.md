# QuizBench miss analysis (ABMS20260101)

This folder contains R scripts used for downstream statistical analysis. The primary “misses by topic” summary for the ABMS subset is produced by the Python script `r_analysis/analyze_misses.py`, which writes CSVs/PNGs under `eval_results/quizbench/quizzes_ABMS20260101/` and prints three tables to stdout.

## What “ABMS20260101” is

`ABMS20260101` is a *quota-balanced* subset of judge-valid QuizBench questions built by `quizbench/build_abms_valid_subset.py`. Each generator gets the same per-specialty quotas from `data/ABMS_specialties.csv` (total 50 questions per generator, chunked into 5×10-question quizzes).

In these subset runs, each per-question row inside `*_result.json` includes ABMS topic metadata such as:
- `abms_specialty` (canonical ABMS specialty label)
- `target_topic` (also set to the canonical specialty)

Because of that, `r_analysis/analyze_misses.py` automatically switches to “ABMS topic mode” when `--quiz_batch_tag` contains `ABMS`, and it **does not require** any `topics_*.json` files.

## Outputs produced by `r_analysis/analyze_misses.py`

Run:
```bash
uv run r_analysis/analyze_misses.py \
  --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
  --quiz_batch_tag ABMS20260101
```

### Files written (default locations)

All artifacts land in `eval_results/quizbench/quizzes_ABMS20260101/` (the parent of `runs/`):
- `eval_results/quizbench/quizzes_ABMS20260101/misses_detail.csv`: one row per **miss event** (generator, answer_model, question_id, topic, gold, pred).
- `eval_results/quizbench/quizzes_ABMS20260101/misses_by_model.csv`: miss rate summary per answer model.
- `eval_results/quizbench/quizzes_ABMS20260101/misses_by_generator_topic.csv`: “topic share of missed set” table as a CSV.
- `eval_results/quizbench/quizzes_ABMS20260101/misses_topic_miss_rate_radar.png`: 3-panel radar plot of `topic_miss_rate_%` with **ALL + per-generator** series.
- `eval_results/quizbench/quizzes_ABMS20260101/misses_topic_miss_rate_radar_all_only.png`: 3-panel radar plot of `topic_miss_rate_%` with **ALL only** (different color per panel).

Tip: capture the printed tables too:
```bash
uv run r_analysis/analyze_misses.py \
  --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
  --quiz_batch_tag ABMS20260101 \
  | tee eval_results/quizbench/quizzes_ABMS20260101/misses_report.txt
```

## Interpreting the tables

`r_analysis/analyze_misses.py` prints three tables:

### 1) Miss rate by answer model

`n_missed / n_questions` for each answer model across all included quizzes (and after any judge filtering if enabled).

### 2) Topic share of missed questions by generator

This table answers: “Among the questions that were missed, what fraction were in each topic?”

- In the default `--count_mode unique_questions`, a question is counted as “missed” for a generator if **any** answer model missed it (each question counted once per generator).
- The `ALL` row aggregates across all generators.

### 3) Topic miss rate and over/under-representation by generator

This table adds denominators and two key diagnostics:

- `topic_prevalence_%`:
  - `topic_total_questions / total_questions` (or attempts if `--count_mode events`)
  - “How common is this topic in the evaluated set?”

- `topic_miss_rate_%`:
  - `topic_misses / topic_total_questions`
  - “Of the questions in this topic, what fraction were missed (by ≥1 model in `unique_questions` mode)?”

- `over_under`:
  - `(topic_misses/total_misses) / (topic_total_questions/total_questions)`
  - Values **> 1** mean misses are *overrepresented* in this topic relative to how often the topic appears.
  - Values **< 1** mean misses are *underrepresented*.

## Radar plots (topic miss rates)

Both radar plots visualize `topic_miss_rate_%` by topic, split across 3 subplots:
1. **Surgical specialties**: anything containing “surgery” + Ob/Gyn + HEENT (Otolaryngology) + Urology + Ophthalmology
2. **Internal Medicine/Family Medicine**: everything else not in (1) or (3)
3. **Other specialties**: Anes, Derm, EM, Neuro, Rad Onc, Psych

For readability the radar axis labels are abbreviated (e.g., `HEENT`, `NSG`, `Neuro`, `Psych`, `ObGyn`, `Nephro`). The mapping lives in `r_analysis/analyze_misses.py` in `_abbreviate_topic_label`.

## Reproducing ABMS20260101 from scratch (optional)

If you only need to re-run the analysis/plots and the runs already exist, skip to the `r_analysis/analyze_misses.py` command above.

1) Build the ABMS subset (writes quizzes + per-generator manifests under `eval_results/quizbench/quizzes_ABMS20260101/runs/`):
```bash
uv run quizbench/build_abms_valid_subset.py \
  --subset_tag ABMS20260101 \
  --source_runs_roots_csv eval_results/quizbench/quizzes_Jan2026/runs
```

2) Evaluate answer models on the subset (produces `*_result.json` / `*_summary.json`):
```bash
SUBSET_TAG=ABMS20260101 bash eval_round_robin_subset.sh
```

3) Run the miss/topic analysis and generate plots:
```bash
uv run r_analysis/analyze_misses.py \
  --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
  --quiz_batch_tag ABMS20260101
```

## Using the outputs in R

From this folder (`r_analysis/`), you can load the CSVs via:
```r
library(readr)
misses <- read_csv("../eval_results/quizbench/quizzes_ABMS20260101/misses_detail.csv")
by_model <- read_csv("../eval_results/quizbench/quizzes_ABMS20260101/misses_by_model.csv")
by_topic <- read_csv("../eval_results/quizbench/quizzes_ABMS20260101/misses_by_generator_topic.csv")
```

## Paired item bootstrap (overall + pairwise differences)

This implements a simple paired item-block bootstrap to compare overall accuracy across answer models without fitting a parametric model.

Run:
```bash
Rscript r_analysis/paired_item_bootstrap.R \
  --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
  --outdir R_results/paired_bootstrap_ABMS20260101 \
  --boot_reps 5000 \
  --delta 5 \
  --seed 2026
```

Default behavior: the script restricts to **quiz-generator models** (i.e., answer models whose names match a generator directory under `runs/`). To include all answer models, add `--include_all_llms`.

Outputs written under `--outdir`:
- `overall_accuracy_bootstrap.csv`
- `pairwise_accuracy_diffs_bootstrap.csv`
- `topic_accuracy_wilson.csv` (raw topic accuracies + Wilson 95% CIs)
- `topic_accuracy_pooled_wilson.csv` (pooled across included LLMs)
- `topic_global_test_permutation.csv` (global “do topics differ?” permutation test using item-level means)
- `bootstrap_summary.txt` (text summary + “within ±Δ” flags)
