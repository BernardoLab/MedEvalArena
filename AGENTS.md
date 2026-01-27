# Repository Guidelines

## Project Structure & Module Organization
The repo centers on `quizbench/` (quiz generation/eval logic), bash entrypoints in `v1_scripts/`, `v2_scripts/`, and prompts kept under `cot_prompt_lib/`. Published datasets reside in `data_medARCv1/`, while model outputs go to `results/` and `eval_results/quizbench/runs/`. Keep experimental notebooks or metrics in `eval_uq_results/` instead of polluting source folders.

## Build, Test, and Development Commands
Install deps via `python -m pip install -r requirements.txt`, adding `python -m pip install -r quizbench/requirements_quizbench.txt` before touching QuizBench modules. Run `NUM_QUIZZES=2 NUM_QUESTIONS=5 bash v1_scripts/eval_quizbench_triplet.sh` to sanity-check generator, evaluators, and aggregation. For manual loops, chain `python quizbench/generate_quiz.py`, `python quizbench/eval_quiz.py`, `python quizbench/aggregate_results.py`, then recompute stats through `python compute_accuracy.py results/<run_dir>`.

## Coding Style & Naming Conventions
Stick to PEP 8: four-space indent, `snake_case` functions, docstrings for new CLIs, and type hints when sharing payloads between modules. Keep long prompts or rubrics in `cot_prompt_lib/` or JSON fixtures rather than inline literals. Bash launchers stay `eval_<model>.sh` and expose env overrides at the top; extend them by mirroring existing guard/cleanup blocks.

## Testing Guidelines
No automated suite exists, so run a minimal QuizBench cycle (`NUM_QUIZZES=1 NUM_QUESTIONS=3 bash v1_scripts/eval_quizbench_triplet.sh`) before submitting. When touching scoring or aggregation, rerun `python quizbench/run_quizbench.py --num_quizzes 3 --num_questions 5` and record any accuracy deltas plus updated CSV paths. For standalone helpers, execute them against toy inputs and paste the observed stdout or artifact path into the PR.

## Commit & Pull Request Guidelines
Commits mirror current history—short, imperative summaries such as “Fix repository name”—and expanded rationale belongs in the body. Pull requests should explain motivation, list the commands executed, and reference any dataset or leaderboard issue while attaching accuracy tables or result file paths. Request reviewers aligned with the area (`quizbench`, `v2_scripts`, UQ) and track future work via GitHub issues instead of inline TODOs.

## Security & Configuration Tips
Never hardcode secrets; export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` (or source an ignored `.env`) before running scripts. Check bash files for stray `set -x` or echo statements that might leak headers, and scrub `results/` artifacts before sharing. Large datasets such as `data_medARCv1/` stay outside Git—reference institutional storage rather than committing PHI-like files.
