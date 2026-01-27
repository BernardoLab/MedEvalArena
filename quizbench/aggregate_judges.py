#!/usr/bin/env python3
"""
Aggregate judge summaries from QuizBench runs and print a compact table.

By default, scans `eval_results/quizbench/runs/` and groups results by
generator_model and judge_model. Optional debugging mode inspects per-item
judge results to help troubleshoot parsing failures.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Ensure package imports succeed whether run from repo root or quizbench/ dir
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.utils import extract_json_block
from quizbench.manifest_utils import resolve_quizbench_manifest_path


DEFAULT_ENSEMBLE_JUDGES: List[str] = [
    "claude-opus-4-5-20251101",
    "gemini-3-pro-preview",
    "gpt-5.1-2025-11-13",
    "grok-4-0709",
    "deepseek-v3.2",
    "kimi-k2-thinking"
]


def load_quiz_generators(runs_root: Path) -> Dict[str, str]:
    """
    Build a mapping from quiz_id -> generator_model using quizbench manifests.
    """
    mapping: Dict[str, str] = {}

    for generator_dir in runs_root.iterdir():
        if not generator_dir.is_dir():
            continue

        try:
            manifest_path = resolve_quizbench_manifest_path(generator_dir)
        except SystemExit:
            continue

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        quizzes = manifest.get("quizzes", [])
        for q in quizzes:
            quiz_id = q.get("quiz_id")
            generator_model = q.get("generator_model") or manifest.get(
                "generator_model", generator_dir.name
            )
            if quiz_id:
                mapping[quiz_id] = str(generator_model)

    return mapping


def iter_judge_summaries(
    runs_root: Path,
) -> Iterable[Tuple[str, Dict[str, Any], Path]]:
    """
    Yield (quiz_id, summary_row, run_dir) triples from all summary_all_judges.json
    files under runs_root.
    """
    for generator_dir in runs_root.iterdir():
        if not generator_dir.is_dir():
            continue

        for run_dir in generator_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Prefer explicit per-judge *_judge_summary.json files when present.
            judge_summary_paths = sorted(run_dir.glob("*_judge_summary.json"))
            if judge_summary_paths:
                for js_path in judge_summary_paths:
                    try:
                        with js_path.open("r", encoding="utf-8") as f:
                            row = json.load(f)
                    except Exception:  # noqa: BLE001
                        continue

                    quiz_id = row.get("quiz_id") or run_dir.name
                    yield quiz_id, row, run_dir
                continue

            # Fallback to legacy summary_all_judges.json (list of rows).
            summary_path = run_dir / "summary_all_judges.json"
            if not summary_path.exists():
                continue

            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception:  # noqa: BLE001
                continue

            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                quiz_id = row.get("quiz_id") or run_dir.name
                yield quiz_id, row, run_dir


def aggregate(
    runs_root: Path,
    judge_model_filter: str | None,
    generator_model_filter: str | None,
    allowed_judges: set[str] | None = None,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], set[Path]],
]:
    """
    Aggregate judge summary rows into per-(generator, judge) stats.

    If allowed_judges is not None, only include rows whose judge model is in
    that set. Also returns the set of run directories that contributed to each
    group.
    """
    quiz_to_gen = load_quiz_generators(runs_root)

    stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "n_quizzes": 0,
            "total_items": 0,
            "total_valid": 0,
            "total_invalid": 0,
            "total_pass": 0,
            "total_fail": 0,
            "total_safety_true": 0,
            "score_sum": 0.0,
            "score_count": 0,
            "fail_reason_counts": Counter(),
        }
    )
    group_runs: Dict[Tuple[str, str], set[Path]] = defaultdict(set)

    for quiz_id, row, run_dir in iter_judge_summaries(runs_root):
        judge_model = str(row.get("model", "")).strip()
        if allowed_judges is not None and judge_model not in allowed_judges:
            continue
        generator_model = quiz_to_gen.get(quiz_id, "unknown")

        if judge_model_filter and judge_model != judge_model_filter:
            continue
        if generator_model_filter and generator_model != generator_model_filter:
            continue

        key = (generator_model, judge_model)
        agg = stats[key]
        group_runs[key].add(run_dir)

        n_items = int(row.get("n_items", 0) or 0)
        n_valid = int(row.get("n_valid_responses", 0) or 0)
        n_invalid = int(row.get("n_invalid_responses", 0) or 0)
        n_pass = int(row.get("n_pass", 0) or 0)
        n_fail = int(row.get("n_fail", 0) or 0)
        n_safety_true = int(row.get("n_safety_flag_true", 0) or 0)

        agg["n_quizzes"] += 1
        agg["total_items"] += n_items
        agg["total_valid"] += n_valid
        agg["total_invalid"] += n_invalid
        agg["total_pass"] += n_pass
        agg["total_fail"] += n_fail
        agg["total_safety_true"] += n_safety_true

        score_mean = row.get("medical_score_mean")
        if score_mean is not None and n_valid > 0:
            agg["score_sum"] += float(score_mean) * n_valid
            agg["score_count"] += n_valid

        fr_counts = row.get("fail_reason_counts") or {}
        for reason, count in fr_counts.items():
            agg["fail_reason_counts"][reason] += int(count or 0)

    return stats, group_runs


def print_table(stats: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    if not stats:
        print("No judge summaries found under the specified runs_root.")
        return

    header = (
        "Generator",
        "Judge",
        "Quizzes",
        "Items",
        "Parsed",
        "ParseErr",
        "Pass%",
        "MedScore",
        "Safety%",
    )
    print(
        f"{header[0]:<18} {header[1]:<28} "
        f"{header[2]:>7} {header[3]:>7} {header[4]:>7} {header[5]:>8} "
        f"{header[6]:>7} {header[7]:>9} {header[8]:>8}"
    )
    print("-" * 95)

    for (generator_model, judge_model), agg in sorted(stats.items()):
        total_items = agg["total_items"]
        total_valid = agg["total_valid"]
        total_invalid = agg["total_invalid"]
        total_pass = agg["total_pass"]
        total_safety_true = agg["total_safety_true"]

        pass_rate = (100.0 * total_pass / total_valid) if total_valid else 0.0
        safety_rate = (
            100.0 * total_safety_true / total_valid if total_valid else 0.0
        )
        med_score = (
            agg["score_sum"] / agg["score_count"] if agg["score_count"] > 0 else 0.0
        )

        print(
            f"{generator_model:<18} {judge_model:<28} "
            f"{agg['n_quizzes']:7d} {total_items:7d} {total_valid:7d} {total_invalid:8d} "
            f"{pass_rate:7.1f} {med_score:9.3f} {safety_rate:8.1f}"
        )

    print(
        "\nLogical validity / fail-reason codes (aggregated across items):\n"
        "  NA = logically valid (PASS)\n"
        "  C = contradiction, N = no defensible answer,\n"
        "  M = multiple defensible answers, U = underspecified, K = miskeyed\n"
    )
    for (generator_model, judge_model), agg in sorted(stats.items()):
        fr = agg["fail_reason_counts"]
        if not fr:
            continue
        reasons = ", ".join(f"{r}={c}" for r, c in sorted(fr.items()))
        print(f"- {generator_model} / {judge_model}: {reasons}")


def debug_parse_errors(
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    group_runs: Dict[Tuple[str, str], set[Path]],
    max_items: int,
) -> None:
    """
    Print sample parse errors and extracted JSON blocks for groups
    where all judge responses are currently invalid.
    """
    print("\nDebugging parse errors for groups with zero valid responses:\n")
    any_group = False

    for (generator_model, judge_model), agg in sorted(stats.items()):
        if agg["total_valid"] > 0:
            continue

        run_dirs = sorted(group_runs.get((generator_model, judge_model), []))
        if not run_dirs:
            continue

        any_group = True
        print(f"== Generator: {generator_model} | Judge: {judge_model} ==")
        printed = 0

        for run_dir in run_dirs:
            result_path = run_dir / f"{judge_model}_judge_result.json"
            if not result_path.exists():
                continue

            with result_path.open("r", encoding="utf-8") as f:
                rows = json.load(f)

            for row in rows:
                if row.get("judge_output_valid"):
                    continue

                parse_error = row.get("judge_parse_error")
                raw = row.get("judge_model_outputs_raw") or ""
                json_block = extract_json_block(str(raw))

                try:
                    json.loads(json_block)
                    json_ok = True
                    json_exc = ""
                except Exception as exc:  # noqa: BLE001
                    json_ok = False
                    json_exc = str(exc)

                qid = row.get("question_id")
                print(f"- question_id: {qid}")
                print(f"  judge_parse_error: {parse_error}")
                print(f"  extracted_json_block (trunc): {json_block[:200]!r}")
                print(
                    f"  json.loads success: {json_ok}"
                    f"{'' if json_ok else ' | ' + json_exc}"
                )
                print(f"  raw_prefix: {str(raw)[:200]!r}")
                printed += 1

                if printed >= max_items:
                    break

            if printed >= max_items:
                break

        if printed == 0:
            print("  (no invalid records with parse errors found)")

        print()

    if not any_group:
        print("No groups with zero valid responses found; nothing to debug.")


def filter_by_judge(
    run_dir: Path,
    judge_models: Iterable[str],
    *,
    min_medical_score: int | None = None,
    require_logical_valid: bool = True,
    logical_mode: str = "all",
) -> set[str] | None:
    """
    Return the set of question_ids that pass judge-based filters for a run.

    When require_logical_valid is False, filters are applied independently for
    each judge and then intersected across the ensemble:
      - judge_output_valid must be True, and
      - the per-judge medical_accuracy_score must meet min_medical_score
        (if specified).

    When require_logical_valid is True, verdict-based logical validity is also
    enforced. The logical_mode parameter controls how judge verdicts are
    aggregated across the ensemble:
      - logical_mode == "all" (default): a question is kept only if every judge
        with a result for that question has verdict PASS and meets the score
        threshold (intersection semantics, the historical behavior).
      - logical_mode == "majority": for each question, consider only judges
        whose outputs are valid and whose scores meet the threshold; the
        question is kept if a strict majority of those judges have verdict
        PASS. Judges with invalid outputs or missing results are ignored for
        majority counting.

    If no judge_result files are found for any of judge_models, returns None
    to signal that no judge-based filtering should be applied for this run.
    """
    judge_models = [m.strip() for m in judge_models if str(m).strip()]
    if not judge_models:
        return None

    logical_mode = (logical_mode or "all").lower()
    if logical_mode not in {"all", "majority"}:
        raise ValueError(f"Unsupported logical_mode: {logical_mode!r}")

    any_judge = False
    allowed_across_judges: set[str] | None = None

    # In majority mode we aggregate counts per question_id instead of
    # intersecting per-judge sets, but only when logical validity is required.
    majority_pass_counts: dict[str, int] = {}
    majority_total_counts: dict[str, int] = {}

    for judge_model in judge_models:
        result_path = run_dir / f"{judge_model}_judge_result.json"
        if not result_path.exists():
            continue

        try:
            with result_path.open("r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:  # noqa: BLE001
            continue

        if not isinstance(rows, list):
            continue

        any_judge = True
        allowed_for_judge: set[str] = set()
        for row in rows:
            qid = row.get("question_id")
            if not qid:
                continue

            if not row.get("judge_output_valid"):
                continue

            score = row.get("judge_medical_accuracy_score")
            verdict = row.get("judge_verdict")

            if score is None or verdict is None:
                judge_json = row.get("judge_json") or {}
                if score is None:
                    score = judge_json.get("medical_accuracy_score")
                if verdict is None:
                    verdict = judge_json.get("verdict")

            score_int: int | None
            try:
                score_int = int(score) if score is not None else None
            except (TypeError, ValueError):
                score_int = None

            # Score-based filter (applied in all modes when specified).
            if min_medical_score is not None:
                if score_int is None or score_int < min_medical_score:
                    continue

            verdict_str = verdict.strip().upper() if isinstance(verdict, str) else ""
            is_pass = verdict_str == "PASS"
            qid_str = str(qid)

            if not require_logical_valid:
                # Historical behavior when logical validity is disabled:
                # intersect per-judge sets based only on score + validity.
                allowed_for_judge.add(qid_str)
            elif logical_mode == "all":
                # Require PASS for each judge (intersection semantics).
                if is_pass:
                    allowed_for_judge.add(qid_str)
            else:  # logical_mode == "majority" and require_logical_valid
                # In majority mode, count how many judges PASS vs how many are
                # eligible (valid output + score >= threshold) for this qid.
                majority_total_counts[qid_str] = (
                    majority_total_counts.get(qid_str, 0) + 1
                )
                if is_pass:
                    majority_pass_counts[qid_str] = (
                        majority_pass_counts.get(qid_str, 0) + 1
                    )

        if not require_logical_valid or logical_mode == "all":
            if allowed_across_judges is None:
                allowed_across_judges = allowed_for_judge
            else:
                allowed_across_judges &= allowed_for_judge

    if not any_judge:
        return None

    if require_logical_valid and logical_mode == "majority":
        # Keep questions where a strict majority of eligible judges PASS.
        allowed_majority: set[str] = set()
        for qid, total in majority_total_counts.items():
            passed = majority_pass_counts.get(qid, 0)
            if passed > total / 2.0:
                allowed_majority.add(qid)
        return allowed_majority

    return allowed_across_judges or set()


def mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and (sample) standard deviation for a list of floats.

    Returns (mean, std). For n == 0, both are NaN; for n == 1, std is NaN.
    """
    if not values:
        return float("nan"), float("nan")

    n = len(values)
    mean_val = sum(values) / n
    if n == 1:
        return mean_val, float("nan")

    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return mean_val, math.sqrt(var)


def aggregate_across_judges(
    runs_root: Path,
    ensemble_judges: List[str],
    generator_model_filter: str | None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-quiz judge summaries across multiple judge models.

    For each generator_model, collects per-quiz medical_score_mean, pass rate,
    and safety rate for all rows whose judge model is in ensemble_judges, then
    computes mean and standard deviation for each metric.
    """
    if not ensemble_judges:
        return {}

    quiz_to_gen = load_quiz_generators(runs_root)
    ensemble_set = {m.strip() for m in ensemble_judges if m.strip()}
    if not ensemble_set:
        return {}

    per_generator: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "quiz_ids": set(),
            "med_scores": [],
            "pass_rates": [],
            "safety_rates": [],
        }
    )

    for quiz_id, row, _ in iter_judge_summaries(runs_root):
        judge_model = str(row.get("model", "")).strip()
        if judge_model not in ensemble_set:
            continue

        generator_model = quiz_to_gen.get(quiz_id, "unknown")
        if generator_model_filter and generator_model != generator_model_filter:
            continue

        n_valid = int(row.get("n_valid_responses", 0) or 0)
        n_pass = int(row.get("n_pass", 0) or 0)
        n_safety_true = int(row.get("n_safety_flag_true", 0) or 0)
        if n_valid <= 0:
            continue

        med_score = row.get("medical_score_mean")
        pass_rate = n_pass / n_valid
        safety_rate = n_safety_true / n_valid

        slot = per_generator[generator_model]
        slot["quiz_ids"].add(quiz_id)
        slot["pass_rates"].append(pass_rate)
        slot["safety_rates"].append(safety_rate)
        if med_score is not None:
            try:
                slot["med_scores"].append(float(med_score))
            except (TypeError, ValueError):
                pass

    summary: Dict[str, Dict[str, float]] = {}
    for generator_model, data in per_generator.items():
        med_mean, med_std = mean_std(data["med_scores"])
        pass_mean, pass_std = mean_std(data["pass_rates"])
        safety_mean, safety_std = mean_std(data["safety_rates"])

        summary[generator_model] = {
            "n_quizzes": float(len(data["quiz_ids"])),
            "n_judge_evals": float(len(data["med_scores"])),
            "med_score_mean": med_mean,
            "med_score_std": med_std,
            "pass_rate_mean": pass_mean,
            "pass_rate_std": pass_std,
            "safety_rate_mean": safety_mean,
            "safety_rate_std": safety_std,
        }

    return summary


def print_ensemble_table(
    ensemble_stats: Dict[str, Dict[str, float]], ensemble_judges: List[str]
) -> None:
    """
    Print a compact table summarizing mean/std across multiple judge models
    for each generator.
    """
    if not ensemble_stats:
        print(
            "\nNo ensemble judge statistics found for the requested models; "
            "nothing to aggregate across judges."
        )
        return

    judges_label = ", ".join(sorted(ensemble_judges))
    print(
        f"\nEnsemble judge statistics (aggregated across: {judges_label})\n"
        "Per-generator mean and std are computed over per-quiz judge summaries."
    )

    header = (
        "Generator",
        "Quizzes",
        "JudgeEvals",
        "MedMean",
        "MedStd",
        "PassMean%",
        "PassStd%",
        "SafetyMean%",
        "SafetyStd%",
    )
    print(
        f"{header[0]:<18} {header[1]:>7} {header[2]:>10} "
        f"{header[3]:>9} {header[4]:>9} "
        f"{header[5]:>11} {header[6]:>11} "
        f"{header[7]:>12} {header[8]:>12}"
    )
    print("-" * 115)

    for generator_model, stats in sorted(ensemble_stats.items()):
        n_quizzes = int(stats.get("n_quizzes", 0))
        n_evals = int(stats.get("n_judge_evals", 0))

        med_mean = stats.get("med_score_mean", float("nan"))
        med_std = stats.get("med_score_std", float("nan"))
        pass_mean = stats.get("pass_rate_mean", float("nan")) * 100.0
        pass_std = stats.get("pass_rate_std", float("nan")) * 100.0
        safety_mean = stats.get("safety_rate_mean", float("nan")) * 100.0
        safety_std = stats.get("safety_rate_std", float("nan")) * 100.0

        def fmt(x: float) -> str:
            if isinstance(x, float) and math.isnan(x):
                return "   nan"
            return f"{x:9.3f}"

        print(
            f"{generator_model:<18} {n_quizzes:7d} {n_evals:10d} "
            f"{fmt(med_mean)} {fmt(med_std)} "
            f"{fmt(pass_mean)} {fmt(pass_std)} "
            f"{fmt(safety_mean)} {fmt(safety_std)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate QuizBench judge summaries and print a table."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root directory containing per-generator run folders.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Optional judge model filter (exact match).",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default=None,
        help="Optional generator model filter (exact match).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample parse errors and extracted JSON blocks.",
    )
    parser.add_argument(
        "--debug_items",
        type=int,
        default=5,
        help="Max number of invalid items to show when --debug is set.",
    )
    parser.add_argument(
        "--ensemble_judges",
        nargs="+",
        default=DEFAULT_ENSEMBLE_JUDGES,
        help=(
            "List of judge models to aggregate across when computing "
            "ensemble statistics (mean/std per generator). "
            "Defaults to grok-4-0709, gemini-3-pro-preview, gpt-5.1-2025-11-13."
        ),
    )

    args = parser.parse_args()
    runs_root = Path(args.runs_root).expanduser()

    if not runs_root.exists():
        raise SystemExit(f"runs_root does not exist: {runs_root}")

    if args.judge_model:
        allowed_judges: set[str] | None = None
    else:
        allowed_judges = {m.strip() for m in args.ensemble_judges if m.strip()}

    stats, group_runs = aggregate(
        runs_root,
        judge_model_filter=args.judge_model,
        generator_model_filter=args.generator_model,
        allowed_judges=allowed_judges,
    )
    print_table(stats)

    ensemble_stats = aggregate_across_judges(
        runs_root,
        ensemble_judges=args.ensemble_judges,
        generator_model_filter=args.generator_model,
    )
    print_ensemble_table(ensemble_stats, args.ensemble_judges)

    if args.debug:
        debug_parse_errors(stats, group_runs, max_items=args.debug_items)


if __name__ == "__main__":
    main()
