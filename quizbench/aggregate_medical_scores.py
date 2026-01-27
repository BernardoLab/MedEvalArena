#!/usr/bin/env python3
"""
Aggregate QuizBench judge `medical_accuracy_score` and print means.

This script scans `*_judge_result.json` files under a runs root (default:
`eval_results/quizbench/runs`) and computes the mean medical score (1–5) across
all *valid* per-question judge outputs.

Examples:
  python3 quizbench/aggregate_medical_scores.py
  python3 quizbench/aggregate_medical_scores.py --group_by generator
  python3 quizbench/aggregate_medical_scores.py --group_by generator_judge
  python3 quizbench/aggregate_medical_scores.py --judges gpt-5.1-2025-11-13
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_JUDGE_RESULT_SUFFIX = "_judge_result.json"


@dataclass
class ScoreAccumulator:
    score_sum: int = 0
    score_count: int = 0
    quiz_dirs: set[Path] = field(default_factory=set)

    def add(self, score: int, *, quiz_dir: Path) -> None:
        self.score_sum += int(score)
        self.score_count += 1
        self.quiz_dirs.add(quiz_dir)

    def mean(self) -> float:
        if self.score_count <= 0:
            return float("nan")
        return self.score_sum / float(self.score_count)


def _parse_judge_model_from_filename(filename: str) -> str | None:
    if not filename.endswith(_JUDGE_RESULT_SUFFIX):
        return None
    judge = filename[: -len(_JUDGE_RESULT_SUFFIX)]
    return judge.strip() or None


def _infer_generator_from_result_path(result_path: Path, runs_root: Path) -> str:
    """
    Best-effort generator model inference from on-disk layout.

    Expected layout for QuizBench end-to-end runs:
      <runs_root>/<generator_model>/<quiz_id>/<judge_model>_judge_result.json
    """
    quiz_dir = result_path.parent
    generator_dir = quiz_dir.parent
    if generator_dir.resolve() != runs_root.resolve():
        name = generator_dir.name.strip()
        if name:
            return name

    # Fallback: parse from quiz_id naming convention:
    #   <timestamp>_<sanitized_generator_model>_seed<seed>
    quiz_id = quiz_dir.name
    parts = quiz_id.split("_")
    if len(parts) >= 3 and parts[-1].startswith("seed"):
        gen = "_".join(parts[1:-1]).strip()
        if gen:
            return gen

    return "unknown"


def _iter_judge_result_paths(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists() or not runs_root.is_dir():
        return []
    return runs_root.rglob(f"*{_JUDGE_RESULT_SUFFIX}")


def _iter_valid_medical_scores(result_path: Path) -> Iterable[int]:
    try:
        with result_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return []

    if not isinstance(rows, list):
        return []

    out: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        # Skip explicitly invalid outputs; tolerate missing flags for older files.
        if row.get("judge_output_valid") is False:
            continue

        score: Any = row.get("judge_medical_accuracy_score")
        if score is None:
            judge_json = row.get("judge_json")
            if isinstance(judge_json, dict):
                score = judge_json.get("medical_accuracy_score")

        try:
            score_int = int(score)
        except (TypeError, ValueError):
            continue
        if not 1 <= score_int <= 5:
            continue
        out.append(score_int)

    return out


def _print_single_key_table(
    rows: list[tuple[str, ScoreAccumulator]],
    *,
    label: str,
) -> None:
    if not rows:
        print("[WARN] No medical_accuracy_score values found to aggregate.")
        return

    name_width = max(len(label), max(len(name) for name, _ in rows))
    header = (
        f"{label:<{name_width}}  "
        f"{'Quizzes':>7}  "
        f"{'Items':>7}  "
        f"{'Mean':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, acc in rows:
        print(
            f"{name:<{name_width}}  "
            f"{len(acc.quiz_dirs):7d}  "
            f"{acc.score_count:7d}  "
            f"{acc.mean():6.3f}"
        )


def _print_pair_key_table(
    rows: list[tuple[tuple[str, str], ScoreAccumulator]],
    *,
    left_label: str,
    right_label: str,
) -> None:
    if not rows:
        print("[WARN] No medical_accuracy_score values found to aggregate.")
        return

    left_width = max(len(left_label), max(len(k[0]) for k, _ in rows))
    right_width = max(len(right_label), max(len(k[1]) for k, _ in rows))
    header = (
        f"{left_label:<{left_width}}  "
        f"{right_label:<{right_width}}  "
        f"{'Quizzes':>7}  "
        f"{'Items':>7}  "
        f"{'Mean':>6}"
    )
    print(header)
    print("-" * len(header))

    for (left, right), acc in rows:
        print(
            f"{left:<{left_width}}  "
            f"{right:<{right_width}}  "
            f"{len(acc.quiz_dirs):7d}  "
            f"{acc.score_count:7d}  "
            f"{acc.mean():6.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate medical_accuracy_score across QuizBench judge outputs."
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root directory to scan for *_judge_result.json files.",
    )
    ap.add_argument(
        "--group_by",
        type=str,
        choices=["judge", "generator", "generator_judge"],
        default="judge",
        help=(
            "Aggregation grouping: by judge model, by generator model "
            "(inferred from directory layout), or by generator+judge pair."
        ),
    )
    ap.add_argument(
        "--judges",
        nargs="*",
        default=None,
        help="Optional list of judge model names to include (exact match).",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser()
    judge_filter = {j.strip() for j in (args.judges or []) if j.strip()} or None

    if args.group_by == "generator_judge":
        accs: dict[tuple[str, str], ScoreAccumulator] = defaultdict(ScoreAccumulator)
    else:
        accs: dict[str, ScoreAccumulator] = defaultdict(ScoreAccumulator)

    n_files = 0
    for result_path in _iter_judge_result_paths(runs_root):
        judge_model = _parse_judge_model_from_filename(result_path.name)
        if judge_model is None:
            continue
        if judge_filter is not None and judge_model not in judge_filter:
            continue

        quiz_dir = result_path.parent
        generator_model = _infer_generator_from_result_path(result_path, runs_root)
        scores = _iter_valid_medical_scores(result_path)
        if not scores:
            continue

        n_files += 1
        if args.group_by == "judge":
            key = judge_model
        elif args.group_by == "generator":
            key = generator_model
        else:
            key = (generator_model, judge_model)

        acc = accs[key]
        for s in scores:
            acc.add(s, quiz_dir=quiz_dir)

    if n_files == 0:
        print(f"[WARN] No usable judge result files found under: {runs_root}")
        return

    if args.group_by == "generator_judge":
        rows = sorted(
            accs.items(),
            key=lambda kv: (-kv[1].mean(), kv[0][0], kv[0][1]),
        )
        _print_pair_key_table(
            rows,
            left_label="Generator",
            right_label="Judge",
        )
    elif args.group_by == "generator":
        rows = sorted(accs.items(), key=lambda kv: (-kv[1].mean(), kv[0]))
        _print_single_key_table(rows, label="Generator")
    else:
        rows = sorted(accs.items(), key=lambda kv: (-kv[1].mean(), kv[0]))
        _print_single_key_table(rows, label="Judge")


if __name__ == "__main__":
    main()

