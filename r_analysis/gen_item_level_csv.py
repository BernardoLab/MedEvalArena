#!/usr/bin/env python3
"""
Generate an item-level CSV (model, quiz_id, question_id, correct) from eval outputs.

This script reads per-model eval files produced by quizbench (i.e., *_result.json),
computes correctness from pred vs answer, and writes a long-format CSV suitable
for mixed-effects analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple


CSV_HEADER = ("model", "quiz_id", "question_id", "correct")


def _iter_result_paths(runs_root: Path) -> Iterable[Path]:
    for path in runs_root.rglob("*_result.json"):
        if path.name.endswith("_judge_result.json"):
            continue
        yield path


def _rows_from_result(path: Path) -> List[Tuple[str, str, str, int]]:
    model = path.name[: -len("_result.json")]
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    rows: List[Tuple[str, str, str, int]] = []
    for item in items:
        quiz_id = item.get("quiz_id")
        question_id = item.get("question_id")
        gold = item.get("answer")
        pred = item.get("pred")
        if not quiz_id or not question_id or not gold:
            continue
        correct = 1 if pred == gold else 0
        rows.append((model, quiz_id, question_id, correct))
    return rows


def generate_item_level_csv(runs_root: Path, out_path: Path) -> int:
    rows: List[Tuple[str, str, str, int]] = []
    for result_path in _iter_result_paths(runs_root):
        rows.extend(_rows_from_result(result_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate an item-level CSV from quizbench eval outputs "
            "(*_result.json files)."
        )
    )
    ap.add_argument(
        "--runs_root",
        required=True,
        help="Root directory containing quizbench run folders.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path (e.g., eval_results/.../judge_item_level.csv).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)

    count = generate_item_level_csv(runs_root, out_path)
    print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
