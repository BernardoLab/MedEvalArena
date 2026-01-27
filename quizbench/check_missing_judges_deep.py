#!/usr/bin/env python3
"""
Deep judge-output validity scan for QuizBench runs.

Mirrors inter_judge_reliability.py filtering for logical verdict units.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.judge_utils import build_unit_id, extract_verdict, is_judge_output_valid  # noqa: E402


def _load_json_list(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _format_counter(counts: Counter[Any]) -> str:
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return ", ".join(parts) if parts else "none"


def _format_missing_summary(missing: Counter[tuple[str, str]]) -> str:
    if not missing:
        return "none"
    parts = []
    for (judge, status), count in sorted(
        missing.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        parts.append(f"{judge}:{status}={count}")
    return ", ".join(parts)


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    return [
        p for p in sorted(runs_root.iterdir()) if p.is_dir() and "seed" in p.name
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deep judge-output validity scan for QuizBench runs."
    )
    ap.add_argument("--runs_root", type=str, required=True, help="Generator run root.")
    ap.add_argument(
        "--judge_models",
        nargs="+",
        required=True,
        help="Judge model names to include.",
    )
    ap.add_argument(
        "--max_units",
        type=int,
        default=50,
        help="Max units to list for missing verdicts (0 disables listing).",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    judge_models = [m.strip() for m in args.judge_models if str(m).strip()]
    if not judge_models:
        raise SystemExit("[FATAL] No judge models provided.")
    if not runs_root.exists():
        raise SystemExit(f"[FATAL] runs_root does not exist: {runs_root}")

    run_dirs = _iter_run_dirs(runs_root)
    if not run_dirs:
        print(f"[WARN] No run dirs found under {runs_root}")
        return

    unit_status: dict[str, dict[str, str]] = defaultdict(dict)
    units_seen: set[str] = set()
    missing_unit_id = 0
    invalid_output_counts: Counter[str] = Counter()
    missing_verdict_counts: Counter[str] = Counter()

    for run_dir in run_dirs:
        quiz_id = run_dir.name
        for judge_model in judge_models:
            result_path = run_dir / f"{judge_model}_judge_result.json"
            if not result_path.exists():
                continue
            rows = _load_json_list(result_path)
            for row in rows:
                if not is_judge_output_valid(row):
                    unit_id = build_unit_id(row, quiz_id)
                    if unit_id is None:
                        missing_unit_id += 1
                        continue
                    units_seen.add(unit_id)
                    unit_status[unit_id][judge_model] = "invalid_output"
                    invalid_output_counts[judge_model] += 1
                    continue

                unit_id = build_unit_id(row, quiz_id)
                if unit_id is None:
                    missing_unit_id += 1
                    continue
                units_seen.add(unit_id)

                verdict = extract_verdict(row)
                if verdict is None:
                    unit_status[unit_id][judge_model] = "missing_verdict"
                    missing_verdict_counts[judge_model] += 1
                    continue

                unit_status[unit_id][judge_model] = "verdict"

    units_with_verdict = {
        uid
        for uid, per_judge in unit_status.items()
        if any(status == "verdict" for status in per_judge.values())
    }

    missing_rows: Counter[str] = Counter()
    for unit_id in units_with_verdict:
        per_judge = unit_status.get(unit_id, {})
        for judge_model in judge_models:
            if judge_model not in per_judge:
                per_judge[judge_model] = "missing_row"
                missing_rows[judge_model] += 1

    raters_hist: Counter[int] = Counter()
    missing_reason_counts: Counter[tuple[str, str]] = Counter()
    units_missing: list[tuple[str, dict[str, str]]] = []
    for unit_id in units_with_verdict:
        per_judge = unit_status.get(unit_id, {})
        raters = sum(1 for status in per_judge.values() if status == "verdict")
        raters_hist[raters] += 1
        missing = {j: s for j, s in per_judge.items() if s != "verdict"}
        if missing:
            units_missing.append((unit_id, missing))
            for judge_model, status in missing.items():
                missing_reason_counts[(judge_model, status)] += 1

    print("[DEEP] Judge-output validity scan (inter_judge_reliability-style)")
    print(f"  run_dirs: {len(run_dirs)}")
    print(f"  judge_models: {len(judge_models)}")
    print(f"  rows_missing_unit_id: {missing_unit_id}")
    print(
        "  logical verdict units: "
        f"{len(units_with_verdict)} (units by #raters: {_format_counter(raters_hist)})"
    )
    print(
        f"  invalid_output by judge: {_format_counter(invalid_output_counts)}"
    )
    print(
        f"  missing_verdict by judge: {_format_counter(missing_verdict_counts)}"
    )
    print(f"  missing_row by judge: {_format_counter(missing_rows)}")
    print(
        f"  missing verdict reasons: {_format_missing_summary(missing_reason_counts)}"
    )

    if args.max_units > 0 and units_missing:
        print(f"  units with missing verdicts (showing {min(len(units_missing), args.max_units)}):")
        for unit_id, missing in sorted(units_missing)[: args.max_units]:
            missing_bits = ", ".join(f"{j}:{s}" for j, s in sorted(missing.items()))
            print(f"    {unit_id} -> {missing_bits}")
        if len(units_missing) > args.max_units:
            print(f"  ... {len(units_missing) - args.max_units} more")


if __name__ == "__main__":
    main()
