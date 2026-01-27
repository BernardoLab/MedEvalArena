#!/usr/bin/env python3
"""
Run QuizBench Phase 2 multiple times for uncertainty quantification
using the quizzes listed in a quizbench_manifest*.json file.
"""

import argparse
import os
import subprocess
import sys

from quizbench.utils import ensure_dir
from quizbench.run_gen_quiz import sanitize_model_name
from quizbench.manifest_utils import resolve_quizbench_manifest_path


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run QuizBench evaluation multiple times per model "
            "to support uncertainty quantification over the same quizzes."
        )
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root containing quizbench_manifest*.json and generated quizzes.",
    )
    ap.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional explicit manifest path (defaults to newest quizbench_manifest*.json under --runs_root).",
    )
    ap.add_argument(
        "--eval_model",
        type=str,
        required=True,
        help="Quiz-taker model to evaluate (e.g. gpt-5-nano-2025-08-07).",
    )
    ap.add_argument(
        "--num_runs",
        type=int,
        default=15,
        help="Number of repeated evaluation runs.",
    )
    ap.add_argument(
        "--uq_root",
        type=str,
        default="eval_uq_results/quizbench",
        help="Root directory to hold UQ outputs (per-model/run subdirs are created inside).",
    )
    ap.add_argument(
        "--use_batch_api",
        action="store_true",
        help="Pass through to run_eval: use Batch API where supported.",
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Passed through to run_eval/eval_quiz (for batch mode).",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Max tokens for eval_quiz calls.",
    )
    ap.add_argument(
        "--only_quiz_ids_csv",
        type=str,
        default=None,
        help="Optional comma-separated quiz_ids subset to evaluate.",
    )
    return ap.parse_args()


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit(f"[FATAL] {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def main() -> None:
    args = _parse_args()

    manifest_path = resolve_quizbench_manifest_path(args.runs_root, manifest_path=args.manifest_path)

    model_slug = sanitize_model_name(args.eval_model)
    model_uq_root = os.path.join(args.uq_root, model_slug)
    ensure_dir(model_uq_root)

    print(
        f"[INFO] Running UQ eval for model '{args.eval_model}' "
        f"using quizzes in {args.runs_root}."
    )
    print(f"[INFO] UQ outputs will be written under: {model_uq_root}")

    for run_idx in range(args.num_runs):
        eval_out_root = os.path.join(model_uq_root, f"run_{run_idx}")
        ensure_dir(eval_out_root)

        print(
            f"[UQ] Run {run_idx + 1}/{args.num_runs}: "
            f"eval_out_root={eval_out_root}"
        )

        cmd = [
            sys.executable,
            "-m",
            "quizbench.run_eval",
            "--runs_root",
            args.runs_root,
            "--manifest_path",
            str(manifest_path),
            "--eval_model",
            args.eval_model,
            "--eval_out_root",
            eval_out_root,
            "--max_tokens",
            str(args.max_tokens),
            "--reasoning_effort",
            args.reasoning_effort,
        ]
        if args.use_batch_api:
            cmd.append("--use_batch_api")
        if args.only_quiz_ids_csv:
            cmd.extend(["--only_quiz_ids_csv", args.only_quiz_ids_csv])

        _run(cmd)

    print(
        f"[INFO] Completed {args.num_runs} UQ runs for model '{args.eval_model}'.\n"
        f"[INFO] UQ tree root: {model_uq_root}"
    )


if __name__ == "__main__":
    main()
