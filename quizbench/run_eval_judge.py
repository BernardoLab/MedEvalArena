#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from quizbench.run_gen_quiz import load_config, collect_env
from quizbench.utils import ensure_dir, now_utc_iso
from quizbench.manifest_utils import resolve_quizbench_manifest_path


def run(cmd: list) -> str:
    """
    Run a subprocess and return its stdout (stripped).
    If the command fails, raise with stderr attached.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit(f"[FATAL] {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def parse_csv_arg(val: str):
    return [x.strip() for x in val.split(",") if x.strip()]


def prefer_non_backup_path(path_str: str) -> tuple[str, bool]:
    """
    Prefer a non-backup sibling path when the input points into a *_backup directory.

    Example:
      quizzes/quizzes_Jan2026_backup/foo.jsonl -> quizzes/quizzes_Jan2026/foo.jsonl
    """
    p = Path(path_str)
    parts: list[str] = []
    changed = False
    for part in p.parts:
        if part.endswith("_backup"):
            parts.append(part[: -len("_backup")])
            changed = True
        else:
            parts.append(part)

    if not changed:
        return path_str, False

    candidate = Path(*parts)
    if candidate.exists():
        return str(candidate), True

    return path_str, False


def main():
    ap = argparse.ArgumentParser(
        description="Judge MCQ validity/accuracy for all quizzes listed in a quizbench manifest."
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config (falls back to manifest.config_path when omitted).",
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs/",
        help="Root directory containing quizbench_manifest*.json and quiz runs.",
    )
    ap.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional explicit manifest path (defaults to newest quizbench_manifest*.json under --runs_root).",
    )
    ap.add_argument(
        "--judge_out_root",
        type=str,
        default=None,
        help="Directory to write judge outputs; defaults to --runs_root when unset.",
    )
    ap.add_argument(
        "--judge_model",
        type=str,
        required=True,
        help="Judge model to call for all quizzes (e.g., gpt-4o).",
    )
    ap.add_argument(
        "--use_batch_api",
        action="store_true",
        help="Request batch judge evaluation where supported (fallback to per-call for now).",
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Reasoning effort passed through to eval_quiz_judge (reserved for batch).",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Max tokens for non-batch eval_quiz_judge calls.",
    )
    ap.add_argument(
        "--only_quiz_ids_csv",
        type=str,
        default=None,
        help="Optional CSV of quiz_ids to judge (default: all quizzes in manifest).",
    )
    ap.add_argument(
        "--only_generator_models_csv",
        type=str,
        default=None,
        help="Optional CSV of generator_models to include (default: all generators).",
    )
    ap.add_argument(
        "--max_quizzes",
        type=int,
        default=None,
        help="Optional cap on number of quizzes to judge (useful for debugging).",
    )

    args = ap.parse_args()

    if args.use_batch_api and args.judge_model.strip().startswith("claude-"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "[FATAL] --use_batch_api was set with an Anthropic model "
                f"('{args.judge_model}'), but ANTHROPIC_API_KEY is not set. "
                "Please export ANTHROPIC_API_KEY before running run_eval_judge.py."
            )

    ensure_dir(args.runs_root)
    judge_out_root = args.judge_out_root or args.runs_root
    ensure_dir(judge_out_root)

    manifest_path = resolve_quizbench_manifest_path(args.runs_root, manifest_path=args.manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cfg = None
    config_path = args.config or manifest.get("config_path")
    if config_path:
        resolved_cfg = Path(config_path).expanduser().resolve()
        cfg = load_config(str(resolved_cfg))
        env_overrides = collect_env(cfg)
        if env_overrides:
            os.environ.update(env_overrides)
            print(f"[INFO] Applied env overrides from config: {resolved_cfg}")
        else:
            print(f"[INFO] Loaded config with no env overrides: {resolved_cfg}")

    mn_lower = args.judge_model.strip().lower()
    # automatically use openrouter for deepseek and kimi
    use_openrouter_for_judge = mn_lower.startswith(("deepseek-", "kimi-"))

    quizzes = manifest.get("quizzes", [])
    if not quizzes:
        print("[WARN] No quizzes listed in manifest; nothing to judge.")
        print(args.runs_root)
        return

    if args.only_generator_models_csv:
        allowed_generators = set(parse_csv_arg(args.only_generator_models_csv))
        quizzes = [
            q for q in quizzes if q.get("generator_model") in allowed_generators
        ]
        if not quizzes:
            print("[WARN] After generator_model filtering, no quizzes remain to judge.")
            print(args.runs_root)
            return

    if args.only_quiz_ids_csv:
        allowed_ids = {q.strip() for q in args.only_quiz_ids_csv.split(",") if q.strip()}
        quizzes = [q for q in quizzes if q.get("quiz_id") in allowed_ids]
        if not quizzes:
            print("[WARN] After quiz_id filtering, no quizzes remain to judge.")
            print(args.runs_root)
            return

    if args.max_quizzes is not None and args.max_quizzes > 0:
        quizzes = quizzes[: args.max_quizzes]

    print(f"[INFO] Judging {len(quizzes)} quizzes with model '{args.judge_model}'")

    for q in quizzes:
        quiz_path = q["quiz_path"]
        quiz_id = q["quiz_id"]

        # If manifests reference older *_backup quiz dirs, prefer the canonical path
        # (without the suffix) when it exists.
        non_backup_quiz_path, changed = prefer_non_backup_path(quiz_path)
        if changed:
            print(f"[INFO] Rewriting quiz_path to avoid '_backup': {quiz_path} -> {non_backup_quiz_path}")
            quiz_path = non_backup_quiz_path
            q["quiz_path"] = non_backup_quiz_path

        if not os.path.exists(quiz_path):
            print(f"[WARN] Quiz file missing, skipping: {quiz_path}")
            continue

        print(f"[JUDGE] model={args.judge_model} quiz_id={quiz_id} path={quiz_path}")

        eval_cmd = [
            sys.executable,
            "-m",
            "quizbench.eval_quiz_judge",
            "--quiz_file",
            quiz_path,
            "--judge_model",
            args.judge_model,
            "--out_dir",
            judge_out_root,
            "--max_tokens",
            str(args.max_tokens),
            "--reasoning_effort",
            args.reasoning_effort,
        ]
        if args.use_batch_api:
            eval_cmd.append("--use_batch_api")
        if use_openrouter_for_judge:
            eval_cmd.append("--use_openrouter")
            print(f"[JUDGE] using OpenRouter")

        run_dir = run(eval_cmd)
        print(f"[OK] Completed judge eval for quiz_id={quiz_id}, run_dir={run_dir}")

    judge_models = set(manifest.get("judge_models", []))
    judge_models.add(args.judge_model)
    manifest["judge_models"] = sorted(judge_models)

    judge_run_ids = set(manifest.get("judge_run_ids", []))
    for q in quizzes:
        judge_run_ids.add(q["quiz_id"])
    manifest["judge_run_ids"] = sorted(judge_run_ids)
    manifest["last_judge_eval_at"] = now_utc_iso()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Updated manifest at {manifest_path}")
    print(judge_out_root)


if __name__ == "__main__":
    main()
