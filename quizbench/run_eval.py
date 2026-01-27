#!/usr/bin/env python3
import argparse, os, subprocess, sys, json
from pathlib import Path
from quizbench.utils import ensure_dir, now_utc_iso
from quizbench.run_gen_quiz import load_config, collect_env
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


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: have a single quiz-taker model take all quizzes generated in Phase 1. Supports OpenRouter for DeepSeek/Kimi when configured."
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
        default="eval_uq_results/",
        help="Root directory used by run_quizbench (contains quizbench_manifest*.json and run dirs).",
    )
    ap.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional explicit manifest path (defaults to newest quizbench_manifest*.json under --runs_root).",
    )
    ap.add_argument(
        "--eval_out_root",
        type=str,
        default=None,
        help="Optional directory to write per-quiz eval outputs; defaults to --runs_root if unset.",
    )
    ap.add_argument(
        "--eval_model",
        type=str,
        required=True,
        help="Single quiz-taker model to evaluate on all quizzes (e.g. gpt-4o).",
    )
    ap.add_argument(
        "--use_batch_api",
        action="store_true",
        help=(
            "If set, quizbench/eval_quiz.py will use batch APIs where supported: "
            "OpenAI/Gemini Batch for those providers and Anthropic Message Batches "
            "for Claude models."
        ),
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Reasoning effort passed through to eval_quiz.py in batch mode.",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Max tokens for non-batch eval_quiz.py calls.",
    )
    ap.add_argument(
        "--only_quiz_ids_csv",
        type=str,
        default=None,
        help="Optional CSV of quiz_ids to evaluate (default: all quizzes in manifest).",
    )
    ap.add_argument(
        "--only_generator_models_csv",
        type=str,
        default=None,
        help=(
            "Optional CSV of generator_models; only quizzes produced by these "
            "generators will be evaluated (default: all generators)."
        ),
    )

    args = ap.parse_args()

    if args.use_batch_api and args.eval_model.strip().startswith("claude-"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "[FATAL] --use_batch_api was set with an Anthropic model "
                f"('{args.eval_model}'), but ANTHROPIC_API_KEY is not set. "
                "Please export ANTHROPIC_API_KEY before running run_eval.py."
            )

    ensure_dir(args.runs_root)
    eval_out_root = args.eval_out_root or args.runs_root
    ensure_dir(eval_out_root)

    manifest_path = resolve_quizbench_manifest_path(args.runs_root, manifest_path=args.manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load config (prefer CLI, fall back to manifest) to pull env overrides such as OPENROUTER_API_KEY.
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

    mn_lower = args.eval_model.strip().lower()
    use_openrouter_for_eval = None
    if cfg is not None and cfg.get("use_openrouter") is not None:
        use_openrouter_for_eval = bool(cfg.get("use_openrouter"))
    if use_openrouter_for_eval is None:
        # Default to OpenRouter for DeepSeek/Kimi when not specified in config.
        use_openrouter_for_eval = mn_lower.startswith(("deepseek-", "kimi-"))

    quizzes = manifest.get("quizzes", [])
    if not quizzes:
        print("[WARN] No quizzes listed in manifest; nothing to evaluate.")
        print(args.runs_root)
        return

    # Optional subset filter by generator model
    if args.only_generator_models_csv:
        allowed_generators = set(parse_csv_arg(args.only_generator_models_csv))
        quizzes = [
            q
            for q in quizzes
            if q.get("generator_model") in allowed_generators
        ]
        if not quizzes:
            print(
                "[WARN] After generator_model filtering, no quizzes remain to evaluate."
            )
            print(args.runs_root)
            return

    # Optional subset filter
    if args.only_quiz_ids_csv:
        allowed_ids = {q.strip() for q in args.only_quiz_ids_csv.split(",") if q.strip()}
        quizzes = [q for q in quizzes if q.get("quiz_id") in allowed_ids]
        if not quizzes:
            print("[WARN] After filtering, no quizzes remain to evaluate.")
            print(args.runs_root)
            return

    print(f"[INFO] Evaluating {len(quizzes)} quizzes with model '{args.eval_model}'")

    for q in quizzes:
        quiz_path = q["quiz_path"]
        quiz_id = q["quiz_id"]

        if not os.path.exists(quiz_path):
            print(f"[WARN] Quiz file missing, skipping: {quiz_path}")
            continue

        print(f"[EVAL] model={args.eval_model} quiz_id={quiz_id} path={quiz_path}")

        eval_cmd = [
            sys.executable,
            "-m",
            "quizbench.eval_quiz",
            "--quiz_file",
            quiz_path,
            "--test_models_csv",
            args.eval_model,
            "--out_dir",
            eval_out_root,
            "--max_tokens",
            str(args.max_tokens),
            "--reasoning_effort",
            args.reasoning_effort,
        ]
        if args.use_batch_api:
            eval_cmd.append("--use_batch_api")
        if use_openrouter_for_eval and mn_lower.startswith(("deepseek-", "kimi-")):
            eval_cmd.append("--use_openrouter")

        run_dir = run(eval_cmd)
        print(f"[OK] Completed eval for quiz_id={quiz_id}, run_dir={run_dir}")

    # Update manifest with evaluation metadata
    eval_models = set(manifest.get("eval_models", []))
    eval_models.add(args.eval_model)
    manifest["eval_models"] = sorted(eval_models)

    run_ids = set(manifest.get("run_ids", []))
    for q in quizzes:
        run_ids.add(q["quiz_id"])
    manifest["run_ids"] = sorted(run_ids)
    manifest["last_eval_at"] = now_utc_iso()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Updated manifest at {manifest_path}")
    # Machine-readable output (same as run_quizbench)
    print(args.runs_root)


if __name__ == "__main__":
    main()
