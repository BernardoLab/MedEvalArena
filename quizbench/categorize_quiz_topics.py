#!/usr/bin/env python3
"""
Classify QuizBench questions into medicine topics using a chosen evaluation model.

This CLI mirrors eval_quiz.py's per-call flow (no batch API usage) and writes one
topics JSON per quiz inside the corresponding run folder under
eval_results/quizbench/runs/<generator_model>/<quiz_id>/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from quizbench.clients import call_llm
from quizbench.manifest_utils import resolve_quizbench_manifest_path
from quizbench.target_planning import load_integer_targets_csv
from quizbench.topic_mapping import normalize_label
from quizbench.utils import ensure_dir, now_utc_iso, read_jsonl

# Keep this in sync with generator defaults in batch_generate_quiz.py.
DEFAULT_TOPICS_CSV = (
    "cardiology,endocrinology,infectious disease,hematology/oncology,neurology,"
    "nephrology,pulmonology,obstetrics,gynecology,pediatrics,geriatrics,"
    "dermatology,rheumatology,emergency medicine,critical care"
)
DEFAULT_TOPICS = tuple(t.strip() for t in DEFAULT_TOPICS_CSV.split(",") if t.strip())


def parse_topics(topics_file: str | None, fallback_csv: str = DEFAULT_TOPICS_CSV) -> List[str]:
    """
    Return a list of topics from a file (CSV or newline-separated) or the fallback CSV.
    """
    if not topics_file:
        return [t.strip() for t in fallback_csv.split(",") if t.strip()]

    text = Path(topics_file).read_text(encoding="utf-8")
    # If the file contains commas, treat it as CSV; otherwise split on newlines.
    raw_parts = text.split(",") if "," in text else text.splitlines()
    topics = [p.strip() for p in raw_parts if p.strip()]
    if not topics:
        raise SystemExit(f"[FATAL] No topics could be parsed from {topics_file}")
    return topics


def sanitize_for_filename(value: str) -> str:
    """
    Make a safe filename fragment from a model name or identifier.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def resolve_path(path_str: str, base: Path) -> Path:
    """
    Resolve absolute or relative paths against the provided base directory.
    """
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = resolve_quizbench_manifest_path(run_dir)
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, options: Sequence[str], topics: Sequence[str]) -> str:
    """
    Build a concise classification prompt that asks for exactly one topic.
    """
    options_block = ""
    if options:
        formatted_opts = "\n".join(f"- {opt}" for opt in options)
        options_block = f"\nOptions:\n{formatted_opts}"

    prompt = (
        "You are a medical domain classifier.\n"
        "Given one multiple-choice question, choose the single best-matching topic "
        "from the allowed list below. Respond with exactly one topic string "
        "from the list and nothing else.\n\n"
        f"Allowed topics: {', '.join(topics)}\n\n"
        f"Question:\n{question.strip()}"
        f"{options_block}\n\n"
        "Answer with one topic from the allowed list."
    )
    return prompt


def _extract_candidate_tokens(text: str) -> List[str]:
    cleaned = text.strip().strip("`").strip()
    # Focus on the first line to reduce chatter in multi-line answers.
    first_line = cleaned.splitlines()[0] if cleaned else ""
    tokens = [first_line]
    # Split on separators to pull simple tokens.
    tokens.extend(part.strip() for part in re.split(r"[;,:|/]", first_line) if part.strip())
    return tokens


def normalize_topic(raw_response: str, allowed: Sequence[str]) -> str | None:
    """
    Map a raw model response to one of the allowed topics, or None if no match.
    """
    allowed_lower = {t.lower(): t for t in allowed}
    allowed_norm: Dict[str, str] = {}
    for topic in allowed:
        key = normalize_label(topic)
        if not key:
            continue
        allowed_norm.setdefault(key, topic)

    candidates = _extract_candidate_tokens(raw_response)
    for cand in candidates:
        cand_norm = cand.lower().strip(" .\"'")
        if cand_norm.startswith("topic "):
            cand_norm = cand_norm.replace("topic", "", 1).strip(" .:-")
        if cand_norm in allowed_lower:
            return allowed_lower[cand_norm]
        cand_norm2 = normalize_label(cand)
        if cand_norm2 in allowed_norm:
            return allowed_norm[cand_norm2]

    # Substring fallback (e.g., "The topic is cardiology.")
    for cand in candidates:
        cand_norm = cand.lower()
        for key, original in allowed_lower.items():
            if key in cand_norm:
                return original
        cand_norm2 = normalize_label(cand)
        for key_norm, original in allowed_norm.items():
            if key_norm and key_norm in cand_norm2:
                return original
    return None


def summarize_topics(per_question: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    topics = [row.get("topic") for row in per_question if row.get("topic")]
    counts = Counter(topics)
    total = sum(counts.values())
    majority_topic = None
    if counts:
        majority_topic = counts.most_common(1)[0][0]
    return {
        "topic_counts": counts,
        "majority_topic": majority_topic,
        "num_questions": total,
        "num_unclassified": sum(1 for row in per_question if not row.get("topic")),
    }


def process_quiz(
    quiz: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    eval_model: str,
    topics: Sequence[str],
    max_questions: int | None,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    overwrite: bool,
    use_openrouter: bool,
) -> None:
    quiz_id = quiz["quiz_id"]
    quiz_path = resolve_path(quiz["quiz_path"], repo_root)
    if not quiz_path.exists():
        print(f"[WARN] Quiz file missing, skipping: {quiz_path}")
        return

    quiz_items = read_jsonl(str(quiz_path))
    if max_questions is not None:
        quiz_items = quiz_items[:max_questions]

    quiz_out_dir = run_dir / quiz_id
    ensure_dir(str(quiz_out_dir))

    outfile = quiz_out_dir / f"topics_{sanitize_for_filename(eval_model)}.json"
    if outfile.exists() and not overwrite:
        print(f"[SKIP] Found existing topics file (use --overwrite to replace): {outfile}")
        return

    per_question: List[Dict[str, Any]] = []
    for idx, item in enumerate(quiz_items, start=1):
        qid = item.get("question_id") or f"{quiz_id}-{idx:03d}"
        prompt = build_prompt(item.get("question", ""), item.get("options", []), topics)

        raw_response = ""
        topic = None
        if dry_run:
            raw_response = "[dry-run skipped model call]"
        else:
            raw_response, _reasoning_trace = call_llm(
                eval_model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                openrouter=use_openrouter,
            )
            topic = normalize_topic(raw_response, topics) or "unknown"

        per_question.append(
            {
                "question_id": qid,
                "topic": topic,
                "raw_response": raw_response,
            }
        )
        if topic == "unknown":
            print(f"[WARN] Could not parse topic for {qid}; stored as 'unknown'.")

    summary = summarize_topics(per_question)
    payload = {
        "quiz_id": quiz_id,
        "generator_model": quiz.get("generator_model"),
        "eval_model": eval_model,
        "default_topics": list(topics),
        "created_at": now_utc_iso(),
        "per_question": per_question,
        "summary": summary,
    }

    if dry_run:
        print(f"[DRY-RUN] Would write {outfile} with {len(per_question)} records.")
        return

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote topics to {outfile}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Assign medicine topics to QuizBench questions using a specified evaluation model. "
            "This tool runs synchronously (no batch APIs) and writes one JSON per quiz."
        )
    )
    ap.add_argument(
        "--eval_model",
        required=True,
        help="Evaluation model used for classification (e.g., gpt-4o).",
    )
    ap.add_argument(
        "--generator_model",
        type=str,
        help="Name of the generator model (used to locate runs under --runs_root).",
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a generator run directory "
            "(overrides --runs_root/--generator_model)."
        ),
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root directory containing generator run folders.",
    )
    ap.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help=(
            "Optional explicit manifest path (absolute or relative to the run_dir). "
            "Defaults to the newest quizbench_manifest*.json under the run_dir."
        ),
    )
    ap.add_argument(
        "--topics_file",
        type=str,
        default=None,
        help="Optional path to a CSV or newline-separated topics file.",
    )
    ap.add_argument(
        "--targets_csv",
        type=str,
        default=None,
        help=(
            "Optional targets CSV (Specialty,Number). When set, uses its Specialty column "
            "as the allowed topics list (overrides --topics_file/default topics)."
        ),
    )
    ap.add_argument(
        "--max_quizzes",
        type=int,
        default=None,
        help="Process at most this many quizzes (after filtering).",
    )
    ap.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Process at most this many questions per quiz.",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=120,
        help="Max tokens for each classification call (non-batch).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for classification calls.",
    )
    ap.add_argument(
        "--quiz_ids_csv",
        type=str,
        default=None,
        help="Optional CSV of quiz_ids to process (defaults to all in manifest).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="List planned work and skip model calls / file writes.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing topics JSON files when set.",
    )
    ap.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Route DeepSeek/Kimi calls through OpenRouter when set.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    else:
        if not args.generator_model:
            raise SystemExit("[FATAL] Provide --generator_model or --run_dir.")
        run_dir = Path(args.runs_root) / args.generator_model

    repo_root = Path(".").resolve()
    run_dir = run_dir if run_dir.is_absolute() else (repo_root / run_dir)
    manifest_path = resolve_quizbench_manifest_path(run_dir, manifest_path=args.manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.targets_csv:
        topics = load_integer_targets_csv(Path(args.targets_csv)).categories
    else:
        topics = parse_topics(args.topics_file)
    quiz_records = manifest.get("quizzes", [])
    if args.generator_model:
        quiz_records = [
            q for q in quiz_records if q.get("generator_model") == args.generator_model
        ]

    if args.quiz_ids_csv:
        allowed = {q.strip() for q in args.quiz_ids_csv.split(",") if q.strip()}
        quiz_records = [q for q in quiz_records if q.get("quiz_id") in allowed]

    if args.max_quizzes is not None:
        quiz_records = quiz_records[: args.max_quizzes]

    if not quiz_records:
        print("[WARN] No quizzes to process after filtering.")
        return

    print(
        f"[INFO] Classifying {len(quiz_records)} quizzes "
        f"with model '{args.eval_model}' using {len(topics)} topics."
    )

    for quiz in quiz_records:
        process_quiz(
            quiz=quiz,
            run_dir=run_dir,
            repo_root=repo_root,
            eval_model=args.eval_model,
            topics=topics,
            max_questions=args.max_questions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            use_openrouter=args.use_openrouter,
        )


if __name__ == "__main__":
    main()
