#!/usr/bin/env python3
"""
Anthropic Message Batches-based wrapper around quizbench/batch_generate_quiz.py.

This script mirrors run_batch_gen_quiz.py but targets Anthropic Claude models
via the Anthropic Message Batches API instead of the OpenAI / Gemini Batch APIs.

For each generator model, we keep submitting batch jobs until we have produced
at least `num_questions_per_quiz` valid questions in total.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.batch_generate_quiz import (  # noqa: E402
    DEFAULT_TOPICS,
    QuizRequest,
    QuizResult,
    _parse_quiz_from_text,
    _render_prompt,
)
from quizbench.run_gen_quiz import (  # noqa: E402
    build_quiz_id,
    coerce_int,
    collect_env,
    load_config,
    parse_models_field,
    sanitize_model_name,
)
from quizbench.utils import ensure_dir, now_utc_iso, compact_utc_ts, write_jsonl  # noqa: E402


# Polling cadence (seconds) when waiting for Anthropic Message Batches
ANTHROPIC_BATCH_POLL_INTERVAL_SECONDS = 60


def _is_anthropic_model(model_name: str) -> bool:
    """
    Lightweight guard to keep this script focused on Anthropic Claude models.
    """
    return model_name.startswith("claude-")


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Returns an Anthropic client configured for Message Batches.

    Reads the API key from the ANTHROPIC_API_KEY environment variable
    (optionally populated from config via quizbench.run_gen_quiz.collect_env).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Anthropic API key is not set. Please export ANTHROPIC_API_KEY "
            "or provide `anthropic_api_key` in your config."
        )
    return anthropic.Anthropic(api_key=api_key)


def _extract_text_from_anthropic_message(message) -> str:
    """
    Given an Anthropic Message object, extract the assistant text content
    in a robust way.
    """
    content = getattr(message, "content", "")

    # Some SDK versions may expose content as a list of blocks.
    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_val = getattr(block, "text", None)
                if text_val is None and isinstance(block, dict):
                    text_val = block.get("text", "")
                if text_val:
                    text_parts.append(text_val)
        if text_parts:
            return "".join(text_parts)
        # Fallback to stringifying the whole content list
        return "".join(str(b) for b in content)

    # Already a string
    if isinstance(content, str):
        return content

    # Fallback
    return str(content)


def _build_anthropic_batch_requests(
    model_name: str,
    quiz_requests: List[QuizRequest],
    batch_input_path: str,
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 4000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build a list of Anthropic Message Batch requests for the given quizzes,
    and write a JSONL sidecar file for reproducibility.

    Returns:
        (requests, custom_id_to_meta)
        - requests: list of {custom_id, params} dicts for the Batches API.
        - custom_id_to_meta: mapping custom_id -> {"request": QuizRequest, "sampled_topics": [...]}
    """
    if not quiz_requests:
        return [], {}

    ensure_dir(str(Path(batch_input_path).parent))

    requests: List[Dict[str, Any]] = []
    custom_id_to_meta: Dict[str, Dict[str, Any]] = {}
    effective_max_tokens = max_tokens or 4000

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in quiz_requests:
            prompt, sampled_topics = _render_prompt(req)

            params: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": effective_max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }
            # Sampling controls (kept minimal; extend if needed).
            if temperature is not None:
                params["temperature"] = float(temperature)

            custom_id = f"quiz__{req.quiz_id}"
            request_obj = {
                "custom_id": custom_id,
                "params": params,
            }

            f.write(json.dumps(request_obj) + "\n")
            requests.append(request_obj)
            custom_id_to_meta[custom_id] = {
                "request": req,
                "sampled_topics": sampled_topics,
            }

    return requests, custom_id_to_meta


def _create_message_batch(client: anthropic.Anthropic, requests: List[Dict[str, Any]]):
    """
    Create an Anthropic Message Batch from an in-memory list of requests.
    """
    print(f"[INFO] Creating Anthropic Message Batch with {len(requests)} requests...")
    message_batch = client.messages.batches.create(requests=requests)
    print(f"[INFO] Created Message Batch with id: {message_batch.id}")
    print(f"[INFO] Initial processing_status: {message_batch.processing_status}")
    return message_batch


def _wait_for_batch_completion(
    client: anthropic.Anthropic,
    message_batch,
    poll_interval: int = ANTHROPIC_BATCH_POLL_INTERVAL_SECONDS,
):
    """
    Poll the Message Batch until processing_status == 'ended'.
    """
    batch_id = message_batch.id
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        status = message_batch.processing_status
        counts = getattr(message_batch, "request_counts", None)
        print(f"[INFO] Batch {batch_id} status: {status}. Request counts: {counts}")
        if status == "ended":
            break
        time.sleep(poll_interval)

    print(f"[INFO] Batch {batch_id} reached terminal status: {message_batch.processing_status}")
    return message_batch


def _collect_batch_results(client: anthropic.Anthropic, message_batch):
    """
    Stream and collect all results for a completed Message Batch.

    Returns:
        dict custom_id -> MessageBatchIndividualResponse
    """
    batch_id = message_batch.id
    print(f"[INFO] Collecting results for Message Batch {batch_id}...")
    combined_results: Dict[str, Any] = {}

    for entry in client.messages.batches.results(batch_id):
        combined_results[entry.custom_id] = entry

    print(f"[INFO] Collected results for {len(combined_results)} requests.")
    return combined_results


def generate_quizzes_via_anthropic_batch(
    model_name: str,
    quiz_requests: List[QuizRequest],
    *,
    batch_input_path: str,
    poll_interval: int = ANTHROPIC_BATCH_POLL_INTERVAL_SECONDS,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 4000,
    overwrite: bool = False,
) -> Tuple[List[QuizResult], Optional[str]]:
    """
    Submit an Anthropic Message Batch job to generate quizzes, then parse and write outputs.

    Each quiz is sent as a single request (no chunking).
    Returns (results, batch_id).
    """
    if not quiz_requests:
        return [], None

    if not _is_anthropic_model(model_name):
        raise SystemExit(
            f"[FATAL] generate_quizzes_via_anthropic_batch is intended for Anthropic Claude "
            f"models (claude-*). Got: {model_name!r}"
        )

    client = get_anthropic_client()

    requests, custom_id_to_meta = _build_anthropic_batch_requests(
        model_name=model_name,
        quiz_requests=quiz_requests,
        batch_input_path=batch_input_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not custom_id_to_meta:
        return [], None

    message_batch = _create_message_batch(client, requests)
    message_batch = _wait_for_batch_completion(client, message_batch, poll_interval=poll_interval)
    batch_results = _collect_batch_results(client, message_batch)

    results: List[QuizResult] = []

    for custom_id, meta in custom_id_to_meta.items():
        req: QuizRequest = meta["request"]
        quiz_id = req.quiz_id

        entry = batch_results.get(custom_id)
        if entry is None:
            error_msg = f"No result returned for quiz {quiz_id} (custom_id={custom_id})"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            continue

        result_obj = getattr(entry, "result", None)
        result_type = getattr(result_obj, "type", None)

        if result_type == "succeeded":
            message = result_obj.message
            content = _extract_text_from_anthropic_message(message)
            items, err_msg = _parse_quiz_from_text(content, req, model_name)
            if err_msg:
                results.append(
                    QuizResult(
                        quiz_id=quiz_id,
                        seed=req.seed,
                        generator_model=model_name,
                        quiz_path=None,
                        status="error",
                        error=err_msg,
                        batch_id=getattr(message_batch, "id", None),
                        batch_input_path=batch_input_path,
                    )
                )
                continue

            items = items or []
            if not items:
                error_msg = f"Generator returned zero valid items for {quiz_id}"
                results.append(
                    QuizResult(
                        quiz_id=quiz_id,
                        seed=req.seed,
                        generator_model=model_name,
                        quiz_path=None,
                        status="error",
                        error=error_msg,
                        batch_id=getattr(message_batch, "id", None),
                        batch_input_path=batch_input_path,
                    )
                )
                continue

            ensure_dir(req.out_dir)
            out_path = Path(req.out_dir) / f"{quiz_id}.jsonl"
            try:
                write_jsonl(str(out_path), items, overwrite=overwrite)
            except FileExistsError:
                results.append(
                    QuizResult(
                        quiz_id=quiz_id,
                        seed=req.seed,
                        generator_model=model_name,
                        quiz_path=None,
                        status="error",
                        error=f"Refusing to overwrite existing quiz file: {out_path}",
                        batch_id=getattr(message_batch, "id", None),
                        batch_input_path=batch_input_path,
                    )
                )
                continue

            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=str(out_path),
                    status="ok",
                    error=None,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            print(f"[OK] Generated quiz {quiz_id} at {out_path}")
        elif result_type == "errored":
            err = getattr(result_obj, "error", None)
            err_type = getattr(err, "type", "unknown_error")
            err_msg = getattr(err, "message", "no error message provided")
            error_msg = f"Batch error for quiz {quiz_id} ({err_type}): {err_msg}"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
        elif result_type == "canceled":
            error_msg = "Batch request was canceled before processing."
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
        elif result_type == "expired":
            error_msg = "Batch request expired before it could be processed."
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
        else:
            error_msg = f"Unknown result type for quiz {quiz_id}: {result_type}"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(message_batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )

    return results, getattr(message_batch, "id", None)


def _build_requests_for_model(
    *,
    quiz_id: str,
    num_questions: int,
    num_choices: int,
    seed: int,
    topics_csv: str,
    quizzes_dir: str,
) -> List[QuizRequest]:
    """
    Build a single QuizRequest for the provided quiz metadata.
    """
    return [
        QuizRequest(
            quiz_id=quiz_id,
            seed=seed,
            num_questions=num_questions,
            num_choices=num_choices,
            topics_csv=topics_csv,
            out_dir=quizzes_dir,
        )
    ]


def _count_questions_in_quiz(quiz_path: str) -> int:
    """
    Count how many valid question records ended up in a quiz JSONL file.

    We conservatively treat each non-empty, parseable JSON line as one
    question. If the file is missing or malformed we return 0.
    """
    count = 0
    try:
        with open(quiz_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines; they shouldn't normally appear.
                    continue
                count += 1
    except FileNotFoundError:
        return 0
    return count


def _load_cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate quizzes via the Anthropic Message Batches API (one batch per generator model)."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    ap.add_argument(
        "--quiz_collection",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under config quizzes_dir for this run. "
            "Use 'auto' to name the collection by the run timestamp."
        ),
    )
    ap.add_argument(
        "--lock_quiz_collection",
        action="store_true",
        help=(
            "If set, write a .quizbench_readonly marker file into the output quizzes directory at the end of the run. "
            "Future runs will refuse to generate into a directory that contains this marker."
        ),
    )
    ap.add_argument(
        "--allow_locked_quiz_collection",
        action="store_true",
        help="If set, allow generation into a quizzes directory even if it contains a .quizbench_readonly marker.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow overwriting existing quiz JSONL files in the output directory.",
    )
    ap.add_argument(
        "--batch_input_dir",
        type=str,
        default=None,
        help="Directory to place batch input JSONL files.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generation models.",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens per Anthropic completion (defaults to config or 4000 if unset).",
    )
    ap.add_argument(
        "--poll_seconds",
        type=int,
        default=ANTHROPIC_BATCH_POLL_INTERVAL_SECONDS,
        help="Polling interval while waiting for batch completion.",
    )
    ap.add_argument(
        "--topics_csv",
        type=str,
        default=None,
        help="Override topics CSV (falls back to config, then generator default list).",
    )
    ap.add_argument(
        "--num_choices",
        type=int,
        default=None,
        help="Number of answer choices per question (default 5).",
    )
    return ap.parse_args()


def _resolve_quiz_collection(raw: object | None, *, run_ts: datetime) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip()
    if not val or val.lower() in {"none", "null"}:
        return None
    if val.lower() == "auto":
        return compact_utc_ts(run_ts)
    return val


def main():
    args = _load_cli_args()
    run_ts = datetime.utcnow()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(str(config_path))
    print(f"[INFO] Loaded config from {config_path}")

    generator_models = parse_models_field(
        cfg.get("generator_models")
        or cfg.get("generator_models_csv")
        or cfg.get("generator_model"),
        field_label="generator_models",
        required=True,
    )
    test_models = parse_models_field(
        cfg.get("test_models") or cfg.get("test_models_csv") or cfg.get("test_model"),
        field_label="test_models",
        required=False,
    )
    test_models_csv = ",".join(test_models) if test_models else None

    # Anthropic-only guard to keep API usage siloed.
    for m in generator_models:
        if not _is_anthropic_model(m):
            raise SystemExit(
                f"[FATAL] run_batch_gen_quiz_anthropic.py expects Anthropic Claude "
                f"generator models (claude-*). Found: {m!r}"
            )

    # Question counts:
    num_questions_per_batch = coerce_int(cfg, "num_questions_per_batch", 10)
    num_questions_per_quiz = coerce_int(cfg, "num_questions_per_quiz", 50)

    if num_questions_per_batch <= 0:
        raise ValueError("num_questions_per_batch must be > 0")
    if num_questions_per_quiz <= 0:
        raise ValueError("num_questions_per_quiz must be > 0")

    num_choices = coerce_int(cfg, "num_choices", 5) if args.num_choices is None else int(args.num_choices)
    seed0 = coerce_int(cfg, "seed0", 123)
    quizzes_root_dir = str(cfg.get("quizzes_dir", "quizzes/"))
    runs_root = str(cfg.get("runs_root", "eval_uq_results/"))
    topics_csv = args.topics_csv or cfg.get("topics_csv") or DEFAULT_TOPICS

    if args.max_tokens is not None:
        max_tokens = int(args.max_tokens)
    elif "max_tokens" in cfg:
        max_tokens = coerce_int(cfg, "max_tokens", 4000)
    else:
        max_tokens = coerce_int(cfg, "max_output_tokens", 4000)

    quiz_collection = _resolve_quiz_collection(
        args.quiz_collection
        if args.quiz_collection is not None
        else cfg.get("quiz_collection") or cfg.get("quizzes_collection"),
        run_ts=run_ts,
    )
    quizzes_dir = (
        str(Path(quizzes_root_dir) / quiz_collection)
        if quiz_collection
        else quizzes_root_dir
    )

    lock_path = Path(quizzes_dir) / ".quizbench_readonly"
    if lock_path.exists() and not args.allow_locked_quiz_collection:
        raise SystemExit(
            f"[FATAL] Quizzes directory is locked (found {lock_path}). "
            "Choose a different --quiz_collection (or set quiz_collection in config), "
            "or delete the lock file to proceed."
        )

    batch_input_dir = args.batch_input_dir or cfg.get("batch_input_dir")
    if not batch_input_dir:
        batch_input_dir = str(Path(quizzes_dir) / "anthropic_batch_inputs")

    env_overrides = collect_env(cfg)
    if env_overrides:
        os.environ.update(env_overrides)

    ensure_dir(quizzes_dir)
    ensure_dir(runs_root)
    ensure_dir(batch_input_dir)

    # Theoretical minimum number of batches if each batch yields the full request.
    num_quizzes_per_generator = (num_questions_per_quiz + num_questions_per_batch - 1) // num_questions_per_batch

    manifest: Dict[str, Any] = {
        "created_at": now_utc_iso(),
        "config_path": str(config_path),
        "generator_models": generator_models,
        "generator_model": generator_models[0],
        "test_models": test_models,
        "test_models_csv": test_models_csv,
        "num_quizzes_per_generator": num_quizzes_per_generator,
        "num_questions_per_batch": num_questions_per_batch,
        "num_questions_per_quiz": num_questions_per_quiz,
        "runs_root": runs_root,
        "quizzes_root_dir": quizzes_root_dir,
        "quiz_collection": quiz_collection,
        "quizzes_dir": quizzes_dir,
        "api_provider": "anthropic_message_batches",
        "quizzes": [],
        "run_ids": [],
        "batch_jobs": [],
    }

    quizzes_info: List[Dict[str, Any]] = []

    print("\n=== Generating quizzes via Anthropic Message Batches API ===")

    for g_idx, gen_model in enumerate(generator_models):
        print(f"\n-- Generator model: {gen_model} --")

        safe_model = sanitize_model_name(gen_model)
        base_seed_for_model = seed0 + g_idx

        total_questions_for_model = 0
        quiz_idx_for_model = 0

        while total_questions_for_model < num_questions_per_quiz:
            remaining_needed = num_questions_per_quiz - total_questions_for_model
            to_request = min(num_questions_per_batch, remaining_needed)
            if to_request <= 0:
                break

            seed_for_quiz = base_seed_for_model + quiz_idx_for_model
            quiz_id = build_quiz_id(gen_model, seed_for_quiz, at=run_ts)

            requests = _build_requests_for_model(
                quiz_id=quiz_id,
                num_questions=to_request,
                num_choices=num_choices,
                seed=seed_for_quiz,
                topics_csv=topics_csv,
                quizzes_dir=quizzes_dir,
            )

            batch_input_path = str(
                Path(batch_input_dir) / f"{safe_model}_anthropic_batch_input_{quiz_idx_for_model:03d}.jsonl"
            )

            print(
                f"  [INFO] Submitting Anthropic batch {quiz_idx_for_model} for {gen_model} "
                f"(current total {total_questions_for_model}, remaining {remaining_needed}; "
                f"requesting up to {to_request})."
            )

            results, batch_id = generate_quizzes_via_anthropic_batch(
                gen_model,
                requests,
                batch_input_path=batch_input_path,
                poll_interval=args.poll_seconds,
                temperature=args.temperature,
                max_tokens=max_tokens,
                overwrite=bool(args.overwrite),
            )

            if batch_id:
                manifest["batch_jobs"].append(
                    {
                        "generator_model": gen_model,
                        "batch_id": batch_id,
                        "batch_input_path": batch_input_path,
                        "num_requests": len(requests),
                    }
                )

            batch_questions = 0

            for res in results:
                if res.status != "ok" or not res.quiz_path:
                    print(f"  [ERR] {res.quiz_id}: {res.error}")
                    continue

                n_valid = _count_questions_in_quiz(res.quiz_path)
                batch_questions += n_valid

                quizzes_info.append(
                    {
                        "quiz_id": res.quiz_id,
                        "quiz_path": res.quiz_path,
                        "generator_model": gen_model,
                        "seed": res.seed,
                    }
                )
                print(
                    f"  [OK] Generated quiz {res.quiz_id} at {res.quiz_path} "
                    f"({n_valid} valid questions returned)."
                )

            usable_from_batch = min(batch_questions, num_questions_per_quiz - total_questions_for_model)
            total_questions_for_model += usable_from_batch

            print(
                f"  [INFO] Batch produced {batch_questions} valid; "
                f"used {usable_from_batch}. "
                f"Progress: {total_questions_for_model} / {num_questions_per_quiz}"
            )

            if batch_questions == 0:
                print(
                    f"  [WARN] Anthropic batch {quiz_idx_for_model} for {gen_model} produced 0 valid questions; "
                    "stopping early to avoid an infinite loop."
                )
                break

            quiz_idx_for_model += 1

        print(
            f"  [INFO] Finished generator {gen_model}: "
            f"{total_questions_for_model} valid questions generated "
            f"(target was {num_questions_per_quiz})."
        )

    manifest["quizzes"] = quizzes_info

    manifest_path = os.path.join(runs_root, "quizbench_manifest.json")
    legacy_manifest_path = os.path.join(runs_root, "quizbench_manifest_anthropic.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    # Keep a legacy copy for backward compatibility with earlier runs.
    if legacy_manifest_path != manifest_path:
        with open(legacy_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    if legacy_manifest_path != manifest_path:
        print(f"[INFO] Legacy manifest copy written to: {legacy_manifest_path}")
    if args.lock_quiz_collection:
        lock_payload = {
            "locked_at": now_utc_iso(),
            "config_path": str(config_path),
            "runs_root": runs_root,
            "quizzes_dir": quizzes_dir,
            "quiz_collection": quiz_collection,
        }
        lock_path.write_text(
            json.dumps(lock_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Locked quiz collection: {lock_path}")
    print(runs_root)


if __name__ == "__main__":
    main()
