#!/usr/bin/env python3
"""
Anthropic Message Batches transport helpers for QuizBench generation.

Extracted from run_batch_gen_quiz_anthropic.py so the Anthropic path can be
reused by other entrypoints (e.g., the unified run_batch_gen_quiz.py).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from quizbench.batch_generate_quiz import (  # noqa: E402
    QuizRequest,
    QuizResult,
    _parse_quiz_from_text,
    _render_prompt,
)
from quizbench.utils import ensure_dir, write_jsonl  # noqa: E402

# Polling cadence (seconds) when waiting for Anthropic Message Batches
ANTHROPIC_BATCH_POLL_INTERVAL_SECONDS = 60


def _is_anthropic_model(model_name: str) -> bool:
    """
    Lightweight guard to keep this transport focused on Anthropic Claude models.
    """
    return model_name.startswith("claude-")


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Returns an Anthropic client configured for Message Batches.

    Reads the API key from the ANTHROPIC_API_KEY environment variable.
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
    max_output_tokens: Optional[int] = 4000,
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
    effective_max_tokens = max_output_tokens or 4000

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
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[List[QuizResult], Optional[str]]:
    """
    Submit an Anthropic Message Batch job to generate quizzes, then parse and write outputs.

    Each quiz is sent as a single request (no chunking). Accepts the same knobs
    as generate_quizzes_via_batch; max_tokens is kept as a backward-compatible
    alias for max_output_tokens.
    Returns (results, batch_id).
    """
    if not quiz_requests:
        return [], None

    effective_max_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else (max_tokens if max_tokens is not None else 4000)
    )

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
        max_output_tokens=effective_max_tokens,
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
            error_msg = f"Unknown batch result type for {quiz_id}: {result_type}"
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
