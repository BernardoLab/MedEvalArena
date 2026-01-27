#!/usr/bin/env python3
"""
Sequential (non-batch, non-async) quiz generation utilities for DeepSeek or
other providers that do not support Batch / async APIs.

This mirrors the parsing and normalization behavior used by
quizbench.batch_generate_quiz.generate_quizzes_via_batch, but issues one
request at a time via quizbench.clients.call_llm.
"""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from quizbench.batch_generate_quiz import (
    QuizRequest,
    QuizResult,
    _render_prompt,
    _parse_quiz_from_text,
)
from quizbench.clients import call_llm
from quizbench.utils import ensure_dir, write_jsonl


def _pseudo_batch_id(model_name: str) -> str:
    """
    Build a synthetic batch id for sequential runs so downstream manifests
    can treat these similarly to true Batch jobs.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
    return f"sequential-{int(time.time())}-{slug}"


def generate_quizzes_via_sequential(
    model_name: str,
    requests: List[QuizRequest],
    *,
    temperature: float = 0.2,
    openrouter: bool = False,
    max_output_tokens: Optional[int] = 4000,
    overwrite: bool = False,
) -> Tuple[List[QuizResult], Optional[str]]:
    """
    Sequential "pseudo-batch" quiz generation.

    For each QuizRequest we:
      - Render the quiz-generation prompt using the same spec builder as
        the Batch pipeline.
      - Call the model synchronously via call_llm.
      - Parse and normalize quiz items using the same JSON handling as
        the Batch pipeline.
      - Write a JSONL quiz file and return a QuizResult.

    Returns (results, pseudo_batch_id).
    """
    if not requests:
        return [], None

    batch_id = _pseudo_batch_id(model_name)
    results: List[QuizResult] = []
    effective_max_tokens = max_output_tokens or 4000

    for req in requests:
        out_dir = Path(req.out_dir).expanduser().resolve()
        out_path = out_dir / f"{req.quiz_id}.jsonl"
        if out_path.exists() and not overwrite:
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=f"Refusing to overwrite existing quiz file: {out_path}",
                    batch_id=batch_id,
                    batch_input_path=None,
                )
            )
            continue

        prompt, _sampled_topics = _render_prompt(req)

        try:
            print(
                f"  [INFO] Starting LLM call for {req.quiz_id} "
                f"(model={model_name}, save_path={out_path}, max_tokens={effective_max_tokens}, "
                f"temperature={temperature:.3f})."
            )
            start_time = time.time()
            raw_text, _reasoning_trace = call_llm(
                model_name=model_name,
                prompt=prompt,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                openrouter=openrouter,
                judge_mode=False,
            )
            elapsed = time.time() - start_time
            # Lightweight timing visibility for long-running calls.
            print(
                f"  [INFO] LLM call for {req.quiz_id} "
                f"(model={model_name}) succeeded in {elapsed:.1f}s; "
                f"{len(raw_text)} characters returned."
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            print(
                f"  [ERR] LLM call for {req.quiz_id} "
                f"(model={model_name}) failed after {elapsed:.1f}s with "
                f"{type(exc).__name__}: {exc}"
            )
            tb = traceback.format_exc()
            print(f"  [DEBUG] Traceback for failed LLM call:\n{tb}")
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=str(exc),
                    batch_id=batch_id,
                    batch_input_path=None,
                )
            )
            continue

        items, err_msg = _parse_quiz_from_text(raw_text, req, model_name)
        if err_msg:
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=err_msg,
                    batch_id=batch_id,
                    batch_input_path=None,
                )
            )
            continue

        items = items or []
        if not items:
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=f"Generator returned zero valid items for {req.quiz_id}",
                    batch_id=batch_id,
                    batch_input_path=None,
                )
            )
            continue

        ensure_dir(str(out_dir))
        try:
            write_jsonl(out_path, items, overwrite=overwrite)
        except FileExistsError:
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=f"Refusing to overwrite existing quiz file: {out_path}",
                    batch_id=batch_id,
                    batch_input_path=None,
                )
            )
            continue

        results.append(
            QuizResult(
                quiz_id=req.quiz_id,
                seed=req.seed,
                generator_model=model_name,
                quiz_path=str(out_path),
                status="ok",
                error=None,
                batch_id=batch_id,
                batch_input_path=None,
            )
        )

    return results, batch_id
