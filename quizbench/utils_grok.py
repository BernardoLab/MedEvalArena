"""
Utilities for Grok (xAI) models, providing an async \"batch-like\" pipeline
that mirrors existing Batch behavior without relying on provider batch APIs.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from openai import AsyncOpenAI

from quizbench.generate_quiz_spec import GENERATOR_SYSTEM_INSTRUCTIONS, build_json_spec
from quizbench.utils import (
    ensure_dir,
    extract_json_block,
    normalize_quiz_items,
    now_utc_iso,
    write_jsonl,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from quizbench.batch_generate_quiz import QuizRequest, QuizResult


DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
SMART_PUNCT_TRANS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2039": "'",
        "\u203a": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


@dataclass
class GrokRequest:
    custom_id: str
    prompt: str
    model_name: str
    max_tokens: int
    temperature: float
    extra: Dict[str, Any] | None = None


@dataclass
class GrokResponse:
    custom_id: str
    ok: bool
    raw_text: Optional[str]
    error: Optional[str]


def is_grok_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith("grok-")


def _require_grok_api_key() -> str:
    key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("Set GROK_API_KEY (or XAI_API_KEY) to call Grok models.")
    return key


def _grok_base_url() -> str:
    return os.getenv("GROK_API_BASE_URL") or os.getenv("XAI_API_BASE_URL") or DEFAULT_GROK_BASE_URL


async def _create_grok_async_client() -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client configured for Grok (xAI).
    """
    key = _require_grok_api_key()
    base_url = _grok_base_url()
    return AsyncOpenAI(api_key=key, base_url=base_url)


async def _call_grok_once(client: AsyncOpenAI, req: GrokRequest) -> GrokResponse:
    payload = {
        "model": req.model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
    }
    if req.max_tokens:
        payload["max_tokens"] = req.max_tokens

    resp = await client.chat.completions.create(**payload)
    text: str = ""
    try:
        choice = resp.choices[0]
        content = choice.message.content
        if isinstance(content, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        else:
            text = content or ""
    except Exception as exc:  # pragma: no cover - defensive
        return GrokResponse(custom_id=req.custom_id, ok=False, raw_text=None, error=str(exc))

    return GrokResponse(custom_id=req.custom_id, ok=True, raw_text=text, error=None)


async def _call_grok_with_retry(
    client: AsyncOpenAI,
    req: GrokRequest,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    base_backoff: float = 1.0,
) -> GrokResponse:
    attempt = 0
    delay = base_backoff
    while True:
        try:
            async with semaphore:
                return await _call_grok_once(client, req)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                return GrokResponse(custom_id=req.custom_id, ok=False, raw_text=None, error=str(exc))
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


async def run_grok_async_batch(
    requests: List[GrokRequest],
    *,
    max_concurrency: int = 8,
    max_retries: int = 3,
) -> Dict[str, GrokResponse]:
    if not requests:
        return {}
    client = await _create_grok_async_client()
    try:
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        tasks = [
            _call_grok_with_retry(
                client,
                req,
                sem,
                max_retries=max_retries,
            )
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
        return {r.custom_id: r for r in results}
    finally:
        await client.close()


def run_grok_batch_sync(
    requests: List[GrokRequest],
    *,
    max_concurrency: int = 8,
    max_retries: int = 3,
) -> Dict[str, GrokResponse]:
    """
    Synchronous wrapper around run_grok_async_batch for CLI-style callers.
    """
    if not requests:
        return {}
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_grok_batch_sync cannot be called from an active event loop.")
    return asyncio.run(
        run_grok_async_batch(
            requests,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
        )
    )


def grok_responses_to_batch_lines(responses: Dict[str, GrokResponse]) -> Dict[str, Dict[str, Any]]:
    """
    Map GrokResponse objects into a Batch-like line dict (custom_id -> {response|error}).
    """
    lines: Dict[str, Dict[str, Any]] = {}
    for custom_id, resp in responses.items():
        if not resp.ok:
            lines[custom_id] = {
                "custom_id": custom_id,
                "error": {"code": "grok_error", "message": resp.error or "unknown error"},
            }
            continue

        body = {
            "choices": [
                {
                    "message": {
                        "content": resp.raw_text or "",
                    }
                }
            ]
        }
        lines[custom_id] = {"custom_id": custom_id, "response": {"body": body}}
    return lines


def _pseudo_batch_id(model_name: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
    return f"grok-async-{int(time.time())}-{safe_model}"


def _normalize_jsonish_text(text: str) -> str:
    return text.translate(SMART_PUNCT_TRANS)


def _render_prompt_for_request(req: "QuizRequest") -> Tuple[str, List[str]]:
    topics_list = [t.strip() for t in req.topics_csv.split(",") if t.strip()]

    # Target-driven generation: use explicit plan/counts instead of random sampling.
    if getattr(req, "topic_plan", None) or getattr(req, "topic_counts", None):
        if getattr(req, "topic_plan", None):
            plan = [str(t).strip() for t in (req.topic_plan or []) if str(t).strip()]
            if len(plan) != int(req.num_questions):
                raise ValueError(
                    f"topic_plan length {len(plan)} != num_questions {req.num_questions} for {req.quiz_id}"
                )
            topics_in_order: List[str] = []
            seen: set[str] = set()
            for t in plan:
                if t in seen:
                    continue
                topics_in_order.append(t)
                seen.add(t)
            counts_from_plan: Dict[str, int] = {}
            for t in plan:
                counts_from_plan[t] = int(counts_from_plan.get(t, 0)) + 1
            counts = counts_from_plan
            if getattr(req, "topic_counts", None):
                explicit_counts = {
                    str(k): int(v) for k, v in (req.topic_counts or {}).items() if str(k).strip() and int(v) > 0
                }
                if sum(explicit_counts.values()) != int(req.num_questions):
                    raise ValueError(
                        f"topic_counts sum {sum(explicit_counts.values())} != num_questions {req.num_questions} "
                        f"for {req.quiz_id}"
                    )
                if explicit_counts != counts_from_plan:
                    raise ValueError(
                        f"topic_counts mismatch topic_plan for {req.quiz_id}: "
                        f"counts_from_plan={counts_from_plan} explicit_counts={explicit_counts}"
                    )
                counts = explicit_counts
            spec = build_json_spec(
                topics_in_order,
                req.num_questions,
                req.quiz_id,
                num_choices=req.num_choices,
                topic_plan=plan,
                topic_counts=counts,
            )
            prompt = (
                "SYSTEM:\n"
                + GENERATOR_SYSTEM_INSTRUCTIONS
                + "\n\nTASK:\nGenerate the quiz now per the JSON schema.\n\n"
                + spec
            )
            return prompt, topics_in_order

        counts = {str(k): int(v) for k, v in (req.topic_counts or {}).items() if str(k).strip() and int(v) > 0}
        if sum(counts.values()) != int(req.num_questions):
            raise ValueError(
                f"topic_counts sum {sum(counts.values())} != num_questions {req.num_questions} for {req.quiz_id}"
            )
        plan: List[str] = []
        for t, n in counts.items():
            plan.extend([t] * int(n))
        if len(plan) != int(req.num_questions):
            raise ValueError(
                f"topic_counts expand to {len(plan)} topics, expected num_questions {req.num_questions} for {req.quiz_id}"
            )
        spec = build_json_spec(
            list(counts.keys()),
            req.num_questions,
            req.quiz_id,
            num_choices=req.num_choices,
            topic_plan=plan,
            topic_counts=counts,
        )
        prompt = (
            "SYSTEM:\n"
            + GENERATOR_SYSTEM_INSTRUCTIONS
            + "\n\nTASK:\nGenerate the quiz now per the JSON schema.\n\n"
            + spec
        )
        return prompt, list(counts.keys())

    rng = random.Random(req.seed)
    sample_k = min(len(topics_list), max(5, req.num_questions // 2))
    sampled_topics = rng.sample(topics_list, k=sample_k)
    spec = build_json_spec(sampled_topics, req.num_questions, req.quiz_id, num_choices=req.num_choices)
    prompt = (
        "SYSTEM:\n"
        + GENERATOR_SYSTEM_INSTRUCTIONS
        + "\n\nTASK:\nGenerate the quiz now per the JSON schema.\n\n"
        + spec
    )
    return prompt, sampled_topics


def _parse_quiz_from_text(raw_text: str, req: "QuizRequest", generator_model: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    blob = extract_json_block(raw_text)
    blob_norm = _normalize_jsonish_text(blob)
    try:
        data = json.loads(blob_norm)
    except Exception as exc:
        return None, f"Could not parse JSON for quiz {req.quiz_id}: {exc}"

    data["quiz_id"] = req.quiz_id
    data["difficulty"] = "very-hard"
    target_topics = None
    if getattr(req, "topic_plan", None):
        target_topics = [str(t).strip() for t in (req.topic_plan or [])]
    elif getattr(req, "topic_counts", None):
        expanded: List[str] = []
        for topic, n in (req.topic_counts or {}).items():
            topic_str = str(topic).strip()
            if not topic_str:
                continue
            try:
                count = int(n)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            expanded.extend([topic_str] * count)
        target_topics = expanded or None
    data_norm = normalize_quiz_items(
        data,
        num_choices=req.num_choices,
        generator_model=generator_model,
        seed=req.seed,
        target_topics=target_topics,
    )
    items = data_norm.get("items") or []
    data_norm["quiz_id"] = req.quiz_id
    for idx, it in enumerate(items, start=1):
        it["quiz_id"] = req.quiz_id
        it["question_id"] = f"{req.quiz_id}-{idx:03d}"

    if len(items) != req.num_questions:
        if len(items) < 1:
            return None, f"Generator returned zero valid items for {req.quiz_id}"
    return items, None


def generate_quizzes_via_grok_async(
    model_name: str,
    requests: List["QuizRequest"],
    *,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = 16000,
    overwrite: bool = False,
    max_concurrency: int = 8,
) -> Tuple[List["QuizResult"], Optional[str]]:
    """
    Grok async generation that mimics Batch output shapes used elsewhere.
    """
    if not requests:
        return [], None

    from quizbench.batch_generate_quiz import QuizResult, _extract_text_from_body  # type: ignore

    pseudo_batch_id = _pseudo_batch_id(model_name)

    grok_requests: List[GrokRequest] = []
    custom_id_to_meta: Dict[str, Dict[str, Any]] = {}
    for req in requests:
        prompt, sampled_topics = _render_prompt_for_request(req)
        custom_id = f"quiz__{req.quiz_id}"
        grok_requests.append(
            GrokRequest(
                custom_id=custom_id,
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_output_tokens or 16000,
                temperature=temperature,
                extra={"request": req, "sampled_topics": sampled_topics},
            )
        )
        custom_id_to_meta[custom_id] = {"request": req, "sampled_topics": sampled_topics}

    responses = run_grok_batch_sync(
        grok_requests,
        max_concurrency=max_concurrency,
    )
    batch_lines = grok_responses_to_batch_lines(responses)

    results: List[QuizResult] = []

    for custom_id, meta in custom_id_to_meta.items():
        req: "QuizRequest" = meta["request"]
        line = batch_lines.get(custom_id)
        if line is None:
            error_msg = f"No result returned for quiz {req.quiz_id} (custom_id={custom_id})"
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=pseudo_batch_id,
                    batch_input_path=None,
                )
            )
            continue

        if line.get("error"):
            err = line["error"]
            code = err.get("code", "unknown_code")
            msg = err.get("message", "no error message provided")
            error_msg = f"Batch error for quiz {req.quiz_id} ({code}): {msg}"
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=pseudo_batch_id,
                    batch_input_path=None,
                )
            )
            continue

        body = line.get("response", {}).get("body")
        if body is None:
            error_msg = f"Missing response body for quiz {req.quiz_id}"
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=pseudo_batch_id,
                    batch_input_path=None,
                )
            )
            continue

        content = _extract_text_from_body(body)
        items, err_msg = _parse_quiz_from_text(content, req, model_name)
        if err_msg:
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=err_msg,
                    batch_id=pseudo_batch_id,
                    batch_input_path=None,
                )
            )
            continue

        items = items or []
        if not items:
            error_msg = f"Generator returned zero valid items for {req.quiz_id}"
            results.append(
                QuizResult(
                    quiz_id=req.quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=pseudo_batch_id,
                    batch_input_path=None,
                )
            )
            continue

        ensure_dir(req.out_dir)
        out_path = Path(req.out_dir) / f"{req.quiz_id}.jsonl"
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
                    batch_id=pseudo_batch_id,
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
                batch_id=pseudo_batch_id,
                batch_input_path=None,
            )
        )

    return results, pseudo_batch_id


def call_grok_sync(
    model_name: str,
    prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> str:
    """
    Convenience wrapper to mimic call_llm-style synchronous usage.
    """
    req = GrokRequest(
        custom_id="inline",
        prompt=prompt,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    responses = run_grok_batch_sync(
        [req],
        max_concurrency=1,
    )
    resp = responses.get("inline")
    if resp is None:
        raise RuntimeError("No response returned from Grok.")
    if not resp.ok:
        raise RuntimeError(f"Grok error: {resp.error}")
    return resp.raw_text or ""


def run_on_model_batch_grok(
    model: str,
    items: List[Dict[str, Any]],
    quiz_id: str,
    out_dir: str,
    *,
    build_prompt,
    max_tokens: int,
    max_concurrency: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate one quiz on Grok via async fan-out, returning the same summary shape
    as run_on_model_batch.
    """
    from quizbench.eval_quiz import _sanitize_custom_id_for_anthropic  # reuse helper for ID hygiene
    from quizbench.utils import extract_answer_letter, letters

    ensure_dir(out_dir)

    allowed = letters(len(items[0]["options"])) if items else letters(5)
    pseudo_batch_id = _pseudo_batch_id(model)

    grok_requests: List[GrokRequest] = []
    for it in items:
        prompt = build_prompt(it)
        custom_id = f"{quiz_id}__{it['question_id']}"
        grok_requests.append(
            GrokRequest(
                custom_id=_sanitize_custom_id_for_anthropic(custom_id),
                prompt=prompt,
                model_name=model,
                max_tokens=max_tokens,
                temperature=0.0,
                extra={"question": it},
            )
        )

    responses = run_grok_batch_sync(
        grok_requests,
        max_concurrency=max_concurrency,
    )
    batch_lines = grok_responses_to_batch_lines(responses)

    per_item: List[Dict[str, Any]] = []
    corr = 0
    wrong = 0

    from quizbench.batch_generate_quiz import _extract_text_from_body  # type: ignore

    for it, grok_req in zip(items, grok_requests):
        custom_id = grok_req.custom_id
        line = batch_lines.get(custom_id)

        if line is None:
            raw = "Error: no result returned for this request."
            pred = None
        elif line.get("error"):
            err = line["error"]
            code = err.get("code", "unknown_code")
            msg = err.get("message", "no error message provided")
            raw = f"Batch error ({code}): {msg}"
            pred = None
        else:
            body = line["response"]["body"]
            raw = _extract_text_from_body(body).replace("**", "")
            pred = extract_answer_letter(raw, allowed)

        is_correct = (pred == it["answer"])
        corr += int(is_correct)
        wrong += int(not is_correct)

        rec = dict(it)
        rec.update({"pred": pred, "model_outputs": raw})
        per_item.append(rec)

    acc = corr / max(1, (corr + wrong))

    result_path = os.path.join(out_dir, f"{model}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(per_item, ensure_ascii=False))

    summary = {
        "model": model,
        "created_at": now_utc_iso(),
        "total_corr": corr,
        "total_wrong": wrong,
        "acc": acc,
        "n_items": (corr + wrong),
        "batch_input_file": None,
        "batch_id": pseudo_batch_id,
        "used_batch_api": True,
        "batch_provider": "grok-async",
    }
    summ_path = os.path.join(out_dir, f"{model}_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False))

    return summary
