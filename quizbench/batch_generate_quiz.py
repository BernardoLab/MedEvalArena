#!/usr/bin/env python3
"""
Batch-compatible quiz generator for QuizBench.

Supports:
  - OpenAI Batch API via /v1/responses
  - Gemini Batch API via the OpenAI compatibility layer (/v1/chat/completions)

Designed after scripts/evaluate_from_batch_api.py to keep behavior consistent with the
existing synchronous generator (generate_quiz.py). This CLI builds and submits
a single quiz-generation request per Batch job.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.generate_quiz_spec import (  # noqa: E402
    GENERATOR_SYSTEM_INSTRUCTIONS,
    build_json_spec,
)
from quizbench.run_gen_quiz import build_quiz_id  # noqa: E402
from quizbench.utils import (  # noqa: E402
    ensure_dir,
    extract_json_block,
    normalize_quiz_items,
    compact_utc_ts,
    write_jsonl,
)
from quizbench.utils_grok import generate_quizzes_via_grok_async, is_grok_model  # noqa: E402

DEFAULT_TOPICS = (
    "cardiology,endocrinology,infectious disease,hematology/oncology,neurology,"
    "nephrology,pulmonology,obstetrics,gynecology,pediatrics,geriatrics,"
    "dermatology,rheumatology,emergency medicine,critical care"
)

# Batch polling cadence (seconds) when waiting for completion
BATCH_POLL_INTERVAL_SECONDS = 60


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


def _normalize_jsonish_text(text: str) -> str:
    """
    Normalize common “smart” punctuation to ASCII so that slightly
    non-literal JSON (e.g., curly quotes around keys/strings) can still
    be parsed by the strict json library.
    """
    return text.translate(SMART_PUNCT_TRANS)


def _is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini-")


def _openai_model_supports_sampling_params(model_name: str) -> bool:
    # GPT-5 variants restrict sampling knobs; omit for those.
    return not model_name.startswith("gpt-5")


def get_openai_client(model_name: str) -> OpenAI:
    """
    Returns an OpenAI client configured for Batch API operations.

    - For OpenAI models, talks to api.openai.com (or OPENAI_API_BASE_URL override).
    - For Gemini models, uses the Gemini OpenAI-compatible endpoint.
    """
    if _is_gemini_model(model_name):
        api_key = os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("GEMINI_API_BASE_URL", "").strip() or "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to use Gemini batch generation.")
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to use OpenAI batch generation.")
    openai_base_url = os.environ.get("OPENAI_API_BASE_URL", "").strip()
    if openai_base_url:
        return OpenAI(api_key=api_key, base_url=openai_base_url)
    return OpenAI(api_key=api_key)


def get_gemini_file_client():
    """
    Returns a google-genai Client for file upload/download when using
    Gemini via the OpenAI compatibility layer.
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'google-genai' package is required for Gemini batch generation. "
            "Install it with: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to use Gemini batch generation.")
    return genai.Client(api_key=api_key)


def submit_batch_job(openai_client: OpenAI, batch_input_path: str, *, is_gemini_model: bool = False, gemini_file_client=None, description: str | None = None):
    """
    Uploads the JSONL file and submits a Batch job.
    Mirrors the logic from scripts/evaluate_from_batch_api.py for compatibility.
    """
    print(f"Uploading batch input file: {batch_input_path}")
    endpoint = "/v1/chat/completions" if is_gemini_model else "/v1/responses"

    if is_gemini_model:
        if gemini_file_client is None:
            raise RuntimeError(
                "Gemini file client is required when submitting a Gemini Batch job. "
                "Create one with get_gemini_file_client()."
            )
        display_name = os.path.basename(batch_input_path)
        upload_config = {
            "display_name": display_name,
            "mime_type": "jsonl",
        }
        uploaded_file = gemini_file_client.files.upload(
            file=batch_input_path,
            config=upload_config,
        )
        input_file_id = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "file", None) or uploaded_file
        print(f"Uploaded batch input file to Gemini Files API with id/name: {input_file_id}")

        batch = openai_client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={"description": description or f"QuizBench quiz generation: {display_name}"},
        )
    else:
        with open(batch_input_path, "rb") as f:
            batch_input_file = openai_client.files.create(file=f, purpose="batch")
        print(f"Created batch input file with id: {batch_input_file.id}")
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={"description": description or f"QuizBench quiz generation: {os.path.basename(batch_input_path)}"},
        )

    print(f"Created batch with id: {batch.id}")
    print(f"Initial batch status: {batch.status}")
    return batch


def wait_for_batch_completion(openai_client: OpenAI, batch, poll_interval: int = BATCH_POLL_INTERVAL_SECONDS):
    """
    Polls the Batch job until it reaches a terminal state.
    """
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    while getattr(batch, "status", None) not in terminal_states:
        print(f"Batch {batch.id} status: {batch.status}. Sleeping {poll_interval}s...")
        time.sleep(poll_interval)
        batch = openai_client.batches.retrieve(batch.id)

    print(f"Batch {batch.id} reached terminal status: {batch.status}")
    if getattr(batch, "errors", None):
        print("Batch reported errors:", batch.errors)
    return batch


def _parse_batch_file(openai_client: OpenAI, file_id: Optional[str], *, is_gemini_model: bool = False, gemini_file_client=None) -> Dict[str, dict]:
    """
    Given an output or error file id, return dict custom_id -> line data.
    """
    results: Dict[str, dict] = {}
    if not file_id:
        return results
    if is_gemini_model:
        if gemini_file_client is None:
            raise RuntimeError(
                "Gemini file client is required when downloading Gemini Batch results."
            )
        print(f"Downloading batch file via Gemini Files API: {file_id}")
        file_bytes = gemini_file_client.files.download(file=file_id)
        content = file_bytes.decode("utf-8")
    else:
        print(f"Downloading batch file: {file_id}")
        file_response = openai_client.files.content(file_id)
        content = file_response.text

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        custom_id = data["custom_id"]
        results[custom_id] = data
    return results


def collect_batch_results(openai_client: OpenAI, batch, *, is_gemini_model: bool = False, gemini_file_client=None) -> Dict[str, dict]:
    """
    Retrieves and merges results from the batch's output and error files.
    """
    output_results = _parse_batch_file(
        openai_client,
        getattr(batch, "output_file_id", None),
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
    )
    error_results = _parse_batch_file(
        openai_client,
        getattr(batch, "error_file_id", None),
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
    )
    combined = {}
    combined.update(output_results)
    combined.update(error_results)
    return combined


def _extract_text_from_body(body) -> str:
    """
    Extract assistant text from a Batch response body.
    Supports both Responses API output and Chat Completions output.
    """
    try:
        if not isinstance(body, dict):
            return str(body)

        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output_items = body.get("output")
        if isinstance(output_items, list):
            text_chunks: List[str] = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "message":
                    for part in item.get("content", []) or []:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        if part_type in ("output_text", "text", "input_text"):
                            text = part.get("text")
                            if isinstance(text, str):
                                text_chunks.append(text)
                elif item_type in ("output_text", "text") and isinstance(item.get("text"), str):
                    text_chunks.append(item["text"])
            if text_chunks:
                return "".join(text_chunks)

        if "choices" in body:
            choice = body["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                content = "".join(parts)
            if not isinstance(content, str):
                content = str(content)
            return content

        return json.dumps(body)[:2000]
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error parsing response body: {exc}. Raw body (truncated): {json.dumps(body)[:2000]}"


@dataclass
class QuizRequest:
    quiz_id: str
    seed: int
    num_questions: int
    num_choices: int = 5
    topics_csv: str = DEFAULT_TOPICS
    topic_plan: Optional[List[str]] = None
    topic_counts: Optional[Dict[str, int]] = None
    out_dir: str = "data_medARCv1/quizbench_quizzes"


@dataclass
class QuizResult:
    quiz_id: str
    seed: int
    generator_model: str
    quiz_path: Optional[str]
    status: str
    error: Optional[str] = None
    batch_id: Optional[str] = None
    batch_input_path: Optional[str] = None


def _render_prompt(request: QuizRequest) -> Tuple[str, List[str]]:
    """
    Build the generator prompt and return (prompt, sampled_topics).
    Sampling is deterministic per request.seed.
    """
    topics_list = [t.strip() for t in request.topics_csv.split(",") if t.strip()]

    # Target-driven generation: use explicit plan/counts instead of random sampling.
    if request.topic_plan or request.topic_counts:
        if request.topic_plan:
            plan = [str(t).strip() for t in request.topic_plan if str(t).strip()]
            if len(plan) != int(request.num_questions):
                raise ValueError(
                    f"topic_plan length {len(plan)} != num_questions {request.num_questions} for {request.quiz_id}"
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
            if request.topic_counts:
                explicit_counts = {
                    str(k): int(v) for k, v in request.topic_counts.items() if str(k).strip() and int(v) > 0
                }
                if sum(explicit_counts.values()) != int(request.num_questions):
                    raise ValueError(
                        f"topic_counts sum {sum(explicit_counts.values())} != num_questions {request.num_questions} "
                        f"for {request.quiz_id}"
                    )
                if explicit_counts != counts_from_plan:
                    raise ValueError(
                        f"topic_counts mismatch topic_plan for {request.quiz_id}: "
                        f"counts_from_plan={counts_from_plan} explicit_counts={explicit_counts}"
                    )
                counts = explicit_counts
            spec = build_json_spec(
                topics_in_order,
                request.num_questions,
                request.quiz_id,
                num_choices=request.num_choices,
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

        counts = {str(k): int(v) for k, v in (request.topic_counts or {}).items() if str(k).strip() and int(v) > 0}
        if sum(counts.values()) != int(request.num_questions):
            raise ValueError(
                f"topic_counts sum {sum(counts.values())} != num_questions {request.num_questions} for {request.quiz_id}"
            )
        plan: List[str] = []
        for t, n in counts.items():
            plan.extend([t] * int(n))
        if len(plan) != int(request.num_questions):
            raise ValueError(
                f"topic_counts expand to {len(plan)} topics, expected num_questions {request.num_questions} for {request.quiz_id}"
            )
        spec = build_json_spec(
            list(counts.keys()),
            request.num_questions,
            request.quiz_id,
            num_choices=request.num_choices,
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

    rng = random.Random(request.seed)
    sample_k = min(len(topics_list), max(5, request.num_questions // 2))
    sampled_topics = rng.sample(topics_list, k=sample_k)
    spec = build_json_spec(sampled_topics, request.num_questions, request.quiz_id, num_choices=request.num_choices)
    prompt = (
        "SYSTEM:\n"
        + GENERATOR_SYSTEM_INSTRUCTIONS
        + "\n\nTASK:\nGenerate the quiz now per the JSON schema.\n\n"
        + spec
    )
    return prompt, sampled_topics


def _target_topics_for_request(request: QuizRequest) -> List[str] | None:
    """
    Return the per-item target topic list that was embedded into the prompt.

    In target-driven generation we pass either:
      - request.topic_plan (explicit per-item topics), or
      - request.topic_counts (expanded deterministically by insertion order).

    In flat/sampled generation this returns None.
    """
    if request.topic_plan:
        return [str(t).strip() for t in request.topic_plan]

    if request.topic_counts:
        plan: List[str] = []
        for topic, n in request.topic_counts.items():
            topic_str = str(topic).strip()
            if not topic_str:
                continue
            try:
                count = int(n)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            plan.extend([topic_str] * count)
        return plan or None

    return None


def build_batch_input_file(
    model_name: str,
    requests: List[QuizRequest],
    batch_input_path: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = 4000,
) -> Dict[str, dict]:
    """
    Build a JSONL file with exactly one request per quiz.

    Returns mapping custom_id -> metadata for downstream parsing.
    """
    ensure_dir(str(Path(batch_input_path).parent))
    is_gemini_model = _is_gemini_model(model_name)
    custom_id_to_meta: Dict[str, dict] = {}

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in requests:
            prompt, sampled_topics = _render_prompt(req)

            if is_gemini_model:
                body = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if _openai_model_supports_sampling_params(model_name):
                    body["temperature"] = temperature
                    body["top_p"] = 1
                if max_output_tokens:
                    body["max_tokens"] = max_output_tokens
                url = "/v1/chat/completions"
            else:
                body = {
                    "model": model_name,
                    "input": prompt,
                }
                if _openai_model_supports_sampling_params(model_name):
                    body["temperature"] = temperature
                    body["top_p"] = 1
                if max_output_tokens:
                    body["max_output_tokens"] = max_output_tokens
                url = "/v1/responses"

            custom_id = f"quiz__{req.quiz_id}"
            line_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": url,
                "body": body,
            }
            f.write(json.dumps(line_obj) + "\n")
            custom_id_to_meta[custom_id] = {
                "request": req,
                "sampled_topics": sampled_topics,
            }

    return custom_id_to_meta


def _parse_quiz_from_text(raw_text: str, req: QuizRequest, generator_model: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Parse model raw text -> normalized quiz items.

    Returns: (items, error_message)
    """
    blob = extract_json_block(raw_text)
    blob_norm = _normalize_jsonish_text(blob)
    try:
        data = json.loads(blob_norm)
    except Exception as exc:
        return None, f"Could not parse JSON for quiz {req.quiz_id}: {exc}"

    data["quiz_id"] = req.quiz_id
    data["difficulty"] = "very-hard"
    target_topics = _target_topics_for_request(req)
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
        print(f"[WARN] Requested {req.num_questions}, got {len(items)} valid for {req.quiz_id}. Continuing.")
    return items, None


def generate_quizzes_via_batch(
    model_name: str,
    requests: List[QuizRequest],
    *,
    batch_input_path: str,
    poll_interval: int = BATCH_POLL_INTERVAL_SECONDS,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = 4000,
    overwrite: bool = False,
) -> Tuple[List[QuizResult], Optional[str]]:
    """
    Submit a Batch job to generate quizzes, then parse and write outputs.

    Each quiz is sent as a single request (no chunking).
    Returns (results, batch_id).
    """
    if not requests:
        return [], None

    if is_grok_model(model_name):
        results, pseudo_batch_id = generate_quizzes_via_grok_async(
            model_name,
            requests,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            overwrite=overwrite,
            max_concurrency=int(os.getenv("GROK_MAX_CONCURRENCY", "8")),
        )
        return results, pseudo_batch_id

    is_gemini_model = _is_gemini_model(model_name)
    openai_client = get_openai_client(model_name)
    gemini_file_client = get_gemini_file_client() if is_gemini_model else None

    custom_id_to_meta = build_batch_input_file(
        model_name,
        requests,
        batch_input_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    if not custom_id_to_meta:
        return [], None

    batch = submit_batch_job(
        openai_client,
        batch_input_path,
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
        description=f"QuizBench batch quiz generation ({model_name})",
    )
    batch = wait_for_batch_completion(openai_client, batch, poll_interval=poll_interval)
    batch_results = collect_batch_results(
        openai_client,
        batch,
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
    )

    results: List[QuizResult] = []

    for custom_id, meta in custom_id_to_meta.items():
        req: QuizRequest = meta["request"]
        quiz_id = req.quiz_id

        line = batch_results.get(custom_id)
        if line is None:
            error_msg = f"No result returned for quiz {quiz_id} (custom_id={custom_id})"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            continue

        if line.get("error"):
            err = line["error"]
            code = err.get("code", "unknown_code")
            msg = err.get("message", "no error message provided")
            error_msg = f"Batch error for quiz {quiz_id} ({code}): {msg}"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            continue

        body = line.get("response", {}).get("body")
        if body is None:
            error_msg = f"Missing response body for quiz {quiz_id}"
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=error_msg,
                    batch_id=getattr(batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            continue

        content = _extract_text_from_body(body)
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
                    batch_id=getattr(batch, "id", None),
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
                    batch_id=getattr(batch, "id", None),
                    batch_input_path=batch_input_path,
                )
            )
            continue

        ensure_dir(req.out_dir)
        out_path = Path(req.out_dir) / f"{quiz_id}.jsonl"
        try:
            write_jsonl(out_path, items, overwrite=overwrite)
        except FileExistsError:
            results.append(
                QuizResult(
                    quiz_id=quiz_id,
                    seed=req.seed,
                    generator_model=model_name,
                    quiz_path=None,
                    status="error",
                    error=f"Refusing to overwrite existing quiz file: {out_path}",
                    batch_id=getattr(batch, "id", None),
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
                batch_id=getattr(batch, "id", None),
                batch_input_path=batch_input_path,
            )
        )
        print(f"[OK] Generated quiz {quiz_id} at {out_path}")

    return results, getattr(batch, "id", None)


def _default_topics(cfg: dict) -> str:
    return str(cfg.get("topics_csv") or DEFAULT_TOPICS)


def _cli_requests_from_args(args) -> List[QuizRequest]:
    """
    Build a single QuizRequest object from CLI args for standalone use.

    The Batch job created by this script always contains exactly one
    quiz-generation request.
    """
    if args.quiz_id:
        base_quiz_id = f"{args.quiz_id}_{compact_utc_ts()}"
    else:
        base_quiz_id = build_quiz_id(args.generator_model, args.seed0)

    req = QuizRequest(
        quiz_id=base_quiz_id,
        seed=args.seed0,
        num_questions=args.num_questions,
        num_choices=args.num_choices,
        topics_csv=args.topics_csv,
        out_dir=args.out_dir,
    )
    return [req]


def main():
    ap = argparse.ArgumentParser(
        description="Batch-based quiz generator using OpenAI or Gemini Batch APIs."
    )
    ap.add_argument("--generator_model", required=True, help="e.g., gpt-4o, gemini-2.5-pro")
    ap.add_argument("--num_questions", type=int, default=10)
    ap.add_argument("--num_choices", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--quiz_id", type=str, default=None, help="Optional quiz_id (defaults to timestamp-based).")
    ap.add_argument("--topics_csv", type=str, default=DEFAULT_TOPICS)
    ap.add_argument("--out_dir", type=str, default="data_medARCv1/quizbench_quizzes")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow overwriting an existing <quiz_id>.jsonl in --out_dir.",
    )
    ap.add_argument("--batch_input_path", type=str, default=None, help="Where to write the Batch input JSONL. Defaults to <out_dir>/<generator_model>_batch_input.jsonl.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_output_tokens", type=int, default=4000)
    ap.add_argument("--poll_seconds", type=int, default=BATCH_POLL_INTERVAL_SECONDS)
    args = ap.parse_args()

    requests = _cli_requests_from_args(args)
    batch_input_path = args.batch_input_path
    if not batch_input_path:
        safe_model = args.generator_model.replace("/", "-")
        batch_input_path = str(Path(args.out_dir) / f"{safe_model}_batch_input.jsonl")

    results, batch_id = generate_quizzes_via_batch(
        args.generator_model,
        requests,
        batch_input_path=batch_input_path,
        poll_interval=args.poll_seconds,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        overwrite=bool(args.overwrite),
    )

    for res in results:
        status = res.status.upper()
        path_fragment = f" -> {res.quiz_path}" if res.quiz_path else ""
        print(f"[{status}] {res.quiz_id}{path_fragment}")

    if batch_id:
        print(f"Batch job id: {batch_id}")


if __name__ == "__main__":
    main()
