#!/usr/bin/env python3
import argparse, os, json, re, hashlib
from typing import List, Dict, Any
from tqdm import tqdm
from quizbench.clients import call_llm
from quizbench.utils import (
    read_jsonl, ensure_dir, format_mcq_prompt, extract_answer_letter, letters, now_utc_iso
)
from quizbench.prompt_templates import EVAL_HEADER
from quizbench.utils_grok import is_grok_model, run_on_model_batch_grok

from scripts.evaluate_from_batch_api import (
    BATCH_COMPATIBLE_MODELS,
    get_openai_client,
    get_gemini_file_client,
    _is_gemini_model,
    _openai_model_supports_sampling_params,
    submit_batch_job,
    wait_for_batch_completion,
    collect_batch_results,
    _extract_text_from_body,
)
from quizbench.anthropic_utils import (
    get_anthropic_client,
    create_message_batch as create_anthropic_message_batch,
    wait_for_batch_completion as wait_for_anthropic_batch_completion,
    collect_batch_results as collect_anthropic_batch_results,
    extract_text_from_anthropic_message,
)


def is_anthropic_model(model: str) -> bool:
    """
    Light-weight detector for Anthropic models (Claude family).
    """
    return model.strip().startswith("claude-")


def _sanitize_custom_id_for_anthropic(raw_id: str) -> str:
    """
    Anthropic Message Batch custom_id must match ^[a-zA-Z0-9_-]{1,64}$.
    Replace invalid chars, and shorten deterministically if needed.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", raw_id)
    if len(safe) <= 64:
        return safe

    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    prefix = safe[: 64 - len(digest) - 1]
    return f"{prefix}_{digest}"

def run_on_model(model: str,
                 items: List[Dict[str, Any]],
                 out_dir: str,
                 max_tokens: int = 600,
                 use_openrouter: bool = False) -> Dict[str, Any]:
    ensure_dir(out_dir)
    allowed = letters(5)
    per_item = []
    corr, wrong = 0, 0
    for it in tqdm(items, desc=f"[{model}]"):
        q = it["question"]; opts = it["options"]; gold = it["answer"]
        prompt = EVAL_HEADER + "\n" + format_mcq_prompt(q, opts, allow_rationale=True)
        raw, _reasoning_trace = call_llm(
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            openrouter=use_openrouter,
        )
        pred = extract_answer_letter(raw, allowed)
        is_correct = (pred == gold)
        corr += int(is_correct); wrong += int(not is_correct)
        rec = dict(it)
        rec.update({"pred": pred, "model_outputs": raw})
        per_item.append(rec)
    acc = corr / max(1, (corr + wrong))
    result_path = os.path.join(out_dir, f"{model}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(per_item, ensure_ascii=False))
    summary = {
        "model": model, "created_at": now_utc_iso(),
        "total_corr": corr, "total_wrong": wrong, "acc": acc,
        "n_items": (corr+wrong)
    }
    summ_path = os.path.join(out_dir, f"{model}_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False))
    return summary


def run_on_model_batch_anthropic(
    model: str,
    items: List[Dict[str, Any]],
    quiz_id: str,
    out_dir: str,
    args,
) -> Dict[str, Any]:
    """
    Evaluate one quiz on one Anthropic model using the Message Batches API.
    """
    ensure_dir(out_dir)

    allowed = letters(len(items[0]["options"])) if items else letters(5)
    client = get_anthropic_client()
    max_tokens = args.max_tokens if args.max_tokens is not None else 600

    batch_input_path = os.path.join(out_dir, f"{quiz_id}_{model}_anthropic_batch_requests.jsonl")
    requests = []
    custom_id_map = {}

    # Build batch input JSONL
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for it in items:
            qid = it["question_id"]
            prompt = EVAL_HEADER + "\n" + format_mcq_prompt(
                it["question"], it["options"], allow_rationale=True
            )
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            custom_id = _sanitize_custom_id_for_anthropic(qid)
            custom_id_map[qid] = custom_id
            req_obj = {"custom_id": custom_id, "params": params}
            requests.append(req_obj)
            f.write(json.dumps(req_obj) + "\n")

    message_batch = create_anthropic_message_batch(client, requests)
    message_batch = wait_for_anthropic_batch_completion(client, message_batch)
    batch_results = collect_anthropic_batch_results(client, message_batch)

    per_item = []
    corr, wrong = 0, 0

    for it in items:
        custom_id = custom_id_map.get(it["question_id"])
        entry = batch_results.get(custom_id) if custom_id else None

        if entry is None:
            raw = "Error: no result returned for this request."
            pred = None
        else:
            result_obj = entry.result
            result_type = getattr(result_obj, "type", None)

            if result_type == "succeeded":
                message = result_obj.message
                content = extract_text_from_anthropic_message(message)
                raw = content.replace("**", "")
                pred = extract_answer_letter(raw, allowed)
            elif result_type == "errored":
                err = result_obj.error
                err_type = getattr(err, "type", "unknown_error")
                err_msg = getattr(err, "message", "no error message provided")
                raw = f"Batch error ({err_type}): {err_msg}"
                pred = None
            elif result_type == "canceled":
                raw = "Batch request was canceled before processing."
                pred = None
            elif result_type == "expired":
                raw = "Batch request expired before it could be processed."
                pred = None
            else:
                raw = f"Unknown result type: {result_type}"
                pred = None

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
        "batch_input_file": os.path.basename(batch_input_path),
        "batch_id": getattr(message_batch, "id", None),
        "used_batch_api": True,
        "batch_provider": "anthropic",
    }
    summ_path = os.path.join(out_dir, f"{model}_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False))

    return summary

def run_on_model_batch(model: str,
                       items: List[Dict[str, Any]],
                       quiz_id: str,
                       out_dir: str,
                       args) -> Dict[str, Any]:
    """
    Evaluate one quiz on one model using the provider's batch API.

    - Anthropic (Claude): Message Batches API.
    - OpenAI: /v1/responses via Batch API.
    - Gemini: /v1/chat/completions via Gemini OpenAI-compat Batch API.
    - Fallback to per-call `run_on_model` if model is not batch-capable.
    """
    ensure_dir(out_dir)

    if is_grok_model(model):
        print(f"[INFO] Using Grok async batch-like flow for model '{model}'")

        def _build_prompt(item: Dict[str, Any]) -> str:
            return EVAL_HEADER + "\n" + format_mcq_prompt(
                item["question"], item["options"], allow_rationale=True
            )

        return run_on_model_batch_grok(
            model=model,
            items=items,
            quiz_id=quiz_id,
            out_dir=out_dir,
            build_prompt=_build_prompt,
            max_tokens=args.max_tokens if args.max_tokens is not None else 600,
            max_concurrency=int(os.getenv("GROK_MAX_CONCURRENCY", "8")),
        )

    if is_anthropic_model(model):
        print(f"[INFO] Using Anthropic Message Batches API for model '{model}'")
        return run_on_model_batch_anthropic(model, items, quiz_id=quiz_id, out_dir=out_dir, args=args)

    if model not in BATCH_COMPATIBLE_MODELS:
        print(
            f"[WARN] Model '{model}' not in BATCH_COMPATIBLE_MODELS; "
            f"falling back to per-call eval."
        )
        return run_on_model(
            model,
            items,
            out_dir=out_dir,
            max_tokens=args.max_tokens,
            use_openrouter=getattr(args, "use_openrouter", False),
        )

    allowed = letters(len(items[0]["options"])) if items else letters(5)

    is_gemini = _is_gemini_model(model)
    openai_client = get_openai_client(model)
    gemini_file_client = get_gemini_file_client() if is_gemini else None

    batch_input_path = os.path.join(out_dir, f"{quiz_id}_{model}_batch_input.jsonl")

    # 1) Build batch input JSONL
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for it in items:
            qid = it["question_id"]
            prompt = EVAL_HEADER + "\n" + format_mcq_prompt(
                it["question"], it["options"], allow_rationale=True
            )

            if is_gemini:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "top_p": 1,
                    "reasoning_effort": args.reasoning_effort,
                }
                url = "/v1/chat/completions"
            else:
                body = {"model": model, "input": prompt}
                if args.reasoning_effort:
                    body["reasoning"] = {"effort": args.reasoning_effort}
                if _openai_model_supports_sampling_params(model):
                    body["top_p"] = 1
                url = "/v1/responses"

            line_obj = {
                "custom_id": f"{quiz_id}__{qid}",
                "method": "POST",
                "url": url,
                "body": body,
            }
            f.write(json.dumps(line_obj) + "\n")

    # 2) Submit batch
    batch = submit_batch_job(
        openai_client,
        batch_input_path,
        is_gemini_model=is_gemini,
        gemini_file_client=gemini_file_client,
    )

    # 3) Wait for completion
    batch = wait_for_batch_completion(openai_client, batch)

    # 4) Collect results (output + error files merged)
    batch_results = collect_batch_results(
        openai_client,
        batch,
        is_gemini_model=is_gemini,
        gemini_file_client=gemini_file_client,
    )

    # 5) Build per-item records
    per_item = []
    corr, wrong = 0, 0

    for it in items:
        custom_id = f"{quiz_id}__{it['question_id']}"
        line = batch_results.get(custom_id)

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

    # 6) Save results & summary (same shapes as non-batch)
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
        "batch_input_file": os.path.basename(batch_input_path),
        "batch_id": getattr(batch, "id", None),
        "used_batch_api": True,
    }
    summ_path = os.path.join(out_dir, f"{model}_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False))

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiz_file", required=True, help="path to quiz JSONL (from generator)")
    ap.add_argument("--test_models_csv", required=True, help="comma-separated model names")
    ap.add_argument("--out_dir", type=str, default="eval_results/quizbench/runs")
    ap.add_argument(
        "--use_batch_api",
        action="store_true",
        help=(
            "Use Batch APIs where supported: OpenAI/Gemini Batch or "
            "Anthropic Message Batches for Claude models."
        ),
    )
    ap.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Route DeepSeek/Kimi calls through OpenRouter when set.",
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Reasoning effort for Responses/Gemini (e.g., low, medium, high).",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=600,
        help="Max tokens for per-call (non-batch) mode.",
    )

    args = ap.parse_args()

    items = read_jsonl(args.quiz_file)
    quiz_id = os.path.splitext(os.path.basename(args.quiz_file))[0]
    run_dir = os.path.join(args.out_dir, quiz_id)
    ensure_dir(run_dir)

    test_models = [m.strip() for m in args.test_models_csv.split(",") if m.strip()]
    manifest = {"quiz_id": quiz_id, "n_items": len(items), "models": test_models, "created_at": now_utc_iso()}
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # [ ] TODO - FIX/SIMPLIFY BATCH HANDLING LOGIC
    all_summ = []
    for m in test_models:
        is_batch_capable = (m in BATCH_COMPATIBLE_MODELS) or is_anthropic_model(m) or is_grok_model(m)
        is_deepseek_or_kimi = m.lower().startswith(("deepseek-", "kimi-"))

        if args.use_batch_api:
            if is_batch_capable:
                summ = run_on_model_batch(m, items, quiz_id=quiz_id, out_dir=run_dir, args=args)
            elif is_deepseek_or_kimi:
                print(
                    f"[WARN] Model '{m}' is not batch-capable; falling back to per-call evaluation with OpenRouter={args.use_openrouter}."
                )
                summ = run_on_model(
                    m,
                    items,
                    out_dir=run_dir,
                    max_tokens=args.max_tokens,
                    use_openrouter=args.use_openrouter,
                )
            else:
                raise SystemExit(
                    f"[FATAL] Model '{m}' is not batch-capable and no fallback is defined when --use_batch_api is set."
                )
        else:
            summ = run_on_model(
                m,
                items,
                out_dir=run_dir,
                max_tokens=args.max_tokens,
                use_openrouter=args.use_openrouter,
            )
        all_summ.append(summ)


    with open(os.path.join(run_dir, "summary_all_models.json"), "w") as f:
        json.dump(all_summ, f)
    print(run_dir)  # machine-readable

if __name__ == "__main__":
    main()
