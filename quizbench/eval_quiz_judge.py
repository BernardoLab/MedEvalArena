#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from quizbench.clients import call_llm
from quizbench.utils import (
    read_jsonl,
    ensure_dir,
    extract_json_block,
    now_utc_iso,
    letters,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "cot_prompt_lib" / "judge_mcq_validity_system_prompt.txt"
_PROMPT_CACHE: Optional[str] = None


def load_judge_system_prompt() -> str:
    """
    Read and cache the judge system prompt from cot_prompt_lib.
    """
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        try:
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                _PROMPT_CACHE = f.read().strip()
        except FileNotFoundError as exc:
            raise SystemExit(f"[FATAL] Missing judge system prompt at {PROMPT_PATH}") from exc
    return _PROMPT_CACHE


def build_judge_prompt(item: Dict[str, Any], system_prompt: str) -> str:
    """
    Construct the user prompt for a single MCQ item, embedding the system
    instructions at the top as required by call_llm's single-string contract.
    """
    options = item.get("options") or []
    labels = letters(len(options)) if options else []

    lines: List[str] = []
    qid = item.get("question_id") or item.get("id")
    assert qid
    lines.append(f"Question ID: {qid}")

    question_text = str(item.get("question", "")).strip()
    lines.append(f"Question: {question_text}")
    lines.append("Options:")
    for idx, opt in enumerate(options):
        label = labels[idx] if idx < len(labels) else f"Opt{idx+1}"
        lines.append(f"{label}. {opt}")

    ans = str(item.get("answer", "")).strip()
    assert ans
    lines.append(f"Correct answer key: {ans}")

    explanation = str(item.get("explanation", "")).strip()
    assert explanation
    lines.append(f"Answer explanation: {explanation}")

    mcq_block = "\n".join(lines)
    return (
        system_prompt.strip()
        + "\n\n--- MCQ ITEM ---\n"
        + mcq_block
        + "\n\nReturn only the JSON object specified above; do not include markdown."
    )


def _normalize_bool(val: Any) -> Tuple[Optional[bool], Optional[str]]:
    if isinstance(val, bool):
        return val, None
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true", None
    return None, "safety_flag must be a boolean"


def validate_judge_json(payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate and normalize the parsed judge JSON against required fields.
    Returns (normalized_payload, error_message).
    """
    if not isinstance(payload, dict):
        return None, "Parsed JSON is not an object"

    # New minimal schema: used by current judge prompts
    minimal_keys = {"analysis", "medical_accuracy_score", "logical_validity", "logical_false_reason"}
    assert minimal_keys.issubset(payload.keys())

    analysis = payload.get("analysis")
    if not isinstance(analysis, str):
        return None, "analysis must be a string"

    logical_valid = payload.get("logical_validity")
    if not isinstance(logical_valid, bool):
        return None, "logical_validity must be a boolean"

    lfr_raw = str(payload.get("logical_false_reason", "")).strip().upper()
    valid_lfr = {"C", "N", "M", "U", "K", "T"}
    if lfr_raw not in valid_lfr:
        return None, "logical_false_reason must be one of C/N/M/U/K/T"

    score_raw = payload.get("medical_accuracy_score")
    try:
        score_int = int(score_raw)
    except (TypeError, ValueError):
        return None, "medical_accuracy_score must be an integer between 1 and 5"
    if not 1 <= score_int <= 5:
        return None, "medical_accuracy_score must be between 1 and 5"

    # Map logical_validity / logical_false_reason into legacy verdict/fail_reason.
    # When the item is logically valid, logical_false_reason must be "T";
    # we record a PASS with fail_reason="NA". When invalid, logical_false_reason
    # must be one of C/N/M/U/K and becomes the fail_reason.
    if logical_valid and lfr_raw != "T":
        return None, "logical_false_reason must be T when logical_validity is true"
    if (not logical_valid) and lfr_raw not in {"C", "N", "M", "U", "K"}:
        return None, "logical_false_reason must be C/N/M/U/K when logical_validity is false"

    verdict = "PASS" if logical_valid else "FAIL"
    fail_reason = "NA" if logical_valid else lfr_raw

    normalized = {
        "analysis": analysis,
        "medical_accuracy_score": score_int,
        "logical_validity": logical_valid,
        "logical_false_reason": lfr_raw,
        "verdict": verdict,
        "fail_reason": fail_reason,
    }
    return normalized, None


def run_judge_on_model(
    model: str,
    items: List[Dict[str, Any]],
    quiz_id: Optional[str],
    out_dir: str,
    max_tokens: int = 1200,
    use_openrouter: bool = False,
    judge_mode: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a quiz with the judge model and persist per-item and summary JSON.
    """
    ensure_dir(out_dir)
    system_prompt = load_judge_system_prompt()

    per_item: List[Dict[str, Any]] = []
    full_traces: List[Dict[str, Any]] = []
    valid_responses = 0
    n_pass = 0
    n_fail = 0
    fail_reason_counts: Dict[str, int] = {}
    score_values: List[int] = []
    safety_true = 0

    for idx, item in enumerate(tqdm(items, desc=f"[{model}] judge")):
        prompt = build_judge_prompt(item, system_prompt)
        raw, reasoning_trace = call_llm(
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            openrouter=use_openrouter,
            judge_mode=judge_mode
        )

        json_block = extract_json_block(raw)
        parsed_json: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        output_valid = False

        try:
            parsed_obj = json.loads(json_block)
        except Exception as exc:  # noqa: BLE001
            parse_error = f"json decode error: {exc}"
        else:
            normalized, err = validate_judge_json(parsed_obj)
            if err:
                parse_error = err
            else:
                parsed_json = normalized
                output_valid = True

        rec = dict(item)
        if quiz_id:
            rec.setdefault("quiz_id", quiz_id)
        rec.setdefault("question_id", rec.get("question_id") or f"item-{idx+1:03d}")

        rec.update(
            {
                "judge_model": model,
                "judge_model_outputs_raw": raw,
                "judge_output_valid": output_valid,
                "judge_parse_error": parse_error,
            }
        )

        if parsed_json:
            verdict = parsed_json["verdict"]
            fail_reason = parsed_json["fail_reason"]
            score = parsed_json["medical_accuracy_score"]
            safety = parsed_json.get("safety_flag")

            rec["judge_json"] = parsed_json
            rec["judge_verdict"] = verdict
            rec["judge_fail_reason"] = fail_reason
            rec["judge_medical_accuracy_score"] = score
            if safety is not None:
                rec["judge_safety_flag"] = safety

            valid_responses += 1
            if verdict == "PASS":
                n_pass += 1
            else:
                n_fail += 1
            fail_reason_counts[fail_reason] = fail_reason_counts.get(fail_reason, 0) + 1
            score_values.append(score)
            if safety:
                safety_true += 1

        if reasoning_trace is not None:
            full_traces.append(
                {
                    "quiz_id": rec.get("quiz_id"),
                    "question_id": rec["question_id"],
                    "judge_model": model,
                    "provider_response": reasoning_trace,
                }
            )

        per_item.append(rec)

    result_path = os.path.join(out_dir, f"{model}_judge_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(per_item, ensure_ascii=False))

    score_mean = None
    if score_values:
        score_mean = sum(score_values) / len(score_values)

    summary = {
        "model": model,
        "quiz_id": quiz_id,
        "created_at": now_utc_iso(),
        "n_items": len(items),
        "n_valid_responses": valid_responses,
        "n_invalid_responses": len(items) - valid_responses,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "fail_reason_counts": fail_reason_counts,
        "medical_score_mean": score_mean,
        "n_safety_flag_true": safety_true,
        "used_batch_api": False,
    }

    summ_path = os.path.join(out_dir, f"{model}_judge_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False))

    if full_traces:
        trace_path = os.path.join(out_dir, f"{model}_judge_full_reasoning_traces.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(full_traces, f, ensure_ascii=False, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser(description="Judge MCQ validity and accuracy for a single quiz JSONL.")
    ap.add_argument("--quiz_file", required=True, help="Path to quiz JSONL (from generator).")
    ap.add_argument("--judge_model", required=True, help="Judge LLM to call (e.g., gpt-4o).")
    ap.add_argument("--out_dir", type=str, default="eval_results/quizbench_judge/runs", help="Root output directory.")
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Max tokens for judge responses (per call).",
    )
    ap.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Route DeepSeek/Kimi judge calls through OpenRouter when set.",
    )
    ap.add_argument(
        "--use_batch_api",
        action="store_true",
        help="(Reserved) Use batch APIs where supported; currently falls back to per-call.",
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort parameter for batch-compatible APIs (not yet used).",
    )

    args = ap.parse_args()

    items = read_jsonl(args.quiz_file)
    quiz_id = os.path.splitext(os.path.basename(args.quiz_file))[0]
    run_dir = os.path.join(args.out_dir, quiz_id)
    ensure_dir(run_dir)

    manifest = {
        "quiz_id": quiz_id,
        "quiz_file": args.quiz_file,
        "n_items": len(items),
        "judge_model": args.judge_model,
        "created_at": now_utc_iso(),
    }
    with open(os.path.join(run_dir, "judge_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.use_batch_api:
        print("[WARN] Batch judge API not yet implemented; falling back to per-call mode.")

    summary = run_judge_on_model(
        args.judge_model,
        items,
        quiz_id=quiz_id,
        out_dir=run_dir,
        max_tokens=args.max_tokens,
        use_openrouter=args.use_openrouter,
        judge_mode=True # ALWAYS TRUE IN THIS CASE, GETS PASSED TO clients.py
    )

    with open(os.path.join(run_dir, "summary_all_judges.json"), "w", encoding="utf-8") as f:
        json.dump([summary], f, ensure_ascii=False)

    print(run_dir)  # machine-readable for chaining


if __name__ == "__main__":
    main()
