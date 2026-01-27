#!/usr/bin/env python3
import argparse, os, json, uuid, random
from datetime import datetime

from quizbench.clients import call_llm
from quizbench.generate_quiz_spec import (
    GENERATOR_SYSTEM_INSTRUCTIONS,
    build_json_spec,
)
from quizbench.utils import (
    ensure_dir,
    extract_json_block,
    normalize_quiz_items,
    write_jsonl,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_model", required=True,
                    help="e.g., gpt-4o, gemini-1.5-pro-latest, claude-3-sonnet-20240229")
    ap.add_argument("--num_questions", type=int, default=10)
    ap.add_argument("--num_choices", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topics_csv", type=str, default="cardiology,endocrinology,infectious disease,hematology/oncology,neurology,nephrology,pulmonology,obstetrics,gynecology,pediatrics,geriatrics,dermatology,rheumatology,emergency medicine,critical care")
    ap.add_argument("--out_dir", type=str, default="data_medARCv1/quizbench_quizzes")
    ap.add_argument("--quiz_id", type=str, default=None,
                    help="Optional explicit quiz_id to use (helps align filenames/manifests).")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow overwriting an existing <quiz_id>.jsonl in --out_dir.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out_dir)
    quiz_id = args.quiz_id or f"quiz_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

    topics_list = [t.strip() for t in args.topics_csv.split(",") if t.strip()]
    sampled = random.sample(topics_list, k=min(len(topics_list), max(5, args.num_questions//2)))
    spec = build_json_spec(sampled, args.num_questions, quiz_id, num_choices=args.num_choices)

    prompt = (
        "SYSTEM:\n" + GENERATOR_SYSTEM_INSTRUCTIONS + "\n\n"
        "TASK:\nGenerate the quiz now per the JSON schema.\n\n"
        + spec
    )

    raw, _reasoning_trace = call_llm(
        args.generator_model,
        prompt,
        max_tokens=4000,
        temperature=0.0,
    )
    blob = extract_json_block(raw)
    try:
        data = json.loads(blob)
    except Exception as e:
        raise SystemExit(f"[FATAL] Could not parse JSON from generator. Error: {e}\nRaw:\n{raw[:1200]}")

    data["quiz_id"] = quiz_id
    data["difficulty"] = "very-hard"
    data_norm = normalize_quiz_items(
        data,
        num_choices=args.num_choices,
        generator_model=args.generator_model,
        seed=args.seed,
    )
    items = data_norm["items"]
    data_norm["quiz_id"] = quiz_id
    for idx, it in enumerate(items, start=1):
        it["quiz_id"] = quiz_id
        it["question_id"] = f"{quiz_id}-{idx:03d}"

    if len(items) != args.num_questions:
        # Accept fewer if model didn't reach target, but enforce >=1
        if len(items) < 1:
            raise SystemExit("[FATAL] Generator returned zero valid items.")
        print(f"[WARN] Requested {args.num_questions}, got {len(items)} valid. Continuing.")

    out_path = os.path.join(args.out_dir, f"{data_norm['quiz_id']}.jsonl")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"[FATAL] Refusing to overwrite existing quiz file: {out_path}")
    write_jsonl(out_path, items, overwrite=bool(args.overwrite))
    print(out_path)  # machine-readable path for caller

if __name__ == "__main__":
    main()
