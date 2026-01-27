import re, json, datetime, uuid, os
from typing import List, Dict, Any, Optional

CHOICE_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def letters(n:int)->List[str]:
    return list(CHOICE_MAP[:n])

def hard_exit(msg:str, code:int=1):
    raise SystemExit(f"[FATAL] {msg}")

def now_utc_iso()->str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def compact_utc_ts(at:datetime.datetime|None=None)->str:
    """
    Return a file/name-safe UTC timestamp with milliseconds.
    Format: YYYYMMDDTHHMMSSmmmZ (no colons or dashes).
    """
    ts = at or datetime.datetime.utcnow()
    return f"{ts.strftime('%Y%m%dT%H%M%S')}{ts.microsecond//1000:03d}Z"

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def extract_json_block(text:str)->str:
    """
    Pull the first well-formed JSON object or array from text.
    Handles ```json fences``` and stray commentary.
    """
    # Strip fences if present
    m = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # Fallback: first {...} or [...]
    m2 = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return text.strip()

def extract_answer_letter(text:str, allowed:List[str])->Optional[str]:
    """
    Robustly pull a single letter choice from model output.
    Priority:
      - 'The answer is (X)'  (case-insensitive)
      - 'Answer: X'
      - last standalone capital [A..] in text
    """
    pats = [
        r"(?i)the\s+answer\s+is\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)\banswer\s*:\s*([A-Z])\b",
        r"\(([A-Z])\)\s*$",
    ]
    for p in pats:
        m = re.search(p, text.strip())
        if m:
            v = m.group(1).upper()
            return v if v in allowed else None
    # Last single capital letter
    m = re.findall(r"\b([A-Z])\b", text.upper())
    if m:
        v = m[-1]
        return v if v in allowed else None
    return None

def format_mcq_prompt(question:str, options:List[str], allow_rationale:bool=True)->str:
    letters_list = letters(len(options))
    body = "Question: " + question.strip() + "\nOptions:\n"
    for i, opt in enumerate(options):
        body += f"{letters_list[i]}. {opt}\n"
    tail = (
        "Pick the single best answer.\n"
        "Output MUST end with: The answer is (X)\n"
        "Use only one capital letter (A, B, C, D, E)."
    )
    if not allow_rationale:
        tail = "Respond with a single line ending in: The answer is (X)"
    return body + tail

def validate_quiz_item(item:Dict[str, Any], num_choices:int=5)->Optional[str]:
    # We synthesize stable question_ids downstream, so do not require
    # the model to supply them. Only core content fields are mandatory.
    required = ["question", "options", "answer", "explanation"]
    for k in required:
        if k not in item:
            return f"Missing field: {k}"
    if not isinstance(item["options"], list) or len(item["options"]) != num_choices:
        return f"Options must have exactly {num_choices}"
    allowed = letters(num_choices)
    ans = str(item["answer"]).strip().upper()
    if ans not in allowed:
        return f"Answer must be one of {allowed}, got {ans}"
    return None

def normalize_quiz_items(
    quiz: Dict[str, Any],
    num_choices: int = 5,
    generator_model: str | None = None,
    seed: int | None = None,
    target_topics: List[str] | None = None,
) -> Dict[str, Any]:
    items = []
    for idx, it in enumerate(quiz.get("items", [])):
        err = validate_quiz_item(it, num_choices=num_choices)
        if err:
            continue
        ans_letter = str(it["answer"]).strip().upper()
        ans_index = letters(num_choices).index(ans_letter)
        out = {
            "quiz_id": quiz.get("quiz_id"),
            "question_id": it.get("question_id"),
            "question": it.get("question"),
            "options": it.get("options"),
            "answer": ans_letter,
            "answer_index": ans_index,
            "explanation": it.get("explanation"),
            "generator_model": generator_model,
            "seed": seed,
            "category": "medicine",
            "difficulty": quiz.get("difficulty", "very-hard"),
            "not_for_clinical_use": True,
        }
        if target_topics is not None:
            topic_val = None
            if idx < len(target_topics):
                raw_topic = target_topics[idx]
                if raw_topic is not None:
                    topic_str = str(raw_topic).strip()
                    topic_val = topic_str or None
            out["target_topic"] = topic_val
        items.append(out)
    quiz["items"] = items
    return quiz

def write_jsonl(path: str, rows: List[Dict[str, Any]], *, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path:str)->List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
