#!/usr/bin/env python3
"""
Inspect the raw JSON text for a specific quiz in a completed Batch job.

This script:
  1. Re-downloads the Batch output/error JSONL files (via fetch_batch_output.py).
  2. Locates the line for the requested quiz (custom_id == f"quiz__<quiz_id>").
  3. Extracts the model text, pulls out the JSON block, and attempts json.loads.
  4. If parsing fails, prints the exception details and a small context snippet
     around the offending character position.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure imports work whether run from repo root or the quizbench/ directory.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.fetch_batch_output import fetch_batch_output  # noqa: E402
from quizbench.batch_generate_quiz import _extract_text_from_body  # noqa: E402
from quizbench.utils import extract_json_block, ensure_dir  # noqa: E402


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-download Batch output/error JSONL files for a quiz generation job "
            "and inspect the JSON for a specific quiz_id."
        )
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch id to inspect, e.g., batch_abc123.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for the batch (controls which API/key is used).",
    )
    parser.add_argument(
        "--quiz-id",
        required=True,
        help="Quiz id to inspect (without the 'quiz__' prefix).",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_uq_results/batch_outputs",
        help="Directory to store/download batch output/error JSONL files.",
    )
    return parser.parse_args()


def _load_batch_lines(out_dir: Path, batch_id: str) -> List[Dict]:
    """
    Load all JSONL lines written by fetch_batch_output for the given batch.
    """
    entries: List[Dict] = []
    for label in ("output", "error"):
        path = out_dir / f"{batch_id}_{label}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Could not parse {label} line as JSON: {exc}. "
                        f"Line prefix: {line[:200]}",
                        file=sys.stderr,
                    )
    return entries


def _find_entry_for_custom_id(entries: List[Dict], custom_id: str) -> Optional[Dict]:
    for entry in entries:
        if entry.get("custom_id") == custom_id:
            return entry
    return None


def _print_json_error_context(blob: str, exc: json.JSONDecodeError) -> None:
    """
    Print details and a short context snippet around the JSON decoding error.
    """
    print(
        f"[ERR] json.loads failed: {exc}. "
        f"(line={exc.lineno}, col={exc.colno}, pos={exc.pos})"
    )
    # Show a small window of characters around the failing position.
    radius = 80
    start = max(0, exc.pos - radius)
    end = min(len(blob), exc.pos + radius)
    snippet = blob[start:end]
    pointer_line = " " * (exc.pos - start) + "^"
    print("\n[CONTEXT] Around error position:")
    print(snippet)
    print(pointer_line)

    if 0 <= exc.pos < len(blob):
        offending = blob[exc.pos]
        print(
            f"[DETAIL] Offending char repr: {repr(offending)} "
            f"(ord={ord(offending)}, hex=0x{ord(offending):x})"
        )


def _normalize_jsonish_text(text: str) -> str:
    """
    Normalize common smart punctuation to ASCII to help json.loads succeed.
    """
    return text.translate(SMART_PUNCT_TRANS)


def _find_control_chars(s: str) -> List[int]:
    """
    Return indexes of control characters (ord < 32) other than \n, \r, \t.
    Useful to spot stray control bytes that break JSON parsing.
    """
    bad = []
    for i, ch in enumerate(s):
        if ord(ch) < 32 and ch not in ("\n", "\r", "\t"):
            bad.append(i)
    return bad


def _try_json_load(label: str, text: str) -> bool:
    """
    Attempt json.loads and print debug info on failure.
    """
    try:
        json.loads(text)
        print(f"[OK] {label} parsed successfully.")
        return True
    except json.JSONDecodeError as exc:
        print(f"[FAIL] {label} did not parse.")
        _print_json_error_context(text, exc)
        bad = _find_control_chars(text)
        if bad:
            preview = ", ".join(str(i) for i in bad[:10])
            extra = " (truncated)" if len(bad) > 10 else ""
            print(f"[INFO] Found control chars at positions: {preview}{extra}")
        return False


def main() -> None:
    args = _parse_args()
    batch_id: str = args.batch_id
    model_name: str = args.model
    quiz_id: str = args.quiz_id
    out_dir = Path(args.out_dir)

    ensure_dir(str(out_dir))

    # Step 1: (Re)download Batch output and error files.
    print(
        f"[INFO] Fetching batch output for {batch_id} "
        f"using model {model_name} into {out_dir}..."
    )
    fetch_batch_output(batch_id, model_name=model_name, out_dir=out_dir)

    # Step 2: Load all JSONL lines and locate the requested quiz entry.
    entries = _load_batch_lines(out_dir, batch_id)
    custom_id = f"quiz__{quiz_id}"
    entry = _find_entry_for_custom_id(entries, custom_id)
    if entry is None:
        print(
            f"[FATAL] No entry found for custom_id={custom_id}. "
            "Check that the quiz_id and batch-id are correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    if entry.get("error"):
        err = entry["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        print("[INFO] Entry contains an error object from the Batch job:")
        if code:
            print(f"  code   : {code}")
        print(f"  message: {msg}")
        return

    body = entry.get("response", {}).get("body")
    if body is None:
        print("[FATAL] Entry has no 'response.body'; nothing to inspect.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Extract the assistant text and JSON block.
    raw_text = _extract_text_from_body(body)
    print("\n[INFO] Raw model text (truncated to 2000 chars):")
    print(raw_text[:2000])

    blob = extract_json_block(raw_text)
    print("\n[INFO] Extracted JSON block (truncated to 2000 chars):")
    print(blob[:2000])

    blob_norm = _normalize_jsonish_text(blob)
    if blob_norm != blob:
        print("\n[INFO] Normalized JSON block differs from raw; showing normalized (truncated to 2000 chars):")
        print(blob_norm[:2000])

    # Step 4: Attempt to parse raw first, then normalized fallback.
    ok_raw = _try_json_load("Raw JSON block", blob)
    if ok_raw:
        print("\n[OK] Extracted JSON parses successfully (raw).")
        return

    ok_norm = _try_json_load("Normalized JSON block", blob_norm)
    if ok_norm:
        print("\n[OK] Extracted JSON parses successfully after normalization.")
        return

    print("\n[FATAL] JSON still failed to parse after normalization.")
    sys.exit(1)


if __name__ == "__main__":
    main()
