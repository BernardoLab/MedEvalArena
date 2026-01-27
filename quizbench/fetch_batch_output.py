#!/usr/bin/env python3
"""
Download the output/error files for a completed OpenAI (or Gemini) Batch job and
summarize the lines that came back. Useful for debugging quiz generation runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure imports work whether run from repo root or the quizbench/ directory.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.batch_generate_quiz import (  # noqa: E402
    get_gemini_file_client,
    get_openai_client,
)
from quizbench.utils import ensure_dir  # noqa: E402


def _is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini-")


def _download_batch_file(
    batch_id: str,
    file_id: Optional[str],
    label: str,
    *,
    out_dir: Path,
    is_gemini_model: bool,
    openai_client,
    gemini_file_client=None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Download a batch output/error file and write it locally.
    Returns (path, raw_text).
    """
    if not file_id:
        return None, None

    print(f"[INFO] Downloading {label} file for {batch_id}: {file_id}")
    if is_gemini_model:
        if gemini_file_client is None:
            raise RuntimeError("Gemini file client is required when is_gemini_model=True.")
        file_bytes = gemini_file_client.files.download(file=file_id)
        raw_text = file_bytes.decode("utf-8")
    else:
        file_response = openai_client.files.content(file_id)
        raw_text = file_response.text

    out_path = out_dir / f"{batch_id}_{label}.jsonl"
    ensure_dir(str(out_path.parent))
    out_path.write_text(raw_text, encoding="utf-8")
    return out_path, raw_text


def _parse_jsonl_lines(raw_text: Optional[str]) -> Dict[str, dict]:
    parsed: Dict[str, dict] = {}
    if not raw_text:
        return parsed
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Could not parse line: {exc}. Line content: {line[:200]}")
            continue
        custom_id = entry.get("custom_id") or f"line_{len(parsed)}"
        parsed[custom_id] = entry
    return parsed


def fetch_batch_output(batch_id: str, *, model_name: str, out_dir: Path) -> None:
    is_gemini_model = _is_gemini_model(model_name)
    openai_client = get_openai_client(model_name)
    gemini_file_client = get_gemini_file_client() if is_gemini_model else None

    batch = openai_client.batches.retrieve(batch_id)
    print(f"[INFO] Batch {batch_id} status: {getattr(batch, 'status', 'unknown')}")
    counts = getattr(batch, "request_counts", None)
    if counts:
        print(f"[INFO] Request counts: {counts}")
    errors = getattr(batch, "errors", None)
    if errors:
        print(f"[INFO] Batch reported errors object: {errors}")

    output_path, output_text = _download_batch_file(
        batch_id,
        getattr(batch, "output_file_id", None),
        "output",
        out_dir=out_dir,
        is_gemini_model=is_gemini_model,
        openai_client=openai_client,
        gemini_file_client=gemini_file_client,
    )
    error_path, error_text = _download_batch_file(
        batch_id,
        getattr(batch, "error_file_id", None),
        "error",
        out_dir=out_dir,
        is_gemini_model=is_gemini_model,
        openai_client=openai_client,
        gemini_file_client=gemini_file_client,
    )

    combined: Dict[str, dict] = {}
    combined.update(_parse_jsonl_lines(output_text))
    combined.update(_parse_jsonl_lines(error_text))

    ok_count = sum(1 for entry in combined.values() if not entry.get("error"))
    err_count = len(combined) - ok_count
    print(f"[INFO] Parsed {len(combined)} total lines ({ok_count} ok, {err_count} error).")

    if err_count:
        print("[INFO] First few errors:")
        shown = 0
        for custom_id, entry in combined.items():
            if not entry.get("error"):
                continue
            err = entry["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            print(f"  - {custom_id}: {msg}")
            shown += 1
            if shown >= 5:
                break

    if output_path:
        print(f"[OK] Wrote batch output lines to {output_path}")
    else:
        print("[WARN] Batch has no output_file_id; nothing downloaded.")

    if error_path:
        print(f"[OK] Wrote batch error lines to {error_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download output/error files for a Batch job and print a quick summary."
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="The Batch id to fetch, e.g., batch_abc123.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for the batch (controls which API base/key is used).",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_uq_results/batch_outputs",
        help="Directory to store downloaded JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    fetch_batch_output(args.batch_id, model_name=args.model, out_dir=out_dir)


if __name__ == "__main__":
    main()
