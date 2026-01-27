"""
Split a global `topic_mapping_report_*.json` into per-generator "shim" reports.

Why this exists
---------------
`quizbench/run_batch_gen_quiz.py` accepts `--topic_mapping_report` (a JSON report
emitted by `quizbench/apply_topic_mapping.py`). That report is typically created
for a *base* runs root (e.g. `.../runs/`), but some configs set `runs_root` to a
*per-generator* directory (e.g. `.../runs/deepseek-v3.2/`). In that case,
`run_batch_gen_quiz.py` warns when the report's `runs_root` does not exactly
match the config `runs_root`, even though the mapping metadata is still valid.

This script reads a single global `topic_mapping_report_*.json` and writes one
small per-generator JSON file under each generator directory with the same
mapping defaults but with `runs_root` rewritten to that generator directory.
These "shim" files can be passed back to `run_batch_gen_quiz.py` via
`--topic_mapping_report` to avoid the mismatch warning.

Example
-------
python quizbench/split_topic_mapping_report.py \
  --report topic_mapping_report_20251214T202513Z.json \
  --generator deepseek-v3.2

Then run:
python quizbench/run_batch_gen_quiz.py ... \
  --topic_mapping_report eval_results/quizbench/quizzes_Jan2026/runs/deepseek-v3.2/accumulated_mapped_topics.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REQUIRED_KEYS: Sequence[str] = (
    "targets_csv",
    "topic_map_path",
    "mode",
    "unmapped_policy",
    "runs_root",
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[FATAL] Failed to parse JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] Expected a JSON object at: {path}")
    return {str(k): v for k, v in payload.items()}


def _ensure_keys(payload: Mapping[str, Any], *, path: Path) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SystemExit(
            f"[FATAL] Report is missing required key(s) {missing} in {path}. "
            "Expected a topic_mapping_report_*.json from quizbench/apply_topic_mapping.py."
        )


def _iter_generator_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        raise SystemExit(f"[FATAL] runs_root does not exist: {runs_root}")
    if not runs_root.is_dir():
        raise SystemExit(f"[FATAL] runs_root is not a directory: {runs_root}")
    return [p for p in sorted(runs_root.iterdir()) if p.is_dir()]


def _parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _maybe_relativize(path_str: str | None, *, base: Path | None) -> str | None:
    if path_str is None:
        return None
    if base is None:
        return path_str
    try:
        abs_path = Path(str(path_str)).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return path_str
    try:
        return str(abs_path.relative_to(base))
    except ValueError:
        return path_str


def _build_shim_report(
    global_report: Mapping[str, Any],
    *,
    generator_model: str,
    generator_runs_root: Path,
    source_report_path: Path,
    relativize_base: Path | None,
) -> Dict[str, Any]:
    created_at = str(global_report.get("created_at") or "").strip() or _now_utc_iso()
    return {
        "created_at": created_at,
        "source_report_path": str(source_report_path),
        "generator_model": str(generator_model),
        # Critical: this must match the config's runs_root to avoid warnings.
        "runs_root": _maybe_relativize(str(generator_runs_root), base=relativize_base),
        # Defaults consumed by run_batch_gen_quiz.py when provided via --topic_mapping_report.
        "targets_csv": _maybe_relativize(str(global_report.get("targets_csv") or ""), base=relativize_base),
        "topic_map_path": _maybe_relativize(str(global_report.get("topic_map_path") or ""), base=relativize_base),
        "mode": str(global_report.get("mode") or ""),
        "unmapped_policy": str(global_report.get("unmapped_policy") or ""),
        # Extra provenance (ignored by run_batch_gen_quiz.py but useful for debugging).
        "global_runs_root": _maybe_relativize(str(global_report.get("runs_root") or ""), base=relativize_base),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create per-generator topic-mapping shim reports from a global topic_mapping_report_*.json. "
            "Each shim is written under <runs_root>/<generator>/<out_name> and can be passed to "
            "quizbench/run_batch_gen_quiz.py via --topic_mapping_report."
        )
    )
    ap.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to a topic_mapping_report_*.json (from quizbench/apply_topic_mapping.py).",
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help=(
            "Override the base runs_root to scan for generator subdirectories. "
            "If omitted, uses the report's runs_root."
        ),
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="accumulated_mapped_topics.json",
        help="Output filename written inside each generator directory (default: accumulated_mapped_topics.json).",
    )
    ap.add_argument(
        "--generator",
        type=str,
        action="append",
        default=[],
        help=(
            "Optional generator directory name to process (can be repeated). "
            "If omitted, processes all generator dirs under runs_root."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shim files if present.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be written without writing files.",
    )
    ap.add_argument(
        "--relative_to",
        type=str,
        default=None,
        help=(
            "If set, attempt to write paths relative to this directory (e.g. repo root). "
            "When unset, paths are written as-is (typically absolute)."
        ),
    )
    args = ap.parse_args(argv)

    report_path = Path(args.report).expanduser().resolve()
    report = _load_json_object(report_path)
    _ensure_keys(report, path=report_path)

    report_runs_root = str(report.get("runs_root") or "").strip()
    if args.runs_root:
        runs_root = Path(args.runs_root).expanduser().resolve()
    elif report_runs_root:
        runs_root = Path(report_runs_root).expanduser().resolve()
    else:
        raise SystemExit("[FATAL] No runs_root provided (missing in report and --runs_root not set).")

    relativize_base = Path(args.relative_to).expanduser().resolve() if args.relative_to else None

    generator_dirs = _iter_generator_dirs(runs_root)
    if not generator_dirs:
        raise SystemExit(f"[FATAL] No generator directories found under: {runs_root}")

    requested = [g for g in (args.generator or []) if str(g).strip()]
    if requested:
        wanted = set()
        for raw in requested:
            wanted.update(_parse_csv_list(raw))
        available = {p.name for p in generator_dirs}
        missing = sorted(wanted - available)
        if missing:
            raise SystemExit(
                f"[FATAL] Requested generator(s) not found under {runs_root}: {missing}. "
                f"Available: {sorted(available)}"
            )
        generator_dirs = [p for p in generator_dirs if p.name in wanted]

    wrote = 0
    skipped = 0
    for generator_dir in generator_dirs:
        out_path = generator_dir / str(args.out_name)
        payload = _build_shim_report(
            report,
            generator_model=generator_dir.name,
            generator_runs_root=generator_dir,
            source_report_path=report_path,
            relativize_base=relativize_base,
        )

        if out_path.exists() and not bool(args.overwrite):
            skipped += 1
            print(f"[INFO] Exists, skipping: {out_path}")
            continue

        if bool(args.dry_run):
            print(f"[DRY-RUN] Would write: {out_path}")
            continue

        _write_json(out_path, payload)
        wrote += 1
        print(f"[OK] Wrote: {out_path}")

    if bool(args.dry_run):
        print(f"[DRY-RUN] Complete. Would write {len(generator_dirs) - skipped} file(s); skipped_existing={skipped}.")
    else:
        print(f"[OK] Complete. Wrote {wrote} file(s); skipped_existing={skipped}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

