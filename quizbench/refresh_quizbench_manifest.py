#!/usr/bin/env python3
"""
Refresh (merge) QuizBench manifest files after adding quizzes.

Why this exists
---------------
QuizBench generation scripts often write *snapshot* manifests (e.g.,
`quizbench_manifest_Jan2026.json`, `quizbench_manifest_Jan2026_<ts>.json`)
that list only the quizzes produced in that specific run. In iterative
workflows where you add more quizzes over time, downstream steps sometimes
benefit from having a single "consolidated" manifest that contains the union
of all quiz_ids for a batch tag.

This utility:
  - Merges all `quizbench_manifest*.json` files in each generator directory.
  - Deduplicates quizzes by `quiz_id`.
  - Optionally fixes missing `quiz_path` entries by locating `<quiz_id>.jsonl`
    under the collection's `quizzes_dir` (or `quizzes/`).
  - Optionally fills `target_topic` by re-reading quiz JSONL files.

Typical usage
-------------
Consolidate all generator manifests under a batch runs root:

  uv run quizbench/refresh_quizbench_manifest.py \\
    --runs_root eval_results/quizbench/quizzes_Jan2026/runs \\
    --quiz_batch_tag Jan2026
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.utils import now_utc_iso  # noqa: E402


_SEED_RE = re.compile(r"(?:^|_)seed(\d+)$")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge quizbench_manifest*.json files into a single consolidated manifest per generator.",
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        required=True,
        help=(
            "Either a generator directory containing quizbench_manifest*.json, "
            "or a batch runs root containing generator subdirectories."
        ),
    )
    ap.add_argument(
        "--quiz_batch_tag",
        type=str,
        required=True,
        help="Batch tag used for output filename: quizbench_manifest_<TAG>.json (e.g., Jan2026).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would change, but do not write any files.",
    )
    ap.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to back up an existing output manifest before overwriting "
            "(default: True)."
        ),
    )
    ap.add_argument(
        "--backup_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write backups into (default: tmp/manifest_backups/<ts>/). "
            "Only used when --backup is enabled and the output manifest exists."
        ),
    )
    ap.add_argument(
        "--fix_quiz_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to fix quiz entries whose quiz_path does not exist by searching "
            "for <quiz_id>.jsonl under quizzes_dir/quizzes_root_dir (default: True)."
        ),
    )
    ap.add_argument(
        "--recompute_target_topics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to (re)read quiz JSONL files and set target_topic for each quiz "
            "(default: False)."
        ),
    )
    ap.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to sort merged quizzes by quiz_id for determinism (default: True).",
    )
    return ap.parse_args()


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT_DIR / p)


def _to_repo_relative(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT_DIR.resolve())
    except Exception:  # noqa: BLE001
        return str(path)
    return rel.as_posix()


def _prefer_non_backup_path(path: Path) -> Path:
    """
    If `path` points into a directory component ending with "_backup", prefer the
    sibling path with that suffix stripped when it exists.
    """
    parts = list(path.parts)
    if not any(p.endswith("_backup") for p in parts):
        return path

    alt_parts = [p[: -len("_backup")] if p.endswith("_backup") else p for p in parts]
    alt = Path(*alt_parts)
    return alt if alt.exists() else path


def _safe_int(val: object) -> int | None:
    try:
        return int(val)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _parse_seed_from_quiz_id(quiz_id: str) -> int | None:
    parts = str(quiz_id).strip().split("_")
    if not parts:
        return None
    m = _SEED_RE.search(parts[-1])
    if not m:
        return None
    return _safe_int(m.group(1))


def _read_target_topics_from_quiz_jsonl(path: Path) -> list[str | None]:
    topics: list[str | None] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    topics.append(None)
                    continue
                raw = row.get("target_topic")
                if raw is None:
                    topics.append(None)
                else:
                    val = str(raw).strip()
                    topics.append(val or None)
    except FileNotFoundError:
        return []
    return topics


def _as_str_set(val: object) -> set[str]:
    if not isinstance(val, list):
        return set()
    out: set[str] = set()
    for item in val:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return out


def _best_quizzes_dir_from_manifest(manifest: dict[str, Any]) -> Path | None:
    """
    Best-effort resolve to a quizzes directory that exists on disk.
    """
    candidates: list[Path] = []

    quizzes_dir = manifest.get("quizzes_dir")
    if isinstance(quizzes_dir, str) and quizzes_dir.strip():
        candidates.append(_resolve_repo_path(quizzes_dir.strip()))

    quizzes_root_dir = manifest.get("quizzes_root_dir")
    quiz_collection = manifest.get("quiz_collection")
    if isinstance(quizzes_root_dir, str) and quizzes_root_dir.strip():
        root = _resolve_repo_path(quizzes_root_dir.strip())
        if isinstance(quiz_collection, str) and quiz_collection.strip():
            candidates.append(root / quiz_collection.strip())
        candidates.append(root)

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def _index_quiz_files(root: Path, *, recursive: bool) -> dict[str, Path]:
    """
    Return mapping quiz_id -> absolute path for *.jsonl files.
    """
    if not root.exists() or not root.is_dir():
        return {}
    paths = root.rglob("*.jsonl") if recursive else root.glob("*.jsonl")
    out: dict[str, Path] = {}
    for p in paths:
        if p.is_file():
            out[p.stem] = p.resolve()
    return out


@dataclass(frozen=True)
class _MergedQuiz:
    quiz_id: str
    payload: dict[str, Any]
    score: int


def _score_quiz_entry(entry: dict[str, Any], *, quiz_path_exists: bool) -> int:
    score = 0
    if quiz_path_exists:
        score += 10
    if entry.get("target_topic"):
        score += 2
    if entry.get("seed") is not None:
        score += 1
    if entry.get("generator_model") or entry.get("generator"):
        score += 1
    return score


def _merge_quiz_entries(
    *,
    existing: _MergedQuiz | None,
    incoming: dict[str, Any],
    quiz_path_exists: bool,
) -> _MergedQuiz:
    quiz_id = str(incoming.get("quiz_id") or "").strip()
    payload = dict(incoming)

    # Normalize quiz_id.
    payload["quiz_id"] = quiz_id

    score = _score_quiz_entry(payload, quiz_path_exists=quiz_path_exists)
    if existing is None:
        return _MergedQuiz(quiz_id=quiz_id, payload=payload, score=score)

    # Prefer higher-scoring payloads; on ties, prefer payload with more keys.
    if score > existing.score or (score == existing.score and len(payload) > len(existing.payload)):
        return _MergedQuiz(quiz_id=quiz_id, payload=payload, score=score)

    # Otherwise, keep existing but fill missing keys opportunistically.
    merged = dict(existing.payload)
    for k, v in payload.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return _MergedQuiz(quiz_id=quiz_id, payload=merged, score=existing.score)


def _discover_generator_dirs(runs_root: Path) -> list[Path]:
    """
    If runs_root itself is a generator dir, return [runs_root].
    Otherwise, return its immediate subdirs that contain quizbench_manifest*.json.
    """
    if any(runs_root.glob("quizbench_manifest*.json")):
        return [runs_root]

    out: list[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("quizbench_manifest*.json")):
            out.append(child)
    return out


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _pick_base_manifest(
    manifests: list[tuple[Path, dict[str, Any]]],
    *,
    preferred_name: str,
) -> tuple[Path, dict[str, Any]]:
    for p, m in manifests:
        if p.name == preferred_name:
            return p, m

    # Fallback: newest by mtime, tie-break by name.
    return max(manifests, key=lambda pm: (pm[0].stat().st_mtime, pm[0].name))


def _merge_manifests_for_generator(
    *,
    generator_dir: Path,
    batch_tag: str,
    fix_quiz_paths: bool,
    recompute_target_topics: bool,
    sort_quizzes: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """
    Return (out_path, merged_manifest, stats).
    """
    manifest_paths = sorted([p for p in generator_dir.glob("quizbench_manifest*.json") if p.is_file()])
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for p in manifest_paths:
        m = _load_manifest(p)
        if m is not None:
            loaded.append((p, m))

    if not loaded:
        raise SystemExit(f"[FATAL] No readable quizbench_manifest*.json under {generator_dir}")

    preferred_out_name = f"quizbench_manifest_{batch_tag}.json"
    base_path, base = _pick_base_manifest(loaded, preferred_name=preferred_out_name)

    # Build quiz file indices for path-fixing and optional target_topic recompute.
    quizzes_dir = _best_quizzes_dir_from_manifest(base)
    if quizzes_dir is None:
        # Try other manifests.
        for _, m in loaded:
            quizzes_dir = _best_quizzes_dir_from_manifest(m)
            if quizzes_dir is not None:
                break

    # Always build a fallback global index under quizzes/ (repo-local).
    global_quizzes_root = ROOT_DIR / "quizzes"
    global_index = _index_quiz_files(global_quizzes_root, recursive=True)
    local_index: dict[str, Path] = {}
    if quizzes_dir is not None:
        local_index = _index_quiz_files(quizzes_dir, recursive=False)

    merged_by_id: dict[str, _MergedQuiz] = {}

    missing_quiz_id_rows = 0
    initial_quiz_rows = 0
    fixed_paths = 0
    recomputed_topics = 0

    for _, manifest in loaded:
        quizzes = manifest.get("quizzes") or []
        if not isinstance(quizzes, list):
            continue
        for q in quizzes:
            if not isinstance(q, dict):
                continue
            initial_quiz_rows += 1
            quiz_id = str(q.get("quiz_id") or "").strip()
            if not quiz_id:
                missing_quiz_id_rows += 1
                continue

            entry = dict(q)
            entry["quiz_id"] = quiz_id

            # Ensure seed exists.
            if entry.get("seed") is None:
                seed = _parse_seed_from_quiz_id(quiz_id)
                if seed is not None:
                    entry["seed"] = seed

            # Resolve quiz_path existence and optionally fix it.
            raw_quiz_path = entry.get("quiz_path")
            quiz_path: Path | None = None
            if isinstance(raw_quiz_path, str) and raw_quiz_path.strip():
                raw_quiz_path = raw_quiz_path.strip()
                candidate = _resolve_repo_path(raw_quiz_path)
                candidate = _prefer_non_backup_path(candidate)
                if candidate.exists():
                    quiz_path = candidate.resolve()
                    if candidate != _resolve_repo_path(raw_quiz_path):
                        # Normalize to repo-relative when we rewrite away from *_backup.
                        entry["quiz_path"] = (
                            str(candidate) if Path(raw_quiz_path).is_absolute() else _to_repo_relative(candidate)
                        )

            if quiz_path is None and fix_quiz_paths:
                found = local_index.get(quiz_id) or global_index.get(quiz_id)
                if found is not None and found.exists():
                    preferred = _prefer_non_backup_path(found)
                    entry["quiz_path"] = _to_repo_relative(preferred)
                    quiz_path = preferred
                    fixed_paths += 1

            if quiz_path is not None and recompute_target_topics:
                entry["target_topic"] = _read_target_topics_from_quiz_jsonl(quiz_path)
                recomputed_topics += 1

            merged = _merge_quiz_entries(
                existing=merged_by_id.get(quiz_id),
                incoming=entry,
                quiz_path_exists=quiz_path is not None and quiz_path.exists(),
            )
            merged_by_id[quiz_id] = merged

    merged_quizzes = [mq.payload for mq in merged_by_id.values()]
    if sort_quizzes:
        merged_quizzes = sorted(merged_quizzes, key=lambda row: str(row.get("quiz_id") or ""))

    merged_manifest = dict(base)
    merged_manifest["quizzes"] = merged_quizzes

    # Union some common metadata fields across manifests.
    eval_models: set[str] = set()
    judge_models: set[str] = set()
    run_ids: set[str] = set()
    judge_run_ids: set[str] = set()
    for _, m in loaded:
        eval_models |= _as_str_set(m.get("eval_models"))
        judge_models |= _as_str_set(m.get("judge_models"))
        run_ids |= _as_str_set(m.get("run_ids"))
        judge_run_ids |= _as_str_set(m.get("judge_run_ids"))

    if eval_models:
        merged_manifest["eval_models"] = sorted(eval_models)
    if judge_models:
        merged_manifest["judge_models"] = sorted(judge_models)
    if run_ids:
        merged_manifest["run_ids"] = sorted(run_ids)
    if judge_run_ids:
        merged_manifest["judge_run_ids"] = sorted(judge_run_ids)

    merged_manifest["manifest_refreshed_at"] = now_utc_iso()
    merged_manifest["manifest_inputs"] = [p.name for p, _ in loaded]

    # Prefer a quizzes_dir that exists (and is repo-relative).
    if quizzes_dir is not None:
        merged_manifest["quizzes_dir"] = _to_repo_relative(quizzes_dir)

    out_path = generator_dir / preferred_out_name
    stats: dict[str, Any] = {
        "generator_dir": str(generator_dir),
        "input_manifests": [str(p) for p, _ in loaded],
        "base_manifest": str(base_path),
        "out_path": str(out_path),
        "initial_quiz_rows": initial_quiz_rows,
        "merged_quiz_ids": len(merged_by_id),
        "missing_quiz_id_rows": missing_quiz_id_rows,
        "fixed_quiz_paths": fixed_paths,
        "recomputed_target_topics": recomputed_topics,
    }
    return out_path, merged_manifest, stats


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _backup_if_needed(out_path: Path, *, backup_dir: Path) -> Path | None:
    if not out_path.exists():
        return None
    _ensure_dir(backup_dir)
    backup_path = backup_dir / out_path.name
    backup_path.write_bytes(out_path.read_bytes())
    return backup_path


def main() -> None:
    args = _parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists() or not runs_root.is_dir():
        raise SystemExit(f"[FATAL] runs_root is not a directory: {runs_root}")

    batch_tag = str(args.quiz_batch_tag).strip()
    if not batch_tag:
        raise SystemExit("[FATAL] --quiz_batch_tag cannot be empty.")

    generator_dirs = _discover_generator_dirs(runs_root)
    if not generator_dirs:
        raise SystemExit(f"[FATAL] No generator dirs with quizbench_manifest*.json under: {runs_root}")

    backup_dir: Path | None = None
    if args.backup:
        if args.backup_dir:
            backup_dir = Path(args.backup_dir).expanduser().resolve()
        else:
            backup_dir = (ROOT_DIR / "tmp" / "manifest_backups" / now_utc_iso().replace(":", "").replace("-", ""))

    for gen_dir in generator_dirs:
        out_path, merged_manifest, stats = _merge_manifests_for_generator(
            generator_dir=gen_dir,
            batch_tag=batch_tag,
            fix_quiz_paths=bool(args.fix_quiz_paths),
            recompute_target_topics=bool(args.recompute_target_topics),
            sort_quizzes=bool(args.sort),
        )

        print(f"[INFO] {gen_dir.name}: {stats['merged_quiz_ids']} quizzes (from {len(stats['input_manifests'])} manifests)")
        print(f"       fixed_quiz_paths={stats['fixed_quiz_paths']} recomputed_target_topics={stats['recomputed_target_topics']}")
        print(f"       out_path={out_path}")

        if args.dry_run:
            continue

        if args.backup and backup_dir is not None:
            per_gen_backup_dir = backup_dir / gen_dir.name
            backup_path = _backup_if_needed(out_path, backup_dir=per_gen_backup_dir)
            if backup_path is not None:
                print(f"[INFO] Backed up existing manifest to: {backup_path}")

        _write_json(out_path, merged_manifest)
        print(f"[OK] Wrote consolidated manifest: {out_path}")

    print(str(runs_root))


if __name__ == "__main__":
    main()
