#!/usr/bin/env python3
"""
Build a per-generator ABMS-quota subset of *judge-valid* questions.

This script:
  - scans one or more source QuizBench runs roots (each containing generator dirs),
  - filters questions using the same criteria as aggregate_filtered_results.sh:
      - min_medical_score=1
      - require_logical_valid=True
      - logical_mode=majority
  - assigns each candidate question to a canonical ABMS specialty with precedence:
      1) per-question `target_topic` field in the quiz JSONL (preferred)
      2) else `topics_*_mapped.json` in the run dir (uses `topic_mapped`)
      3) else hard error
  - selects exactly the quotas in data/ABMS_specialties.csv (total=50),
  - writes 5 synthetic 10-question quiz JSONL files per generator (chunk_size=10),
  - writes a minimal quizbench_manifest_<TAG>.json per generator under a new runs tree.

Artifacts are written under:
  quizzes/quizzes_<TAG>/
  eval_results/quizbench/quizzes_<TAG>/runs/

The TAG should include date info (e.g., ABMS2026011).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


# Ensure imports work when invoked as a script (python quizbench/build_abms_valid_subset.py).
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES, filter_by_judge  # noqa: E402
from quizbench.target_planning import IntegerTargets, load_integer_targets_csv  # noqa: E402
from quizbench.topic_mapping import build_canonical_index, normalize_label  # noqa: E402
from quizbench.utils import ensure_dir, now_utc_iso, read_jsonl, write_jsonl  # noqa: E402


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_id_fragment(value: str) -> str:
    """
    Make a safe identifier fragment usable in filenames and quiz_ids.
    """
    text = (value or "").strip()
    text = _SAFE_ID_RE.sub("_", text)
    return text.strip("_.-") or "unknown"


def _today_utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_csv_arg(val: str | None) -> List[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


def _load_yaml_string_list(path: Path) -> List[str]:
    """
    Minimal YAML list loader for files like:
      - item1
      - item2

    Avoids hard dependency on PyYAML for this simple case.
    """
    text = path.read_text(encoding="utf-8")
    out: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not line.startswith("-"):
            raise ValueError(f"Unsupported YAML (expected list items): {path} :: {raw_line!r}")
        item = line[1:].strip()
        if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
            item = item[1:-1]
        item = item.strip()
        if item:
            out.append(item)
    return out


def _load_topic_map_overrides(path: Path) -> Tuple[bool, Dict[str, str]]:
    """
    Parse a small topic-map YAML file (e.g., data/topic_to_abms.yaml) without PyYAML.

    We only need:
      - top-level `normalize: true/false` (default: True)
      - the `map:` block of `src: dst` pairs
    """
    text = path.read_text(encoding="utf-8")
    normalize = True
    in_map = False
    map_indent: int | None = None
    overrides: Dict[str, str] = {}

    for raw_line in text.splitlines():
        # Preserve indentation for block parsing; strip comments afterward.
        if "#" in raw_line:
            raw_line = raw_line.split("#", 1)[0]
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if not in_map and indent == 0 and line.startswith("normalize:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in {"true", "false"}:
                normalize = val == "true"
            continue

        if not in_map and line == "map:":
            in_map = True
            map_indent = indent
            continue

        if in_map:
            if map_indent is not None and indent <= map_indent:
                in_map = False
                map_indent = None
                continue
            if ":" not in line:
                continue
            raw_src, raw_dst = line.split(":", 1)
            src = raw_src.strip()
            dst = raw_dst.strip()
            if (dst.startswith('"') and dst.endswith('"')) or (dst.startswith("'") and dst.endswith("'")):
                dst = dst[1:-1].strip()
            if not src or not dst:
                continue
            key = normalize_label(src) if normalize else src
            overrides[key] = dst

    return normalize, overrides


def _resolve_existing_path(path_str: str) -> Path | None:
    """
    Resolve a path that may be absolute or repo-root-relative.
    Returns the resolved path if it exists, else None.
    """
    raw = (path_str or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    else:
        p = p.resolve()
    return p if p.exists() else None


def _build_quiz_jsonl_index(quizzes_root: Path) -> Dict[str, List[Path]]:
    """
    Build a mapping quiz_id -> list of quiz JSONL paths under quizzes_root.
    """
    index: Dict[str, List[Path]] = defaultdict(list)
    if not quizzes_root.exists():
        return index
    for path in quizzes_root.rglob("*.jsonl"):
        if not path.is_file():
            continue
        index[path.stem].append(path.resolve())
    return index


def _load_manifest_quiz_paths(generator_dir: Path) -> Dict[str, List[str]]:
    """
    Load quiz_id -> [quiz_path,...] from any quizbench_manifest*.json under generator_dir.
    """
    out: Dict[str, List[str]] = defaultdict(list)
    for manifest_path in sorted(generator_dir.glob("quizbench_manifest*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        quizzes = payload.get("quizzes") or []
        if not isinstance(quizzes, list):
            continue
        for q in quizzes:
            if not isinstance(q, dict):
                continue
            quiz_id = str(q.get("quiz_id") or "").strip()
            quiz_path = str(q.get("quiz_path") or "").strip()
            if quiz_id and quiz_path:
                out[quiz_id].append(quiz_path)
    return out


def _resolve_quiz_jsonl_path(
    *,
    quiz_id: str,
    manifest_quiz_paths: Mapping[str, Sequence[str]],
    quiz_index: Mapping[str, Sequence[Path]],
) -> Path:
    """
    Resolve <quiz_id>.jsonl using manifest hints first, then a global index.
    """
    candidates: List[Path] = []
    for path_str in manifest_quiz_paths.get(quiz_id, []):
        p = _resolve_existing_path(path_str)
        if p is not None and p.is_file():
            candidates.append(p)

    if candidates:
        # Prefer determinism: newest by mtime, then name.
        return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))

    indexed = list(quiz_index.get(quiz_id, []))
    if not indexed:
        raise SystemExit(f"[FATAL] Could not locate quiz JSONL for quiz_id={quiz_id!r}")
    if len(indexed) == 1:
        return indexed[0]

    # Deterministic tie-break: newest mtime, then path string.
    return max(indexed, key=lambda p: (p.stat().st_mtime, str(p)))


def _is_quiz_run_dir(path: Path) -> bool:
    """
    Heuristic for per-quiz run directories under a generator folder.
    """
    if not path.is_dir():
        return False
    if (path / "manifest.json").exists():
        return True
    # Judge/eval outputs.
    for pat in ("*_judge_result.json", "*_result.json", "*_summary.json", "summary_all_judges.json"):
        if any(path.glob(pat)):
            return True
    return False


def _canonicalize_abms(raw: str, canonical_index: Mapping[str, str]) -> str:
    """
    Canonicalize a raw specialty label to one from data/ABMS_specialties.csv.
    """
    norm = normalize_label(str(raw))
    if norm in canonical_index:
        return str(canonical_index[norm])
    raise ValueError(f"Non-canonical ABMS specialty: {raw!r}")


def _canonicalize_abms_with_overrides(
    raw: str,
    *,
    canonical_index: Mapping[str, str],
    topic_overrides: Mapping[str, str],
    topic_overrides_normalize: bool,
) -> str:
    """
    Canonicalize a label to ABMS categories, optionally using topic-map overrides.
    """
    norm = normalize_label(str(raw)) if topic_overrides_normalize else str(raw).strip()
    if norm in canonical_index:
        return str(canonical_index[norm])
    mapped = topic_overrides.get(norm)
    if mapped is None:
        raise ValueError(f"Non-canonical ABMS specialty: {raw!r}")
    return _canonicalize_abms(str(mapped), canonical_index)


def _select_topics_mapped_file(run_dir: Path) -> Path:
    candidates = [p for p in run_dir.glob("topics_*_mapped.json") if p.is_file()]
    if not candidates:
        raise SystemExit(
            f"[FATAL] Missing topics_*_mapped.json fallback in run dir: {run_dir}"
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _load_topic_mapped_by_qid(topics_mapped_path: Path) -> Dict[str, str]:
    try:
        payload = json.loads(topics_mapped_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[FATAL] Failed to parse {topics_mapped_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] topics payload is not a dict: {topics_mapped_path}")

    per_question = payload.get("per_question") or []
    if not isinstance(per_question, list):
        raise SystemExit(f"[FATAL] topics payload missing per_question list: {topics_mapped_path}")

    out: Dict[str, str] = {}
    for row in per_question:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id") or "").strip()
        topic_mapped = row.get("topic_mapped")
        if not qid:
            continue
        if topic_mapped is None or not str(topic_mapped).strip():
            continue
        out[qid] = str(topic_mapped).strip()
    return out


@dataclass(frozen=True)
class CandidateQuestion:
    generator_model: str
    abms_specialty: str
    label_source: str  # "target_topic" or "topics_mapped"
    source_label_raw: str
    source_priority: int
    source_runs_root: str
    source_generator_dir: str
    source_run_dir: str
    source_quiz_id: str
    source_quiz_path: str
    source_question_id: str
    item: Dict[str, Any]


def _candidate_sort_key(cand: CandidateQuestion) -> Tuple[int, int, str, str]:
    # Prefer target_topic over topics_mapped; prefer earlier source_runs_roots.
    src_rank = 0 if cand.label_source == "target_topic" else 1
    return (src_rank, int(cand.source_priority), cand.source_quiz_id, cand.source_question_id)


class UnassignableTopicError(Exception):
    """
    A topic label exists, but cannot be canonicalized to an ABMS specialty.
    """


def _assign_abms_specialty(
    *,
    item: Mapping[str, Any],
    question_id: str,
    run_dir: Path,
    canonical_index: Mapping[str, str],
    topic_overrides: Mapping[str, str],
    topic_overrides_normalize: bool,
    topics_mapped_cache: MutableMapping[Path, Dict[str, str]],
) -> Tuple[str, str, str]:
    """
    Return (abms_specialty, label_source, raw_label_used).
    """
    raw_target = item.get("target_topic")
    if raw_target is not None:
        raw_str = str(raw_target).strip()
        if raw_str:
            try:
                abms = _canonicalize_abms_with_overrides(
                    raw_str,
                    canonical_index=canonical_index,
                    topic_overrides=topic_overrides,
                    topic_overrides_normalize=topic_overrides_normalize,
                )
            except ValueError as exc:
                raise UnassignableTopicError(
                    f"target_topic not canonical for question_id={question_id}: {raw_str!r} ({exc})"
                ) from exc
            return abms, "target_topic", raw_str

    topics_path = _select_topics_mapped_file(run_dir)
    if topics_path not in topics_mapped_cache:
        topics_mapped_cache[topics_path] = _load_topic_mapped_by_qid(topics_path)
    mapped = topics_mapped_cache[topics_path].get(question_id)
    if mapped is None or not str(mapped).strip():
        raise SystemExit(
            f"[FATAL] Missing topic_mapped for question_id={question_id} in {topics_path}"
        )
    try:
        abms = _canonicalize_abms_with_overrides(
            str(mapped),
            canonical_index=canonical_index,
            topic_overrides=topic_overrides,
            topic_overrides_normalize=topic_overrides_normalize,
        )
    except ValueError as exc:
        raise UnassignableTopicError(
            f"topic_mapped not canonical for question_id={question_id}: {mapped!r} ({exc}). "
            "If this is a raw topic label (e.g., 'cardiology'), add an override in --topic_map."
        ) from exc
    return abms, "topics_mapped", str(mapped)


def _select_for_targets(
    *,
    candidates: Sequence[CandidateQuestion],
    targets: IntegerTargets,
) -> Tuple[List[CandidateQuestion], Dict[str, int]]:
    """
    Select exactly targets.total candidates satisfying per-category quotas.
    Returns (selected_list, available_counts_by_category).
    """
    by_cat: Dict[str, List[CandidateQuestion]] = defaultdict(list)
    for cand in candidates:
        by_cat[cand.abms_specialty].append(cand)

    available_counts = {cat: len(by_cat.get(cat, [])) for cat in targets.categories}

    selected: List[CandidateQuestion] = []
    shortfalls: List[str] = []
    for cat in targets.categories:
        need = int(targets.targets_by_category.get(cat, 0) or 0)
        if need <= 0:
            continue
        pool = sorted(by_cat.get(cat, []), key=_candidate_sort_key)
        if len(pool) < need:
            shortfalls.append(f"{cat}: need {need}, have {len(pool)}")
            continue
        selected.extend(pool[:need])

    if shortfalls:
        msg = "Insufficient candidates to meet ABMS quotas:\n  " + "\n  ".join(shortfalls)
        raise SystemExit(f"[FATAL] {msg}")

    # Deterministic interleaving across categories for chunk balance.
    queues: Dict[str, List[CandidateQuestion]] = {}
    for cat in targets.categories:
        need = int(targets.targets_by_category.get(cat, 0) or 0)
        if need <= 0:
            continue
        pool = [c for c in selected if c.abms_specialty == cat]
        queues[cat] = sorted(pool, key=_candidate_sort_key)

    interleaved: List[CandidateQuestion] = []
    while any(queues.values()):
        for cat in targets.categories:
            q = queues.get(cat) or []
            if q:
                interleaved.append(q.pop(0))
                queues[cat] = q

    if len(interleaved) != targets.total:
        raise SystemExit(
            f"[FATAL] Internal error: selected {len(interleaved)} != targets.total {targets.total}"
        )

    return interleaved, available_counts


def _write_generator_outputs(
    *,
    generator_model: str,
    subset_tag: str,
    selected: Sequence[CandidateQuestion],
    targets: IntegerTargets,
    out_quizzes_root: Path,
    out_runs_root: Path,
    source_runs_roots: Sequence[str],
    judge_models: Sequence[str],
    min_medical_score: int,
    logical_mode: str,
    chunk_size: int,
    overwrite: bool,
    targets_csv_path: str,
) -> None:
    generator_dir = out_runs_root / generator_model
    ensure_dir(str(generator_dir))

    quiz_dir = out_quizzes_root / generator_model
    ensure_dir(str(quiz_dir))

    total = targets.total
    if chunk_size <= 0:
        raise SystemExit("[FATAL] chunk_size must be > 0")
    if total % chunk_size != 0:
        raise SystemExit(
            f"[FATAL] targets.total={total} is not divisible by chunk_size={chunk_size}"
        )
    n_chunks = total // chunk_size

    generator_slug = _sanitize_id_fragment(generator_model)
    quizzes: List[Dict[str, Any]] = []

    selection_rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(selected, start=1):
        selection_rows.append(
            {
                "index": idx,
                "abms_specialty": cand.abms_specialty,
                "label_source": cand.label_source,
                "source_label_raw": cand.source_label_raw,
                "source_runs_root": cand.source_runs_root,
                "source_generator_dir": cand.source_generator_dir,
                "source_run_dir": cand.source_run_dir,
                "source_quiz_id": cand.source_quiz_id,
                "source_quiz_path": cand.source_quiz_path,
                "source_question_id": cand.source_question_id,
            }
        )

    selection_report = {
        "created_at": now_utc_iso(),
        "subset_tag": subset_tag,
        "generator_model": generator_model,
        "targets_csv_total": total,
        "targets_by_specialty": dict(targets.targets_by_category),
        "judge_models": list(judge_models),
        "filter_min_med_score": min_medical_score,
        "filter_require_logical_valid": True,
        "filter_logical_mode": logical_mode,
        "chunk_size": chunk_size,
        "num_chunks": n_chunks,
        "source_runs_roots": list(source_runs_roots),
        "selected": selection_rows,
    }
    report_path = generator_dir / f"selection_report_{subset_tag}.json"
    report_path.write_text(json.dumps(selection_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for chunk_idx in range(n_chunks):
        part = chunk_idx + 1
        quiz_id = f"{subset_tag}_{generator_slug}_p{part:02d}"
        quiz_path = quiz_dir / f"{quiz_id}.jsonl"

        start = chunk_idx * chunk_size
        chunk = list(selected[start : start + chunk_size])
        out_rows: List[Dict[str, Any]] = []
        for q_idx, cand in enumerate(chunk, start=1):
            out_item = dict(cand.item)
            out_item["quiz_id"] = quiz_id
            out_item["question_id"] = f"{quiz_id}-{q_idx:03d}"
            out_item["target_topic"] = cand.abms_specialty
            out_item["abms_specialty"] = cand.abms_specialty
            out_item["abms_source"] = cand.label_source
            out_item["source_label_raw"] = cand.source_label_raw
            out_item["source_quiz_id"] = cand.source_quiz_id
            out_item["source_question_id"] = cand.source_question_id
            out_item["source_runs_root"] = cand.source_runs_root
            out_item["source_run_dir"] = cand.source_run_dir
            out_item["source_quiz_path"] = cand.source_quiz_path
            out_rows.append(out_item)

        write_jsonl(str(quiz_path), out_rows, overwrite=overwrite)

        quizzes.append(
            {
                "quiz_id": quiz_id,
                "quiz_path": str(quiz_path.relative_to(ROOT_DIR)),
                "generator_model": generator_model,
                "seed": None,
            }
        )

    manifest = {
        "created_at": now_utc_iso(),
        "subset_tag": subset_tag,
        "generator_model": generator_model,
        "generator_models": [generator_model],
        "runs_root": str(out_runs_root),
        "quizzes_dir": str(out_quizzes_root),
        "quizzes": quizzes,
        "targets_csv": targets_csv_path,
        "judge_models": list(judge_models),
        "filter_min_med_score": min_medical_score,
        "filter_require_logical_valid": True,
        "filter_logical_mode": logical_mode,
        "chunk_size": chunk_size,
        "num_quizzes_per_generator": len(quizzes),
        "num_questions_per_quiz": chunk_size,
        "num_questions_total": total,
        "source_runs_roots": list(source_runs_roots),
        "selection_report": str(report_path.relative_to(ROOT_DIR)),
    }
    manifest_path = generator_dir / f"quizbench_manifest_{subset_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build per-generator ABMS-quota subsets of judge-valid questions (5x10 chunking)."
    )
    ap.add_argument(
        "--source_runs_roots_csv",
        type=str,
        default=None,
        help=(
            "Comma-separated list of source runs roots (each contains generator dirs). "
            "Order matters: earlier roots are preferred when selecting questions."
        ),
    )
    ap.add_argument(
        "--subset_tag",
        type=str,
        default=None,
        help="Artifact tag (recommended: ABMSYYYYMMDD, e.g. ABMS20260101).",
    )
    ap.add_argument(
        "--subset_date",
        type=str,
        default=None,
        help="If --subset_tag is omitted, build tag as ABMS<YYYYMMDD> using this date.",
    )
    ap.add_argument(
        "--targets_csv",
        type=str,
        default="data/ABMS_specialties.csv",
        help="CSV defining ABMS specialty quotas (defaults to data/ABMS_specialties.csv).",
    )
    ap.add_argument(
        "--topic_map",
        type=str,
        default="data/topic_to_abms.yaml",
        help=(
            "Optional topic-map file used to canonicalize non-ABMS labels when falling back "
            "to topics_*_mapped.json (default: data/topic_to_abms.yaml)."
        ),
    )
    ap.add_argument(
        "--judge_models_yaml",
        type=str,
        default="data/judge_models.yaml",
        help="YAML list of judge models used for validity filtering.",
    )
    ap.add_argument(
        "--min_medical_score",
        type=int,
        default=1,
        help="Minimum medical_accuracy_score for judge eligibility (default: 1).",
    )
    ap.add_argument(
        "--logical_mode",
        type=str,
        default="majority",
        choices=["all", "majority"],
        help="Judge aggregation mode for logical validity (default: majority).",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Questions per synthetic quiz file (default: 10).",
    )
    ap.add_argument(
        "--only_generators_csv",
        type=str,
        default=None,
        help="Optional CSV of generator directory names to process (default: all found).",
    )
    ap.add_argument(
        "--out_quizzes_root",
        type=str,
        default=None,
        help="Override output quizzes root (default: quizzes/quizzes_<TAG>).",
    )
    ap.add_argument(
        "--out_runs_root",
        type=str,
        default=None,
        help="Override output runs root (default: eval_results/quizbench/quizzes_<TAG>/runs).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output quiz JSONL files if they already exist.",
    )
    ap.add_argument(
        "--unassignable_policy",
        type=str,
        default="skip",
        choices=["skip", "error"],
        help=(
            "Behavior when a valid question has a topic label but it cannot be canonicalized "
            "to an ABMS specialty (e.g., topic_mapped='Misc'): 'skip' (default) or 'error'."
        ),
    )
    ap.add_argument(
        "--insufficient_policy",
        type=str,
        default="error",
        choices=["skip", "error"],
        help=(
            "Behavior when a generator does not have sufficient judge-valid candidates to "
            "meet all ABMS quotas: 'error' (default) or 'skip' (continue to next generator)."
        ),
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write outputs; only print per-generator feasibility summaries.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    subset_tag = (args.subset_tag or "").strip()
    if not subset_tag:
        date_str = (args.subset_date or "").strip() or _today_utc_yyyymmdd()
        if not re.fullmatch(r"\d{8}", date_str):
            raise SystemExit(f"[FATAL] --subset_date must be YYYYMMDD, got: {date_str!r}")
        subset_tag = f"ABMS{date_str}"

    subset_tag = _sanitize_id_fragment(subset_tag)

    source_roots = _parse_csv_arg(args.source_runs_roots_csv)
    if not source_roots:
        default_candidate = ROOT_DIR / "eval_results" / "quizbench" / "quizzes_Jan2026" / "runs"
        if default_candidate.exists():
            source_roots = [str(default_candidate)]
        else:
            raise SystemExit(
                "[FATAL] --source_runs_roots_csv is required (no default source runs root found)."
            )
    source_runs_roots = [Path(p).expanduser().resolve() for p in source_roots]
    for p in source_runs_roots:
        if not p.exists() or not p.is_dir():
            raise SystemExit(f"[FATAL] Source runs root does not exist or is not a dir: {p}")

    targets_csv_path = str(Path(args.targets_csv).expanduser().resolve())
    targets = load_integer_targets_csv(Path(targets_csv_path))
    canonical_index = build_canonical_index(targets.categories, normalize=True)

    topic_map_path = Path(args.topic_map).expanduser().resolve()
    topic_overrides_normalize = True
    topic_overrides: Dict[str, str] = {}
    if topic_map_path.exists():
        try:
            topic_overrides_normalize, topic_overrides = _load_topic_map_overrides(topic_map_path)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[FATAL] Failed to parse --topic_map {topic_map_path}: {exc}") from exc

    judge_models_yaml = Path(args.judge_models_yaml).expanduser().resolve()
    if judge_models_yaml.exists():
        judge_models = _load_yaml_string_list(judge_models_yaml)
    else:
        judge_models = list(DEFAULT_ENSEMBLE_JUDGES)

    only_generators = set(_parse_csv_arg(args.only_generators_csv)) if args.only_generators_csv else None

    out_quizzes_root = Path(
        args.out_quizzes_root or (ROOT_DIR / "quizzes" / f"quizzes_{subset_tag}")
    ).expanduser().resolve()
    out_runs_root = Path(
        args.out_runs_root
        or (ROOT_DIR / "eval_results" / "quizbench" / f"quizzes_{subset_tag}" / "runs")
    ).expanduser().resolve()

    quiz_index = _build_quiz_jsonl_index(ROOT_DIR / "quizzes")

    # Map generator_model -> list[(source_priority, generator_dir)]
    generators: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for pri, src_root in enumerate(source_runs_roots):
        for gen_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            gen_name = gen_dir.name
            if only_generators is not None and gen_name not in only_generators:
                continue
            generators[gen_name].append((pri, gen_dir))

    if not generators:
        raise SystemExit(
            f"[FATAL] No generator directories found under source roots: {source_roots}"
        )

    print(f"[INFO] subset_tag={subset_tag}")
    print(f"[INFO] source_runs_roots={', '.join(str(p) for p in source_runs_roots)}")
    print(f"[INFO] targets_total={targets.total} chunk_size={args.chunk_size}")
    print(f"[INFO] judge_models={', '.join(judge_models)}")

    topics_mapped_cache: Dict[Path, Dict[str, str]] = {}

    for generator_model, gen_dirs in sorted(generators.items()):
        # Build candidate pool across source roots, preferring earlier roots.
        seen_qids: set[str] = set()
        candidates: List[CandidateQuestion] = []
        skipped_runs: Dict[str, str] = {}
        skipped_questions: Dict[str, int] = defaultdict(int)

        for pri, gen_dir in sorted(gen_dirs, key=lambda t: t[0]):
            manifest_quiz_paths = _load_manifest_quiz_paths(gen_dir)
            for run_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
                if not _is_quiz_run_dir(run_dir):
                    continue
                quiz_id = run_dir.name

                allowed_qids = filter_by_judge(
                    run_dir,
                    judge_models,
                    min_medical_score=args.min_medical_score,
                    require_logical_valid=True,
                    logical_mode=args.logical_mode,
                )
                if allowed_qids is None:
                    skipped_runs[str(run_dir)] = "no_judge_results_found"
                    continue

                quiz_path = _resolve_quiz_jsonl_path(
                    quiz_id=quiz_id, manifest_quiz_paths=manifest_quiz_paths, quiz_index=quiz_index
                )
                items = read_jsonl(str(quiz_path))

                qid_to_item: Dict[str, Dict[str, Any]] = {}
                for idx, item in enumerate(items, start=1):
                    if not isinstance(item, dict):
                        continue
                    qid = str(item.get("question_id") or "").strip() or f"{quiz_id}-{idx:03d}"
                    if qid in qid_to_item:
                        raise SystemExit(
                            f"[FATAL] Duplicate question_id={qid!r} in quiz file: {quiz_path}"
                        )
                    fixed = dict(item)
                    fixed.setdefault("quiz_id", quiz_id)
                    fixed["question_id"] = qid
                    qid_to_item[qid] = fixed

                for qid in sorted(allowed_qids):
                    if qid in seen_qids:
                        continue
                    src_item = qid_to_item.get(str(qid))
                    if src_item is None:
                        raise SystemExit(
                            f"[FATAL] Judge returned question_id={qid!r} not found in quiz JSONL: {quiz_path}"
                        )

                    try:
                        abms, label_source, raw_label = _assign_abms_specialty(
                            item=src_item,
                            question_id=qid,
                            run_dir=run_dir,
                            canonical_index=canonical_index,
                            topic_overrides=topic_overrides,
                            topic_overrides_normalize=topic_overrides_normalize,
                            topics_mapped_cache=topics_mapped_cache,
                        )
                    except UnassignableTopicError as exc:
                        if args.unassignable_policy == "error":
                            raise SystemExit(f"[FATAL] {exc}") from exc
                        skipped_questions[str(exc)] += 1
                        continue

                    seen_qids.add(qid)
                    candidates.append(
                        CandidateQuestion(
                            generator_model=generator_model,
                            abms_specialty=abms,
                            label_source=label_source,
                            source_label_raw=raw_label,
                            source_priority=int(pri),
                            source_runs_root=str(gen_dir.parent),
                            source_generator_dir=str(gen_dir),
                            source_run_dir=str(run_dir),
                            source_quiz_id=quiz_id,
                            source_quiz_path=str(quiz_path),
                            source_question_id=qid,
                            item=src_item,
                        )
                    )

        available_total = len(candidates)
        avail_by_cat = defaultdict(int)
        for cand in candidates:
            avail_by_cat[cand.abms_specialty] += 1

        print(f"\n[INFO] Generator={generator_model} candidates={available_total}")
        if skipped_runs:
            print(f"[INFO]   skipped_runs={len(skipped_runs)} (missing judge outputs)")
        if skipped_questions:
            skipped_total = sum(int(v or 0) for v in skipped_questions.values())
            print(
                f"[INFO]   skipped_questions={skipped_total} "
                f"(unassignable topics; policy={args.unassignable_policy})"
            )

        shortfalls: List[str] = []
        available_counts_by_category: Dict[str, int] = {}
        for cat in targets.categories:
            have = int(avail_by_cat.get(cat, 0) or 0)
            available_counts_by_category[cat] = have
            need = int(targets.targets_by_category.get(cat, 0) or 0)
            if need <= 0:
                continue
            if have < need:
                shortfalls.append(f"{cat}: need {need}, have {have}")

        if shortfalls:
            if args.dry_run:
                print("[DRY-RUN]   would_select=0")
                for row in shortfalls:
                    print(f"[DRY-RUN]   {row}  <-- SHORTFALL")
                continue

            msg = "Insufficient candidates to meet ABMS quotas:\n  " + "\n  ".join(shortfalls)
            if args.insufficient_policy == "skip":
                print(f"[SKIP] {msg}")
                continue
            raise SystemExit(f"[FATAL] {msg}")

        selected: List[CandidateQuestion]
        selected, available_counts_by_category = _select_for_targets(
            candidates=candidates,
            targets=targets,
        )

        if args.dry_run:
            print(f"[DRY-RUN]   would_select={len(selected)}")
            # Print a compact per-category availability/need summary.
            for cat in targets.categories:
                need = int(targets.targets_by_category.get(cat, 0) or 0)
                have = int(available_counts_by_category.get(cat, 0) or 0)
                if need <= 0:
                    continue
                if have < need:
                    print(f"[DRY-RUN]   {cat}: need={need} have={have}  <-- SHORTFALL")
            continue

        _write_generator_outputs(
            generator_model=generator_model,
            subset_tag=subset_tag,
            selected=selected,
            targets=targets,
            out_quizzes_root=out_quizzes_root,
            out_runs_root=out_runs_root,
            source_runs_roots=[str(p) for p in source_runs_roots],
            judge_models=judge_models,
            min_medical_score=args.min_medical_score,
            logical_mode=args.logical_mode,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
            targets_csv_path=targets_csv_path,
        )

        print(f"[OK] Wrote subset for generator={generator_model}")

    if not args.dry_run:
        print(f"\n[INFO] Done. Output quizzes root: {out_quizzes_root}")
        print(f"[INFO] Done. Output runs root:   {out_runs_root}")


if __name__ == "__main__":
    main()
