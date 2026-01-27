#!/usr/bin/env python3
"""
Utilities for computing topic distributions over *majority-valid* questions.

This module is meant to be imported by generation/eval orchestration scripts
and does not depend on pandas/seaborn.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from quizbench.aggregate_judges import filter_by_judge

from quizbench.manifest_utils import resolve_quizbench_manifest_path
from quizbench.topic_mapping import (  # noqa: E402
    TopicMappingConfig,
    build_canonical_index,
    load_topic_mapping,
    map_topic_label,
)
from quizbench.utils import now_utc_iso, read_jsonl

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ValidTopicDistribution:
    """
    Aggregated topic counts over questions that pass judge filters.

    All fields are JSON-serializable via `to_json_dict()`.
    """

    created_at: str
    generator_model: str
    generator_run_root: str
    topics_eval_model: str | None
    judge_models: List[str]
    min_medical_score: int | None
    logical_mode: str
    valid_total: int
    valid_counts_by_topic: Dict[str, int]
    runs_included: List[str]
    runs_skipped: Dict[str, str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "generator_model": self.generator_model,
            "generator_run_root": self.generator_run_root,
            "topics_eval_model": self.topics_eval_model,
            "judge_models": list(self.judge_models),
            "min_medical_score": self.min_medical_score,
            "logical_mode": self.logical_mode,
            "valid_total": self.valid_total,
            "valid_counts_by_topic": dict(self.valid_counts_by_topic),
            "runs_included": list(self.runs_included),
            "runs_skipped": dict(self.runs_skipped),
        }


def _iter_topics_files(generator_run_root: Path) -> Iterable[Path]:
    yield from sorted(generator_run_root.rglob("topics_*.json"))


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _looks_like_topic_mapping_output(payload: Mapping[str, Any] | None) -> bool:
    """
    Heuristic: files written by `apply_topic_mapping.py` include a `topic_mapping` block.
    """
    if not payload:
        return False
    topic_mapping = payload.get("topic_mapping")
    return isinstance(topic_mapping, dict) and bool(topic_mapping.get("applied_at"))


def _group_topics_files_by_run(
    generator_run_root: Path,
) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for path in _iter_topics_files(generator_run_root):
        grouped[path.parent].append(path)
    return grouped


def _select_topics_file(
    *,
    run_dir: Path,
    candidates: Sequence[Path],
    topics_eval_model: str | None,
    prefer_topic_mapping_outputs: bool,
) -> tuple[Path | None, str | None, str | None]:
    """
    Choose exactly one topics_*.json file for a run directory.

    Returns (path, eval_model, error_reason).
    """
    if not candidates:
        return None, None, "no_topics_files"

    if topics_eval_model is None:
        if len(candidates) == 1:
            payload = _load_json(candidates[0]) or {}
            return candidates[0], str(payload.get("eval_model") or "unknown"), None

        # Common case: `apply_topic_mapping.py` created an extra `*_mapped.json`
        # alongside the original. Prefer the original input file.
        payloads = {p: _load_json(p) for p in candidates}
        if prefer_topic_mapping_outputs:
            mapping_outputs = [p for p in candidates if _looks_like_topic_mapping_output(payloads[p])]
            if len(mapping_outputs) == 1:
                payload = payloads[mapping_outputs[0]] or {}
                return mapping_outputs[0], str(payload.get("eval_model") or "unknown"), None
            if len(mapping_outputs) > 1:
                # When multiple topic-mapping outputs exist (e.g., repeated runs
                # producing *_mapped_mapped.json), select the newest mapping
                # output deterministically instead of skipping the run.
                def _mapping_sort_key(path: Path) -> tuple[str, float, str]:
                    payload = payloads.get(path) or {}
                    topic_mapping = payload.get("topic_mapping") if isinstance(payload, dict) else {}
                    applied_at = ""
                    if isinstance(topic_mapping, dict):
                        applied_at = str(topic_mapping.get("applied_at") or "").strip()
                    try:
                        mtime = float(path.stat().st_mtime)
                    except Exception:  # noqa: BLE001
                        mtime = 0.0
                    return (applied_at, mtime, path.name)

                chosen = max(mapping_outputs, key=_mapping_sort_key)
                payload = payloads.get(chosen) or {}
                return chosen, str((payload or {}).get("eval_model") or "unknown"), None
        non_mapping = [p for p in candidates if not _looks_like_topic_mapping_output(payloads[p])]
        if len(non_mapping) == 1:
            payload = payloads[non_mapping[0]] or {}
            return non_mapping[0], str(payload.get("eval_model") or "unknown"), None

        non_mapped_suffix = [p for p in candidates if not p.stem.endswith("_mapped")]
        if len(non_mapped_suffix) == 1:
            payload = payloads.get(non_mapped_suffix[0]) or _load_json(non_mapped_suffix[0]) or {}
            return non_mapped_suffix[0], str(payload.get("eval_model") or "unknown"), None

        return None, None, "multiple_topics_files (set --topics_eval_model to disambiguate)"

    # Filter candidates by payload eval_model.
    matched: List[Path] = []
    matched_payloads: Dict[Path, Mapping[str, Any]] = {}
    for p in candidates:
        payload = _load_json(p)
        if not payload:
            continue
        if str(payload.get("eval_model") or "").strip() == topics_eval_model:
            matched.append(p)
            matched_payloads[p] = payload

    if not matched:
        return None, None, f"no_topics_file_for_eval_model={topics_eval_model!r}"
    if len(matched) > 1:
        if prefer_topic_mapping_outputs:
            mapping = [p for p in matched if _looks_like_topic_mapping_output(matched_payloads.get(p))]
            if len(mapping) == 1:
                payload = matched_payloads.get(mapping[0]) or {}
                return mapping[0], str(payload.get("eval_model") or "unknown"), None
            if len(mapping) > 1:
                def _mapping_sort_key(path: Path) -> tuple[str, float, str]:
                    payload = matched_payloads.get(path) or {}
                    topic_mapping = payload.get("topic_mapping") if isinstance(payload, dict) else {}
                    applied_at = ""
                    if isinstance(topic_mapping, dict):
                        applied_at = str(topic_mapping.get("applied_at") or "").strip()
                    try:
                        mtime = float(path.stat().st_mtime)
                    except Exception:  # noqa: BLE001
                        mtime = 0.0
                    return (applied_at, mtime, path.name)

                chosen = max(mapping, key=_mapping_sort_key)
                payload = matched_payloads.get(chosen) or {}
                return chosen, str(payload.get("eval_model") or "unknown"), None

        # Prefer original when `*_mapped.json` exists alongside the true topics file.
        non_mapping = [p for p in matched if not _looks_like_topic_mapping_output(matched_payloads.get(p))]
        if len(non_mapping) == 1:
            payload = matched_payloads.get(non_mapping[0]) or {}
            return non_mapping[0], str(payload.get("eval_model") or "unknown"), None

        non_mapped_suffix = [p for p in matched if not p.stem.endswith("_mapped")]
        if len(non_mapped_suffix) == 1:
            payload = matched_payloads.get(non_mapped_suffix[0]) or _load_json(non_mapped_suffix[0]) or {}
            return non_mapped_suffix[0], str(payload.get("eval_model") or "unknown"), None

        return None, None, f"multiple_topics_files_for_eval_model={topics_eval_model!r}"

    payload = matched_payloads.get(matched[0]) or _load_json(matched[0]) or {}
    return matched[0], str(payload.get("eval_model") or "unknown"), None


def _qid_to_topic_from_topics_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    per_question = payload.get("per_question") or []
    qid_to_topic: Dict[str, str] = {}
    if not isinstance(per_question, list):
        return qid_to_topic
    for row in per_question:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue
        topic = row.get("topic") or "unknown"
        qid_to_topic[qid] = str(topic)
    return qid_to_topic


def _qid_to_mapped_topic_from_topics_payload(
    payload: Mapping[str, Any],
    *,
    canonical_categories: Sequence[str],
    canonical_index: Mapping[str, str],
    config: TopicMappingConfig | None,
    mapping_mode: str,
    unmapped_policy: str,
) -> Dict[str, str]:
    per_question = payload.get("per_question") or []
    qid_to_topic: Dict[str, str] = {}
    if not isinstance(per_question, list):
        return qid_to_topic

    for row in per_question:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue

        # Prefer precomputed field if present; otherwise map from raw topic.
        raw_topic = row.get("topic_mapped")
        if raw_topic is None:
            raw_topic = row.get("topic")

        mapped = map_topic_label(
            str(raw_topic) if raw_topic is not None else None,
            canonical_categories=canonical_categories,
            canonical_index=canonical_index,
            config=config,
            mode=mapping_mode,
            unmapped_policy=unmapped_policy,
        )
        if mapped is None:
            continue
        qid_to_topic[qid] = str(mapped)

    return qid_to_topic


def _resolve_quizzes_dir_for_fallback(generator_run_root: Path) -> Path | None:
    """
    Best-effort resolve a quizzes directory for locating <quiz_id>.jsonl files.

    Uses the batch tag path segment `.../quizzes_<TAG>/runs/...` when present,
    otherwise falls back to the repo-root `quizzes/` directory.
    """
    generator_run_root = generator_run_root.expanduser().resolve()
    for part in generator_run_root.parts:
        if part.startswith("quizzes_") and part != "quizzes":
            candidate = (ROOT_DIR / "quizzes" / part).resolve()
            if candidate.exists() and candidate.is_dir():
                return candidate

    fallback = (ROOT_DIR / "quizzes").resolve()
    if fallback.exists() and fallback.is_dir():
        return fallback
    return None


def _build_quiz_jsonl_index(quizzes_dir: Path) -> Dict[str, Path]:
    quizzes_dir = quizzes_dir.expanduser().resolve()
    recursive = quizzes_dir.name == "quizzes"
    paths = quizzes_dir.rglob("*.jsonl") if recursive else quizzes_dir.glob("*.jsonl")

    index: Dict[str, Path] = {}
    for p in paths:
        if not p.is_file():
            continue
        quiz_id = p.stem
        if not quiz_id:
            continue
        existing = index.get(quiz_id)
        if existing is None:
            index[quiz_id] = p
            continue
        try:
            if (p.stat().st_mtime, str(p)) > (existing.stat().st_mtime, str(existing)):
                index[quiz_id] = p
        except Exception:  # noqa: BLE001
            continue
    return index


def _load_quiz_paths_from_manifest(generator_run_root: Path) -> Dict[str, Path]:
    """
    Load quiz_id -> quiz_path from the newest quizbench_manifest*.json under generator_run_root.
    """
    try:
        manifest_path = resolve_quizbench_manifest_path(generator_run_root)
    except SystemExit:
        return {}

    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Path] = {}
    quizzes = payload.get("quizzes") or []
    if not isinstance(quizzes, list):
        return out
    for q in quizzes:
        if not isinstance(q, dict):
            continue
        quiz_id = str(q.get("quiz_id") or "").strip()
        quiz_path = str(q.get("quiz_path") or "").strip()
        if not quiz_id or not quiz_path:
            continue
        p = Path(quiz_path).expanduser()
        p = p if p.is_absolute() else (ROOT_DIR / p)
        try:
            p = p.resolve()
        except Exception:  # noqa: BLE001
            continue
        if p.exists() and p.is_file():
            out[quiz_id] = p
    return out


def _qid_to_mapped_topic_from_quiz_jsonl(
    quiz_path: Path,
    *,
    quiz_id: str,
    canonical_categories: Sequence[str],
    canonical_index: Mapping[str, str],
    config: TopicMappingConfig | None,
    mapping_mode: str,
    unmapped_policy: str,
) -> Dict[str, str]:
    """
    Read per-question topics from a quiz JSONL file's `target_topic` field.
    """
    qid_to_topic: Dict[str, str] = {}
    try:
        items = read_jsonl(str(quiz_path))
    except Exception:  # noqa: BLE001
        return qid_to_topic

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id") or "").strip() or f"{quiz_id}-{idx:03d}"
        raw_topic = item.get("target_topic")
        if raw_topic is None:
            raw_topic = item.get("topic")

        mapped = map_topic_label(
            str(raw_topic) if raw_topic is not None else None,
            canonical_categories=canonical_categories,
            canonical_index=canonical_index,
            config=config,
            mode=mapping_mode,
            unmapped_policy=unmapped_policy,
        )
        if mapped is None:
            continue
        qid_to_topic[qid] = str(mapped)

    return qid_to_topic


def compute_valid_topic_distribution(
    *,
    generator_run_root: Path,
    generator_model: str,
    judge_models: Sequence[str],
    topics_eval_model: str | None = None,
    min_medical_score: int | None = None,
    logical_mode: str = "majority",
    allow_missing_judges: bool = False,
    canonical_categories: Sequence[str] | None = None,
    topic_map_path: Path | None = None,
    mapping_mode: str = "map",
    unmapped_policy: str = "keep",
    prefer_topic_mapping_outputs: bool = False,
    fallback_to_quiz_target_topics: bool = True,
) -> ValidTopicDistribution:
    """
    Compute a topic distribution over questions deemed valid by the judge ensemble.

    Validity is determined by `quizbench.aggregate_judges.filter_by_judge(...)` with:
      - require_logical_valid=True
      - logical_mode (default: "majority")

    Topic labels are read from `topics_*.json` files produced by
    `quizbench/categorize_quiz_topics.py` under each run directory.
    """
    generator_run_root = generator_run_root.expanduser().resolve()
    logical_mode = (logical_mode or "majority").lower()
    if logical_mode not in {"all", "majority"}:
        raise ValueError(f"Unsupported logical_mode: {logical_mode!r}")

    judge_models = [str(m).strip() for m in judge_models if str(m).strip()]
    if not judge_models and not allow_missing_judges:
        raise ValueError("judge_models must be non-empty unless allow_missing_judges=True")

    mapping_config = load_topic_mapping(topic_map_path) if topic_map_path else None
    cfg = mapping_config or TopicMappingConfig()
    canonical_categories_list = list(canonical_categories or [])
    canonical_index = (
        build_canonical_index(canonical_categories_list, normalize=cfg.normalize)
        if canonical_categories_list
        else {}
    )

    quiz_paths_from_manifest: Dict[str, Path] = {}
    quiz_index: Dict[str, Path] = {}
    if fallback_to_quiz_target_topics:
        quiz_paths_from_manifest = _load_quiz_paths_from_manifest(generator_run_root)
        quizzes_dir = _resolve_quizzes_dir_for_fallback(generator_run_root)
        if quizzes_dir is not None:
            quiz_index = _build_quiz_jsonl_index(quizzes_dir)

    counts: Counter[str] = Counter()
    included: List[str] = []
    skipped: MutableMapping[str, str] = {}

    observed_eval_model: str | None = None
    run_dirs = [p for p in sorted(generator_run_root.iterdir()) if p.is_dir()]
    for run_dir in run_dirs:
        candidates = sorted(run_dir.glob("topics_*.json"))
        qid_to_topic: Dict[str, str] = {}

        if candidates:
            topics_path, eval_model, err = _select_topics_file(
                run_dir=run_dir,
                candidates=candidates,
                topics_eval_model=topics_eval_model,
                prefer_topic_mapping_outputs=prefer_topic_mapping_outputs,
            )
            if err or not topics_path:
                skipped[str(run_dir)] = err or "unknown_error"
                continue

            payload = _load_json(topics_path)
            if not payload:
                skipped[str(run_dir)] = f"failed_to_parse_topics_json: {topics_path.name}"
                continue

            # Keep for metadata/debugging.
            if observed_eval_model is None:
                observed_eval_model = eval_model

            qid_to_topic = _qid_to_mapped_topic_from_topics_payload(
                payload,
                canonical_categories=canonical_categories_list,
                canonical_index=canonical_index,
                config=mapping_config,
                mapping_mode=mapping_mode,
                unmapped_policy=unmapped_policy,
            )
            if not qid_to_topic:
                skipped[str(run_dir)] = "no_topics_rows"
                continue
        else:
            if not fallback_to_quiz_target_topics:
                skipped[str(run_dir)] = "no_topics_files"
                continue

            quiz_id = run_dir.name
            quiz_path = quiz_paths_from_manifest.get(quiz_id) or quiz_index.get(quiz_id)
            if quiz_path is None or not quiz_path.exists():
                skipped[str(run_dir)] = "no_topics_files_and_missing_quiz_jsonl"
                continue

            qid_to_topic = _qid_to_mapped_topic_from_quiz_jsonl(
                quiz_path,
                quiz_id=quiz_id,
                canonical_categories=canonical_categories_list,
                canonical_index=canonical_index,
                config=mapping_config,
                mapping_mode=mapping_mode,
                unmapped_policy=unmapped_policy,
            )
            if not qid_to_topic:
                skipped[str(run_dir)] = "no_topics_found_in_quiz_jsonl"
                continue

        allowed_qids = filter_by_judge(
            run_dir,
            judge_models,
            min_medical_score=min_medical_score,
            require_logical_valid=True,
            logical_mode=logical_mode,
        )
        if allowed_qids is None:
            if not allow_missing_judges:
                skipped[str(run_dir)] = "no_judge_results_found"
                continue
            allowed_qids = set(qid_to_topic.keys())

        for qid in allowed_qids:
            topic = qid_to_topic.get(str(qid))
            if not topic:
                continue
            counts[str(topic)] += 1

        included.append(str(run_dir))

    # In case topics_eval_model wasn't provided, capture the one we observed.
    topics_eval_model_out = topics_eval_model or observed_eval_model

    return ValidTopicDistribution(
        created_at=now_utc_iso(),
        generator_model=generator_model,
        generator_run_root=str(generator_run_root),
        topics_eval_model=topics_eval_model_out,
        judge_models=list(judge_models),
        min_medical_score=min_medical_score,
        logical_mode=logical_mode,
        valid_total=int(sum(counts.values())),
        valid_counts_by_topic={k: int(v) for k, v in sorted(counts.items())},
        runs_included=included,
        runs_skipped=dict(skipped),
    )
