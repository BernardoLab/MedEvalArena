#!/usr/bin/env python3
"""
Apply a human-guided topic→ABMS mapping across existing `topics_*.json` files.

Default behavior is safe:
  - adds `topic_mapped` per question
  - writes a new file alongside the original (suffix `_mapped`)
  - produces a JSON report with unmapped labels for iterative cleanup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.target_planning import load_integer_targets_csv  # noqa: E402
from quizbench.topic_mapping import (  # noqa: E402
    TopicMappingConfig,
    build_canonical_index,
    load_topic_mapping,
    map_topic_label,
    map_topic_label_explained,
    normalize_label,
)
from quizbench.manifest_utils import resolve_quizbench_manifest_path  # noqa: E402
from quizbench.utils import now_utc_iso, read_jsonl  # noqa: E402


_SLASH_SPACING_RE = re.compile(r"\s*/\s*")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Apply a topic→ABMS mapping to topics_*.json files."
    )
    ap.add_argument(
        "--operation",
        type=str,
        default="apply",
        choices=["apply", "accumulate"],
        help=(
            "Operation to run: 'apply' maps topics files (default); "
            "'accumulate' only writes per-generator topic distributions."
        ),
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root directory to scan for topics_*.json files.",
    )
    ap.add_argument(
        "--targets_csv",
        type=str,
        default="data/ABMS_specialties.csv",
        help="CSV containing canonical ABMS specialty names (Specialty column).",
    )
    ap.add_argument(
        "--topic_map",
        type=str,
        default=None,
        help="YAML/JSON mapping file path (recommended).",
    )
    ap.add_argument(
        "--topics_eval_model",
        type=str,
        default=None,
        help=(
            "For --operation accumulate: when multiple topics_*.json exist per run, "
            "select the one whose payload eval_model matches this string."
        ),
    )
    ap.add_argument(
        "--prefer_topic_mapping_outputs",
        action="store_true",
        help=(
            "For --operation accumulate: when multiple topics files exist, prefer "
            "inputs produced by apply_topic_mapping.py (payload contains 'topic_mapping')."
        ),
    )
    ap.add_argument(
        "--judge_models",
        type=str,
        default=None,
        help=(
            "For --operation accumulate: comma-separated judge model names used to determine "
            "majority-valid questions (default: data/judge_models.yaml or quizbench.aggregate_judges.DEFAULT_ENSEMBLE_JUDGES)."
        ),
    )
    ap.add_argument(
        "--judge_models_yaml",
        type=str,
        default="data/judge_models.yaml",
        help=(
            "For --operation accumulate: YAML list of judge models. Ignored if --judge_models is provided "
            "(default: data/judge_models.yaml)."
        ),
    )
    ap.add_argument(
        "--logical_mode",
        type=str,
        default="majority",
        choices=["all", "majority"],
        help="For --operation accumulate: judge aggregation mode (default: majority).",
    )
    ap.add_argument(
        "--min_medical_score",
        type=int,
        default=None,
        help="For --operation accumulate: optional minimum medical_accuracy_score for judge eligibility.",
    )
    ap.add_argument(
        "--allow_missing_judges",
        action="store_true",
        help=(
            "For --operation accumulate: if set, treat runs with no judge outputs as unfiltered "
            "(all questions are treated as valid)."
        ),
    )
    ap.add_argument(
        "--accumulated_out_name",
        type=str,
        default="accumulated_topic_distribution.json",
        help=(
            "For --operation accumulate: output filename written under each generator directory "
            "(default: accumulated_topic_distribution.json)."
        ),
    )
    ap.add_argument(
        "--fallback_to_quiz_target_topics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --operation accumulate: if a run directory contains no topics_*.json, "
            "fall back to reading per-question `target_topic` directly from the quiz JSONL "
            "(default: True)."
        ),
    )
    ap.add_argument(
        "--quizzes_dir",
        type=str,
        default=None,
        help=(
            "For --operation accumulate with --fallback_to_quiz_target_topics: directory containing "
            "<quiz_id>.jsonl files (e.g., quizzes/quizzes_Jan2026). When omitted, this is inferred "
            "from --runs_root (quizzes_<TAG>) or falls back to quizzes/."
        ),
    )
    ap.add_argument(
        "--skip_existing_accumulated",
        action="store_true",
        help="For --operation accumulate: skip writing if the output file already exists.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="map",
        choices=["exact", "normalize", "map"],
        help="Mapping mode (default: map).",
    )
    ap.add_argument(
        "--unmapped_policy",
        type=str,
        default="keep",
        choices=["error", "keep", "misc", "drop"],
        help="Behavior for unmapped labels (default: keep).",
    )
    ap.add_argument(
        "--assert_in_targets",
        action="store_true",
        help=(
            "If set, fail when a topic label cannot be resolved to a category in --targets_csv "
            "(excluding the mapping config's misc_category). Useful when topics are expected to "
            "already be canonical (e.g., ABMS categories)."
        ),
    )
    ap.add_argument(
        "--write_mode",
        type=str,
        default="newfile",
        choices=["newfile", "inplace"],
        help="Write strategy (default: newfile).",
    )
    ap.add_argument(
        "--out_suffix",
        type=str,
        default="_mapped",
        help="Suffix appended to filename in newfile mode (default: _mapped).",
    )
    ap.add_argument(
        "--include_out_suffix_inputs",
        action="store_true",
        help=(
            "Include input files whose filename already ends with --out_suffix "
            "(these are usually previous outputs)."
        ),
    )
    ap.add_argument(
        "--backup_dir",
        type=str,
        default=None,
        help="Backup directory for inplace mode (default: tmp/topic_mapping_backups/<timestamp>).",
    )
    ap.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Overwrite existing topic_mapped fields when present.",
    )
    ap.add_argument(
        "--overwrite_topic",
        action="store_true",
        help="Overwrite the original `topic` field with the mapped value.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and report mappings without writing any files.",
    )
    ap.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional JSON report output path (default: topic_mapping_report_<ts>.json).",
    )
    ap.add_argument(
        "--refine_misc_heme_onc",
        action="store_true",
        help=(
            "For questions that map to Misc from a combined heme/onc label (e.g., 'hematology/oncology'), "
            "call an LLM to split into 'Hematology' vs 'Medical Oncology'."
        ),
    )
    ap.add_argument(
        "--refine_misc_model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Model to use for --refine_misc_heme_onc (default: claude-3-7-sonnet-20250219).",
    )
    ap.add_argument(
        "--refine_misc_max_tokens",
        type=int,
        default=512,
        help="Max tokens for --refine_misc_heme_onc model calls (default: 512).",
    )
    ap.add_argument(
        "--refine_misc_temperature",
        type=float,
        default=0.0,
        help="Temperature for --refine_misc_heme_onc model calls (default: 0.0).",
    )
    return ap.parse_args()


def _load_topics_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_quizzes_dir_for_fallback(*, runs_root: Path, override: str | None) -> Path | None:
    """
    Best-effort resolve a quizzes directory for locating <quiz_id>.jsonl files.

    Preference order:
      1) Explicit --quizzes_dir override
      2) Infer from runs_root segment `.../quizzes_<TAG>/runs` → `quizzes/quizzes_<TAG>`
      3) Fallback to repo-root `quizzes/`
    """
    if override:
        p = Path(override).expanduser()
        p = p if p.is_absolute() else (ROOT_DIR / p)
        p = p.resolve()
        return p if p.exists() and p.is_dir() else None

    for part in runs_root.parts:
        if part.startswith("quizzes_") and part != "quizzes":
            candidate = (ROOT_DIR / "quizzes" / part).resolve()
            if candidate.exists() and candidate.is_dir():
                return candidate

    fallback = (ROOT_DIR / "quizzes").resolve()
    if fallback.exists() and fallback.is_dir():
        return fallback
    return None


def _build_quiz_jsonl_index(quizzes_dir: Path) -> Dict[str, Path]:
    """
    Return mapping quiz_id -> quiz JSONL path for fast lookup.

    If quizzes_dir is the repo-wide `quizzes/` folder, this scans recursively.
    Otherwise, it scans only that directory.
    """
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
        # Prefer newest by mtime; tie-break by path for determinism.
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


def _infer_generator_model(payload: Dict[str, Any], *, path: Path, runs_root: Path) -> str:
    gen = payload.get("generator_model")
    if isinstance(gen, str) and gen.strip():
        return gen.strip()

    try:
        rel = path.relative_to(runs_root)
    except Exception:  # noqa: BLE001
        rel = path

    if rel.parts:
        return str(rel.parts[0]).strip() or "unknown"
    return "unknown"


def _canonicalize_slash_spacing(label: str | None) -> str | None:
    """
    Normalize harmless formatting differences around '/'.

    Example: "Allergy / Immunology" -> "Allergy/Immunology"
    """
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    return _SLASH_SPACING_RE.sub("/", text)


def _parse_csv_list(value: str | None) -> List[str] | None:
    if not value:
        return None
    out = [v.strip() for v in str(value).split(",") if v.strip()]
    return out or None


def _parse_yaml_list(path: str | None) -> List[str] | None:
    if not path:
        return None

    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None

    text = p.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None

    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, list):
            out = [str(item).strip() for item in data if str(item).strip()]
            return out or None

    # Fallback: line-based parsing for a simple YAML list:
    #   - model-a
    #   - model-b
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            item = line.lstrip("-").strip()
            if item:
                out.append(item)
    return out or None


def _looks_like_topic_mapping_output(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    topic_mapping = payload.get("topic_mapping")
    return isinstance(topic_mapping, dict) and bool(topic_mapping.get("applied_at"))


def _select_topics_file_for_run(
    *,
    run_dir: Path,
    candidates: Sequence[Path],
    topics_eval_model: str | None,
    prefer_topic_mapping_outputs: bool,
    out_suffix: str,
) -> tuple[Path | None, str | None, str | None]:
    """
    Choose exactly one topics_*.json file for a run directory.

    Returns (path, eval_model, error_reason).
    """
    if not candidates:
        return None, None, "no_topics_files"

    payloads: Dict[Path, Mapping[str, Any] | None] = {p: _load_topics_json(p) for p in candidates}
    parsed = [(p, payloads[p]) for p in candidates if isinstance(payloads[p], dict)]
    if not parsed:
        return None, None, "failed_to_parse_topics_json"

    if topics_eval_model is not None:
        matched = [
            (p, payload)
            for p, payload in parsed
            if str((payload or {}).get("eval_model") or "").strip() == topics_eval_model
        ]
        if not matched:
            return None, None, f"no_topics_file_for_eval_model={topics_eval_model!r}"
        parsed = matched

    if len(parsed) == 1:
        p, payload = parsed[0]
        return p, str((payload or {}).get("eval_model") or "unknown"), None

    mapping_outputs = [(p, payload) for p, payload in parsed if _looks_like_topic_mapping_output(payload)]
    non_mapping = [(p, payload) for p, payload in parsed if not _looks_like_topic_mapping_output(payload)]

    preferred = mapping_outputs if prefer_topic_mapping_outputs else non_mapping
    fallback = non_mapping if prefer_topic_mapping_outputs else mapping_outputs

    if len(preferred) == 1:
        p, payload = preferred[0]
        return p, str((payload or {}).get("eval_model") or "unknown"), None

    # If multiple remain, use the suffix heuristic.
    if out_suffix:
        suff = str(out_suffix)
        preferred_suffix = [
            (p, payload) for p, payload in preferred if p.stem.endswith(suff)
        ]
        if len(preferred_suffix) == 1:
            p, payload = preferred_suffix[0]
            return p, str((payload or {}).get("eval_model") or "unknown"), None

        non_suffix = [
            (p, payload) for p, payload in preferred if not p.stem.endswith(suff)
        ]
        if len(non_suffix) == 1:
            p, payload = non_suffix[0]
            return p, str((payload or {}).get("eval_model") or "unknown"), None

        fallback_suffix = [
            (p, payload) for p, payload in fallback if p.stem.endswith(suff)
        ]
        if len(fallback_suffix) == 1:
            p, payload = fallback_suffix[0]
            return p, str((payload or {}).get("eval_model") or "unknown"), None

        fallback_non_suffix = [
            (p, payload) for p, payload in fallback if not p.stem.endswith(suff)
        ]
        if len(fallback_non_suffix) == 1:
            p, payload = fallback_non_suffix[0]
            return p, str((payload or {}).get("eval_model") or "unknown"), None

    return None, None, "multiple_topics_files (set --topics_eval_model or adjust preferences)"


def _is_heme_onc_combined_label(raw_topic: str | None) -> bool:
    """
    Return True when a raw label looks like the combined heme/onc bucket.
    """
    if raw_topic is None:
        return False
    norm = normalize_label(str(raw_topic))
    return "hematology" in norm and "oncology" in norm


def _build_heme_onc_prompt(question: str, options: Sequence[str]) -> str:
    """
    Build a constrained prompt that forces a 2-way choice.
    """
    options_block = ""
    if options:
        formatted_opts = "\n".join(f"- {opt}" for opt in options)
        options_block = f"\nOptions:\n{formatted_opts}"

    return (
        "You are a medical domain classifier.\n"
        "Classify the following multiple-choice question as either:\n"
        "- Hematology (blood disorders, coagulation, transfusion medicine, hemoglobinopathies, etc)\n"
        "- Medical Oncology (solid tumors, cancer staging, chemotherapy/targeted/immunotherapy, etc)\n\n"
        "Respond with exactly one label from this list and nothing else:\n"
        "Hematology\n"
        "Medical Oncology\n\n"
        f"Question:\n{str(question or '').strip()}"
        f"{options_block}\n\n"
        "Answer with exactly one label."
    )


def _normalize_heme_onc_response(raw_response: str) -> str | None:
    """
    Map a raw model response to either 'Hematology' or 'Medical Oncology'.
    """
    text = (raw_response or "").strip().strip("`").strip()
    if not text:
        return None

    first_line = text.splitlines()[0].strip()
    first_line = first_line.strip().strip("\"'").strip()
    first_norm = re.sub(r"[^a-zA-Z ]+", " ", first_line).strip().casefold()
    first_norm = re.sub(r"\s+", " ", first_norm).strip()

    if first_norm == "hematology":
        return "Hematology"
    if first_norm in {"medical oncology", "oncology"}:
        return "Medical Oncology"

    # Substring fallback (handles "The answer is Hematology.")
    full_norm = re.sub(r"[^a-zA-Z ]+", " ", text).strip().casefold()
    full_norm = re.sub(r"\s+", " ", full_norm).strip()
    if "medical oncology" in full_norm:
        return "Medical Oncology"
    if "hematology" in full_norm and "oncology" not in full_norm:
        return "Hematology"
    if "oncology" in full_norm and "hematology" not in full_norm:
        return "Medical Oncology"

    return None


def categorize_question(
    question: str,
    options: Sequence[str],
    *,
    model: str = "claude-3-7-sonnet-20250219",
    max_tokens: int = 512,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> tuple[str | None, str]:
    """
    Classify a single question as 'Hematology' or 'Medical Oncology'.

    Returns (label_or_none, raw_model_response).
    """
    prompt = _build_heme_onc_prompt(question, options)
    if dry_run:
        return None, "[dry-run skipped model call]"

    # Lazy import so this script can still run in mapping-only mode without
    # requiring full LLM client dependencies to be installed.
    from quizbench.clients import call_llm  # noqa: E402

    raw_response, _trace = call_llm(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        judge_mode=False,
    )
    label = _normalize_heme_onc_response(raw_response)
    return label, raw_response


def _run_accumulate_only(
    args: argparse.Namespace,
    *,
    runs_root: Path,
    targets: Any,
    topic_map_path: Path | None,
    config: TopicMappingConfig,
    canonical_index: Mapping[str, str],
    created_at: str,
) -> None:
    from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES, filter_by_judge  # noqa: E402

    judge_models = (
        _parse_csv_list(args.judge_models)
        or _parse_yaml_list(args.judge_models_yaml)
        or list(DEFAULT_ENSEMBLE_JUDGES)
    )
    judge_models = [m for m in (judge_models or []) if str(m).strip()]
    if not judge_models and not bool(args.allow_missing_judges):
        raise SystemExit(
            "[FATAL] No judge models provided. Pass --judge_models/--judge_models_yaml "
            "or set --allow_missing_judges."
        )

    out_name = str(args.accumulated_out_name or "").strip() or "accumulated_topic_distribution.json"
    topics_eval_model = str(args.topics_eval_model).strip() if args.topics_eval_model else None
    prefer_mapping_outputs = bool(args.prefer_topic_mapping_outputs)
    canonical_set = set(targets.categories)

    quizzes_dir = _resolve_quizzes_dir_for_fallback(runs_root=runs_root, override=args.quizzes_dir)
    quiz_index: Dict[str, Path] = {}
    if bool(args.fallback_to_quiz_target_topics):
        if quizzes_dir is None:
            print(
                "[WARN] --fallback_to_quiz_target_topics enabled, but could not resolve quizzes_dir; "
                "runs missing topics_*.json will be skipped."
            )
        else:
            quiz_index = _build_quiz_jsonl_index(quizzes_dir)

    generator_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir()]
    if not generator_dirs:
        raise SystemExit(f"[FATAL] No generator directories found under {runs_root}")

    wrote = 0
    skipped_existing = 0
    for generator_dir in generator_dirs:
        generator_model = generator_dir.name
        out_path = generator_dir / out_name
        if bool(args.skip_existing_accumulated) and out_path.exists():
            skipped_existing += 1
            print(f"[INFO] {generator_model}: exists, skipping write: {out_path}")
            continue

        all_counts: Counter[str] = Counter()
        valid_counts: Counter[str] = Counter()
        non_target_counts: Counter[str] = Counter()
        non_target_examples: Dict[str, Dict[str, str]] = {}
        all_total_questions = 0
        all_total_counted = 0
        all_runs_included: List[str] = []
        all_runs_skipped: Dict[str, str] = {}
        valid_total_questions = 0
        valid_total_counted = 0
        valid_runs_included: List[str] = []
        valid_runs_skipped: Dict[str, str] = {}

        run_dirs = [p for p in sorted(generator_dir.iterdir()) if p.is_dir()]
        for run_dir in run_dirs:
            candidates = sorted(run_dir.glob("topics_*.json"))
            qid_to_topic: Dict[str, str] = {}
            total_questions_for_run: int | None = None

            topics_path, _eval_model, err = _select_topics_file_for_run(
                run_dir=run_dir,
                candidates=candidates,
                topics_eval_model=topics_eval_model,
                prefer_topic_mapping_outputs=prefer_mapping_outputs,
                out_suffix=str(args.out_suffix or ""),
            )
            if topics_path is not None and err is None:
                payload = _load_topics_json(topics_path)
                if not payload:
                    all_runs_skipped[str(run_dir)] = f"failed_to_parse_topics_json: {topics_path.name}"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                per_question = payload.get("per_question")
                if not isinstance(per_question, list):
                    all_runs_skipped[str(run_dir)] = "missing_or_invalid_per_question"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                total_questions_for_run = len(per_question)
                for row in per_question:
                    if not isinstance(row, dict):
                        continue
                    qid = str(row.get("question_id") or "").strip()
                    if not qid:
                        continue

                    raw_topic = row.get("topic_mapped")
                    if raw_topic is None:
                        raw_topic = row.get("topic")

                    raw_topic = _canonicalize_slash_spacing(
                        str(raw_topic) if raw_topic is not None else None
                    )
                    mapped = map_topic_label(
                        str(raw_topic) if raw_topic is not None else None,
                        canonical_categories=targets.categories,
                        canonical_index=canonical_index,
                        config=config,
                        mode=args.mode,
                        unmapped_policy=args.unmapped_policy,
                    )
                    if mapped is None:
                        continue
                    mapped_str = str(mapped)
                    if mapped_str not in canonical_set and mapped_str != config.misc_category:
                        non_target_counts[mapped_str] += 1
                        if mapped_str not in non_target_examples:
                            non_target_examples[mapped_str] = {
                                "raw_topic": str(raw_topic or ""),
                                "question_id": qid,
                                "run_dir": str(run_dir),
                            }
                    qid_to_topic[qid] = str(mapped)
            else:
                if err and err != "no_topics_files":
                    all_runs_skipped[str(run_dir)] = err
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                if not bool(args.fallback_to_quiz_target_topics):
                    all_runs_skipped[str(run_dir)] = err or "unknown_error"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                quiz_id = run_dir.name
                quiz_path = quiz_index.get(quiz_id)
                if quiz_path is None:
                    all_runs_skipped[str(run_dir)] = err or "no_topics_files"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                try:
                    quiz_items = read_jsonl(str(quiz_path))
                except Exception:  # noqa: BLE001
                    all_runs_skipped[str(run_dir)] = f"failed_to_parse_quiz_jsonl: {quiz_path}"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

                total_questions_for_run = len(quiz_items)
                for idx, item in enumerate(quiz_items, start=1):
                    if not isinstance(item, dict):
                        continue
                    qid = str(item.get("question_id") or "").strip() or f"{quiz_id}-{idx:03d}"

                    raw_topic = item.get("target_topic")
                    if raw_topic is None:
                        raw_topic = item.get("topic")

                    raw_topic = _canonicalize_slash_spacing(
                        str(raw_topic) if raw_topic is not None else None
                    )
                    mapped = map_topic_label(
                        str(raw_topic) if raw_topic is not None else None,
                        canonical_categories=targets.categories,
                        canonical_index=canonical_index,
                        config=config,
                        mode=args.mode,
                        unmapped_policy=args.unmapped_policy,
                    )
                    if mapped is None:
                        continue
                    mapped_str = str(mapped)
                    if mapped_str not in canonical_set and mapped_str != config.misc_category:
                        non_target_counts[mapped_str] += 1
                        if mapped_str not in non_target_examples:
                            non_target_examples[mapped_str] = {
                                "raw_topic": str(raw_topic or ""),
                                "question_id": qid,
                                "run_dir": str(run_dir),
                            }
                    qid_to_topic[qid] = str(mapped)

                if not qid_to_topic:
                    all_runs_skipped[str(run_dir)] = "no_topics_found_in_quiz_jsonl"
                    valid_runs_skipped.setdefault(str(run_dir), all_runs_skipped[str(run_dir)])
                    continue

            for topic in qid_to_topic.values():
                all_counts[topic] += 1
            all_total_counted += len(qid_to_topic)
            all_total_questions += int(total_questions_for_run or 0)
            all_runs_included.append(str(run_dir))

            allowed_qids = filter_by_judge(
                run_dir,
                judge_models,
                min_medical_score=args.min_medical_score,
                require_logical_valid=True,
                logical_mode=args.logical_mode,
            )
            if allowed_qids is None:
                if bool(args.allow_missing_judges):
                    allowed_qids = set(qid_to_topic.keys())
                else:
                    valid_runs_skipped[str(run_dir)] = "no_judge_results_found"
                    continue

            valid_total_questions += len(allowed_qids)
            for qid in allowed_qids:
                topic = qid_to_topic.get(str(qid))
                if not topic:
                    continue
                valid_counts[topic] += 1
            valid_runs_included.append(str(run_dir))

        valid_total_counted = int(sum(valid_counts.values()))

        if bool(args.assert_in_targets) and non_target_counts:
            top = ", ".join(f"{k}={v}" for k, v in non_target_counts.most_common(10))
            sample = non_target_examples.get(non_target_counts.most_common(1)[0][0]) or {}
            raise SystemExit(
                "[FATAL] Found non-target topic labels while accumulating distributions for "
                f"{generator_model!r} (top 10: {top}). Example: {sample}. "
                "Fix the classifier outputs or provide/update --topic_map/--mode/--unmapped_policy."
            )

        out_payload = {
            "created_at": created_at,
            "generator_model": generator_model,
            "runs_root": str(runs_root),
            "generator_run_root": str(generator_dir),
            "topics_eval_model": topics_eval_model,
            "topic_map_path": str(topic_map_path) if topic_map_path else None,
            "topic_map_mode": args.mode,
            "unmapped_topic_policy": args.unmapped_policy,
            "misc_category": config.misc_category,
            "judge_models": list(judge_models),
            "logical_mode": args.logical_mode,
            "min_medical_score": args.min_medical_score,
            "all_total_questions": int(all_total_questions),
            "all_total_counted": int(all_total_counted),
            "all_counts_by_topic": {k: int(v) for k, v in all_counts.most_common()},
            "all_runs_included": list(all_runs_included),
            "all_runs_skipped": dict(all_runs_skipped),
            "valid_total_questions": int(valid_total_questions),
            "valid_total_counted": int(valid_total_counted),
            "valid_counts_by_topic": {k: int(v) for k, v in valid_counts.most_common()},
            "valid_runs_included": list(valid_runs_included),
            "valid_runs_skipped": dict(valid_runs_skipped),
        }

        if args.dry_run:
            print(
                f"[DRY-RUN] {generator_model}: would write {out_path} "
                f"(all_counted={all_total_counted}/{all_total_questions}, "
                f"valid_counted={valid_total_counted}/{valid_total_questions})."
            )
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_path, out_payload)
        wrote += 1
        print(
            f"[OK] {generator_model}: wrote {out_path} "
            f"(all_counted={all_total_counted}/{all_total_questions}, "
            f"valid_counted={valid_total_counted}/{valid_total_questions})."
        )

    if args.dry_run:
        print(
            f"[DRY-RUN] Would write accumulated distributions for {len(generator_dirs) - skipped_existing} generator(s) "
            f"(skipped_existing={skipped_existing})."
        )
    else:
        print(
            f"[OK] Wrote accumulated distributions for {wrote} generator(s) "
            f"(skipped_existing={skipped_existing})."
        )


def main() -> None:
    args = _parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    targets = load_integer_targets_csv(Path(args.targets_csv))

    config: TopicMappingConfig
    topic_map_path: Path | None = None
    if args.topic_map:
        topic_map_path = Path(args.topic_map).expanduser().resolve()
        config = load_topic_mapping(topic_map_path)
    else:
        config = TopicMappingConfig()

    canonical_index = build_canonical_index(targets.categories, normalize=config.normalize)
    canonical_set = set(targets.categories)

    created_at = now_utc_iso()

    if str(args.operation).strip().lower() == "accumulate":
        _run_accumulate_only(
            args,
            runs_root=runs_root,
            targets=targets,
            topic_map_path=topic_map_path,
            config=config,
            canonical_index=canonical_index,
            created_at=created_at,
        )
        return

    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else Path(f"topic_mapping_report_{created_at.replace(':', '').replace('-', '')}.json").resolve()
    )

    backup_dir = None
    if args.write_mode == "inplace":
        backup_dir = (
            Path(args.backup_dir).expanduser().resolve()
            if args.backup_dir
            else (Path("tmp") / "topic_mapping_backups" / created_at.replace(":", "").replace("-", "")).resolve()
        )

    if args.refine_misc_heme_onc and args.dry_run:
        print(
            "[WARN] --refine_misc_heme_onc requested with --dry_run; "
            "skipping model calls and leaving those questions mapped as Misc."
        )

    raw_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    raw_counts_by_generator: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    mapped_counts_by_generator: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    dropped_counts_by_generator: Counter[str] = Counter()
    unmapped_raw: Counter[str] = Counter()
    unmapped_examples: MutableMapping[str, List[str]] = defaultdict(list)
    misc_raw: Counter[str] = Counter()
    misc_reason_counts: Counter[str] = Counter()
    misc_reason_by_raw: Dict[str, str] = {}
    misc_raw_samples: List[Dict[str, Any]] = []
    file_errors: Dict[str, str] = {}
    files_total = 0
    files_written = 0
    outputs_overwritten = 0
    outputs_new = 0
    heme_onc_refine_candidates = 0
    heme_onc_refined_counts: Counter[str] = Counter()
    heme_onc_refined_samples: List[Dict[str, Any]] = []
    heme_onc_refine_error_counts: Counter[str] = Counter()

    quiz_id_to_path_cache: Dict[Path, Dict[str, str]] = {}
    quiz_questions_cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}

    all_topics_files = sorted(runs_root.rglob("topics_*.json"))
    topics_files = list(all_topics_files)
    if (
        args.write_mode == "newfile"
        and args.out_suffix
        and not args.include_out_suffix_inputs
    ):
        out_suffix = str(args.out_suffix)
        to_skip = [p for p in topics_files if p.stem.endswith(out_suffix)]
        if to_skip:
            topics_files = [p for p in topics_files if not p.stem.endswith(out_suffix)]
            print(
                f"[INFO] Found {len(all_topics_files)} topics file(s) under {runs_root}; "
                f"excluding {len(to_skip)} file(s) whose stem ends with {out_suffix!r} from input scan "
                "(likely previous outputs). In newfile mode, outputs with that suffix may still be overwritten. "
                "Pass --include_out_suffix_inputs to also treat them as inputs."
            )
    if not topics_files:
        raise SystemExit(f"[FATAL] No topics_*.json found under {runs_root}")

    if not (args.write_mode == "newfile" and args.out_suffix and not args.include_out_suffix_inputs):
        print(f"[INFO] Found {len(all_topics_files)} topics file(s) under {runs_root}; processing {len(topics_files)}.")
    elif len(all_topics_files) == len(topics_files):
        print(f"[INFO] Found {len(all_topics_files)} topics file(s) under {runs_root}; processing {len(topics_files)}.")
    else:
        print(f"[INFO] Processing {len(topics_files)} input topics file(s).")

    for path in topics_files:
        files_total += 1
        payload = _load_topics_json(path)
        if not payload:
            file_errors[str(path)] = "failed_to_parse_json"
            continue

        per_question = payload.get("per_question")
        if not isinstance(per_question, list):
            file_errors[str(path)] = "missing_or_invalid_per_question"
            continue

        generator_model = _infer_generator_model(payload, path=path, runs_root=runs_root)

        file_mapped: Counter[str] = Counter()
        file_unmapped: Counter[str] = Counter()
        dropped = 0
        quiz_questions_by_id: Dict[str, Dict[str, Any]] | None = None

        for row in per_question:
            if not isinstance(row, dict):
                continue

            raw_topic = row.get("topic")
            raw_topic_for_mapping = _canonicalize_slash_spacing(
                str(raw_topic) if raw_topic is not None else None
            )
            if raw_topic is not None:
                raw_counts[str(raw_topic)] += 1
                raw_counts_by_generator[generator_model][str(raw_topic)] += 1

            mapping_reason = "existing"
            if (not args.overwrite_existing) and row.get("topic_mapped") is not None:
                mapped = row.get("topic_mapped")
            else:
                mapped, mapping_reason = map_topic_label_explained(
                    str(raw_topic_for_mapping) if raw_topic_for_mapping is not None else None,
                    canonical_categories=targets.categories,
                    canonical_index=canonical_index,
                    config=config,
                    mode=args.mode,
                    unmapped_policy=args.unmapped_policy,
                )
                row["topic_mapped"] = mapped

            raw_str = str(raw_topic).strip() if raw_topic is not None else ""

            if (
                args.refine_misc_heme_onc
                and mapped is not None
                and str(mapped) == config.misc_category
                and raw_str
                and _is_heme_onc_combined_label(raw_str)
            ):
                heme_onc_refine_candidates += 1
                if not args.dry_run:
                    if quiz_questions_by_id is None:
                        run_dir = path.parent.parent
                        quiz_id = str(payload.get("quiz_id") or path.parent.name).strip()
                        quiz_paths = quiz_id_to_path_cache.get(run_dir)
                        if quiz_paths is None:
                            try:
                                manifest_path = resolve_quizbench_manifest_path(run_dir)
                            except SystemExit:
                                heme_onc_refine_error_counts["missing_manifest"] += 1
                                quiz_paths = {}
                            else:
                                try:
                                    with manifest_path.open("r", encoding="utf-8") as f:
                                        manifest = json.load(f)
                                except Exception:  # noqa: BLE001
                                    heme_onc_refine_error_counts["manifest_parse_error"] += 1
                                    manifest = {}
                                quiz_paths = {}
                                for q in manifest.get("quizzes") or []:
                                    if not isinstance(q, dict):
                                        continue
                                    qid = str(q.get("quiz_id") or "").strip()
                                    qpath = str(q.get("quiz_path") or "").strip()
                                    if qid and qpath:
                                        quiz_paths[qid] = qpath
                            quiz_id_to_path_cache[run_dir] = quiz_paths

                        quiz_path_str = quiz_paths.get(quiz_id)
                        if not quiz_path_str:
                            heme_onc_refine_error_counts["missing_quiz_path"] += 1
                            quiz_questions_by_id = {}
                        else:
                            quiz_path = Path(quiz_path_str)
                            quiz_path = quiz_path if quiz_path.is_absolute() else (ROOT_DIR / quiz_path)
                            quiz_path = quiz_path.expanduser().resolve()
                            cached = quiz_questions_cache.get(quiz_path)
                            if cached is None:
                                if not quiz_path.exists():
                                    heme_onc_refine_error_counts["quiz_file_missing"] += 1
                                    cached = {}
                                else:
                                    try:
                                        quiz_items = read_jsonl(str(quiz_path))
                                    except Exception:  # noqa: BLE001
                                        heme_onc_refine_error_counts["quiz_file_parse_error"] += 1
                                        quiz_items = []
                                    cached = {}
                                    for item in quiz_items:
                                        if not isinstance(item, dict):
                                            continue
                                        qid = str(item.get("question_id") or "").strip()
                                        if qid:
                                            cached[qid] = item
                                quiz_questions_cache[quiz_path] = cached
                            quiz_questions_by_id = cached

                    question_id = str(row.get("question_id") or "").strip()
                    item = (quiz_questions_by_id or {}).get(question_id)
                    if not item:
                        heme_onc_refine_error_counts["question_missing_in_quiz"] += 1
                    else:
                        options = item.get("options")
                        options = options if isinstance(options, list) else []
                        try:
                            label, raw_response = categorize_question(
                                str(item.get("question") or ""),
                                options,
                                model=args.refine_misc_model,
                                max_tokens=args.refine_misc_max_tokens,
                                temperature=args.refine_misc_temperature,
                                dry_run=args.dry_run,
                            )
                        except Exception:  # noqa: BLE001
                            heme_onc_refine_error_counts["model_call_error"] += 1
                            label, raw_response = None, ""
                        else:
                            if not label:
                                heme_onc_refine_error_counts["unparsed_model_output"] += 1
                            else:
                                row["topic_mapped"] = label
                                row["topic_split_from_misc"] = {
                                    "raw_topic": raw_str,
                                    "model": args.refine_misc_model,
                                    "max_tokens": args.refine_misc_max_tokens,
                                    "temperature": args.refine_misc_temperature,
                                    "raw_response": raw_response,
                                    "applied_at": created_at,
                                }
                                mapped = label
                                mapping_reason = "refined_heme_onc"
                                heme_onc_refined_counts[label] += 1
                                heme_onc_refined_samples.append(
                                    {
                                        "question_id": question_id,
                                        "topics_file": str(path),
                                        "raw_topic": raw_str,
                                        "mapped_topic": label,
                                        "model": args.refine_misc_model,
                                        "raw_response": raw_response,
                                    }
                                )

            if args.overwrite_topic:
                row["topic"] = mapped

            if mapped is None:
                dropped += 1
                dropped_counts_by_generator[generator_model] += 1
                continue

            mapped_str = str(mapped)
            if mapped_str == config.misc_category and raw_str:
                misc_raw[raw_str] += 1
                misc_reason_counts[mapping_reason] += 1
                if raw_str not in misc_reason_by_raw:
                    misc_reason_by_raw[raw_str] = mapping_reason
                elif misc_reason_by_raw[raw_str] != mapping_reason:
                    misc_reason_by_raw[raw_str] = "multiple"
                misc_raw_samples.append(
                    {
                        "raw_topic": raw_str,
                        "mapped_topic": config.misc_category,
                        "reason": mapping_reason,
                        "question_id": str(row.get("question_id") or "").strip(),
                        "topics_file": str(path),
                    }
                )
            if mapped_str not in canonical_set and mapped_str != config.misc_category:
                # If the mapped output differs from the raw label, this likely
                # indicates a bad mapping override (typo / non-canonical dest).
                if raw_str and mapped_str != raw_str:
                    file_errors[str(path)] = f"mapped_category_not_in_targets: {mapped_str!r}"
                    # Skip counting/writing this file further.
                    break

            mapped_counts[mapped_str] += 1
            mapped_counts_by_generator[generator_model][mapped_str] += 1
            file_mapped[mapped_str] += 1

            # Track unmapped raw labels when policy is "keep" (or other invalid outputs).
            if mapped_str not in canonical_set and mapped_str != config.misc_category:
                if raw_str and raw_str not in canonical_set:
                    unmapped_raw[raw_str] += 1
                    file_unmapped[raw_str] += 1
                    if len(unmapped_examples[raw_str]) < 5:
                        unmapped_examples[raw_str].append(str(path))

        payload["topic_mapping"] = {
            "applied_at": created_at,
            "topic_map_path": str(topic_map_path) if topic_map_path else None,
            "mode": args.mode,
            "unmapped_policy": args.unmapped_policy,
            "normalize": config.normalize,
            "combined_to_misc": config.combined_to_misc,
            "misc_category": config.misc_category,
            "refine_misc_heme_onc": bool(args.refine_misc_heme_onc),
            "refine_misc_model": str(args.refine_misc_model),
            "refine_misc_max_tokens": int(args.refine_misc_max_tokens),
            "refine_misc_temperature": float(args.refine_misc_temperature),
        }

        payload["summary_mapped"] = {
            "topic_counts": dict(file_mapped),
            "num_questions": len(per_question),
            "num_dropped": dropped,
            "num_unmapped_raw": int(sum(file_unmapped.values())),
            "unmapped_raw_counts": dict(file_unmapped),
        }

        if str(path) in file_errors:
            # Skip writes for files with mapping/validation errors.
            continue

        if args.write_mode == "newfile":
            out_path = path.with_name(f"{path.stem}{args.out_suffix}{path.suffix}")
            if out_path == path:
                file_errors[str(path)] = "out_path_equals_input_path (choose a non-empty --out_suffix)"
                continue
            if out_path.exists():
                outputs_overwritten += 1
            else:
                outputs_new += 1
            if args.dry_run:
                continue
            _write_json(out_path, payload)
            files_written += 1
        else:
            assert backup_dir is not None
            outputs_overwritten += 1
            if args.dry_run:
                continue
            rel = path.relative_to(runs_root)
            backup_path = backup_dir / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            _write_json(path, payload)
            files_written += 1

    report = {
        "created_at": created_at,
        "runs_root": str(runs_root),
        "targets_csv": str(Path(args.targets_csv).expanduser().resolve()),
        "topic_map_path": str(topic_map_path) if topic_map_path else None,
        "mode": args.mode,
        "unmapped_policy": args.unmapped_policy,
        "write_mode": args.write_mode,
        "out_suffix": args.out_suffix,
        "backup_dir": str(backup_dir) if backup_dir else None,
        "files_total": files_total,
        "files_written": files_written,
        "raw_topic_counts": dict(raw_counts),
        "mapped_topic_counts": dict(mapped_counts),
        "raw_topic_counts_by_generator": {
            gen: {k: int(v) for k, v in ctr.most_common()}
            for gen, ctr in sorted(raw_counts_by_generator.items())
        },
        "mapped_topic_counts_by_generator": {
            gen: {k: int(v) for k, v in ctr.most_common()}
            for gen, ctr in sorted(mapped_counts_by_generator.items())
        },
        "dropped_counts_by_generator": dict(dropped_counts_by_generator),
        "unmapped_raw_counts": dict(unmapped_raw),
        "unmapped_raw_examples": dict(unmapped_examples),
        "misc_category": config.misc_category,
        "misc_raw_counts": dict(misc_raw),
        "misc_reason_counts": dict(misc_reason_counts),
        "misc_reason_by_raw": dict(misc_reason_by_raw),
        "misc_raw_samples": list(misc_raw_samples),
        "refine_misc_heme_onc": bool(args.refine_misc_heme_onc),
        "refine_misc_model": str(args.refine_misc_model),
        "refine_misc_max_tokens": int(args.refine_misc_max_tokens),
        "refine_misc_temperature": float(args.refine_misc_temperature),
        "heme_onc_refine_candidates": int(heme_onc_refine_candidates),
        "heme_onc_refined_counts": dict(heme_onc_refined_counts),
        "heme_onc_refine_error_counts": dict(heme_onc_refine_error_counts),
        "heme_onc_refined_samples": list(heme_onc_refined_samples),
        "file_errors": dict(file_errors),
    }

    if args.dry_run:
        would_write_n = files_total - len(file_errors)
        if args.write_mode == "newfile":
            print(
                f"[DRY-RUN] Processed {files_total} input file(s); would write {would_write_n} output file(s) "
                f"(overwrite {outputs_overwritten}, new {outputs_new})."
            )
        else:
            backup_note = f" (backups under {backup_dir})" if backup_dir else ""
            print(
                f"[DRY-RUN] Processed {files_total} input file(s); would overwrite {would_write_n} input file(s)"
                f"{backup_note}."
            )
        print(f"[DRY-RUN] Would write report to: {report_path}")
    else:
        _write_json(report_path, report)
        if args.write_mode == "newfile":
            print(
                f"[OK] Processed {files_total} input file(s); wrote {files_written} output file(s) "
                f"(overwrote {outputs_overwritten}, new {outputs_new})."
            )
        else:
            print(f"[OK] Processed {files_total} input file(s); overwrote {files_written} input file(s).")
        print(f"[OK] Wrote report to: {report_path}")
        if args.write_mode == "inplace":
            assert backup_dir is not None
            print(f"[OK] Backups written under: {backup_dir}")

    if file_errors:
        print(f"[WARN] {len(file_errors)} file(s) had errors (see report).")

    if mapped_counts_by_generator:
        print("[INFO] Cumulative mapped topic counts by generator:")
        for gen, ctr in sorted(mapped_counts_by_generator.items()):
            total_mapped = int(sum(ctr.values()))
            total_dropped = int(dropped_counts_by_generator.get(gen, 0) or 0)
            print(f"  - {gen}: mapped={total_mapped}, dropped={total_dropped}")
            for topic, n in ctr.most_common():
                print(f"      {topic}: {int(n)}")

    if args.refine_misc_heme_onc:
        refined_summary = ", ".join(
            f"{k}={v}" for k, v in heme_onc_refined_counts.most_common()
        ) or "none"
        error_summary = ", ".join(
            f"{k}={v}" for k, v in heme_onc_refine_error_counts.most_common()
        ) or "none"
        print(
            f"[INFO] Heme/onc refinement: candidates={heme_onc_refine_candidates}, "
            f"refined=({refined_summary}), errors=({error_summary})."
        )

    if args.unmapped_policy != "error" and unmapped_raw:
        top = ", ".join(f"{k}={v}" for k, v in unmapped_raw.most_common(10))
        print(f"[WARN] Unmapped raw labels present (top 10): {top}")
    if bool(args.assert_in_targets) and unmapped_raw:
        top = ", ".join(f"{k}={v}" for k, v in unmapped_raw.most_common(10))
        examples = dict(list(unmapped_examples.items())[:5])
        raise SystemExit(
            "[FATAL] Found topic labels that are not in the targets CSV "
            f"(top 10: {top}). Examples: {examples}. "
            "Fix the classifier outputs or provide/update --topic_map/--mode/--unmapped_policy."
        )


if __name__ == "__main__":
    main()
