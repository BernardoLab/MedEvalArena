#!/usr/bin/env python3
"""
Batch-based wrapper around quizbench/batch_generate_quiz.py.

This script mirrors run_gen_quiz.py but uses the OpenAI Batch API (or the
Gemini Batch API via the OpenAI compatibility layer) to generate quizzes for
each generator model. For each generator we keep submitting batch jobs until
we have produced at least `num_questions_per_quiz` valid questions in total.

Fixes:
- Each iteration requests only the remaining number of questions needed by
  default (min(num_questions_per_batch, remaining)), with optional overshoot
  via --planned_batch_request_multiplier.
- Tally caps the number of questions counted from a batch to the remaining need,
  ensuring totals never overshoot the target.
- When re-running into an existing output directory, automatically advances the
  per-generator seed to avoid duplicate seedNNN runs (configurable via
  --no-avoid_existing_seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Keep this in sync with quizbench/batch_generate_quiz.py to avoid importing
# provider SDKs (e.g., openai) when using --plan_targets_only.
DEFAULT_TOPICS = (
    "cardiology,endocrinology,infectious disease,hematology/oncology,neurology,"
    "nephrology,pulmonology,obstetrics,gynecology,pediatrics,geriatrics,"
    "dermatology,rheumatology,emergency medicine,critical care"
)
BATCH_POLL_INTERVAL_SECONDS = 60
MAX_ZERO_VALID_RETRIES = 4
from quizbench.run_gen_quiz import (  # noqa: E402
    build_quiz_id,
    coerce_int,
    collect_env,
    load_config,
    parse_models_field,
    sanitize_model_name,
)
from quizbench.utils import ensure_dir, now_utc_iso, compact_utc_ts  # noqa: E402
from quizbench.manifest_utils import resolve_quizbench_manifest_path  # noqa: E402
from quizbench.topic_mapping import (  # noqa: E402
    TopicMappingConfig,
    build_canonical_index,
    load_topic_mapping,
    map_topic_label,
)
from quizbench.valid_topic_distribution import compute_valid_topic_distribution, _select_topics_file  # noqa: E402
from quizbench.target_planning import (  # noqa: E402
    chunked,
    compute_remainders,
    counts_for_batch,
    expand_to_desired_list,
    load_integer_targets_csv,
)

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES  # noqa: E402


_SEED_RE = re.compile(r"(?:^|[_-])seed(\d+)(?:$|\D)")
_QUIZ_BATCH_RE = re.compile(r"(?:^|[\\/])quizzes_([^\\/]+)")
_PROVIDER_OPENAI = "openai_or_gemini_batch"
_PROVIDER_ANTHROPIC = "anthropic_message_batches"
_PROVIDER_SEQUENTIAL = "sequential_llm"


def _extract_seeds_from_name(name: str) -> List[int]:
    seeds: List[int] = []
    for match in _SEED_RE.finditer(name):
        try:
            seeds.append(int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return seeds


def _extract_quiz_batch_tag_from_runs_root(runs_root: str) -> str | None:
    """
    Extract a "batch" tag from runs_root.

    We treat any path segment like `quizzes_<TAG>` as the batch identifier and
    return `<TAG>`. Example:
      eval_results/quizbench/quizzes_Jan2026/runs/deepseek-v3.2/ -> Jan2026
    """
    text = str(runs_root).strip()
    if not text:
        return None

    matches = list(_QUIZ_BATCH_RE.finditer(text))
    if not matches:
        return None
    tag = matches[-1].group(1).strip()
    return tag or None


def _is_anthropic_model_name(model_name: str) -> bool:
    return str(model_name).startswith("claude-")


def _provider_for_generator(model_name: str, *, use_batch_api: bool) -> str:
    if not use_batch_api:
        return _PROVIDER_SEQUENTIAL
    if _is_anthropic_model_name(model_name):
        return _PROVIDER_ANTHROPIC
    return _PROVIDER_OPENAI


def _batch_input_filename(provider: str, safe_model: str, quiz_idx: int) -> str:
    suffix = "anthropic_batch_input" if provider == _PROVIDER_ANTHROPIC else "batch_input"
    return f"{safe_model}_{suffix}_{quiz_idx:03d}.jsonl"


def _unique_manifest_path(runs_root: str, manifest_basename: str, *, run_ts: datetime) -> Path:
    """
    Pick a manifest path under runs_root that won't overwrite an existing file.

    If `manifest_basename` already exists, we append a UTC timestamp (and, if
    needed, an increment) to keep prior manifests intact.
    """
    root = Path(runs_root).expanduser().resolve()
    base = root / manifest_basename
    if not base.exists():
        return base

    stem, suffix = base.stem, base.suffix
    ts = compact_utc_ts(run_ts)
    candidate = root / f"{stem}_{ts}{suffix}"
    if not candidate.exists():
        return candidate

    for idx in range(1, 1000):
        candidate = root / f"{stem}_{ts}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
    return candidate


def _discover_existing_seeds_for_model(
    *,
    generator_model: str,
    quizzes_dir: str,
    runs_root: str,
) -> Set[int]:
    safe_model = sanitize_model_name(generator_model)
    seeds: Set[int] = set()

    quizzes_path = Path(quizzes_dir).expanduser().resolve()
    if quizzes_path.exists():
        for path in quizzes_path.glob(f"*_{safe_model}_seed*.jsonl"):
            if path.is_file():
                seeds.update(_extract_seeds_from_name(path.name))

    runs_root_path = Path(runs_root).expanduser().resolve()
    gen_runs_root = _resolve_generator_run_root(runs_root_path, generator_model)
    if gen_runs_root.exists():
        filter_by_name = gen_runs_root == runs_root_path and runs_root_path.name not in {safe_model, generator_model}
        for path in gen_runs_root.iterdir():
            if filter_by_name and safe_model not in path.name:
                continue
            seeds.update(_extract_seeds_from_name(path.name))

    return seeds


def _compute_starting_quiz_index(*, base_seed: int, existing_seeds: Set[int]) -> int:
    if not existing_seeds:
        return 0
    next_seed = max(int(base_seed), int(max(existing_seeds)) + 1)
    return max(0, int(next_seed) - int(base_seed))


def _repeat_topic_plan(topic_plan: Sequence[str], *, target_len: int) -> List[str]:
    """
    Repeat a topic plan deterministically to reach `target_len` items.

    Used to "overshoot" planned target batches by requesting extra questions
    while keeping the original per-item topic order.
    """
    if target_len <= 0:
        return []
    base = [str(t).strip() for t in topic_plan if str(t).strip()]
    if not base:
        return []
    if len(base) >= target_len:
        return base[:target_len]
    reps = (target_len + len(base) - 1) // len(base)
    return (base * reps)[:target_len]


def _build_requests_for_model(
    *,
    quiz_id: str,
    num_questions: int,
    num_choices: int,
    seed: int,
    topics_csv: str,
    topic_plan: List[str] | None = None,
    topic_counts: Dict[str, int] | None = None,
    quizzes_dir: str,
) -> List["QuizRequest"]:
    """
    Build a single QuizRequest for the provided quiz metadata.
    """
    # Import lazily so --plan_targets_only works without provider SDK deps.
    from quizbench.batch_generate_quiz import QuizRequest  # noqa: WPS433

    return [
        QuizRequest(
            quiz_id=quiz_id,
            seed=seed,
            num_questions=num_questions,
            num_choices=num_choices,
            topics_csv=topics_csv,
            topic_plan=topic_plan,
            topic_counts=topic_counts,
            out_dir=quizzes_dir,
        )
    ]


def _count_questions_in_quiz(quiz_path: str) -> int:
    """
    Count how many valid question records ended up in a quiz JSONL file.

    We conservatively treat each non-empty, parseable JSON line as one
    question. If the file is missing or malformed we return 0.
    """
    count = 0
    try:
        with open(quiz_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines; they shouldn't normally appear.
                    continue
                count += 1
    except FileNotFoundError:
        return 0
    return count


def _read_target_topics_from_quiz(quiz_path: str) -> List[str | None]:
    """
    Return the per-item target_topic values from a quiz JSONL file.

    If the file is missing or malformed, returns an empty list. Lines that fail
    JSON parsing are skipped (mirroring _count_questions_in_quiz behavior).
    """
    topics: List[str | None] = []
    try:
        with open(quiz_path, "r", encoding="utf-8") as f:
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
                val = row.get("target_topic")
                if val is None:
                    topics.append(None)
                else:
                    topics.append(str(val).strip() or None)
    except FileNotFoundError:
        return []
    return topics


def _load_cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate quizzes via the OpenAI/Gemini Batch API (one batch per generator model)."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    ap.add_argument(
        "--quiz_collection",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under config quizzes_dir for this run. "
            "Use 'auto' to name the collection by the run timestamp."
        ),
    )
    ap.add_argument(
        "--quiz_batch_tag",
        type=str,
        default=None,
        help=(
            "Optional tag used only for naming the manifest file "
            "(quizbench_manifest_<TAG>*.json). When omitted, the tag is inferred from "
            "the config runs_root path segment `quizzes_<TAG>`, if present."
        ),
    )
    ap.add_argument(
        "--lock_quiz_collection",
        action="store_true",
        help=(
            "If set, write a .quizbench_readonly marker file into the output quizzes directory at the end of the run."
        ),
    )
    ap.add_argument(
        "--allow_locked_quiz_collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to allow generation into a quizzes directory even if it contains a .quizbench_readonly marker "
            "(default: True; pass --no-allow_locked_quiz_collection to enforce the marker)."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allow overwriting existing quiz JSONL files in the output directory.",
    )
    ap.add_argument("--batch_input_dir", type=str, default=None, help="Directory to place batch input JSONL files.")
    ap.add_argument("--temperature", type=float, default=0, help="Sampling temperature for generation models.")
    ap.add_argument(
        "--max_output_tokens",
        "--max_tokens",
        dest="max_output_tokens",
        type=int,
        default=None,
        help="Max output tokens per quiz request (alias: --max_tokens; defaults to config or 16000 if unset).",
    )
    ap.add_argument(
        "--poll_seconds",
        type=int,
        default=BATCH_POLL_INTERVAL_SECONDS,
        help="Polling interval while waiting for batch completion.",
    )
    ap.add_argument(
        "--topics_csv",
        type=str,
        default=None,
        help="Override topics CSV (falls back to config, then generator default list).",
    )
    ap.add_argument(
        "--num_choices",
        type=int,
        default=None,
        help="Number of answer choices per question (default 5).",
    )
    ap.add_argument(
        "--plan_targets_only",
        action="store_true",
        help=(
            "Run Steps 0–3 (topic dist + target planning) and exit without "
            "submitting any generation batches."
        ),
    )
    ap.add_argument(
        "--use_target_batches",
        action="store_true",
        help=(
            "Plan target batches (Steps 0–3) and then generate quizzes by iterating "
            "over the planned batches (rather than generating until a flat total is reached)."
        ),
    )
    ap.add_argument(
        "--planned_batch_request_multiplier",
        type=float,
        default=2.0,
        help=(
            "When using --use_target_batches, multiply the number of questions requested for each planned batch by "
            "this factor to 'overshoot' and improve the chance of meeting targets despite invalid generations "
            "(default: 1.0; e.g. 2.0 doubles requests). Progress counting still caps at the planned need."
        ),
    )
    ap.add_argument(
        "--avoid_existing_seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to scan the output quizzes_dir and runs_root for existing seedNNN artifacts and start new "
            "generation at the next unused seed per generator (default: True)."
        ),
    )
    ap.add_argument(
        "--topic_mapping_report",
        type=str,
        default=None,
        help=(
            "Optional topic_mapping_report_*.json produced by quizbench/apply_topic_mapping.py "
            "(or a per-generator shim like accumulated_mapped_topics.json produced by "
            "quizbench/split_topic_mapping_report.py). "
            "If provided, its metadata is used as defaults for --targets_csv/--topic_map/"
            "--topic_map_mode/--unmapped_topic_policy."
        ),
    )
    ap.add_argument(
        "--targets_csv",
        type=str,
        default="data/ABMS_specialties.csv",
        help="CSV with target counts per category (default: data/ABMS_specialties.csv).",
    )
    ap.add_argument(
        "--target_total",
        type=int,
        default=None,
        help=(
            "Optional expected total target questions across categories. "
            "If omitted, uses the sum of the CSV values."
        ),
    )
    ap.add_argument(
        "--topics_eval_model",
        type=str,
        default=None,
        help=(
            "When multiple topics_*.json exist per run, select the one whose "
            "payload eval_model matches this string."
        ),
    )
    ap.add_argument(
        "--prefer_topic_mapping_outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When selecting topics_*.json per run, prefer files produced by "
            "quizbench/apply_topic_mapping.py (they include a topic_mapping block and "
            "may contain refinements like hematology vs oncology)."
        ),
    )
    ap.add_argument(
        "--judge_models_yaml",
        type=str,
        default="data/judge_models.yaml",
        help=(
            "For determining existing question topic distribution."
            "yaml with judge model names to use for majority validity "
            "(default: quizbench.aggregate_judges.DEFAULT_ENSEMBLE_JUDGES)."
        ),
    )
    ap.add_argument(
        "--logical_mode",
        type=str,
        default="majority",
        help=("Mode medical_accuracy_score required for judge eligibility."
              "For determine existing question topic distribution.")
    )
    ap.add_argument(
        "--min_medical_score",
        type=int,
        default=4,
        help=("Optional minimum medical_accuracy_score required for judge eligibility."
              "For determine existing question topic distribution.")
    )
    ap.add_argument(
        "--allow_missing_judges",
        action="store_true",
        help="If set, treat runs with no judge outputs as unfiltered (all questions allowed).",
    )
    ap.add_argument(
        "--dist_out_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write questions_dist_{LLM}.json outputs "
            "(defaults to each generator's run root)."
        ),
    )
    ap.add_argument(
        "--recompute_questions_dist",
        action="store_true",
        help="If set, recompute questions_dist_{LLM}.json even if it already exists.",
    )
    ap.add_argument(
        "--topic_map",
        type=str,
        default=None,
        help="Optional YAML/JSON mapping from topics_*.json labels to target CSV categories.",
    )
    ap.add_argument(
        "--topic_map_mode",
        type=str,
        default="map",
        choices=["exact", "normalize", "map"],
        help="Mapping mode for --topic_map (default: map).",
    )
    ap.add_argument(
        "--unmapped_topic_policy",
        type=str,
        default="keep",
        choices=["error", "keep", "misc", "drop"],
        help="What to do with unmapped topic labels (default: keep).",
    )
    return ap.parse_args()


def _resolve_quiz_collection(raw: object | None, *, run_ts: datetime) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip()
    if not val or val.lower() in {"none", "null"}:
        return None
    if val.lower() == "auto":
        return compact_utc_ts(run_ts)
    return val


def _parse_yaml_list(path: str | None) -> List[str] | None:
    if not path:
        return None

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

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

    # Fallback: line-based parsing for simple YAML lists.
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line.lstrip("-").strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out or None


def _load_topic_mapping_report(path: str) -> Dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(str(report_path))
    with report_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"topic_mapping_report must be a JSON object: {report_path}")
    return {str(k): v for k, v in payload.items()}


def _resolve_generator_run_root(runs_root: Path, generator_model: str) -> Path:
    """
    Resolve the per-generator run root for topic/judge artifacts.

    Supports either:
      - runs_root already being the generator directory, containing quizbench_manifest.json
      - runs_root being a base directory, containing subdirs per generator model
    """
    runs_root = runs_root.expanduser().resolve()
    if (runs_root / "quizbench_manifest.json").exists():
        return runs_root

    direct = runs_root / generator_model
    if (direct / "quizbench_manifest.json").exists() or direct.is_dir():
        return direct

    safe = runs_root / sanitize_model_name(generator_model)
    if (safe / "quizbench_manifest.json").exists() or safe.is_dir():
        return safe

    # Fall back to the configured runs_root; caller can handle missing data.
    return runs_root


def _validate_targets_total(targets_total: int, expected_total: int) -> None:
    if int(targets_total) != int(expected_total):
        raise SystemExit(
            f"[FATAL] targets_csv sums to {targets_total}, expected {expected_total}. "
            "Fix the CSV or pass --target_total accordingly."
        )


def _normalize_counts_by_topic(raw: object) -> Dict[str, int]:
    """
    Coerce a JSON-ish mapping into a {topic: nonnegative int} dict.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw.items():
        topic = str(key).strip()
        if not topic:
            continue
        try:
            n = int(value or 0)
        except (TypeError, ValueError):
            continue
        out[topic] = max(n, 0)
    return out


def _split_topic_counts(
    *,
    counts_by_topic: Mapping[str, int],
    target_categories: Sequence[str] | None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Return (target_items, extra_items) lists, both including only positive counts.
    """
    if not target_categories:
        items = [(str(k), int(v or 0)) for k, v in counts_by_topic.items() if int(v or 0) > 0]
        return items, []

    target_set = set(target_categories)
    target_items = [
        (cat, int(counts_by_topic.get(cat, 0) or 0))
        for cat in target_categories
        if int(counts_by_topic.get(cat, 0) or 0) > 0
    ]
    extra_items = [
        (str(k), int(v or 0))
        for k, v in counts_by_topic.items()
        if str(k) not in target_set and int(v or 0) > 0
    ]
    return target_items, extra_items


def _print_current_topic_distribution(
    *,
    valid_total: int,
    valid_counts_by_topic: Mapping[str, int],
    target_categories: Sequence[str] | None = None,
    max_extra_topics: int = 15,
) -> None:
    """
    Print a human-readable topic distribution summary to stdout.
    """
    counts_by_topic = dict(valid_counts_by_topic)
    total = int(valid_total or 0)
    if total <= 0:
        total = int(sum(int(v or 0) for v in counts_by_topic.values()))

    if total <= 0:
        print("  [INFO] Current valid topic distribution: (no valid questions found)")
        return

    target_items, extra_items = _split_topic_counts(
        counts_by_topic=counts_by_topic,
        target_categories=target_categories,
    )

    if target_categories:
        target_items.sort(key=lambda kv: (-kv[1], kv[0]))
        zero_targets = [
            cat for cat in target_categories if int(counts_by_topic.get(cat, 0) or 0) <= 0
        ]
        print(f"  [INFO] Current valid topic distribution (n={total}, within targets={len(target_items)}):")
        for topic, n in target_items:
            pct = 100.0 * float(n) / float(total)
            print(f"    {topic}: {n} ({pct:.1f}%)")
        if zero_targets:
            print(
                "  [INFO] Target categories with 0 valid questions: "
                f"{len(zero_targets)} / {len(target_categories)}"
            )
    else:
        target_items.sort(key=lambda kv: (-kv[1], kv[0]))
        print(f"  [INFO] Current valid topic distribution (n={total}):")
        for topic, n in target_items:
            pct = 100.0 * float(n) / float(total)
            print(f"    {topic}: {n} ({pct:.1f}%)")
        return

    if extra_items:
        extra_items.sort(key=lambda kv: (-kv[1], kv[0]))
        total_extra = int(sum(n for _, n in extra_items))
        shown = extra_items[: max(0, int(max_extra_topics))]
        print(
            f"  [INFO] Non-target topics: total={total_extra}, unique={len(extra_items)} "
            f"(showing {len(shown)})."
        )
        for topic, n in shown:
            pct = 100.0 * float(n) / float(total)
            print(f"    {topic}: {n} ({pct:.1f}%)")
        remaining = len(extra_items) - len(shown)
        if remaining > 0:
            print(f"    ... ({remaining} more; see questions_dist JSON for full list)")


def _print_target_deltas(
    *,
    targets_by_category: Mapping[str, int],
    valid_counts_by_topic: Mapping[str, int],
    categories_in_order: Sequence[str],
) -> None:
    """
    Print per-category deltas: target - current valid.

    This makes it easy to see how many questions remain (positive) or are
    overshot (negative) for every target category.
    """
    categories = [str(c).strip() for c in categories_in_order if str(c).strip()]
    if not categories:
        return

    print("  [INFO] Target deltas (target - valid) by category:")
    for cat in categories:
        try:
            target_n = int(targets_by_category.get(cat, 0) or 0)
        except (TypeError, ValueError):
            target_n = 0
        try:
            valid_n = int(valid_counts_by_topic.get(cat, 0) or 0)
        except (TypeError, ValueError):
            valid_n = 0

        delta = int(target_n) - int(valid_n)
        if delta > 0:
            print(f"    {cat}: remaining={delta} (target={target_n}, valid={valid_n})")
        elif delta < 0:
            print(f"    {cat}: overshoot={-delta} (target={target_n}, valid={valid_n})")
        else:
            print(f"    {cat}: on_target (target={target_n}, valid={valid_n})")


def _load_quiz_paths_from_manifest(generator_run_root: Path) -> Dict[str, Path]:
    """
    Load quiz_id -> quiz_path from the newest quizbench_manifest*.json under generator_run_root.
    """
    try:
        manifest_path = resolve_quizbench_manifest_path(generator_run_root)
    except SystemExit:
        return {}

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}

    quizzes = payload.get("quizzes") or []
    if not isinstance(quizzes, list):
        return {}

    out: Dict[str, Path] = {}
    for row in quizzes:
        if not isinstance(row, dict):
            continue
        quiz_id = str(row.get("quiz_id") or "").strip()
        quiz_path = str(row.get("quiz_path") or "").strip()
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


def _resolve_quizzes_dir_for_fallback(generator_run_root: Path) -> Path | None:
    """
    Best-effort resolve a quizzes directory for locating <quiz_id>.jsonl files.

    Uses the path segment `.../quizzes_<TAG>/runs/...` when present, otherwise
    falls back to the repo-root `quizzes/` directory.
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


def _load_qid_to_topic_for_run(
    *,
    run_dir: Path,
    topics_eval_model: str | None,
    prefer_topic_mapping_outputs: bool,
    quiz_paths_from_manifest: Mapping[str, Path],
    quiz_index: Mapping[str, Path],
    canonical_categories: Sequence[str],
    canonical_index: Mapping[str, str],
    mapping_config: TopicMappingConfig | None,
    mapping_mode: str,
    unmapped_policy: str,
    fallback_to_quiz_target_topics: bool = True,
) -> tuple[Dict[str, str], str | None]:
    """
    Load per-question mapped topic labels for a run directory.

    Returns (qid_to_topic, error_reason). When error_reason is None,
    qid_to_topic is non-empty.
    """
    candidates = sorted(run_dir.glob("topics_*.json"))
    if candidates:
        topics_path, _, err = _select_topics_file(
            run_dir=run_dir,
            candidates=candidates,
            topics_eval_model=topics_eval_model,
            prefer_topic_mapping_outputs=prefer_topic_mapping_outputs,
        )
        if err or topics_path is None:
            return {}, err or "unknown_error"

        try:
            payload = json.loads(topics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}, f"failed_to_parse_topics_json: {topics_path.name}"
        if not isinstance(payload, dict):
            return {}, f"failed_to_parse_topics_json: {topics_path.name}"

        per_question = payload.get("per_question") or []
        if not isinstance(per_question, list):
            return {}, "no_topics_rows"

        qid_to_topic: Dict[str, str] = {}
        for row in per_question:
            if not isinstance(row, dict):
                continue
            qid = str(row.get("question_id") or "").strip()
            if not qid:
                continue
            raw_topic = row.get("topic_mapped")
            if raw_topic is None:
                raw_topic = row.get("topic")

            mapped = map_topic_label(
                str(raw_topic) if raw_topic is not None else None,
                canonical_categories=canonical_categories,
                canonical_index=canonical_index,
                config=mapping_config,
                mode=mapping_mode,
                unmapped_policy=unmapped_policy,
            )
            if mapped is None:
                continue
            qid_to_topic[qid] = str(mapped)

        if not qid_to_topic:
            return {}, "no_topics_rows"
        return qid_to_topic, None

    if not fallback_to_quiz_target_topics:
        return {}, "no_topics_files"

    quiz_id = run_dir.name
    quiz_path = quiz_paths_from_manifest.get(quiz_id) or quiz_index.get(quiz_id)
    if quiz_path is None or not quiz_path.exists():
        return {}, "no_topics_files_and_missing_quiz_jsonl"

    qid_to_topic = {}
    try:
        with quiz_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
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
                    config=mapping_config,
                    mode=mapping_mode,
                    unmapped_policy=unmapped_policy,
                )
                if mapped is None:
                    continue
                qid_to_topic[qid] = str(mapped)
    except Exception:  # noqa: BLE001
        return {}, "failed_to_read_quiz_jsonl"

    if not qid_to_topic:
        return {}, "no_topics_found_in_quiz_jsonl"

    return qid_to_topic, None


def _print_zero_valid_target_diagnostics(
    *,
    generator_run_root: Path,
    target_categories: Sequence[str],
    valid_counts_by_topic: Mapping[str, int],
    runs_included: Sequence[str],
    runs_skipped: Mapping[str, str],
    judge_models: Sequence[str],
    topics_eval_model: str | None,
    logical_mode: str | None,
    min_medical_score: int | None,
    allow_missing_judges: bool,
    topic_map_path: Path | None,
    mapping_mode: str,
    unmapped_policy: str,
    prefer_topic_mapping_outputs: bool,
) -> None:
    zero_targets = [
        cat for cat in target_categories if int(valid_counts_by_topic.get(cat, 0) or 0) <= 0
    ]
    if not zero_targets:
        return

    included_set = {str(p) for p in runs_included if str(p).strip()}
    skipped_map = {str(k): str(v) for k, v in (runs_skipped or {}).items() if str(k).strip()}

    mapping_config = load_topic_mapping(topic_map_path) if topic_map_path else None
    cfg = mapping_config or TopicMappingConfig()
    canonical_categories = list(target_categories)
    canonical_index = build_canonical_index(canonical_categories, normalize=cfg.normalize)

    quiz_paths_from_manifest = _load_quiz_paths_from_manifest(generator_run_root)
    quizzes_dir = _resolve_quizzes_dir_for_fallback(generator_run_root)
    quiz_index = _build_quiz_jsonl_index(quizzes_dir) if quizzes_dir is not None else {}

    zero_set = set(zero_targets)
    per_cat: Dict[str, Dict[str, Any]] = {}
    for cat in zero_targets:
        per_cat[cat] = {
            "included_total": 0,
            "included_valid": 0,
            "included_invalid": 0,
            "included_invalid_reasons": Counter(),
            "majority_fail_pass_total": Counter(),
            "majority_fail_verdicts": Counter(),
            "no_eligible_ineligible": Counter(),
            "skipped_total": 0,
            "skipped_by_reason": Counter(),
        }

    run_dirs = [p for p in sorted(generator_run_root.iterdir()) if p.is_dir()]
    for run_dir in run_dirs:
        qid_to_topic, _ = _load_qid_to_topic_for_run(
            run_dir=run_dir,
            topics_eval_model=topics_eval_model,
            prefer_topic_mapping_outputs=prefer_topic_mapping_outputs,
            quiz_paths_from_manifest=quiz_paths_from_manifest,
            quiz_index=quiz_index,
            canonical_categories=canonical_categories,
            canonical_index=canonical_index,
            mapping_config=mapping_config,
            mapping_mode=mapping_mode,
            unmapped_policy=unmapped_policy,
        )
        if not qid_to_topic:
            continue

        qids_by_cat: Dict[str, List[str]] = defaultdict(list)
        for qid, topic in qid_to_topic.items():
            if topic in zero_set:
                qids_by_cat[str(topic)].append(str(qid))

        if not qids_by_cat:
            continue

        run_key = str(run_dir)
        if run_key in included_set:
            run_status = "included"
            skip_reason = None
        else:
            run_status = "skipped"
            skip_reason = skipped_map.get(run_key) or "untracked_run"

        if run_status != "included":
            for cat, qids in qids_by_cat.items():
                slot = per_cat.get(cat)
                if slot is None:
                    continue
                slot["skipped_total"] += len(qids)
                slot["skipped_by_reason"][skip_reason] += len(qids)
            continue

        qids_interest: Set[str] = set()
        for qids in qids_by_cat.values():
            qids_interest.update(qids)

        # Load judge rows only for qids we care about.
        rows_by_judge: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        any_judge = False
        for judge_model in judge_models:
            result_path = run_dir / f"{judge_model}_judge_result.json"
            if not result_path.exists():
                continue
            try:
                rows = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rows, list):
                continue
            any_judge = True
            keep: Dict[str, Mapping[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                qid = row.get("question_id")
                if not qid:
                    continue
                qid_str = str(qid)
                if qid_str in qids_interest:
                    keep[qid_str] = row
            rows_by_judge[str(judge_model)] = keep

        for cat, qids in qids_by_cat.items():
            slot = per_cat.get(cat)
            if slot is None:
                continue

            for qid in qids:
                slot["included_total"] += 1

                if not any_judge:
                    if allow_missing_judges:
                        slot["included_valid"] += 1
                    else:
                        slot["included_invalid"] += 1
                        slot["included_invalid_reasons"]["no_judge_results_found"] += 1
                    continue

                eligible = 0
                passed = 0
                verdict_counts: Counter[str] = Counter()
                ineligible_counts: Counter[str] = Counter()

                for judge_model in judge_models:
                    row = (rows_by_judge.get(str(judge_model)) or {}).get(qid)
                    if row is None:
                        ineligible_counts["missing_result"] += 1
                        continue

                    if not row.get("judge_output_valid"):
                        ineligible_counts["invalid_output"] += 1
                        continue

                    score = row.get("judge_medical_accuracy_score")
                    verdict = row.get("judge_verdict")

                    if score is None or verdict is None:
                        judge_json = row.get("judge_json") or {}
                        if score is None:
                            score = judge_json.get("medical_accuracy_score")
                        if verdict is None:
                            verdict = judge_json.get("verdict")

                    score_int: int | None
                    try:
                        score_int = int(score) if score is not None else None
                    except (TypeError, ValueError):
                        score_int = None

                    if min_medical_score is not None:
                        if score_int is None or score_int < min_medical_score:
                            ineligible_counts["below_min_medical_score"] += 1
                            continue

                    eligible += 1
                    verdict_str = verdict.strip().upper() if isinstance(verdict, str) else ""
                    verdict_key = verdict_str or "MISSING"
                    verdict_counts[verdict_key] += 1
                    if verdict_str == "PASS":
                        passed += 1

                is_valid = eligible > 0 and passed > eligible / 2.0
                if is_valid:
                    slot["included_valid"] += 1
                    continue

                slot["included_invalid"] += 1
                if eligible <= 0:
                    slot["included_invalid_reasons"]["no_eligible_judges"] += 1
                    slot["no_eligible_ineligible"].update(ineligible_counts)
                else:
                    slot["included_invalid_reasons"]["majority_fail"] += 1
                    slot["majority_fail_pass_total"][f"{passed}/{eligible}"] += 1
                    slot["majority_fail_verdicts"].update(verdict_counts)

    print("  [INFO] Zero-valid target category diagnostics:")
    for cat in zero_targets:
        slot = per_cat.get(cat) or {}
        included_total = int(slot.get("included_total") or 0)
        included_valid = int(slot.get("included_valid") or 0)
        included_invalid = int(slot.get("included_invalid") or 0)
        skipped_total = int(slot.get("skipped_total") or 0)

        parts: List[str] = []
        if included_total <= 0:
            parts.append("included=0")
        else:
            reasons: Counter[str] = slot.get("included_invalid_reasons") or Counter()
            reason_bits = []
            if reasons:
                for key in sorted(reasons.keys()):
                    reason_bits.append(f"{key}={int(reasons[key])}")
            reason_text = f" ({', '.join(reason_bits)})" if reason_bits else ""
            parts.append(f"included={included_total} (valid={included_valid}, invalid={included_invalid}){reason_text}")

        if skipped_total > 0:
            skipped_reasons: Counter[str] = slot.get("skipped_by_reason") or Counter()
            top_reasons = skipped_reasons.most_common(2)
            top_text = ", ".join(f"{k}={v}" for k, v in top_reasons if k) if top_reasons else ""
            parts.append(f"skipped={skipped_total}" + (f" ({top_text})" if top_text else ""))

        print(f"    {cat}: " + "; ".join(parts))

        majority_fail = int((slot.get("included_invalid_reasons") or {}).get("majority_fail") or 0)
        if majority_fail > 0:
            pass_total: Counter[str] = slot.get("majority_fail_pass_total") or Counter()
            top = ", ".join(f"{k}×{v}" for k, v in pass_total.most_common(3))
            verdicts: Counter[str] = slot.get("majority_fail_verdicts") or Counter()
            verdict_top = ", ".join(f"{k}={v}" for k, v in verdicts.most_common(4))
            extra_bits = []
            if top:
                extra_bits.append(f"pass/eligible: {top}")
            if verdict_top:
                extra_bits.append(f"eligible verdicts: {verdict_top}")
            if extra_bits:
                print(f"      - majority_fail details: " + "; ".join(extra_bits))

        no_eligible = int((slot.get("included_invalid_reasons") or {}).get("no_eligible_judges") or 0)
        if no_eligible > 0:
            ineligible: Counter[str] = slot.get("no_eligible_ineligible") or Counter()
            top = ", ".join(f"{k}={v}" for k, v in ineligible.most_common(4))
            if top:
                print(f"      - no_eligible_judges details: {top}")


def _compute_questions_dist_payload(
    *,
    generator_run_root: Path,
    generator_model: str,
    targets: "IntegerTargets",
    targets_csv: Path,
    expected_total: int,
    judge_models: List[str],
    topics_eval_model: str | None,
    logical_mode: str | None,
    min_medical_score: int | None,
    allow_missing_judges: bool,
    topic_map_path: Path | None,
    mapping_mode: str,
    unmapped_policy: str,
    prefer_topic_mapping_outputs: bool,
    batch_size: int,
) -> Dict[str, Any]:
    dist = compute_valid_topic_distribution(
        generator_run_root=generator_run_root,
        generator_model=generator_model,
        judge_models=judge_models,
        topics_eval_model=topics_eval_model,
        min_medical_score=min_medical_score,
        logical_mode=logical_mode,
        allow_missing_judges=allow_missing_judges,
        canonical_categories=targets.categories,
        topic_map_path=topic_map_path,
        mapping_mode=mapping_mode,
        unmapped_policy=unmapped_policy,
        prefer_topic_mapping_outputs=prefer_topic_mapping_outputs,
    )

    remaining, overshoot, extra_current = compute_remainders(
        targets=targets,
        current_counts_by_category=dist.valid_counts_by_topic,
    )
    desired = expand_to_desired_list(
        categories_in_order=targets.categories,
        counts_by_category=remaining,
    )
    batches = chunked(desired, batch_size)
    batch_category_counts = [
        counts_for_batch(batch=batch, categories_in_order=targets.categories) for batch in batches
    ]

    remaining_total = int(sum(remaining.values()))
    payload = dist.to_json_dict()
    payload.update(
        {
            "targets_csv": str(targets_csv.expanduser().resolve()),
            "target_total": int(expected_total),
            "targets_categories": list(targets.categories),
            "targets_by_category": dict(targets.targets_by_category),
            "topic_map_path": str(topic_map_path) if topic_map_path else None,
            "topic_map_mode": str(mapping_mode),
            "unmapped_topic_policy": str(unmapped_policy),
            "prefer_topic_mapping_outputs": bool(prefer_topic_mapping_outputs),
            "remaining_by_category": dict(remaining),
            "remaining_total": remaining_total,
            "overshoot_by_category": dict(overshoot),
            "extra_current_by_category": dict(extra_current),
            "desired_categories": list(desired),
            "batch_size": int(batch_size),
            "batches": list(batches),
            "batch_category_counts": list(batch_category_counts),
        }
    )
    return payload


def _load_or_compute_questions_dist_payload(
    *,
    out_path: Path,
    recompute: bool,
    compute_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    def _norm_abs_path(value: object) -> str | None:
        if value is None:
            return None
        try:
            return str(Path(str(value)).expanduser().resolve())
        except Exception:  # noqa: BLE001 - defensive normalization
            return str(value)

    def _dist_cache_mismatch_reasons(payload: Dict[str, Any]) -> List[str]:
        expected_targets_csv = _norm_abs_path(compute_kwargs.get("targets_csv"))
        cached_targets_csv = _norm_abs_path(payload.get("targets_csv"))
        reasons: List[str] = []
        if expected_targets_csv != cached_targets_csv:
            reasons.append(f"targets_csv differs ({cached_targets_csv} != {expected_targets_csv})")

        expected_total = compute_kwargs.get("expected_total")
        cached_total = payload.get("target_total")
        if expected_total is not None:
            if cached_total is None or int(expected_total) != int(cached_total):
                reasons.append(f"target_total differs ({cached_total} != {expected_total})")

        expected_map = _norm_abs_path(compute_kwargs.get("topic_map_path"))
        cached_map = _norm_abs_path(payload.get("topic_map_path"))
        if expected_map != cached_map:
            reasons.append(f"topic_map_path differs ({cached_map} != {expected_map})")

        expected_mode = str(compute_kwargs.get("mapping_mode") or "").strip()
        cached_mode = str(payload.get("topic_map_mode") or "").strip()
        if expected_mode and cached_mode != expected_mode:
            reasons.append(f"topic_map_mode differs ({cached_mode} != {expected_mode})")

        expected_unmapped = str(compute_kwargs.get("unmapped_policy") or "").strip()
        cached_unmapped = str(payload.get("unmapped_topic_policy") or "").strip()
        if expected_unmapped and cached_unmapped != expected_unmapped:
            reasons.append(f"unmapped_topic_policy differs ({cached_unmapped} != {expected_unmapped})")

        expected_prefer = compute_kwargs.get("prefer_topic_mapping_outputs")
        cached_prefer = payload.get("prefer_topic_mapping_outputs")
        if expected_prefer is not None and cached_prefer is not None and bool(expected_prefer) != bool(cached_prefer):
            reasons.append(
                f"prefer_topic_mapping_outputs differs ({bool(cached_prefer)} != {bool(expected_prefer)})"
            )
        return reasons

    if out_path.exists() and not recompute:
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"[FATAL] Failed to parse existing questions_dist file: {out_path} ({exc})") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"[FATAL] Existing questions_dist file is not a JSON object: {out_path}")
        if not (isinstance(payload.get("batches"), list) and isinstance(payload.get("remaining_by_category"), dict)):
            raise SystemExit(
                f"[FATAL] Existing questions_dist file is missing required keys "
                f"(batches/remaining_by_category): {out_path}"
            )

        mismatch = _dist_cache_mismatch_reasons(payload)
        if not mismatch:
            return payload
        print(
            "[WARN] Existing questions_dist file metadata does not match current "
            f"targets/mapping args; recomputing ({'; '.join(mismatch)})."
        )

    payload = _compute_questions_dist_payload(**compute_kwargs)
    ensure_dir(str(out_path.parent))
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main():
    args = _load_cli_args()
    run_ts = datetime.now(timezone.utc)
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    print(f"[INFO] Loaded config from {config_path}")

    generator_models = parse_models_field(
        cfg.get("generator_models")
        or cfg.get("generator_models_csv")
        or cfg.get("generator_model"),
        field_label="generator_models",
        required=True,
    )
    test_models = parse_models_field(
        cfg.get("test_models") or cfg.get("test_models_csv") or cfg.get("test_model"),
        field_label="test_models",
        required=False,
    )
    test_models_csv = ",".join(test_models) if test_models else None

    # Question counts:
    num_questions_per_batch = coerce_int(cfg, "num_questions_per_batch", 10)  # Default to 10
    num_questions_per_quiz = coerce_int(cfg, "num_questions_per_quiz", 50)    # Default to 50

    # Basic validation to avoid invalid configs and confusing behavior.
    if num_questions_per_batch <= 0:
        raise ValueError("num_questions_per_batch must be > 0")
    if num_questions_per_quiz <= 0:
        raise ValueError("num_questions_per_quiz must be > 0")

    planned_batch_multiplier = float(args.planned_batch_request_multiplier or 1.0)
    if not math.isfinite(planned_batch_multiplier) or planned_batch_multiplier < 1.0:
        raise ValueError("--planned_batch_request_multiplier must be a finite float >= 1.0")

    num_choices = coerce_int(cfg, "num_choices", 5) if args.num_choices is None else int(args.num_choices)
    seed0 = coerce_int(cfg, "seed0", 123)
    quizzes_root_dir = str(cfg.get("quizzes_dir", "quizzes/"))
    runs_root = str(cfg.get("runs_root", "eval_uq_results/"))
    topics_csv = args.topics_csv or cfg.get("topics_csv") or DEFAULT_TOPICS
    max_output_tokens = (
        args.max_output_tokens
        if args.max_output_tokens is not None
        else (
            coerce_int(cfg, "max_output_tokens", 16000)
            if "max_output_tokens" in cfg
            else coerce_int(cfg, "max_tokens", 16000)
        )
    )

    mapping_report: Dict[str, Any] | None = None
    if args.topic_mapping_report:
        try:
            mapping_report = _load_topic_mapping_report(args.topic_mapping_report)
        except FileNotFoundError as exc:
            missing_path = Path(args.topic_mapping_report).expanduser().resolve()
            candidates = sorted(ROOT_DIR.glob("topic_mapping_report_*.json"))
            if not candidates:
                raise SystemExit(
                    f"[FATAL] topic_mapping_report not found: {missing_path}. "
                    "Pass an existing topic_mapping_report_*.json (from quizbench/apply_topic_mapping.py) "
                    "or create per-generator shim files with quizbench/split_topic_mapping_report.py."
                ) from exc

            try:
                fallback = max(candidates, key=lambda p: p.stat().st_mtime)
            except Exception:  # noqa: BLE001 - mtime best-effort
                fallback = candidates[-1]

            msg = f"[WARN] topic_mapping_report not found: {missing_path}; falling back to {fallback}."
            if missing_path.name == "accumulated_mapped_topics.json":
                msg += " Create this shim with: python quizbench/split_topic_mapping_report.py --report <topic_mapping_report_*.json>"
            print(msg)

            mapping_report = _load_topic_mapping_report(str(fallback))
            args.topic_mapping_report = str(fallback)

        report_targets_csv = mapping_report.get("targets_csv")
        if isinstance(report_targets_csv, str) and args.targets_csv == "data/ABMS_specialties.csv":
            args.targets_csv = report_targets_csv

        report_topic_map = mapping_report.get("topic_map_path")
        if isinstance(report_topic_map, str) and args.topic_map is None:
            args.topic_map = report_topic_map

        report_mode = str(mapping_report.get("mode") or "").strip().lower()
        if report_mode in {"exact", "normalize", "map"} and args.topic_map_mode == "map":
            args.topic_map_mode = report_mode

        report_unmapped = str(mapping_report.get("unmapped_policy") or "").strip().lower()
        if report_unmapped in {"error", "keep", "misc", "drop"} and args.unmapped_topic_policy == "keep":
            args.unmapped_topic_policy = report_unmapped

        report_runs_root = mapping_report.get("runs_root")
        if isinstance(report_runs_root, str):
            report_root_abs = str(Path(report_runs_root).expanduser().resolve())
            cfg_root_abs = str(Path(runs_root).expanduser().resolve())
            if report_root_abs != cfg_root_abs:
                print(f"[WARN] topic_mapping_report runs_root={report_root_abs} != config runs_root={cfg_root_abs}")

    if args.prefer_topic_mapping_outputs is None:
        args.prefer_topic_mapping_outputs = bool(args.topic_mapping_report)

    if args.use_target_batches:
        print(
            "[INFO] Target batches enabled "
            f"(targets_csv={args.targets_csv}, topic_map={args.topic_map or 'none'}, "
            f"topic_map_mode={args.topic_map_mode}, unmapped_topic_policy={args.unmapped_topic_policy})."
        )
    else:
        print(f"[ERROR] Flat generation mode using topics_csv={topics_csv!r} currently disabled.")
        sys.exit()

    # Allow configs (e.g., DeepSeek) to opt out of true Batch API usage and
    # instead use a sequential "pseudo-batch" path driven by the same loop.
    use_batch_api = bool(cfg.get("use_batch_api", True))
    use_openrouter = bool(cfg.get("use_openrouter", False))

    env_overrides = collect_env(cfg)
    if env_overrides:
        os.environ.update(env_overrides)

    quiz_collection = _resolve_quiz_collection(
        args.quiz_collection
        if args.quiz_collection is not None
        else cfg.get("quiz_collection") or cfg.get("quizzes_collection"),
        run_ts=run_ts,
    )
    quizzes_dir = (
        str(Path(quizzes_root_dir) / quiz_collection)
        if quiz_collection
        else quizzes_root_dir
    )

    if args.plan_targets_only:
        targets_csv_path = Path(args.targets_csv)
        targets = load_integer_targets_csv(targets_csv_path)
        expected_total = int(args.target_total) if args.target_total is not None else int(targets.total)
        _validate_targets_total(targets.total, expected_total)

        judge_models = _parse_yaml_list(args.judge_models_yaml) or list(DEFAULT_ENSEMBLE_JUDGES)
        if not judge_models and not args.allow_missing_judges:
            raise SystemExit(
                "[FATAL] No judge models provided. Use --judge_models_yaml or set --allow_missing_judges."
            )

        for gen_model in generator_models:
            gen_runs_root = _resolve_generator_run_root(Path(runs_root), gen_model)
            topic_map_path = Path(args.topic_map).expanduser().resolve() if args.topic_map else None
            out_dir = Path(args.dist_out_dir).expanduser().resolve() if args.dist_out_dir else gen_runs_root
            out_path = out_dir / f"questions_dist_{sanitize_model_name(gen_model)}.json"

            payload = _load_or_compute_questions_dist_payload(
                out_path=out_path,
                recompute=True,
                compute_kwargs={
                    "generator_run_root": gen_runs_root,
                    "generator_model": gen_model,
                    "targets": targets,
                    "targets_csv": targets_csv_path,
                    "expected_total": expected_total,
                    "judge_models": judge_models,
                    "topics_eval_model": args.topics_eval_model,
                    "logical_mode": args.logical_mode,
                    "min_medical_score": args.min_medical_score,
                    "allow_missing_judges": args.allow_missing_judges,
                    "topic_map_path": topic_map_path,
                    "mapping_mode": args.topic_map_mode,
                    "unmapped_policy": args.unmapped_topic_policy,
                    "prefer_topic_mapping_outputs": bool(args.prefer_topic_mapping_outputs),
                    "batch_size": int(num_questions_per_batch),
                },
            )

            remaining_total = int(payload.get("remaining_total") or 0)
            print(
                f"[INFO] {gen_model}: valid_total={payload.get('valid_total')}, "
                f"remaining_total={remaining_total}, planned_batches={len(payload.get('batches') or [])}"
            )
            _print_current_topic_distribution(
                valid_total=int(payload.get("valid_total") or 0),
                valid_counts_by_topic=_normalize_counts_by_topic(payload.get("valid_counts_by_topic")),
                target_categories=list(targets.categories),
            )
            _print_zero_valid_target_diagnostics(
                generator_run_root=gen_runs_root,
                target_categories=list(targets.categories),
                valid_counts_by_topic=_normalize_counts_by_topic(payload.get("valid_counts_by_topic")),
                runs_included=payload.get("runs_included") or [],
                runs_skipped=payload.get("runs_skipped") or {},
                judge_models=judge_models,
                topics_eval_model=args.topics_eval_model,
                logical_mode=args.logical_mode,
                min_medical_score=args.min_medical_score,
                allow_missing_judges=args.allow_missing_judges,
                topic_map_path=topic_map_path,
                mapping_mode=args.topic_map_mode,
                unmapped_policy=args.unmapped_topic_policy,
                prefer_topic_mapping_outputs=bool(args.prefer_topic_mapping_outputs),
            )
            _print_target_deltas(
                targets_by_category=targets.targets_by_category,
                valid_counts_by_topic=_normalize_counts_by_topic(payload.get("valid_counts_by_topic")),
                categories_in_order=list(targets.categories),
            )
            overshoot = payload.get("overshoot_by_category") or {}
            extra_current = payload.get("extra_current_by_category") or {}
            if overshoot:
                print(f"  [INFO] Overshoot vs targets: {overshoot}")
            if extra_current:
                print(f"  [INFO] Current topics not in targets: {extra_current}")

            batch_category_counts = payload.get("batch_category_counts") or []
            for b_idx, counts in enumerate(batch_category_counts, start=1):
                print(f"  [BATCH {b_idx:02d}] {counts} (quizzes_dir={quizzes_dir})")

            print(f"[OK] Wrote {out_path}")

        print("[INFO] plan_targets_only complete; exiting before generation.")
        return

    provider_by_model = {
        model: _provider_for_generator(model, use_batch_api=use_batch_api) for model in generator_models
    }
    provider_values = set(provider_by_model.values())
    manifest_api_provider = provider_values.pop() if len(provider_values) == 1 else "mixed"

    lock_path = Path(quizzes_dir) / ".quizbench_readonly"
    if lock_path.exists() and not args.allow_locked_quiz_collection:
        raise SystemExit(
            f"[FATAL] Quizzes directory is locked (found {lock_path}). "
            "Choose a different --quiz_collection (or set quiz_collection in config), "
            "or delete the lock file to proceed (or pass --allow_locked_quiz_collection)."
        )

    batch_input_dir = args.batch_input_dir or cfg.get("batch_input_dir")
    if not batch_input_dir:
        batch_input_dir = str(Path(quizzes_dir) / "batch_inputs")

    ensure_dir(quizzes_dir)
    ensure_dir(runs_root)
    ensure_dir(batch_input_dir)

    targets: IntegerTargets | None = None
    expected_total: int | None = None
    judge_models: List[str] | None = None
    targets_csv_path: Path | None = None
    topic_map_path: Path | None = None

    if args.use_target_batches:
        targets_csv_path = Path(args.targets_csv)
        targets = load_integer_targets_csv(targets_csv_path)
        expected_total = int(args.target_total) if args.target_total is not None else int(targets.total)
        _validate_targets_total(targets.total, expected_total)

        judge_models = _parse_yaml_list(args.judge_models_yaml) or list(DEFAULT_ENSEMBLE_JUDGES)
        if not judge_models and not args.allow_missing_judges:
            raise SystemExit(
                "[FATAL] No judge models provided. Use --judge_models_yaml or set --allow_missing_judges."
            )

        topic_map_path = Path(args.topic_map).expanduser().resolve() if args.topic_map else None

    # Theoretical minimum number of batches if each batch yields the full request.
    if args.use_target_batches:
        total_for_theory = int(expected_total or 0)
    else:
        total_for_theory = int(num_questions_per_quiz)
    num_quizzes_per_generator = (total_for_theory + num_questions_per_batch - 1) // num_questions_per_batch

    manifest: Dict[str, Any] = {
        "created_at": now_utc_iso(),
        "config_path": str(config_path),
        "generator_models": generator_models,
        "generator_model": generator_models[0],
        "api_providers_by_model": provider_by_model,
        "test_models": test_models,
        "test_models_csv": test_models_csv,
        "num_quizzes_per_generator": num_quizzes_per_generator,  # theoretical minimum
        "num_questions_per_batch": num_questions_per_batch,
        "num_questions_per_quiz": num_questions_per_quiz,
        "runs_root": runs_root,
        "api_provider": manifest_api_provider,
        "quizzes_root_dir": quizzes_root_dir,
        "quiz_collection": quiz_collection,
        "quizzes_dir": quizzes_dir,
        "target_planning": {
            "enabled": bool(args.use_target_batches),
            "topic_mapping_report": (
                str(Path(args.topic_mapping_report).expanduser().resolve()) if args.topic_mapping_report else None
            ),
            "planned_batch_request_multiplier": planned_batch_multiplier,
            "targets_csv": str(Path(args.targets_csv).expanduser().resolve()),
            "target_total": int(expected_total) if expected_total is not None else None,
            "topic_map_path": str(topic_map_path) if topic_map_path else None,
            "topic_map_mode": args.topic_map_mode,
            "unmapped_topic_policy": args.unmapped_topic_policy,
            "topics_eval_model": args.topics_eval_model,
            "prefer_topic_mapping_outputs": bool(args.prefer_topic_mapping_outputs),
            "judge_models_yaml": str(Path(args.judge_models_yaml).expanduser().resolve()),
            "loigcal_mode": args.logical_mode,
            "min_medical_score": args.min_medical_score,
            "allow_missing_judges": bool(args.allow_missing_judges),
            "questions_dist_files": {},
        },
        "quizzes": [],
        "run_ids": [],
        "batch_jobs": [],
    }

    quizzes_info: List[Dict[str, Any]] = []

    # Import generation transports lazily so --plan_targets_only can run without
    # OpenAI/Batch SDK dependencies installed.
    try:
        from quizbench.batch_generate_quiz import generate_quizzes_via_batch  # noqa: WPS433
        from quizbench.utils_deepseek import generate_quizzes_via_sequential  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[FATAL] Missing generation dependencies. Install repo requirements "
            "(and QuizBench requirements) before running generation."
        ) from exc

    anthropic_batch_fn = None

    def _run_generation_for_requests(
        provider: str,
        gen_model: str,
        requests,
        *,
        batch_input_path: str,
    ):
        nonlocal anthropic_batch_fn

        if provider == _PROVIDER_ANTHROPIC and use_batch_api:
            if anthropic_batch_fn is None:
                try:
                    from quizbench.anthropic_message_batches import generate_quizzes_via_anthropic_batch  # noqa: WPS433
                except ModuleNotFoundError as exc:  # pragma: no cover - import guard
                    raise SystemExit(
                        "[FATAL] Missing Anthropic dependencies. Install repo requirements "
                        "before using claude-* generator models."
                    ) from exc

                anthropic_batch_fn = generate_quizzes_via_anthropic_batch
            return anthropic_batch_fn(
                gen_model,
                requests,
                batch_input_path=batch_input_path,
                poll_interval=args.poll_seconds,
                temperature=args.temperature,
                max_output_tokens=max_output_tokens,
                overwrite=bool(args.overwrite),
            )

        if use_batch_api:
            return generate_quizzes_via_batch(
                gen_model,
                requests,
                batch_input_path=batch_input_path,
                poll_interval=args.poll_seconds,
                temperature=args.temperature,
                max_output_tokens=max_output_tokens,
                overwrite=bool(args.overwrite),
            )

        return generate_quizzes_via_sequential(
            gen_model,
            requests,
            temperature=args.temperature,
            openrouter=use_openrouter,
            max_output_tokens=max_output_tokens,
            overwrite=bool(args.overwrite),
        )

    for g_idx, gen_model in enumerate(generator_models):
        print(f"\n-- Generator model: {gen_model} --")

        safe_model = sanitize_model_name(gen_model)
        provider = provider_by_model.get(gen_model, _provider_for_generator(gen_model, use_batch_api=use_batch_api))
        print(f"  [INFO] Using provider: {provider}")
        # Offset seeds per generator to match run_gen_quiz deterministic scheme,
        # then add a per-quiz offset inside the while loop.
        base_seed_for_model = seed0 + g_idx

        total_questions_for_model = 0
        quiz_idx_for_model = 0

        if args.avoid_existing_seeds:
            existing_seeds = _discover_existing_seeds_for_model(
                generator_model=gen_model,
                quizzes_dir=quizzes_dir,
                runs_root=runs_root,
            )
            if existing_seeds:
                quiz_idx_for_model = _compute_starting_quiz_index(
                    base_seed=base_seed_for_model,
                    existing_seeds=existing_seeds,
                )
                if quiz_idx_for_model:
                    start_seed = base_seed_for_model + quiz_idx_for_model
                    print(
                        f"  [INFO] Found {len(existing_seeds)} existing seed(s) for {gen_model}; "
                        f"starting at seed{start_seed}."
                    )

        if args.use_target_batches:
            if targets is None or expected_total is None or judge_models is None or targets_csv_path is None:
                raise SystemExit("[FATAL] Target planning was not initialized; cannot run --use_target_batches.")

            gen_runs_root = _resolve_generator_run_root(Path(runs_root), gen_model)
            out_dir = Path(args.dist_out_dir).expanduser().resolve() if args.dist_out_dir else gen_runs_root
            out_path = out_dir / f"questions_dist_{sanitize_model_name(gen_model)}.json"
            manifest["target_planning"]["questions_dist_files"][gen_model] = str(out_path)

            questions_dist = _load_or_compute_questions_dist_payload(
                out_path=out_path,
                recompute=bool(args.recompute_questions_dist),
                compute_kwargs={
                    "generator_run_root": gen_runs_root,
                    "generator_model": gen_model,
                    "targets": targets,
                    "targets_csv": targets_csv_path,
                    "expected_total": expected_total,
                    "judge_models": judge_models,
                    "topics_eval_model": args.topics_eval_model,
                    "logical_mode": args.logical_mode,
                    "min_medical_score": args.min_medical_score,
                    "allow_missing_judges": args.allow_missing_judges,
                    "topic_map_path": topic_map_path,
                    "mapping_mode": args.topic_map_mode,
                    "unmapped_policy": args.unmapped_topic_policy,
                    "prefer_topic_mapping_outputs": bool(args.prefer_topic_mapping_outputs),
                    "batch_size": int(num_questions_per_batch),
                },
            )

            remaining_total = int(questions_dist.get("remaining_total") or 0)
            planned_batches = questions_dist.get("batches") or []
            planned_questions = sum(len(b) for b in planned_batches if isinstance(b, list))
            if planned_questions != remaining_total:
                print(
                    f"  [WARN] Planned batches cover {planned_questions} questions, but remaining_total={remaining_total}."
                )

            print(
                f"  [INFO] Target batches planned: remaining_total={remaining_total}, "
                f"planned_batches={len(planned_batches)} (batch_size={num_questions_per_batch})."
            )
            if remaining_total <= 0:
                print(f"  [INFO] No remaining target questions for {gen_model}; skipping generation.")
                continue

            stop_early = False
            for b_idx, batch in enumerate(planned_batches, start=1):
                if stop_early:
                    break
                if not isinstance(batch, list):
                    continue
                batch_plan = [str(t).strip() for t in batch if str(t).strip()]
                if not batch_plan:
                    continue

                cursor = 0
                while cursor < len(batch_plan):
                    sub_plan = batch_plan[cursor:]
                    planned_needed = len(sub_plan)
                    if planned_needed <= 0:
                        break

                    zero_valid_attempts = 0
                    while True:
                        request_idx = quiz_idx_for_model
                        seed_for_quiz = base_seed_for_model + request_idx
                        quiz_id = build_quiz_id(gen_model, seed_for_quiz, at=run_ts)

                        requested = planned_needed
                        if planned_batch_multiplier != 1.0:
                            requested = int(math.ceil(planned_needed * planned_batch_multiplier))
                            requested = max(planned_needed, requested)

                        plan_for_request = (
                            _repeat_topic_plan(sub_plan, target_len=requested)
                            if requested != planned_needed
                            else sub_plan
                        )

                        topics_csv_for_request = ",".join(dict.fromkeys(plan_for_request))
                        sub_counts = counts_for_batch(batch=plan_for_request, categories_in_order=targets.categories)

                        requests = _build_requests_for_model(
                            quiz_id=quiz_id,
                            num_questions=requested,
                            num_choices=num_choices,
                            seed=seed_for_quiz,
                            topics_csv=topics_csv_for_request,
                            topic_plan=plan_for_request,
                            topic_counts=sub_counts,
                            quizzes_dir=quizzes_dir,
                        )

                        batch_input_path = str(
                            Path(batch_input_dir)
                            / _batch_input_filename(provider, safe_model, request_idx)
                        )

                        req_detail = f"requesting {requested}"
                        if planned_batch_multiplier != 1.0:
                            req_detail = (
                                f"requesting {requested} (need {planned_needed}; x{planned_batch_multiplier:g})"
                            )
                        print(
                            f"  [INFO] Submitting planned batch {b_idx:02d} (request {request_idx}) "
                            f"for {gen_model} ({req_detail}; progress {total_questions_for_model} / {remaining_total})."
                        )
                        if provider == _PROVIDER_ANTHROPIC and use_batch_api:
                            print("\n=== Generating quizzes via Anthropic Message Batches API ===")
                        elif use_batch_api:
                            print("\n=== Generating quizzes via Batch API ===")
                        else:
                            print("\n=== Generating quizzes via sequential LLM calls (no Batch API) ===")

                        results, batch_id = _run_generation_for_requests(
                            provider,
                            gen_model,
                            requests,
                            batch_input_path=batch_input_path,
                        )

                        if batch_id:
                            manifest["batch_jobs"].append(
                                {
                                    "generator_model": gen_model,
                                    "batch_id": batch_id,
                                    "batch_input_path": batch_input_path,
                                    "num_requests": len(requests),
                                    "provider": provider,
                                }
                            )

                        batch_questions = 0
                        for res in results:
                            if res.status != "ok" or not res.quiz_path:
                                print(f"  [ERR] {res.quiz_id}: {res.error}")
                                continue

                            n_valid = _count_questions_in_quiz(res.quiz_path)
                            batch_questions += n_valid

                            target_topics = _read_target_topics_from_quiz(res.quiz_path)
                            quizzes_info.append(
                                {
                                    "quiz_id": res.quiz_id,
                                    "quiz_path": res.quiz_path,
                                    "generator_model": gen_model,
                                    "seed": res.seed,
                                    "target_topic": target_topics,
                                }
                            )
                            print(
                                f"  [OK] Generated quiz {res.quiz_id} at {res.quiz_path} "
                                f"({n_valid} valid questions returned)."
                            )

                        usable_from_batch = min(batch_questions, planned_needed)
                        total_questions_for_model += usable_from_batch
                        cursor += usable_from_batch

                        print(
                            f"  [INFO] Planned batch {b_idx:02d} produced {batch_questions} valid; "
                            f"used {usable_from_batch}. Progress: {total_questions_for_model} / {remaining_total}"
                        )

                        if batch_questions == 0:
                            zero_valid_attempts += 1
                            if zero_valid_attempts > MAX_ZERO_VALID_RETRIES:
                                print(
                                    f"  [WARN] Planned batch {b_idx:02d} (request {request_idx}) "
                                    f"for {gen_model} produced 0 valid questions after "
                                    f"{MAX_ZERO_VALID_RETRIES} retries; stopping early."
                                )
                                stop_early = True
                                break
                            print(
                                f"  [WARN] Planned batch {b_idx:02d} (request {request_idx}) "
                                f"for {gen_model} produced 0 valid questions; "
                                f"retrying ({zero_valid_attempts}/{MAX_ZERO_VALID_RETRIES})."
                            )
                            continue

                        quiz_idx_for_model += 1
                        break

                    if stop_early:
                        break

            print(
                f"  [INFO] Finished generator {gen_model}: "
                f"{total_questions_for_model} valid questions generated "
                f"(planned target was {remaining_total})."
            )
            continue

        while total_questions_for_model < num_questions_per_quiz:
            remaining_needed = num_questions_per_quiz - total_questions_for_model
            to_request = min(num_questions_per_batch, remaining_needed)
            if to_request <= 0:
                break

            seed_for_quiz = base_seed_for_model + quiz_idx_for_model
            quiz_id = build_quiz_id(gen_model, seed_for_quiz, at=run_ts)

            requests = _build_requests_for_model(
                quiz_id=quiz_id,
                num_questions=to_request,
                num_choices=num_choices,
                seed=seed_for_quiz,
                topics_csv=topics_csv,
                quizzes_dir=quizzes_dir,
            )

            batch_input_path = str(
                Path(batch_input_dir) / _batch_input_filename(provider, safe_model, quiz_idx_for_model)
            )

            print(
                f"  [INFO] Submitting batch {quiz_idx_for_model} for {gen_model} "
                f"(current total {total_questions_for_model}, remaining {remaining_needed}; "
                f"requesting up to {to_request})."
            )
            if provider == _PROVIDER_ANTHROPIC and use_batch_api:
                print("\n=== Generating quizzes via Anthropic Message Batches API ===")
            elif use_batch_api:
                print("\n=== Generating quizzes via Batch API ===")
            else:
                print("\n=== Generating quizzes via sequential LLM calls (no Batch API) ===")

            results, batch_id = _run_generation_for_requests(
                provider,
                gen_model,
                requests,
                batch_input_path=batch_input_path,
            )

            if batch_id:
                manifest["batch_jobs"].append(
                    {
                        "generator_model": gen_model,
                        "batch_id": batch_id,
                        "batch_input_path": batch_input_path,
                        "num_requests": len(requests),
                        "provider": provider,
                    }
                )

            batch_questions = 0

            for res in results:
                if res.status != "ok" or not res.quiz_path:
                    print(f"  [ERR] {res.quiz_id}: {res.error}")
                    continue

                n_valid = _count_questions_in_quiz(res.quiz_path)
                batch_questions += n_valid

                target_topics = _read_target_topics_from_quiz(res.quiz_path)
                quizzes_info.append(
                    {
                        "quiz_id": res.quiz_id,
                        "quiz_path": res.quiz_path,
                        "generator_model": gen_model,
                        "seed": res.seed,
                        "target_topic": target_topics,
                    }
                )
                print(
                    f"  [OK] Generated quiz {res.quiz_id} at {res.quiz_path} "
                    f"({n_valid} valid questions returned)."
                )

            usable_from_batch = min(batch_questions, num_questions_per_quiz - total_questions_for_model)
            total_questions_for_model += usable_from_batch

            print(
                f"  [INFO] Batch produced {batch_questions} valid; "
                f"used {usable_from_batch}. "
                f"Progress: {total_questions_for_model} / {num_questions_per_quiz}"
            )

            if batch_questions == 0:
                print(
                    f"  [WARN] Batch {quiz_idx_for_model} for {gen_model} produced 0 valid questions; "
                    "stopping early to avoid an infinite loop."
                )
                break

            quiz_idx_for_model += 1

        print(
            f"  [INFO] Finished generator {gen_model}: "
            f"{total_questions_for_model} valid questions generated "
            f"(target was {num_questions_per_quiz})."
        )

    manifest["quizzes"] = quizzes_info

    canonical_manifest_path = Path(runs_root).expanduser().resolve() / "quizbench_manifest.json"

    batch_tag = str(args.quiz_batch_tag).strip() if args.quiz_batch_tag else None
    if not batch_tag:
        batch_tag = _extract_quiz_batch_tag_from_runs_root(runs_root)
    if batch_tag:
        safe_tag = sanitize_model_name(batch_tag)
        manifest_basename = f"quizbench_manifest_{safe_tag}.json"
    else:
        manifest_basename = f"quizbench_manifest_{compact_utc_ts(run_ts)}.json"

    manifest_path = _unique_manifest_path(runs_root, manifest_basename, run_ts=run_ts)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    if canonical_manifest_path.exists():
        print(f"[INFO] Left existing manifest untouched: {canonical_manifest_path}")
    if args.lock_quiz_collection:
        lock_payload = {
            "locked_at": now_utc_iso(),
            "config_path": str(config_path),
            "runs_root": runs_root,
            "quizzes_dir": quizzes_dir,
            "quiz_collection": quiz_collection,
        }
        lock_path.write_text(
            json.dumps(lock_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Locked quiz collection: {lock_path}")
    print(runs_root)


if __name__ == "__main__":
    main()
