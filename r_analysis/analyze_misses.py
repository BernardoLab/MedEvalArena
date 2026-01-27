#!/usr/bin/env python3
"""
Analyze QuizBench misses across runs.

This script scans QuizBench run directories (containing `*_summary.json` and
`*_result.json` files), aggregates which questions were missed by each answer
model, and summarizes the topic distribution of missed questions per quiz
generator (and overall).

Topic labels are sourced from `topics_*.json` files written by
`quizbench/categorize_quiz_topics.py` (optionally using `topic_mapped` when
present). Miss aggregation can optionally be restricted to questions that pass
LLM-as-judge validity filters, mirroring `quizbench/aggregate_results.py`.

For ABMS subset tags (i.e., when --quiz_batch_tag contains "ABMS"), topic labels
are sourced from per-question fields embedded in `*_result.json` (e.g.,
`abms_specialty` / `target_topic`), since these subset runs may not include
`topics_*.json` files.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES, filter_by_judge  # noqa: E402


def _select_manifest_paths_for_batch_tag(runs_root: str, quiz_batch_tag: str) -> list[str]:
    tag = (quiz_batch_tag or "").strip()
    if not tag:
        return []
    pattern = f"quizbench_manifest_{tag}*.json"
    candidates = glob.glob(os.path.join(runs_root, "**", pattern), recursive=True)
    return sorted(set(candidates))


def _select_legacy_manifest_paths(runs_root: str) -> list[str]:
    candidates: list[str] = []

    root_manifest = os.path.join(runs_root, "quizbench_manifest.json")
    if os.path.exists(root_manifest):
        candidates.append(root_manifest)

    candidates.extend(glob.glob(os.path.join(runs_root, "*", "quizbench_manifest.json")))
    return sorted(set(candidates))


def _load_quiz_ids_from_manifest_paths(manifest_paths: list[str]) -> set[str]:
    quiz_ids: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue

        quizzes = manifest.get("quizzes") or []
        if not isinstance(quizzes, list):
            continue
        for q in quizzes:
            if not isinstance(q, dict):
                continue
            quiz_id = q.get("quiz_id")
            if isinstance(quiz_id, str) and quiz_id:
                quiz_ids.add(quiz_id)
    return quiz_ids


def load_quizbench_manifest_generators(
    runs_root: str, *, quiz_batch_tag: str | None = None
) -> dict[str, str]:
    """
    Return mapping quiz_id -> generator_model from any quizbench_manifest*.json under runs_root.

    When quiz_batch_tag is provided, treats it as a prefix and includes the union of
    all `quizbench_manifest_<TAG>*.json` discovered under runs_root, plus any legacy
    `quizbench_manifest.json` files.
    """
    if quiz_batch_tag:
        manifest_paths = _select_manifest_paths_for_batch_tag(runs_root, quiz_batch_tag)
        manifest_paths.extend(_select_legacy_manifest_paths(runs_root))
        manifest_paths = sorted(set(manifest_paths))
    else:
        manifest_paths = glob.glob(
            os.path.join(runs_root, "**", "quizbench_manifest*.json"), recursive=True
        )

    mapping: dict[str, str] = {}
    for manifest_path in manifest_paths:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue

        quizzes = manifest.get("quizzes") or []
        if not isinstance(quizzes, list):
            continue

        for q in quizzes:
            if not isinstance(q, dict):
                continue
            quiz_id = q.get("quiz_id")
            gen = q.get("generator_model") or q.get("generator")
            if not (isinstance(quiz_id, str) and quiz_id and isinstance(gen, str) and gen):
                continue
            existing = mapping.get(quiz_id)
            if existing and existing != gen:
                print(
                    f"[WARN] Conflicting generator for quiz_id={quiz_id!r} "
                    f"across manifests: {existing!r} vs {gen!r}; keeping {existing!r}."
                )
                continue
            mapping[quiz_id] = gen
    return mapping


def infer_generator_from_run_dir(run_dir: str, quiz_generators: dict[str, str]) -> str:
    """
    Infer the generator model string for a given run directory.

    Preference order:
    1) quizbench manifest mapping (quiz_id -> generator_model)
    2) Parse from run_dir basename (quiz_id), which is expected to match:
         <timestamp>_<sanitized_generator_model>_seed<seed>
    """
    base = os.path.basename(run_dir.rstrip(os.sep))

    gen = quiz_generators.get(base)
    if isinstance(gen, str) and gen:
        return gen

    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].startswith("seed"):
        gen_name = "_".join(parts[1:-1])
        if gen_name:
            return gen_name

    return "unknown"


def canonicalize_generator_model(name: str, mode: str = "family") -> str:
    raw = name or "unknown"
    if mode == "exact":
        return raw

    low = raw.lower()
    if low.startswith("gpt-") or low.startswith("openai-") or low == "openai":
        return "openai"
    if low.startswith("claude-") or low.startswith("anthropic") or low == "anthropic":
        return "anthropic"
    if low.startswith("gemini-") or low.startswith("google-") or low == "gemini":
        return "gemini"
    return raw


def find_run_dirs(runs_root: str, run_dir_glob: str) -> list[str]:
    """
    Recursively find run directories (contain *_summary.json) matching the glob.
    """
    matches: list[str] = []
    for dirpath, _, _ in os.walk(runs_root):
        base = os.path.basename(dirpath.rstrip(os.sep))
        if not fnmatch.fnmatch(base, run_dir_glob):
            continue
        if glob.glob(os.path.join(dirpath, "*_summary.json")):
            matches.append(dirpath)
    return sorted(matches)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _select_topics_payload(
    run_dir: Path,
    *,
    topics_eval_model: str | None,
) -> dict[str, Any] | None:
    """
    Choose a topics_*.json payload for a run directory.

    Heuristic:
    - If topics_eval_model is provided, prefer a file whose payload eval_model matches exactly.
    - Otherwise, prefer the file with the most `_mapped` suffixes in the filename.
    """
    candidates = sorted(run_dir.glob("topics_*.json"))
    if not candidates:
        return None

    best_payload: dict[str, Any] | None = None
    best_score: tuple[int, int, str] | None = None

    for path in candidates:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue

        eval_model = payload.get("eval_model")
        score_eval = 0
        if topics_eval_model:
            if isinstance(eval_model, str) and eval_model == topics_eval_model:
                score_eval = 1
            else:
                score_eval = 0

        score_mapped = path.name.count("_mapped")
        score = (score_eval, score_mapped, path.name)

        if best_score is None or score > best_score:
            best_score = score
            best_payload = payload

    if topics_eval_model and best_payload:
        if best_payload.get("eval_model") != topics_eval_model:
            print(
                f"[WARN] No topics file matched --topics_eval_model={topics_eval_model!r} "
                f"under {run_dir}; proceeding with best-effort selection."
            )

    return best_payload


def load_topic_map(
    run_dir: Path,
    *,
    topics_eval_model: str | None,
    topic_field: str,
) -> dict[str, str]:
    payload = _select_topics_payload(run_dir, topics_eval_model=topics_eval_model)
    if not payload:
        return {}

    per_question = payload.get("per_question") or []
    if not isinstance(per_question, list):
        return {}

    field = (topic_field or "auto").lower().strip()
    if field not in {"auto", "raw", "mapped"}:
        raise ValueError(f"Unsupported --topic_field: {topic_field!r}")

    out: dict[str, str] = {}
    for row in per_question:
        if not isinstance(row, dict):
            continue
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            continue

        if field == "raw":
            topic = row.get("topic") or row.get("topic_mapped") or "unknown"
        else:  # auto or mapped
            topic = row.get("topic_mapped") or row.get("topic") or "unknown"

        topic_str = str(topic).strip() if topic is not None else "unknown"
        topic_str = " ".join(topic_str.split())
        topic_str = topic_str.casefold() if topic_str else "unknown"
        out[qid] = topic_str or "unknown"

    return out


def _format_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right_align: set[int] | None = None,
) -> str:
    if not headers:
        return ""
    right_align = right_align or set()
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_cell(idx: int, text: str) -> str:
        return text.rjust(widths[idx]) if idx in right_align else text.ljust(widths[idx])

    header_line = " | ".join(_fmt_cell(i, h) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in widths)
    out_lines = [header_line, sep_line]
    for row in rows:
        out_lines.append(" | ".join(_fmt_cell(i, c) for i, c in enumerate(row)))
    return "\n".join(out_lines)


def _default_out_dir(runs_root: str) -> Path:
    p = Path(runs_root)
    return p.parent if p.name == "runs" else p


def _normalize_topic_label(topic: str) -> str:
    return " ".join(str(topic or "").split()).casefold()


def _abbreviate_topic_label(topic: str) -> str:
    raw = " ".join(str(topic or "").split())
    norm = raw.casefold()
    mapping = {
        # ABMS / canonical specialties
        "anesthesiology": "Anes",
        "critical care medicine": "CCM",
        "dermatology": "Derm",
        "emergency medicine": "EM",
        "endocrine": "Endo",
        "gastrointestinal": "GI",
        "general surgery": "Gen Surg",
        "hematology": "Heme",
        "infectious disease": "ID",
        "medical oncology": "Onc",
        "nephrology": "Nephro",
        "neurology": "Neuro",
        "neurological surgery": "NSG",
        "obstetrics and gynecology": "ObGyn",
        "ophthalmology": "Ophtho",
        "orthopedic surgery": "Ortho",
        "otolaryngology": "HEENT",
        "physical medicine and rehabilitation": "PM&R",
        "plastic surgery": "Plast Surg",
        "psychiatry": "Psych",
        "pulmonary": "Pulm",
        "radiation oncology": "Rad Onc",
        "rheumatology/musculoskeletal": "Rheum",
        "thoracic surgery": "Thor Surg",
        "urology": "Uro",
        "vascular surgery": "Vasc Surg",
        "geriatrics": "Geri",
        "pediatrics": "Peds",
        "cardiovascular": "Cards",
        "allergy/immunology": "All/Imm",
        "immunology": "Immuno",
        # QuizBench default topic set (Jan2026-style)
        "cardiology": "Cards",
        "endocrinology": "Endo",
        "hematology/oncology": "Heme/Onc",
        "obstetrics": "Ob",
        "gynecology": "Gyn",
        "critical care": "CC",
        "unknown": "Unknown",
    }
    if norm in mapping:
        return mapping[norm]
    return raw


def _is_surgical_specialty_topic(topic: str) -> bool:
    norm = _normalize_topic_label(topic)
    if "surgery" in norm:
        return True
    if norm in {"otolaryngology", "urology", "ophthalmology"}:
        return True
    # Include OB/Gyn in the surgical panel by request.
    return ("obstetrics" in norm) or ("gynecology" in norm) or ("obgyn" in norm)


def _is_other_specialty_topic(topic: str) -> bool:
    norm = _normalize_topic_label(topic)
    return norm in {
        "anesthesiology",
        "dermatology",
        "emergency medicine",
        "neurology",
        "radiation oncology",
        "psychiatry",
    }


def _split_topics_for_radar(topics_all: list[str]) -> tuple[list[str], list[str], list[str]]:
    topics_surgery = [t for t in topics_all if _is_surgical_specialty_topic(t)]
    surgery_set = set(topics_surgery)

    topics_other_specialties = [
        t for t in topics_all if t not in surgery_set and _is_other_specialty_topic(t)
    ]
    other_specialty_set = set(topics_other_specialties)

    topics_im_fm = [
        t for t in topics_all if t not in surgery_set and t not in other_specialty_set
    ]
    return topics_surgery, topics_im_fm, topics_other_specialties


def save_topic_miss_rate_radar(
    *,
    out_path: str,
    generator_topic_total: dict[str, dict[str, int]],
    generator_topic_missed: dict[str, dict[str, int]],
    generator_total_questions: dict[str, int],
    overall_topic_total: dict[str, int],
    overall_topic_missed: dict[str, int],
    count_mode: str,
) -> None:
    """
    Save a 3-panel radar plot of topic miss rates.

    Left: Surgical specialties (including Ob/Gyn, ENT, Urology, Ophthalmology).
    Middle: Internal Medicine/Family Medicine (remaining topics).
    Right: Other specialties (Anesthesiology, Dermatology, Emergency Medicine, Neurology, Radiation Oncology, Psychiatry).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import colorcet as cc
    except Exception as exc:
        print(f"[WARN] Could not import plotting deps (matplotlib/seaborn/colorcet): {exc}")
        return

    out_path = str(out_path or "").strip()
    if not out_path:
        return

    topics_all = sorted(t for t, n in overall_topic_total.items() if int(n or 0) > 0)
    if not topics_all:
        print("[WARN] No topic totals available for radar plotting.")
        return

    topics_surgery, topics_im_fm, topics_other_specialties = _split_topics_for_radar(topics_all)

    if not topics_surgery:
        print("[WARN] No surgery/ObGyn topics found; radar panel will be empty.")
    if not topics_im_fm:
        print("[WARN] No Internal Medicine/Family Medicine topics found; radar panel will be empty.")
    if not topics_other_specialties:
        print("[WARN] No 'Other specialties' topics found; radar panel will be empty.")

    generators = sorted([g for g, n in generator_total_questions.items() if int(n or 0) > 0])
    series = ["ALL", *generators]

    # Color palette (prefer a qualitative colorcet colormap).
    palette = None
    try:
        cmap = cc.cm["CET_L20"]
    except Exception:
        cmap = None
    if cmap is not None:
        n = max(1, len(series))
        palette = [cmap(i / max(1, n - 1)) for i in range(n)]
    if not palette:
        palette = getattr(cc, "glasbey", None)
    if not palette:
        palette = sns.color_palette("tab20", n_colors=len(series))

    color_by_series = {name: palette[idx % len(palette)] for idx, name in enumerate(series)}

    def _miss_rate_pct(gen: str, topic: str) -> float:
        if gen == "ALL":
            total = int(overall_topic_total.get(topic, 0) or 0)
            missed = int(overall_topic_missed.get(topic, 0) or 0)
        else:
            total = int(generator_topic_total.get(gen, {}).get(topic, 0) or 0)
            missed = int(generator_topic_missed.get(gen, {}).get(topic, 0) or 0)
        if total <= 0:
            return 0.0
        return (missed / total) * 100.0

    def _radar_angles(n: int) -> list[float]:
        if n <= 0:
            return []
        step = (2.0 * math.pi) / n
        return [i * step for i in range(n)]

    sns.set_theme(
        style="whitegrid",
        context="poster",
        rc={
            "axes.titlesize": 26,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "legend.title_fontsize": 17,
        },
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    ax_surg, ax_im_fm, ax_other = axes

    def _topic_label_fontsize(n_topics: int) -> int:
        if n_topics <= 8:
            return 22
        if n_topics <= 12:
            return 20
        if n_topics <= 18:
            return 18
        if n_topics <= 24:
            return 16
        return 14

    def _bring_text_forward(ax) -> None:
        try:
            for gl in list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines()):
                gl.set_zorder(0)
        except Exception:
            pass
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                t.set_zorder(10)
                t.set_clip_on(False)
            except Exception:
                pass
        try:
            ax.title.set_zorder(10)
            ax.title.set_clip_on(False)
        except Exception:
            pass

    def _plot_panel(ax, topics: list[str], title: str) -> list[Any]:
        title_obj = ax.set_title(title, pad=20, fontsize=20)
        try:
            title_obj.set_zorder(10)
            title_obj.set_clip_on(False)
        except Exception:
            pass
        ax.set_theta_offset(math.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_axisbelow(True)
        ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=18)
        ax.tick_params(axis="x", pad=22)
        ax.tick_params(axis="y", pad=10)

        if not topics:
            ax.set_xticks([])
            ax.set_xticklabels([])
            _bring_text_forward(ax)
            return []

        angles = _radar_angles(len(topics))
        angles_closed = angles + [angles[0]]
        labels = [_abbreviate_topic_label(t) for t in topics]
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=_topic_label_fontsize(len(topics)))

        handles: list[Any] = []
        for name in series:
            values = [_miss_rate_pct(name, t) for t in topics]
            values_closed = values + [values[0]]
            color = color_by_series[name]
            (line,) = ax.plot(
                angles_closed,
                values_closed,
                color=color,
                linewidth=2.2 if name == "ALL" else 1.6,
                marker="o",
                markersize=2.5 if name == "ALL" else 2.0,
                alpha=0.95,
                label=name,
                zorder=2,
            )
            handles.append(line)
        _bring_text_forward(ax)
        return handles

    handles_surg = _plot_panel(ax_surg, topics_surgery, "Surgical specialties")
    handles_im_fm = _plot_panel(ax_im_fm, topics_im_fm, "Int Med/Fam Med")
    handles_other = _plot_panel(ax_other, topics_other_specialties, "Other specialties")

    denom_label = "attempts" if count_mode == "events" else "questions"
    fig.suptitle(f"Topic miss rate (%) by generator ({denom_label})", fontsize=28)

    legend_handles = handles_surg or handles_im_fm or handles_other
    if legend_handles:
        fig.legend(
            legend_handles,
            series,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            title="Generator",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def save_topic_miss_rate_radar_all_only(
    *,
    out_path: str,
    overall_topic_total: dict[str, int],
    overall_topic_missed: dict[str, int],
    count_mode: str,
) -> None:
    """
    Save a 3-panel radar plot that only shows the aggregated "ALL" series.

    The "ALL" line color differs per subplot (using a colorcet colormap).
    Plots topic accuracy (% correct) rather than miss rate.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import colorcet as cc
    except Exception as exc:
        print(f"[WARN] Could not import plotting deps (matplotlib/seaborn/colorcet): {exc}")
        return

    out_path = str(out_path or "").strip()
    if not out_path:
        return

    topics_all = sorted(t for t, n in overall_topic_total.items() if int(n or 0) > 0)
    if not topics_all:
        print("[WARN] No topic totals available for ALL-only radar plotting.")
        return

    topics_surgery, topics_im_fm, topics_other_specialties = _split_topics_for_radar(topics_all)

    denom_label = "attempts" if count_mode == "events" else "questions"

    try:
        cmap = cc.cm["CET_L20"]
    except Exception:
        cmap = None
    if cmap is None:
        print("[WARN] Could not load colorcet colormap CET_L20; skipping ALL-only radar plot.")
        return

    # Avoid extremes for better visibility.
    panel_colors = [cmap((i + 1) / 4.0) for i in range(3)]

    def _correct_rate_pct(topic: str) -> float:
        total = int(overall_topic_total.get(topic, 0) or 0)
        missed = int(overall_topic_missed.get(topic, 0) or 0)
        if total <= 0:
            return 0.0
        correct = total - missed
        return (correct / total) * 100.0

    def _radar_angles(n: int) -> list[float]:
        if n <= 0:
            return []
        step = (2.0 * math.pi) / n
        return [i * step for i in range(n)]

    sns.set_theme(
        style="whitegrid",
        context="poster",
        rc={
            "axes.titlesize": 26,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        },
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    ax_surg, ax_im_fm, ax_other = axes

    def _topic_label_fontsize(n_topics: int) -> int:
        if n_topics <= 8:
            return 22
        if n_topics <= 12:
            return 20
        if n_topics <= 18:
            return 18
        if n_topics <= 24:
            return 16
        return 14

    def _bring_text_forward(ax) -> None:
        try:
            for gl in list(ax.xaxis.get_gridlines()) + list(ax.yaxis.get_gridlines()):
                gl.set_zorder(0)
        except Exception:
            pass
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                t.set_zorder(10)
                t.set_clip_on(False)
            except Exception:
                pass
        try:
            ax.title.set_zorder(10)
            ax.title.set_clip_on(False)
        except Exception:
            pass

    def _plot_panel_single(ax, topics: list[str], title: str, color) -> None:
        title_obj = ax.set_title(title, pad=25, fontsize=25)
        try:
            title_obj.set_zorder(10)
            title_obj.set_clip_on(False)
        except Exception:
            pass
        ax.set_theta_offset(math.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_axisbelow(True)
        ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=14)
        ax.tick_params(axis="x", pad=15)
        ax.tick_params(axis="y", pad=10)

        if not topics:
            ax.set_xticks([])
            ax.set_xticklabels([])
            _bring_text_forward(ax)
            return

        angles = _radar_angles(len(topics))
        angles_closed = angles + [angles[0]]
        values = [_correct_rate_pct(t) for t in topics]
        values_closed = values + [values[0]]
        labels = [_abbreviate_topic_label(t) for t in topics]
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=_topic_label_fontsize(len(topics)))
        ax.plot(
            angles_closed,
            values_closed,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=2.5,
            alpha=0.95,
            zorder=2,
        )
        ax.fill_between(
            angles_closed,
            0,
            values_closed,
            color=color,
            alpha=0.18,
            zorder=1,
        )
        _bring_text_forward(ax)

    _plot_panel_single(ax_surg, topics_surgery, "Surgery", panel_colors[0])
    _plot_panel_single(
        ax_im_fm, topics_im_fm, "Medicine", panel_colors[1]
    )
    _plot_panel_single(ax_other, topics_other_specialties, "Other specialties", panel_colors[2])

    denom_label = "attempts" if count_mode == "events" else "questions"
    fig.suptitle(f"Topic accuracy (%) by specialty ({denom_label})", fontsize=28)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="eval_results/quizbench/runs")
    ap.add_argument(
        "--quiz_batch_tag",
        type=str,
        default=None,
        help=(
            "Optional quiz batch tag (e.g., Jan2026). When set, only include quizzes "
            "listed in any quizbench_manifest_<TAG>*.json discovered under --runs_root "
            "(recursively), plus any legacy quizbench_manifest.json files."
        ),
    )
    ap.add_argument(
        "--run_dir_glob",
        type=str,
        default="*",
        help="Glob pattern (matched against quiz run directory basenames).",
    )
    ap.add_argument(
        "--generator_family_mode",
        choices=["family", "exact"],
        default="exact",
        help="Group generators by provider family (openai/anthropic/gemini) or keep exact names.",
    )
    ap.add_argument(
        "--count_mode",
        choices=["events", "unique_questions"],
        default="events",
        help=(
            "How to count misses when aggregating topic distributions per generator. "
            "'events' counts one miss per (model, question); 'unique_questions' counts "
            "each question at most once per generator if any model missed it."
        ),
    )
    ap.add_argument(
        "--topics_eval_model",
        type=str,
        default=None,
        help="Optional eval_model to select which topics_*.json file to use per quiz.",
    )
    ap.add_argument(
        "--topic_field",
        choices=["auto", "raw", "mapped"],
        default="auto",
        help="Which topic field to use from topics files (default: prefer mapped when present).",
    )
    ap.add_argument(
        "--filter_by_judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-question filtering based on ensemble judge decisions when available.",
    )
    ap.add_argument(
        "--filter_judges",
        nargs="+",
        default=DEFAULT_ENSEMBLE_JUDGES,
        help="Ensemble of judge models to use when filtering questions.",
    )
    ap.add_argument(
        "--filter_min_med_score",
        type=int,
        default=3,
        help="Minimum medical_accuracy_score (1–5) required for a question to be kept.",
    )
    ap.add_argument(
        "--filter_require_logical_valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require judge verdict PASS (logical_validity true) for questions to be kept.",
    )
    ap.add_argument(
        "--filter_logical_mode",
        type=str,
        choices=["all", "majority"],
        default="all",
        help="How to aggregate logical validity across judges ('all' or 'majority').",
    )
    ap.add_argument(
        "--answer_models",
        nargs="+",
        default=None,
        help=(
            "Optional allowlist of answer model names to include. "
            "These should match the *_result.json basenames without the '_result.json' suffix. "
            "If omitted, defaults to generator model names inferred from run manifests "
            "(intersection with available answer models)."
        ),
    )
    ap.add_argument(
        "--out_misses_csv",
        type=str,
        default=None,
        help=(
            "Write a detailed CSV of missed questions (one row per miss). "
            "Default: <runs_root parent>/misses_detail.csv. Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_model_csv",
        type=str,
        default=None,
        help=(
            "Write a per-model miss summary CSV. "
            "Default: <runs_root parent>/misses_by_model.csv. Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_topic_csv",
        type=str,
        default=None,
        help=(
            "Write a per-generator topic miss summary CSV. "
            "Default: <runs_root parent>/misses_by_generator_topic.csv. Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_topic_accuracy_csv",
        type=str,
        default=None,
        help=(
            "Write raw topic accuracy (percent correct) by generator and overall, "
            "counted per (answer_model, question) attempt (pooled accuracy). "
            "Default: <runs_root parent>/topic_accuracy_by_generator_topic.csv. "
            "Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_radar_png",
        type=str,
        default=None,
        help=(
            "Write a 3-panel radar plot PNG for topic miss rates. "
            "Default: <runs_root parent>/misses_topic_miss_rate_radar.png. Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_radar_all_only_png",
        type=str,
        default=None,
        help=(
            "Write a 3-panel radar plot PNG that only shows the aggregated 'ALL' series. "
            "Default: <runs_root parent>/misses_topic_miss_rate_radar_all_only.png. "
            "Set to '' to disable."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    quiz_batch_tag = str(args.quiz_batch_tag).strip() if args.quiz_batch_tag else None
    abms_mode = bool(quiz_batch_tag and "abms" in quiz_batch_tag.casefold())
    tagged_quiz_ids: set[str] = set()
    legacy_quiz_ids: set[str] = set()
    if quiz_batch_tag:
        tagged_manifest_paths = _select_manifest_paths_for_batch_tag(args.runs_root, quiz_batch_tag)
        legacy_manifest_paths = _select_legacy_manifest_paths(args.runs_root)
        tagged_quiz_ids = _load_quiz_ids_from_manifest_paths(tagged_manifest_paths)
        legacy_quiz_ids = _load_quiz_ids_from_manifest_paths(legacy_manifest_paths)

    run_dirs = find_run_dirs(args.runs_root, args.run_dir_glob)
    if not run_dirs and args.run_dir_glob != "*":
        fallback_dirs = find_run_dirs(args.runs_root, "*")
        if fallback_dirs:
            print(
                f"[WARN] No run dirs matched glob '{args.run_dir_glob}'. "
                f"Falling back to '*' ({len(fallback_dirs)} dirs)."
            )
            run_dirs = fallback_dirs

    if not run_dirs:
        raise SystemExit(f"[FATAL] No run directories found under {args.runs_root!r}.")

    quiz_generators = load_quizbench_manifest_generators(args.runs_root, quiz_batch_tag=quiz_batch_tag)
    if quiz_batch_tag:
        if not quiz_generators:
            raise SystemExit(
                f"[FATAL] No quizbench manifests found for batch tag {quiz_batch_tag!r} under {args.runs_root}."
            )
        allowed_quiz_ids = set(quiz_generators.keys())
        run_dirs = [rd for rd in run_dirs if os.path.basename(rd.rstrip(os.sep)) in allowed_quiz_ids]
        if not run_dirs:
            raise SystemExit(
                f"[FATAL] No run directories matched batch tag {quiz_batch_tag!r} under {args.runs_root}."
            )

    out_base = _default_out_dir(args.runs_root)
    out_misses_csv = args.out_misses_csv
    out_model_csv = args.out_model_csv
    out_topic_csv = args.out_topic_csv
    out_topic_accuracy_csv = args.out_topic_accuracy_csv
    out_radar_png = args.out_radar_png
    out_radar_all_only_png = args.out_radar_all_only_png
    if out_misses_csv is None:
        out_misses_csv = str(out_base / "misses_detail.csv")
    if out_model_csv is None:
        out_model_csv = str(out_base / "misses_by_model.csv")
    if out_topic_csv is None:
        out_topic_csv = str(out_base / "misses_by_generator_topic.csv")
    if out_topic_accuracy_csv is None:
        out_topic_accuracy_csv = str(out_base / "topic_accuracy_by_generator_topic.csv")
    if out_radar_png is None:
        out_radar_png = str(out_base / "misses_topic_miss_rate_radar.png")
    if out_radar_all_only_png is None:
        out_radar_all_only_png = str(out_base / "misses_topic_miss_rate_radar_all_only.png")

    answer_model_allowlist: set[str] | None = None
    if args.answer_models:
        answer_model_allowlist = set(args.answer_models)
    else:
        inferred_generators = {
            infer_generator_from_run_dir(rd, quiz_generators) for rd in run_dirs
        }
        inferred_generators = {
            gen for gen in inferred_generators if gen and gen != "unknown"
        }
        if inferred_generators:
            available_models: set[str] = set()
            for rd in run_dirs:
                for result_path in glob.glob(os.path.join(rd, "*_result.json")):
                    base = os.path.basename(result_path)
                    if base.endswith("_judge_result.json"):
                        continue
                    model = (
                        base[: -len("_result.json")] if base.endswith("_result.json") else base
                    )
                    if model:
                        available_models.add(model)

            default_allowlist = inferred_generators & available_models
            if default_allowlist:
                answer_model_allowlist = default_allowlist
                print(
                    "[INFO] Defaulting to generator answer models only "
                    f"({len(answer_model_allowlist)})."
                )
            else:
                print(
                    "[WARN] Could not match generator models to answer models; "
                    "including all answer models."
                )
        else:
            print(
                "[WARN] Could not infer generator models; including all answer models."
            )

    # Per-model aggregates (always event-based).
    model_total: dict[str, int] = defaultdict(int)
    model_missed: dict[str, int] = defaultdict(int)

    # Topic aggregates (depends on count_mode).
    generator_topic_missed: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    generator_total_missed: dict[str, int] = defaultdict(int)

    # Topic denominators for miss rates + enrichment.
    # - In events mode: counts are per (answer_model, question) attempt.
    # - In unique_questions mode: counts are per unique question_id.
    generator_topic_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    generator_total_questions: dict[str, int] = defaultdict(int)
    generator_seen_qids: dict[str, set[str]] = defaultdict(set)
    qid_to_topic: dict[str, str] = {}

    # Raw attempt-level accuracy counts ("raw % correct").
    generator_topic_attempts_raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    generator_topic_correct_raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall_topic_attempts_raw: dict[str, int] = defaultdict(int)
    overall_topic_correct_raw: dict[str, int] = defaultdict(int)

    missed_rows: list[dict[str, Any]] = []
    n_missing_topics = 0
    n_filtered_runs = 0
    n_missing_abms_topics = 0

    for rd in run_dirs:
        quiz_id = os.path.basename(rd.rstrip(os.sep))
        generator_model_raw = infer_generator_from_run_dir(rd, quiz_generators)
        generator_model = canonicalize_generator_model(
            generator_model_raw, mode=args.generator_family_mode
        )

        allowed_qids: set[str] | None = None
        if args.filter_by_judge:
            allowed_qids = filter_by_judge(
                Path(rd),
                args.filter_judges,
                min_medical_score=args.filter_min_med_score,
                require_logical_valid=args.filter_require_logical_valid,
                logical_mode=args.filter_logical_mode,
            )
            if allowed_qids is not None:
                n_filtered_runs += 1

        topic_map: dict[str, str] = {}
        if not abms_mode:
            topic_map = load_topic_map(
                Path(rd),
                topics_eval_model=args.topics_eval_model,
                topic_field=args.topic_field,
            )
            if not topic_map:
                n_missing_topics += 1

        for result_path in glob.glob(os.path.join(rd, "*_result.json")):
            base = os.path.basename(result_path)
            if base.endswith("_judge_result.json"):
                continue

            model = base[: -len("_result.json")] if base.endswith("_result.json") else base
            if answer_model_allowlist is not None and model not in answer_model_allowlist:
                continue

            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception:
                continue
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                qid = row.get("question_id")
                if not isinstance(qid, str) or not qid:
                    continue
                if allowed_qids is not None and qid not in allowed_qids:
                    continue

                gold = row.get("answer")
                if not isinstance(gold, str) or not gold.strip():
                    continue
                pred = row.get("pred")
                pred_str = str(pred).strip() if pred is not None else ""

                # Topic label (used for both denominators and missed-set summaries).
                if abms_mode:
                    topic = (
                        row.get("abms_specialty")
                        or row.get("target_topic")
                        or row.get("target_topics")
                    )
                    if isinstance(topic, list):
                        topic = next((t for t in topic if str(t).strip()), None)
                    topic = str(topic).strip() if topic is not None else "unknown"
                    topic = " ".join(topic.split()) or "unknown"
                else:
                    topic = topic_map.get(qid, "unknown")
                    topic = str(topic).strip() if topic is not None else "unknown"
                    topic = " ".join(topic.split())
                    topic = topic.casefold() if topic else "unknown"

                # Denominators: topic totals and overall totals.
                if args.count_mode == "events":
                    generator_total_questions[generator_model] += 1
                    generator_topic_total[generator_model][topic] += 1
                else:  # unique_questions
                    seen = generator_seen_qids[generator_model]
                    if qid not in seen:
                        seen.add(qid)
                        generator_total_questions[generator_model] += 1
                        generator_topic_total[generator_model][topic] += 1

                model_total[model] += 1
                is_correct = pred_str == gold.strip()

                # Raw attempt-level accuracy tallies (pooled across attempts).
                generator_topic_attempts_raw[generator_model][topic] += 1
                overall_topic_attempts_raw[topic] += 1
                if is_correct:
                    generator_topic_correct_raw[generator_model][topic] += 1
                    overall_topic_correct_raw[topic] += 1
                    continue

                model_missed[model] += 1
                if abms_mode and topic == "unknown":
                    n_missing_abms_topics += 1

                missed_rows.append(
                    {
                        "generator_model": generator_model,
                        "raw_generator_model": generator_model_raw,
                        "quiz_id": quiz_id,
                        "answer_model": model,
                        "question_id": qid,
                        "topic": topic,
                        "gold": gold.strip(),
                        "pred": pred_str or None,
                    }
                )

                qid_to_topic[qid] = topic

                if args.count_mode == "events":
                    generator_total_missed[generator_model] += 1
                    generator_topic_missed[generator_model][topic] += 1

    if args.count_mode == "unique_questions":
        generator_qids: dict[str, set[str]] = defaultdict(set)
        for row in missed_rows:
            gen = row["generator_model"]
            qid = row["question_id"]
            generator_qids[gen].add(qid)

        generator_total_missed = defaultdict(int)
        generator_topic_missed = defaultdict(lambda: defaultdict(int))
        for gen, qids in generator_qids.items():
            generator_total_missed[gen] = len(qids)
            for qid in qids:
                topic = qid_to_topic.get(qid, "unknown")
                generator_topic_missed[gen][topic] += 1

    total_miss_events = len(missed_rows)
    total_considered = int(sum(model_total.values()))
    if total_considered <= 0:
        raise SystemExit("[FATAL] No model result rows found after filtering.")

    print(
        f"[INFO] Processed {len(run_dirs)} run(s), "
        f"{len(model_total)} answer model(s), "
        f"{total_miss_events} miss event(s) out of {total_considered} "
        f"scored question(s)."
    )
    if abms_mode:
        print("[INFO] Topic mode: ABMS (using abms_specialty/target_topic in *_result.json)")
        if n_missing_abms_topics:
            print(
                f"[WARN] Missing abms_specialty/target_topic in {n_missing_abms_topics} miss(es); "
                "those misses are labeled topic='unknown'."
            )
    elif n_missing_topics:
        print(
            f"[WARN] Missing topics_*.json in {n_missing_topics} run(s); "
            "those misses are labeled topic='unknown'."
        )
    if args.filter_by_judge:
        print(
            f"[INFO] Judge filtering active in {n_filtered_runs}/{len(run_dirs)} run(s) "
            "(runs without judge outputs fall back to unfiltered)."
        )

    # Table 1: per-model miss rates.
    model_rows = []
    for model in sorted(model_total.keys()):
        total = int(model_total.get(model, 0) or 0)
        missed = int(model_missed.get(model, 0) or 0)
        miss_rate = (missed / total) * 100.0 if total > 0 else 0.0
        model_rows.append((model, total, missed, miss_rate))
    model_rows.sort(key=lambda r: (-r[3], -r[2], r[0]))

    model_table = _format_table(
        headers=["model", "n_questions", "n_missed", "miss_rate_%"],
        rows=[
            [m, str(t), str(k), f"{p:.1f}"]
            for (m, t, k, p) in model_rows
        ],
        right_align={1, 2, 3},
    )
    print("\nMiss rate by answer model")
    print(model_table)

    # Table 2: topic distribution of misses per generator + overall.
    overall_topic_missed: dict[str, int] = defaultdict(int)
    overall_total_missed = 0
    for gen, total in generator_total_missed.items():
        overall_total_missed += int(total or 0)
        for topic, cnt in generator_topic_missed.get(gen, {}).items():
            overall_topic_missed[topic] += int(cnt or 0)

    topic_rows: list[tuple[str, int, str, int, float]] = []
    for topic, cnt in sorted(overall_topic_missed.items(), key=lambda x: (-x[1], x[0])):
        pct = (cnt / overall_total_missed) * 100.0 if overall_total_missed > 0 else 0.0
        topic_rows.append(("ALL", overall_total_missed, topic, cnt, pct))

    for gen in sorted(generator_total_missed.keys()):
        total = int(generator_total_missed.get(gen, 0) or 0)
        if total <= 0:
            continue
        counts = generator_topic_missed.get(gen, {})
        for topic, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            pct = (cnt / total) * 100.0 if total > 0 else 0.0
            topic_rows.append((gen, total, topic, int(cnt or 0), pct))

    topic_table = _format_table(
        headers=["generator", "total_misses", "topic", "topic_misses", "topic_%"],
        rows=[
            [g, str(tot), tp, str(cnt), f"{pct:.1f}"]
            for (g, tot, tp, cnt, pct) in topic_rows
        ],
        right_align={1, 3, 4},
    )
    print(
        "\nTopic share of missed questions by generator "
        f"({args.count_mode.replace('_', ' ')})"
    )
    print(topic_table)

    # Table 3: topic miss rates + over/under-representation (enrichment).
    # Over/under-representation compares:
    #   (topic_misses / total_misses)  vs  (topic_total / total_questions)
    # Values > 1 mean misses are concentrated in that topic beyond prevalence.
    overall_topic_total: dict[str, int] = defaultdict(int)
    overall_total_questions = 0
    for gen, total in generator_total_questions.items():
        overall_total_questions += int(total or 0)
        for topic, cnt in generator_topic_total.get(gen, {}).items():
            overall_topic_total[topic] += int(cnt or 0)

    def _safe_pct(n: int, d: int) -> float:
        return (n / d) * 100.0 if d > 0 else float("nan")

    def _safe_ratio(n: float, d: float) -> float:
        return (n / d) if d and d > 0 else float("nan")

    def _nan_to_neg_inf(x: float) -> float:
        return x if x == x else float("-inf")

    rate_records: list[dict[str, Any]] = []

    def _add_rate_records(
        gen_label: str,
        *,
        total_questions: int,
        topic_totals: dict[str, int],
        total_misses: int,
        topic_misses: dict[str, int],
    ) -> None:
        for topic, missed in topic_misses.items():
            missed_i = int(missed or 0)
            if missed_i <= 0:
                continue
            topic_total_i = int(topic_totals.get(topic, 0) or 0)
            miss_rate_pct = _safe_pct(missed_i, topic_total_i)
            miss_share = (missed_i / total_misses) if total_misses > 0 else float("nan")
            prevalence = (topic_total_i / total_questions) if total_questions > 0 else float("nan")
            prevalence_pct = _safe_pct(topic_total_i, total_questions)
            over_under = _safe_ratio(miss_share, prevalence)
            rate_records.append(
                {
                    "generator": gen_label,
                    "topic": topic,
                    "topic_total": topic_total_i,
                    "topic_prevalence_pct": prevalence_pct,
                    "topic_misses": missed_i,
                    "topic_miss_rate_pct": miss_rate_pct,
                    "over_under": over_under,
                }
            )

    _add_rate_records(
        "ALL",
        total_questions=int(overall_total_questions or 0),
        topic_totals=dict(overall_topic_total),
        total_misses=int(overall_total_missed or 0),
        topic_misses=dict(overall_topic_missed),
    )
    for gen in sorted(generator_total_questions.keys()):
        _add_rate_records(
            gen,
            total_questions=int(generator_total_questions.get(gen, 0) or 0),
            topic_totals=dict(generator_topic_total.get(gen, {})),
            total_misses=int(generator_total_missed.get(gen, 0) or 0),
            topic_misses=dict(generator_topic_missed.get(gen, {})),
        )

    def _rate_sort_key(r: dict[str, Any]) -> tuple:
        gen = str(r.get("generator", ""))
        gen_rank = 0 if gen == "ALL" else 1
        over = float(r.get("over_under") or float("nan"))
        miss_rate = float(r.get("topic_miss_rate_pct") or float("nan"))
        missed = int(r.get("topic_misses") or 0)
        topic = str(r.get("topic") or "")
        return (gen_rank, gen, -_nan_to_neg_inf(over), -_nan_to_neg_inf(miss_rate), -missed, topic)

    rate_records.sort(key=_rate_sort_key)

    denom_label = "attempts" if args.count_mode == "events" else "questions"
    rate_table = _format_table(
        headers=[
            "generator",
            "topic",
            f"topic_total_{denom_label}",
            "topic_prevalence_%",
            "topic_misses",
            "topic_miss_rate_%",
            "over_under",
        ],
        rows=[
            [
                str(r["generator"]),
                str(r["topic"]),
                str(int(r["topic_total"])),
                (
                    ""
                    if r["topic_prevalence_pct"] != r["topic_prevalence_pct"]
                    else f"{r['topic_prevalence_pct']:.1f}"
                ),
                str(int(r["topic_misses"])),
                ("" if r["topic_miss_rate_pct"] != r["topic_miss_rate_pct"] else f"{r['topic_miss_rate_pct']:.1f}"),
                ("" if r["over_under"] != r["over_under"] else f"{r['over_under']:.2f}"),
            ]
            for r in rate_records
        ],
        right_align={2, 3, 4, 5, 6},
    )
    print(
        "\nTopic miss rate and over/under-representation by generator "
        f"({args.count_mode.replace('_', ' ')})"
    )
    print(rate_table)

    # Raw topic accuracy output (pooled across attempts).
    if out_topic_accuracy_csv:
        accuracy_records: list[dict[str, Any]] = []

        def _add_accuracy(
            gen_label: str, topic_attempts: dict[str, int], topic_correct: dict[str, int]
        ) -> None:
            for topic, total in topic_attempts.items():
                total_i = int(total or 0)
                if total_i <= 0:
                    continue
                correct_i = int(topic_correct.get(topic, 0) or 0)
                acc = correct_i / total_i
                accuracy_records.append(
                    {
                        "generator": gen_label,
                        "topic": topic,
                        "n_attempts": total_i,
                        "n_correct": correct_i,
                        "n_incorrect": total_i - correct_i,
                        "accuracy": acc,
                        "accuracy_pct": acc * 100.0,
                        "miss_rate_pct": (1.0 - acc) * 100.0,
                    }
                )

        _add_accuracy("ALL", dict(overall_topic_attempts_raw), dict(overall_topic_correct_raw))
        for gen in sorted(generator_topic_attempts_raw.keys()):
            _add_accuracy(
                gen,
                dict(generator_topic_attempts_raw.get(gen, {})),
                dict(generator_topic_correct_raw.get(gen, {})),
            )

        accuracy_records.sort(
            key=lambda r: (
                0 if r["generator"] == "ALL" else 1,
                str(r["generator"]),
                float(r["accuracy_pct"]),
                -int(r["n_attempts"]),
                str(r["topic"]),
            )
        )

        write_csv(
            out_topic_accuracy_csv,
            [
                "generator",
                "topic",
                "n_attempts",
                "n_correct",
                "n_incorrect",
                "accuracy",
                "accuracy_pct",
                "miss_rate_pct",
            ],
            accuracy_records,
        )
        print(out_topic_accuracy_csv)

        all_rows = [r for r in accuracy_records if r["generator"] == "ALL"]
        if all_rows:
            hardest = min(all_rows, key=lambda r: r["accuracy_pct"])
            easiest = max(all_rows, key=lambda r: r["accuracy_pct"])
            print("\nRaw topic accuracy (ALL; pooled across attempts)")
            print(
                f"Hardest: {hardest['topic']}  {hardest['accuracy_pct']:.1f}% "
                f"(n_attempts={hardest['n_attempts']})"
            )
            print(
                f"Easiest: {easiest['topic']}  {easiest['accuracy_pct']:.1f}% "
                f"(n_attempts={easiest['n_attempts']})"
            )

    if out_radar_png:
        save_topic_miss_rate_radar(
            out_path=out_radar_png,
            generator_topic_total=generator_topic_total,
            generator_topic_missed=generator_topic_missed,
            generator_total_questions=generator_total_questions,
            overall_topic_total=dict(overall_topic_total),
            overall_topic_missed=dict(overall_topic_missed),
            count_mode=args.count_mode,
        )
    if out_radar_all_only_png:
        save_topic_miss_rate_radar_all_only(
            out_path=out_radar_all_only_png,
            overall_topic_total=dict(overall_topic_total),
            overall_topic_missed=dict(overall_topic_missed),
            count_mode=args.count_mode,
        )

    # Optional outputs.
    if out_misses_csv:
        misses_fieldnames = [
            "generator_model",
            "raw_generator_model",
            "quiz_id",
            "answer_model",
            "question_id",
            "topic",
            "gold",
            "pred",
        ]
        write_csv(out_misses_csv, misses_fieldnames, missed_rows)
        print(f"\n{out_misses_csv}")

    if out_model_csv:
        model_summary_rows: list[dict[str, Any]] = []
        for model, total, missed, miss_rate in model_rows:
            model_summary_rows.append(
                {
                    "model": model,
                    "n_questions": total,
                    "n_missed": missed,
                    "miss_rate": (missed / total) if total > 0 else 0.0,
                    "miss_rate_pct": miss_rate,
                }
            )
        write_csv(
            out_model_csv,
            ["model", "n_questions", "n_missed", "miss_rate", "miss_rate_pct"],
            model_summary_rows,
        )
        print(out_model_csv)

    if out_topic_csv:
        topic_summary_rows: list[dict[str, Any]] = []
        for gen, total, topic, cnt, pct in topic_rows:
            topic_summary_rows.append(
                {
                    "generator_model": gen,
                    "total_misses": total,
                    "topic": topic,
                    "topic_misses": cnt,
                    "topic_pct": pct,
                }
            )
        write_csv(
            out_topic_csv,
            ["generator_model", "total_misses", "topic", "topic_misses", "topic_pct"],
            topic_summary_rows,
        )
        print(out_topic_csv)


if __name__ == "__main__":
    main()
