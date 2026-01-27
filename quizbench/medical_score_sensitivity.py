#!/usr/bin/env python3
"""
Sensitivity analysis for QuizBench judge filtering across medical-score cutoffs.

This script scans QuizBench run directories under --runs_root and, for each
minimum medical_accuracy_score cutoff (default: 1..5), counts how many questions
would be kept after applying judge-based filtering.

It prints a small table to stdout, writes a CSV, and saves a PNG visualizing the
tradeoff between cutoff strictness and questions retained. Optionally, it also
computes how quiz-taker model accuracy changes as the cutoff varies and writes a
second CSV + PNG (a heatmap of accuracy vs cutoff by model).

By default, this script uses a "logical-first" (two-stage) interpretation for
cutoffs that yields monotonic question counts:
  1) compute logical validity once (using min_medical_score=1), then
  2) apply medical-score cutoffs using a strict-majority vote over judge scores.

Use --no-logical_first to instead mirror quizbench.aggregate_judges.filter_by_judge
behavior exactly (in majority mode, counts need not be monotonic).

It can also operate in a "subset" mode for ABMS-style subsets built by
quizbench/build_abms_valid_subset.py. In that case, the subset runs tree does
not include judge outputs; instead, judge outputs live in the *source* runs tree
referenced by `selection_report_<TAG>.json`. Use --use_selection_reports to load
the selection reports and compute sensitivity using the original judge outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES, filter_by_judge

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_LOGICAL_FIRST_BASE_SCORE = 3


def _mean_sem(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n == 1:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sem = (var**0.5) / (n**0.5)
    return mean, sem


@dataclass
class AccuracyStats:
    quiz_acc: list[float] = field(default_factory=list)
    total_corr: int = 0
    total_wrong: int = 0

    def add_quiz(self, corr: int, wrong: int) -> None:
        total = int(corr) + int(wrong)
        if total <= 0:
            return
        self.quiz_acc.append(corr / float(total))
        self.total_corr += int(corr)
        self.total_wrong += int(wrong)

    @property
    def total_items(self) -> int:
        return self.total_corr + self.total_wrong

    def mean_sem(self) -> tuple[float, float]:
        return _mean_sem(self.quiz_acc)

    def micro_accuracy(self) -> float:
        total = self.total_items
        if total <= 0:
            return float("nan")
        return self.total_corr / float(total)


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    """
    Return run dirs under runs_root (directories containing any *_summary.json).
    """
    out: set[Path] = set()
    for dirpath, _, filenames in os.walk(runs_root):
        if any(name.endswith("_summary.json") for name in filenames):
            out.add(Path(dirpath))
    return sorted(out)


def _infer_generator_model(run_dir: Path, runs_root: Path) -> str:
    """
    Infer generator model based on the directory layout:
      <runs_root>/<generator>/<quiz_id>/...
    If runs_root points directly at a generator dir, fall back to runs_root.name.
    """
    try:
        rel = run_dir.resolve().relative_to(runs_root.resolve())
    except Exception:
        return run_dir.parent.name
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return runs_root.name


def _safe_int(x: object) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    try:
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _get_total_items_for_run(run_dir: Path) -> int:
    """
    Best-effort question count for a run dir.

    Preference order:
      1) run_dir/manifest.json "n_items"
      2) max "n_items" across any *_summary.json / summary_all_judges.json
    """
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None
        if isinstance(manifest, dict):
            n_items = _safe_int(manifest.get("n_items"))
            if n_items is not None and n_items > 0:
                return n_items

    n_items_values: list[int] = []

    for p in run_dir.glob("*_summary.json"):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        n_items = _safe_int(row.get("n_items"))
        if n_items is not None and n_items > 0:
            n_items_values.append(n_items)

    all_judges_path = run_dir / "summary_all_judges.json"
    if all_judges_path.exists():
        try:
            rows = json.loads(all_judges_path.read_text(encoding="utf-8"))
        except Exception:
            rows = None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                n_items = _safe_int(row.get("n_items"))
                if n_items is not None and n_items > 0:
                    n_items_values.append(n_items)

    return max(n_items_values) if n_items_values else 0


def _sanitize_suffix(value: str) -> str:
    """
    Make a safe identifier fragment usable in filenames.
    """
    text = (value or "").strip()
    text = _SAFE_ID_RE.sub("_", text)
    return text.strip("_.-")


def _infer_label_from_runs_root(runs_root: Path) -> str | None:
    """
    Infer a human label from common QuizBench layout:
      .../eval_results/quizbench/quizzes_<TAG>/runs
    """
    parts = runs_root.parts
    if len(parts) >= 2 and parts[-1] == "runs":
        parent = parts[-2]
        if parent.startswith("quizzes_") and len(parent) > len("quizzes_"):
            tag = parent[len("quizzes_") :]
            tag = _sanitize_suffix(tag)
            return tag or None
    return None


def _default_output_paths(
    *,
    out_dir: Path,
    out_prefix: str,
    label: str | None,
) -> tuple[Path, Path]:
    suffix = ""
    if label:
        safe = _sanitize_suffix(label)
        if safe:
            suffix = f"_{safe}"
    stem = f"{out_prefix}{suffix}"
    return out_dir / f"{stem}.csv", out_dir / f"{stem}.png"


def _localize_repo_path(raw: str) -> Path | None:
    """
    Best-effort localization of an absolute path recorded on another machine.

    build_abms_valid_subset.py stores absolute `source_run_dir` paths in the
    selection report; those may not exist on the current machine. This helper
    attempts to rebase such paths against the current repo root.
    """
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()

    raw_norm = raw.replace("\\", "/")
    for anchor in ("eval_results/quizbench/", "eval_results/"):
        idx = raw_norm.find(anchor)
        if idx == -1:
            continue
        rel = raw_norm[idx:]
        rebased = (ROOT_DIR / rel).resolve()
        if rebased.exists():
            return rebased

    return None


def _find_selection_reports(subset_runs_root: Path, subset_tag: str | None) -> list[Path]:
    if subset_tag:
        wanted = f"selection_report_{subset_tag}.json"
        hits = sorted(subset_runs_root.glob(f"*/{wanted}"))
        if hits:
            return hits
    return sorted(subset_runs_root.glob("*/selection_report_*.json"))


def _load_selected_questions_from_reports(
    subset_runs_root: Path,
    *,
    subset_tag: str | None,
) -> tuple[dict[Path, set[str]], set[str]]:
    """
    Load ABMS subset selection reports and return:

      - mapping: source_run_dir -> set(question_id) selected from that run
      - judge_models union across reports (may be empty if missing in reports)
    """
    report_paths = _find_selection_reports(subset_runs_root, subset_tag)
    if not report_paths:
        raise SystemExit(
            f"[FATAL] No selection_report*.json found under: {subset_runs_root}\n"
            "If you're analyzing a normal (non-ABMS) runs tree, omit --use_selection_reports."
        )

    selected_by_run_dir: dict[Path, set[str]] = defaultdict(set)
    judge_models_union: set[str] = set()

    for report_path in report_paths:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[FATAL] Could not read selection report: {report_path}: {exc}") from exc

        if not isinstance(report, dict):
            raise SystemExit(f"[FATAL] Selection report is not a JSON object: {report_path}")

        generator_model = str(report.get("generator_model") or report_path.parent.name).strip()
        if not generator_model:
            raise SystemExit(f"[FATAL] Could not infer generator model for report: {report_path}")

        jm = report.get("judge_models") or []
        if isinstance(jm, list):
            judge_models_union |= {str(x).strip() for x in jm if str(x).strip()}

        selected = report.get("selected") or []
        if not isinstance(selected, list):
            raise SystemExit(f"[FATAL] selection_report has non-list 'selected': {report_path}")

        for row in selected:
            if not isinstance(row, dict):
                continue

            source_quiz_id = str(row.get("source_quiz_id") or "").strip()
            source_qid = str(row.get("source_question_id") or "").strip()
            if not source_quiz_id or not source_qid:
                continue

            # Prefer the stored source_run_dir, but rebase it to the local repo.
            source_run_dir_raw = str(row.get("source_run_dir") or "").strip()
            source_run_dir = _localize_repo_path(source_run_dir_raw)
            if source_run_dir is None:
                # Fallback: reconstruct from (rebased) source_runs_root + generator + quiz_id.
                source_runs_root = _localize_repo_path(str(row.get("source_runs_root") or "").strip())
                if source_runs_root is None:
                    raise SystemExit(
                        f"[FATAL] Could not localize source_runs_root for selection report row.\n"
                        f"report={report_path}\n"
                        f"source_runs_root={row.get('source_runs_root')!r}"
                    )
                source_run_dir = (source_runs_root / generator_model / source_quiz_id).resolve()

            if not source_run_dir.exists() or not source_run_dir.is_dir():
                raise SystemExit(
                    f"[FATAL] Source run directory does not exist: {source_run_dir}\n"
                    f"(from report {report_path})"
                )

            selected_by_run_dir[source_run_dir].add(source_qid)

    return dict(selected_by_run_dir), judge_models_union


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    _ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iter_result_paths(run_dir: Path) -> list[Path]:
    """
    Return non-judge *_result.json paths within a quiz run directory.
    """
    out: list[Path] = []
    for p in sorted(run_dir.glob("*_result.json")):
        if p.name.endswith("_judge_result.json"):
            continue
        out.append(p)
    return out


def _load_json_list(path: Path) -> list[dict] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    out: list[dict] = []
    for row in obj:
        if isinstance(row, dict):
            out.append(row)
    return out


def _score_rows_by_cutoff(
    rows: list[dict],
    *,
    cutoffs: list[int],
    include_row_for_cutoff,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Return ({cutoff: corr}, {cutoff: wrong}) for a set of eval result rows.

    Matches the filtering semantics used in quizbench/aggregate_results.py:
      - skip rows where gold answer is missing/blank
      - skip rows where pred is None
    """
    corr_by: dict[int, int] = {c: 0 for c in cutoffs}
    wrong_by: dict[int, int] = {c: 0 for c in cutoffs}

    for row in rows:
        gold = str(row.get("answer", "")).strip()
        if not gold:
            continue
        pred = row.get("pred")
        if pred is None:
            continue
        is_corr = pred == gold
        for c in cutoffs:
            if not include_row_for_cutoff(row, c):
                continue
            if is_corr:
                corr_by[c] += 1
            else:
                wrong_by[c] += 1

    return corr_by, wrong_by


def _score_majority_allowed_by_cutoff(
    run_dir: Path,
    judge_models: list[str],
    *,
    cutoffs: list[int],
    base_min_score: int = _LOGICAL_FIRST_BASE_SCORE,
) -> dict[int, set[str]] | None:
    """
    Return {cutoff: allowed_qids} where a question is allowed at cutoff c iff a
    strict majority of *eligible* judge outputs have medical_accuracy_score >= c.

    Eligibility for counting a judge vote is:
      - judge_output_valid is True
      - medical_accuracy_score parses to int and >= base_min_score

    If no judge_result files are found for any judge_models, returns None.
    """
    cutoffs = sorted(set(cutoffs))
    if not cutoffs:
        return {}

    any_judge = False
    total_counts: dict[str, int] = defaultdict(int)
    ge_counts_by_cutoff: dict[int, dict[str, int]] = {c: defaultdict(int) for c in cutoffs}

    for judge_model in judge_models:
        result_path = run_dir / f"{judge_model}_judge_result.json"
        if not result_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        any_judge = True

        for row in payload:
            if not isinstance(row, dict):
                continue
            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid:
                continue

            if not row.get("judge_output_valid"):
                continue

            score = row.get("judge_medical_accuracy_score")
            if score is None:
                judge_json = row.get("judge_json")
                if isinstance(judge_json, dict):
                    score = judge_json.get("medical_accuracy_score")

            score_int: int | None
            try:
                score_int = int(score) if score is not None else None
            except (TypeError, ValueError):
                score_int = None

            if score_int is None or score_int < base_min_score:
                continue

            total_counts[qid] = total_counts.get(qid, 0) + 1
            for c in cutoffs:
                if score_int >= c:
                    ge_counts_by_cutoff[c][qid] = ge_counts_by_cutoff[c].get(qid, 0) + 1

    if not any_judge:
        return None

    out: dict[int, set[str]] = {}
    for c in cutoffs:
        allowed: set[str] = set()
        ge_counts = ge_counts_by_cutoff[c]
        for qid, total in total_counts.items():
            ge = ge_counts.get(qid, 0)
            if total > 0 and ge > total / 2.0:
                allowed.add(qid)
        out[c] = allowed
    return out


def _save_plot(
    out_png: Path,
    *,
    cutoffs: list[int],
    kept_counts: list[int],
    total_items: int,
    title_label: str | None,
    require_matplotlib: bool,
) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if require_matplotlib:
            raise SystemExit(
                f"[FATAL] Could not import matplotlib to write {out_png}: {exc}\n"
                "Install dependencies (e.g., `python -m pip install -r requirements.txt`)."
            ) from exc
        print(
            f"[WARN] Could not import matplotlib; skipping plot at {out_png}: {exc}",
            file=sys.stderr,
        )
        return

    percents = [
        (k / float(total_items) * 100.0) if total_items > 0 else float("nan")
        for k in kept_counts
    ]

    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    ax1.bar(cutoffs, kept_counts, color="#4c72b0", alpha=0.85)
    ax1.set_xlabel("Minimum medical_accuracy_score cutoff")
    ax1.set_ylabel("# questions kept")
    ax1.set_xticks(cutoffs)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cutoffs, percents, color="#dd8452", marker="o", linewidth=2)
    ax2.set_ylabel("Percent kept (%)")
    ax2.set_ylim(0, 100)

    title = "Questions kept vs medical-score cutoff"
    if title_label:
        title += f" ({title_label})"
    ax1.set_title(title)

    fig.tight_layout()
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=300)


def _save_accuracy_heatmap(
    out_png: Path,
    *,
    cutoffs: list[int],
    model_names: list[str],
    values: list[list[float]],
    title_label: str | None,
    require_matplotlib: bool,
) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if require_matplotlib:
            raise SystemExit(
                f"[FATAL] Could not import matplotlib to write {out_png}: {exc}\n"
                "Install dependencies (e.g., `python -m pip install -r requirements.txt`)."
            ) from exc
        print(
            f"[WARN] Could not import matplotlib; skipping plot at {out_png}: {exc}",
            file=sys.stderr,
        )
        return

    try:
        import seaborn as sns  # type: ignore
    except Exception:  # noqa: BLE001
        sns = None  # type: ignore[assignment]

    arr = np.array(values, dtype=float)
    mask = np.isnan(arr)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")

    vmin = 0.6
    vmax = 1.0

    height = max(4.4, 0.34 * len(model_names))
    width = max(8.8, 1.2 + 1.0 * len(cutoffs))

    title = "Model accuracy vs medical-score cutoff"
    if title_label:
        title += f" ({title_label})"

    if sns is not None:
        annot = np.empty(arr.shape, dtype=object)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = float(arr[i, j])
                annot[i, j] = "" if math.isnan(v) else f"{v:.2f}"

        rc = {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }

        with sns.plotting_context("paper", rc=rc), sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(width, height))
            sns.heatmap(
                arr,
                ax=ax,
                mask=mask,
                annot=annot,
                fmt="",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Mean accuracy (across quizzes)"},
            )

            ax.set_xticks([i + 0.5 for i in range(len(cutoffs))])
            ax.set_xticklabels([str(c) for c in cutoffs], rotation=0)
            ax.set_xlabel("Minimum medical_accuracy_score cutoff")

            ax.set_yticks([i + 0.5 for i in range(len(model_names))])
            ax.set_yticklabels(model_names, rotation=0)
            ax.set_ylabel("Answer model")

            ax.set_title(title)
            fig.tight_layout()
            _ensure_parent(out_png)
            fig.savefig(out_png, dpi=300)
        return

    # Fallback: matplotlib-only heatmap with annotations.
    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean accuracy (across quizzes)")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = float(arr[i, j])
            if math.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v >= 0.80 else "black",
            )

    ax.set_xticks(list(range(len(cutoffs))))
    ax.set_xticklabels([str(c) for c in cutoffs])
    ax.set_xlabel("Minimum medical_accuracy_score cutoff")

    ax.set_yticks(list(range(len(model_names))))
    ax.set_yticklabels(model_names)
    ax.set_ylabel("Answer model")
    ax.set_title(title)

    fig.tight_layout()
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=300)


def _save_generator_availability_heatmap(
    out_png: Path,
    *,
    cutoffs: list[int],
    total_by_generator: dict[str, int],
    kept_by_generator: dict[str, dict[int, int]],
    title_label: str | None,
    require_matplotlib: bool,
) -> None:
    """
    Save a seaborn heatmap of questions available per generator by cutoff.

    Intended for "paper-style" reporting (NeurIPS-like): compact fonts, clean
    background, and high DPI output.
    """
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if require_matplotlib:
            raise SystemExit(
                f"[FATAL] Could not import matplotlib/numpy to write {out_png}: {exc}\n"
                "Install dependencies (e.g., `python -m pip install -r requirements.txt`)."
            ) from exc
        print(
            f"[WARN] Could not import matplotlib/numpy; skipping heatmap at {out_png}: {exc}",
            file=sys.stderr,
        )
        return

    try:
        import seaborn as sns  # type: ignore
    except Exception:  # noqa: BLE001
        sns = None  # type: ignore[assignment]

    generators = sorted(set(total_by_generator) | set(kept_by_generator))
    if not generators:
        print("[WARN] No generators found; skipping generator availability heatmap.")
        return

    sort_cutoff = max(cutoffs)

    def _sort_key(name: str) -> tuple[int, str]:
        kept = int((kept_by_generator.get(name) or {}).get(sort_cutoff, 0) or 0)
        return (-kept, name)

    generators = sorted(generators, key=_sort_key)
    totals = [int(total_by_generator.get(g, 0) or 0) for g in generators]
    vmax = max(totals) if any(totals) else None

    data = [
        [int((kept_by_generator.get(g) or {}).get(c, 0) or 0) for c in cutoffs]
        for g in generators
    ]
    arr = np.array(data, dtype=int)

    rc = {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }

    height = max(3.6, 0.45 * len(generators))
    width = max(6.2, 1.0 + 0.9 * len(cutoffs))
    if sns is not None:
        with sns.plotting_context("paper", rc=rc), sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(width, height))
            sns.heatmap(
                arr,
                ax=ax,
                annot=True,
                fmt="d",
                cmap=sns.color_palette("Blues", as_cmap=True),
                vmin=0,
                vmax=vmax,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "# questions available"},
            )

            ax.set_xticks([i + 0.5 for i in range(len(cutoffs))])
            ax.set_xticklabels([f">={c}" for c in cutoffs], rotation=0)
            ax.set_yticks([i + 0.5 for i in range(len(generators))])
            ax.set_yticklabels(
                [
                    f"{g} (n={int(total_by_generator.get(g, 0) or 0)})"
                    if int(total_by_generator.get(g, 0) or 0) > 0
                    else g
                    for g in generators
                ],
                rotation=0,
            )

            ax.set_xlabel("Minimum medical_accuracy_score cutoff")
            ax.set_ylabel("Generator")
            title = "Questions available per generator"
            if title_label:
                title += f" ({title_label})"
            ax.set_title(title)

            fig.tight_layout()
            _ensure_parent(out_png)
            fig.savefig(out_png, dpi=300)
    else:
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(width, height))
            cmap = plt.cm.Blues.copy()
            im = ax.imshow(arr, aspect="auto", vmin=0, vmax=vmax, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("# questions available")

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", fontsize=9)

            ax.set_xticks(list(range(len(cutoffs))))
            ax.set_xticklabels([f">={c}" for c in cutoffs], rotation=0)
            ax.set_yticks(list(range(len(generators))))
            ax.set_yticklabels(
                [
                    f"{g} (n={int(total_by_generator.get(g, 0) or 0)})"
                    if int(total_by_generator.get(g, 0) or 0) > 0
                    else g
                    for g in generators
                ],
                rotation=0,
            )

            ax.set_xlabel("Minimum medical_accuracy_score cutoff")
            ax.set_ylabel("Generator")
            title = "Questions available per generator"
            if title_label:
                title += f" ({title_label})"
            ax.set_title(title)

            fig.tight_layout()
            _ensure_parent(out_png)
            fig.savefig(out_png, dpi=300)


def _print_generator_question_availability_table(
    *,
    cutoffs: list[int],
    total_by_generator: dict[str, int],
    kept_by_generator: dict[str, dict[int, int]],
    sort_cutoff: int | None,
    total_label: str,
) -> None:
    """
    Print a generator x cutoff table of how many questions are available after
    judge filtering at each cutoff.
    """
    generators = sorted(set(total_by_generator) | set(kept_by_generator))
    if not generators:
        print("[WARN] No generators found for per-generator cutoff table.")
        return

    sort_c = sort_cutoff if sort_cutoff is not None else max(cutoffs)

    def _sort_key(name: str) -> tuple[int, str]:
        kept = int((kept_by_generator.get(name) or {}).get(sort_c, 0) or 0)
        return (-kept, name)

    generators = sorted(generators, key=_sort_key)

    col_labels = {c: f">={c}" for c in cutoffs}

    def _kept(gen: str, cutoff: int) -> int:
        return int((kept_by_generator.get(gen) or {}).get(cutoff, 0) or 0)

    gen_w = max(len("Generator"), max(len(g) for g in generators))
    total_w = max(
        len(total_label),
        max(len(str(int(total_by_generator.get(g, 0) or 0))) for g in generators),
    )
    col_w = {
        c: max(
            len(col_labels[c]),
            max(len(str(_kept(g, c))) for g in generators),
        )
        for c in cutoffs
    }

    header = (
        f"{'Generator':<{gen_w}} {total_label:>{total_w}} "
        + " ".join(f"{col_labels[c]:>{col_w[c]}}" for c in cutoffs)
    )
    print("\n[INFO] Questions available per generator by medical-score cutoff:")
    print(header)
    print("-" * len(header))
    for g in generators:
        total = int(total_by_generator.get(g, 0) or 0)
        row = f"{g:<{gen_w}} {total:>{total_w}} " + " ".join(
            f"{_kept(g, c):>{col_w[c]}}" for c in cutoffs
        )
        print(row)


def _warn_if_unexpected_total_per_generator(
    *,
    total_by_generator: dict[str, int],
    expected_total: int | None,
) -> None:
    if expected_total is None:
        return
    bad = {g: int(t or 0) for g, t in total_by_generator.items() if int(t or 0) != expected_total}
    if not bad:
        return
    parts = ", ".join(f"{g}={t}" for g, t in sorted(bad.items()))
    print(
        f"[WARN] Expected {expected_total} selected questions per generator, but got: {parts}",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis for QuizBench judge filtering across medical-score cutoffs."
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default=str(ROOT_DIR / "eval_results" / "quizbench" / "runs"),
        help="Runs root to scan (contains generator dirs and quiz run dirs).",
    )
    ap.add_argument(
        "--use_selection_reports",
        action="store_true",
        help=(
            "Treat --runs_root as an ABMS subset runs tree (e.g., "
            "eval_results/quizbench/quizzes_ABMS20260101/runs). Loads "
            "selection_report_<TAG>.json files under each generator dir and "
            "computes sensitivity on the selected questions using the original "
            "judge outputs referenced by those reports."
        ),
    )
    ap.add_argument(
        "--judge_models",
        nargs="+",
        default=None,
        help="Judge model ensemble to consider (default: DEFAULT_ENSEMBLE_JUDGES).",
    )
    ap.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[4, 5],
        help="Minimum medical_accuracy_score cutoffs to evaluate (default: 1 2 3 4 5).",
    )
    ap.add_argument(
        "--require_logical_valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to require logical validity (PASS) when filtering (default: true).",
    )
    ap.add_argument(
        "--logical_mode",
        type=str,
        choices=["all", "majority"],
        default="majority",
        help="How to aggregate logical validity across judges (default: majority).",
    )
    ap.add_argument(
        "--logical_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, compute logical validity once with min_medical_score=1 and then "
            "apply score cutoffs using a strict-majority vote over judge scores (monotonic). "
            "Use --no-logical_first to mirror quizbench.aggregate_judges.filter_by_judge behavior."
        ),
    )
    ap.add_argument(
        "--strict_judge_results",
        action="store_true",
        help=(
            "If set, error when any run directory has no judge results for the "
            "selected judge models (instead of falling back to unfiltered counts)."
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "eval_results" / "quizbench"),
        help="Directory to write output artifacts (default: eval_results/quizbench).",
    )
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="medscore_sensitivity",
        help="Output filename prefix under --out_dir (default: medscore_sensitivity).",
    )
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label added to the plot title and filenames (e.g., a batch tag).",
    )
    ap.add_argument(
        "--subset_tag",
        type=str,
        default=None,
        help=(
            "Optional subset tag used to locate selection_report_<TAG>.json when "
            "--use_selection_reports is set (e.g., ABMS20260101). Defaults to the "
            "inferred label from --runs_root when possible."
        ),
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional explicit path for the output CSV (overrides --out_dir/--out_prefix).",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Optional explicit path for the output PNG (overrides --out_dir/--out_prefix).",
    )
    ap.add_argument(
        "--out_item_csv",
        type=str,
        default=None,
        help=(
            "Optional path to write per-item, per-judge scores (one row per question_id "
            "per judge model). Most useful with --use_selection_reports."
        ),
    )
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a PNG plot (default: true).",
    )
    ap.add_argument(
        "--accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute model accuracy sensitivity and write an additional plot/CSV (default: true).",
    )
    ap.add_argument(
        "--out_accuracy_csv",
        type=str,
        default=None,
        help="Optional explicit path for the per-model accuracy sensitivity CSV.",
    )
    ap.add_argument(
        "--out_accuracy_png",
        type=str,
        default=None,
        help="Optional explicit path for the per-model accuracy sensitivity PNG.",
    )
    ap.add_argument(
        "--out_generator_availability_png",
        type=str,
        default=None,
        help="Optional explicit path for the per-generator availability heatmap PNG.",
    )
    ap.add_argument(
        "--require_plot",
        action="store_true",
        help="If set, error when matplotlib is unavailable (default: warn + skip plot).",
    )

    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists() or not runs_root.is_dir():
        raise SystemExit(f"[FATAL] runs_root does not exist or is not a directory: {runs_root}")

    label = (args.label or "").strip() or _infer_label_from_runs_root(runs_root)
    out_dir = Path(args.out_dir).expanduser()
    out_prefix = _sanitize_suffix(str(args.out_prefix)) or "medscore_sensitivity"
    default_csv, default_png = _default_output_paths(
        out_dir=out_dir,
        out_prefix=out_prefix,
        label=label,
    )
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else default_csv
    out_png = Path(args.out_png).expanduser() if args.out_png else default_png
    out_gen_png = (
        Path(args.out_generator_availability_png).expanduser()
        if args.out_generator_availability_png
        else out_csv.with_name(f"{out_csv.stem}_generator_availability.png")
    )

    subset_tag = (args.subset_tag or "").strip() or label
    subset_tag = _sanitize_suffix(subset_tag) or None

    if args.use_selection_reports:
        selected_by_run_dir, judges_in_reports = _load_selected_questions_from_reports(
            runs_root,
            subset_tag=subset_tag,
        )

        if args.judge_models is None:
            judge_models = sorted(judges_in_reports) if judges_in_reports else list(DEFAULT_ENSEMBLE_JUDGES)
        else:
            judge_models = [m.strip() for m in (args.judge_models or []) if str(m).strip()]

        if not judge_models:
            raise SystemExit("[FATAL] --judge_models resolved to an empty list.")
    else:
        if args.judge_models is None:
            judge_models = list(DEFAULT_ENSEMBLE_JUDGES)
        else:
            judge_models = [m.strip() for m in (args.judge_models or []) if str(m).strip()]
        if not judge_models:
            raise SystemExit("[FATAL] --judge_models resolved to an empty list.")

    cutoffs = sorted(set(args.cutoffs or []))
    if not cutoffs:
        raise SystemExit("[FATAL] --cutoffs resolved to an empty list.")
    if any(c < 1 or c > 5 for c in cutoffs):
        raise SystemExit(f"[FATAL] Cutoffs must be in [1, 5], got: {cutoffs}")

    total_items_all = 0
    runs_missing_judges = 0
    runs_with_judges = 0
    missing_items_total = 0

    kept_total_by_cutoff: dict[int, int] = {c: 0 for c in cutoffs}

    min_cutoff = min(cutoffs)

    total_by_generator: dict[str, int] = defaultdict(int)
    kept_by_generator: dict[str, dict[int, int]] = defaultdict(lambda: {c: 0 for c in cutoffs})

    item_rows: list[dict] = []
    acc_stats: dict[int, dict[str, AccuracyStats]] = defaultdict(lambda: defaultdict(AccuracyStats))
    acc_missing_models = 0
    subset_eval_run_dirs: list[Path] | None = None

    if args.use_selection_reports:
        run_dirs = sorted(selected_by_run_dir.keys())
        if not run_dirs:
            raise SystemExit(f"[FATAL] No selected questions found under: {runs_root}")

        allowed_cache: dict[Path, dict[int, set[str] | None]] = {}

        for run_dir in run_dirs:
            generator_name = run_dir.parent.name
            selected_qids = selected_by_run_dir.get(run_dir) or set()
            n_items = len(selected_qids)
            total_items_all += n_items
            total_by_generator[generator_name] += n_items

            if args.out_item_csv:
                # Build per-judge indices once per run directory.
                judge_rows_by_model: dict[str, dict[str, dict]] = {}
                for judge_model in judge_models:
                    result_path = run_dir / f"{judge_model}_judge_result.json"
                    if not result_path.exists():
                        judge_rows_by_model[judge_model] = {}
                        continue
                    try:
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = None
                    if not isinstance(payload, list):
                        judge_rows_by_model[judge_model] = {}
                        continue
                    idx: dict[str, dict] = {}
                    for rec in payload:
                        if not isinstance(rec, dict):
                            continue
                        qid = rec.get("question_id")
                        if not isinstance(qid, str) or not qid:
                            continue
                        idx[qid] = rec
                    judge_rows_by_model[judge_model] = idx

                for qid in sorted(selected_qids):
                    for judge_model in judge_models:
                        rec = judge_rows_by_model.get(judge_model, {}).get(qid)
                        judge_output_valid = None
                        score = None
                        logical_validity = None
                        logical_false_reason = None
                        verdict = None
                        fail_reason = None
                        parse_error = None

                        if isinstance(rec, dict):
                            judge_output_valid = rec.get("judge_output_valid")
                            parse_error = rec.get("judge_parse_error")
                            score = rec.get("judge_medical_accuracy_score")
                            verdict = rec.get("judge_verdict")
                            fail_reason = rec.get("judge_fail_reason")

                            judge_json = rec.get("judge_json")
                            if isinstance(judge_json, dict):
                                if score is None:
                                    score = judge_json.get("medical_accuracy_score")
                                logical_validity = judge_json.get("logical_validity")
                                logical_false_reason = judge_json.get("logical_false_reason")
                                if verdict is None:
                                    verdict = judge_json.get("verdict")
                                if fail_reason is None:
                                    fail_reason = judge_json.get("fail_reason")

                        item_rows.append(
                            {
                                "subset_tag": subset_tag or "",
                                "generator_model": run_dir.parent.name,
                                "quiz_id": run_dir.name,
                                "question_id": qid,
                                "judge_model": judge_model,
                                "has_judge_row": bool(rec),
                                "judge_output_valid": judge_output_valid,
                                "judge_parse_error": parse_error,
                                "medical_accuracy_score": score,
                                "logical_validity": logical_validity,
                                "logical_false_reason": logical_false_reason,
                                "verdict": verdict,
                                "fail_reason": fail_reason,
                            }
                        )

            allowed_by_cutoff: dict[int, set[str] | None] | None

            if bool(args.logical_first):
                score_allowed = _score_majority_allowed_by_cutoff(
                    run_dir,
                    judge_models,
                    cutoffs=cutoffs,
                    base_min_score=_LOGICAL_FIRST_BASE_SCORE,
                )
                if score_allowed is None:
                    allowed_by_cutoff = None
                elif bool(args.require_logical_valid):
                    logical_allowed = filter_by_judge(
                        run_dir,
                        judge_models,
                        min_medical_score=_LOGICAL_FIRST_BASE_SCORE,
                        require_logical_valid=True,
                        logical_mode=str(args.logical_mode),
                    )
                    if logical_allowed is None:
                        allowed_by_cutoff = None
                    else:
                        logical_allowed_set = set(logical_allowed)
                        allowed_by_cutoff = {
                            c: set(score_allowed.get(c, set())) & logical_allowed_set
                            for c in cutoffs
                        }
                else:
                    allowed_by_cutoff = {c: set(score_allowed.get(c, set())) for c in cutoffs}
            else:
                allowed_by_cutoff = {}
                allowed_min = filter_by_judge(
                    run_dir,
                    judge_models,
                    min_medical_score=min_cutoff,
                    require_logical_valid=bool(args.require_logical_valid),
                    logical_mode=str(args.logical_mode),
                )
                if allowed_min is None:
                    allowed_by_cutoff = None
                else:
                    allowed_by_cutoff[min_cutoff] = allowed_min
                    for c in cutoffs:
                        if c == min_cutoff:
                            continue
                        allowed = filter_by_judge(
                            run_dir,
                            judge_models,
                            min_medical_score=c,
                            require_logical_valid=bool(args.require_logical_valid),
                            logical_mode=str(args.logical_mode),
                        )
                        allowed_by_cutoff[c] = allowed

            if allowed_by_cutoff is None:
                runs_missing_judges += 1
                missing_items_total += n_items
                if args.strict_judge_results:
                    raise SystemExit(
                        f"[FATAL] No judge result files found under source run_dir: {run_dir}"
                    )
                for c in cutoffs:
                    kept_total_by_cutoff[c] += n_items
                    kept_by_generator[generator_name][c] += n_items
                allowed_cache[run_dir] = {c: None for c in cutoffs}
                continue

            runs_with_judges += 1

            for c in cutoffs:
                allowed = allowed_by_cutoff.get(c)
                if allowed is None:
                    if args.strict_judge_results:
                        raise SystemExit(
                            f"[FATAL] Unexpected missing judge results under source run_dir: {run_dir}"
                        )
                    n_kept = n_items
                    allowed_cache.setdefault(run_dir, {})[c] = None
                else:
                    allowed_selected = set(allowed) & selected_qids
                    n_kept = len(allowed_selected)
                    allowed_cache.setdefault(run_dir, {})[c] = allowed_selected

                kept_total_by_cutoff[c] += n_kept
                kept_by_generator[generator_name][c] += n_kept

        if args.accuracy:
            # Accuracy sensitivity uses subset evaluation outputs under the subset runs root.
            subset_eval_run_dirs = _iter_run_dirs(runs_root)
            if not subset_eval_run_dirs:
                acc_missing_models += 1
            else:
                for eval_run_dir in subset_eval_run_dirs:
                    for result_path in _iter_result_paths(eval_run_dir):
                        model = result_path.name[: -len("_result.json")]
                        rows = _load_json_list(result_path)
                        if not rows:
                            continue

                        def include_for_cutoff(row: dict, cutoff: int) -> bool:
                            src_qid = str(row.get("source_question_id") or "").strip()
                            if not src_qid:
                                return False
                            src_run_dir = _localize_repo_path(str(row.get("source_run_dir") or "").strip())
                            if src_run_dir is None:
                                return False
                            selected = selected_by_run_dir.get(src_run_dir)
                            if selected is None or src_qid not in selected:
                                return False

                            per_run = allowed_cache.get(src_run_dir)
                            if per_run is None:
                                # Lazily compute (should be rare; subset results should
                                # only reference source_run_dir values from selection reports).
                                selected = selected_by_run_dir.get(src_run_dir) or set()
                                if bool(args.logical_first):
                                    score_allowed = _score_majority_allowed_by_cutoff(
                                        src_run_dir,
                                        judge_models,
                                        cutoffs=cutoffs,
                                        base_min_score=_LOGICAL_FIRST_BASE_SCORE,
                                    )
                                    if score_allowed is None:
                                        per_run = {c: None for c in cutoffs}
                                    elif bool(args.require_logical_valid):
                                        logical_allowed = filter_by_judge(
                                            src_run_dir,
                                            judge_models,
                                            min_medical_score=_LOGICAL_FIRST_BASE_SCORE,
                                            require_logical_valid=True,
                                            logical_mode=str(args.logical_mode),
                                        )
                                        if logical_allowed is None:
                                            per_run = {c: None for c in cutoffs}
                                        else:
                                            logical_allowed_set = set(logical_allowed)
                                            per_run = {
                                                c: (set(score_allowed.get(c, set())) & logical_allowed_set & selected)
                                                for c in cutoffs
                                            }
                                    else:
                                        per_run = {
                                            c: (set(score_allowed.get(c, set())) & selected)
                                            for c in cutoffs
                                        }
                                else:
                                    per_run = {}
                                    for c in cutoffs:
                                        allowed = filter_by_judge(
                                            src_run_dir,
                                            judge_models,
                                            min_medical_score=c,
                                            require_logical_valid=bool(args.require_logical_valid),
                                            logical_mode=str(args.logical_mode),
                                        )
                                        per_run[c] = (set(allowed) & selected) if allowed is not None else None
                                allowed_cache[src_run_dir] = per_run
                            elif cutoff not in per_run:
                                selected = selected_by_run_dir.get(src_run_dir) or set()
                                if bool(args.logical_first):
                                    # Compute all cutoffs for consistency.
                                    score_allowed = _score_majority_allowed_by_cutoff(
                                        src_run_dir,
                                        judge_models,
                                        cutoffs=cutoffs,
                                        base_min_score=_LOGICAL_FIRST_BASE_SCORE,
                                    )
                                    if score_allowed is None:
                                        for c in cutoffs:
                                            per_run[c] = None
                                    elif bool(args.require_logical_valid):
                                        logical_allowed = filter_by_judge(
                                            src_run_dir,
                                            judge_models,
                                            min_medical_score=_LOGICAL_FIRST_BASE_SCORE,
                                            require_logical_valid=True,
                                            logical_mode=str(args.logical_mode),
                                        )
                                        if logical_allowed is None:
                                            for c in cutoffs:
                                                per_run[c] = None
                                        else:
                                            logical_allowed_set = set(logical_allowed)
                                            for c in cutoffs:
                                                per_run[c] = (
                                                    set(score_allowed.get(c, set())) & logical_allowed_set & selected
                                                )
                                    else:
                                        for c in cutoffs:
                                            per_run[c] = set(score_allowed.get(c, set())) & selected
                                else:
                                    allowed = filter_by_judge(
                                        src_run_dir,
                                        judge_models,
                                        min_medical_score=cutoff,
                                        require_logical_valid=bool(args.require_logical_valid),
                                        logical_mode=str(args.logical_mode),
                                    )
                                    per_run[cutoff] = (set(allowed) & selected) if allowed is not None else None

                            allowed_set = per_run.get(cutoff)
                            if allowed_set is None:
                                return True
                            return src_qid in allowed_set

                        corr_by, wrong_by = _score_rows_by_cutoff(
                            rows,
                            cutoffs=cutoffs,
                            include_row_for_cutoff=include_for_cutoff,
                        )

                        for c in cutoffs:
                            acc_stats[c][model].add_quiz(corr_by[c], wrong_by[c])

    else:
        run_dirs = _iter_run_dirs(runs_root)
        if not run_dirs:
            raise SystemExit(f"[FATAL] No quiz run directories found under: {runs_root}")

        if args.accuracy:
            for rd in run_dirs:
                if _iter_result_paths(rd):
                    break
            else:
                acc_missing_models += 1

        for run_dir in run_dirs:
            generator_name = _infer_generator_model(run_dir, runs_root)
            n_items = _get_total_items_for_run(run_dir)
            total_items_all += n_items
            total_by_generator[generator_name] += n_items

            allowed_by_cutoff: dict[int, set[str] | None] | None

            if bool(args.logical_first):
                score_allowed = _score_majority_allowed_by_cutoff(
                    run_dir,
                    judge_models,
                    cutoffs=cutoffs,
                    base_min_score=_LOGICAL_FIRST_BASE_SCORE,
                )
                if score_allowed is None:
                    allowed_by_cutoff = None
                elif bool(args.require_logical_valid):
                    logical_allowed = filter_by_judge(
                        run_dir,
                        judge_models,
                        min_medical_score=_LOGICAL_FIRST_BASE_SCORE,
                        require_logical_valid=True,
                        logical_mode=str(args.logical_mode),
                    )
                    if logical_allowed is None:
                        allowed_by_cutoff = None
                    else:
                        logical_allowed_set = set(logical_allowed)
                        allowed_by_cutoff = {
                            c: set(score_allowed.get(c, set())) & logical_allowed_set
                            for c in cutoffs
                        }
                else:
                    allowed_by_cutoff = {c: set(score_allowed.get(c, set())) for c in cutoffs}
            else:
                allowed_by_cutoff = {}
                allowed_min = filter_by_judge(
                    run_dir,
                    judge_models,
                    min_medical_score=min_cutoff,
                    require_logical_valid=bool(args.require_logical_valid),
                    logical_mode=str(args.logical_mode),
                )
                if allowed_min is None:
                    allowed_by_cutoff = None
                else:
                    allowed_by_cutoff[min_cutoff] = allowed_min
                    for c in cutoffs:
                        if c == min_cutoff:
                            continue
                        allowed = filter_by_judge(
                            run_dir,
                            judge_models,
                            min_medical_score=c,
                            require_logical_valid=bool(args.require_logical_valid),
                            logical_mode=str(args.logical_mode),
                        )
                        allowed_by_cutoff[c] = allowed

            if allowed_by_cutoff is None:
                runs_missing_judges += 1
                missing_items_total += n_items
                if args.strict_judge_results:
                    raise SystemExit(
                        f"[FATAL] No judge result files found under run_dir: {run_dir}"
                    )
                for c in cutoffs:
                    kept_total_by_cutoff[c] += n_items
                    kept_by_generator[generator_name][c] += n_items
                allowed_by_cutoff = {c: None for c in cutoffs}
                if args.accuracy:
                    # Unfiltered accuracies (no judge outputs).
                    for result_path in _iter_result_paths(run_dir):
                        model = result_path.name[: -len("_result.json")]
                        rows = _load_json_list(result_path)
                        if not rows:
                            continue

                        def include_for_cutoff(_row: dict, _cutoff: int) -> bool:
                            return True

                        corr_by, wrong_by = _score_rows_by_cutoff(
                            rows,
                            cutoffs=cutoffs,
                            include_row_for_cutoff=include_for_cutoff,
                        )
                        for c in cutoffs:
                            acc_stats[c][model].add_quiz(corr_by[c], wrong_by[c])
                continue

            runs_with_judges += 1

            for c in cutoffs:
                allowed = allowed_by_cutoff.get(c)
                if allowed is None:
                    allowed = filter_by_judge(
                        run_dir,
                        judge_models,
                        min_medical_score=c,
                        require_logical_valid=bool(args.require_logical_valid),
                        logical_mode=str(args.logical_mode),
                    )
                    allowed_by_cutoff[c] = allowed

                if allowed is None:
                    if args.strict_judge_results:
                        raise SystemExit(
                            f"[FATAL] Unexpected missing judge results under run_dir: {run_dir}"
                        )
                    n_kept = n_items
                    allowed_by_cutoff[c] = None
                else:
                    n_kept = len(allowed)
                    allowed_by_cutoff[c] = set(allowed)

                kept_total_by_cutoff[c] += n_kept
                kept_by_generator[generator_name][c] += n_kept

            if args.accuracy:
                for result_path in _iter_result_paths(run_dir):
                    model = result_path.name[: -len("_result.json")]
                    rows = _load_json_list(result_path)
                    if not rows:
                        continue

                    def include_for_cutoff(row: dict, cutoff: int) -> bool:
                        qid = row.get("question_id")
                        if not isinstance(qid, str) or not qid:
                            return False
                        allowed_set = allowed_by_cutoff.get(cutoff)
                        if allowed_set is None:
                            return True
                        return qid in allowed_set

                    corr_by, wrong_by = _score_rows_by_cutoff(
                        rows,
                        cutoffs=cutoffs,
                        include_row_for_cutoff=include_for_cutoff,
                    )
                    for c in cutoffs:
                        acc_stats[c][model].add_quiz(corr_by[c], wrong_by[c])

    if total_items_all <= 0:
        raise SystemExit(
            "[FATAL] Could not infer total question counts from run directories "
            "(missing manifest.json and *_summary.json n_items)."
        )

    print(f"[INFO] runs_root={runs_root}")
    if args.use_selection_reports:
        print("[INFO] mode=subset_selection_reports")
    print(f"[INFO] logical_first={bool(args.logical_first)}")
    if args.use_selection_reports:
        eval_dirs = subset_eval_run_dirs
        if eval_dirs is None:
            eval_dirs = _iter_run_dirs(runs_root)
        print(
            f"[INFO] source_run_dirs={len(run_dirs)} subset_eval_run_dirs={len(eval_dirs)}"
        )
    else:
        print(f"[INFO] run_dirs={len(run_dirs)}")
    print(f"[INFO] runs_with_judges={runs_with_judges} runs_missing_judges={runs_missing_judges}")
    if runs_missing_judges:
        print(
            f"[WARN] Missing judge results for {runs_missing_judges} run(s) "
            f"({missing_items_total} item(s)); counts fall back to unfiltered totals."
        )

    # Summary table (overall)
    print("\nCutoff  KeptQuestions  TotalQuestions  Kept%")
    print("-------------------------------------------")
    for c in cutoffs:
        kept = int(kept_total_by_cutoff.get(c, 0) or 0)
        pct = (kept / float(total_items_all)) * 100.0 if total_items_all > 0 else float("nan")
        print(f"{c:>6}  {kept:>13}  {total_items_all:>14}  {pct:5.1f}%")
    if (
        bool(args.require_logical_valid)
        and str(args.logical_mode) == "majority"
        and not bool(args.logical_first)
    ):
        print(
            "\n[NOTE] In logical_mode=majority, kept counts are not guaranteed to be monotonic "
            "in the cutoff: increasing the cutoff can exclude low-scoring judge votes from the "
            "majority denominator, which can cause some questions to flip from FAIL to PASS."
        )

    if total_by_generator:
        expected_total = 50 if args.use_selection_reports else None
        _warn_if_unexpected_total_per_generator(
            total_by_generator=dict(total_by_generator),
            expected_total=expected_total,
        )
        _print_generator_question_availability_table(
            cutoffs=cutoffs,
            total_by_generator=dict(total_by_generator),
            kept_by_generator=dict(kept_by_generator),
            sort_cutoff=max(cutoffs),
            total_label="TotalSelected" if args.use_selection_reports else "TotalQuestions",
        )

    rows = []
    for c in cutoffs:
        kept = int(kept_total_by_cutoff.get(c, 0) or 0)
        pct = (kept / float(total_items_all)) * 100.0 if total_items_all > 0 else float("nan")
        rows.append(
            {
                "cutoff": c,
                "kept_questions": kept,
                "total_questions": total_items_all,
                "kept_pct": pct,
                "run_dirs": len(run_dirs),
                "runs_with_judges": runs_with_judges,
                "runs_missing_judges": runs_missing_judges,
                "missing_items_total": missing_items_total,
                "logical_mode": args.logical_mode,
                "require_logical_valid": bool(args.require_logical_valid),
                "logical_first": bool(args.logical_first),
                "judge_models": ",".join(judge_models),
                "mode": "subset_selection_reports" if args.use_selection_reports else "runs_root",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(out_csv, rows, fieldnames)
    print(f"\n[OK] Wrote {out_csv}")

    if args.out_item_csv:
        out_item_csv = Path(args.out_item_csv).expanduser()
        if not item_rows:
            print(
                f"[WARN] --out_item_csv was set but no item-level rows were collected; "
                f"not writing {out_item_csv}"
            )
        else:
            item_fieldnames = list(item_rows[0].keys())
            _write_csv(out_item_csv, item_rows, item_fieldnames)
            print(f"[OK] Wrote {out_item_csv}")

    if args.plot:
        kept_counts = [int(kept_total_by_cutoff.get(c, 0) or 0) for c in cutoffs]
        _save_plot(
            out_png,
            cutoffs=cutoffs,
            kept_counts=kept_counts,
            total_items=total_items_all,
            title_label=label,
            require_matplotlib=bool(args.require_plot),
        )
        if out_png.exists():
            print(f"[OK] Wrote {out_png}")
        if total_by_generator:
            _save_generator_availability_heatmap(
                out_gen_png,
                cutoffs=cutoffs,
                total_by_generator=dict(total_by_generator),
                kept_by_generator=dict(kept_by_generator),
                title_label=label,
                require_matplotlib=bool(args.require_plot),
            )
            if out_gen_png.exists():
                print(f"[OK] Wrote {out_gen_png}")

    if args.accuracy:
        if acc_missing_models:
            print(
                "[WARN] No non-judge *_result.json files found for model accuracy sensitivity; "
                "skipping accuracy outputs."
            )
            return

        out_acc_csv = (
            Path(args.out_accuracy_csv).expanduser()
            if args.out_accuracy_csv
            else out_csv.with_name(f"{out_csv.stem}_model_accuracy.csv")
        )
        out_acc_png = (
            Path(args.out_accuracy_png).expanduser()
            if args.out_accuracy_png
            else out_csv.with_name(f"{out_csv.stem}_model_accuracy.png")
        )

        acc_rows: list[dict] = []
        for model in sorted({m for c in cutoffs for m in acc_stats.get(c, {})}):
            for c in cutoffs:
                stats = acc_stats.get(c, {}).get(model)
                if stats is None:
                    continue
                mean, sem = stats.mean_sem()
                acc_rows.append(
                    {
                        "model": model,
                        "cutoff": c,
                        "num_quizzes": len(stats.quiz_acc),
                        "mean_accuracy": mean,
                        "sem": sem,
                        "micro_accuracy": stats.micro_accuracy(),
                        "total_items": stats.total_items,
                        "logical_mode": args.logical_mode,
                        "require_logical_valid": bool(args.require_logical_valid),
                        "logical_first": bool(args.logical_first),
                        "judge_models": ",".join(judge_models),
                        "mode": "subset_selection_reports" if args.use_selection_reports else "runs_root",
                    }
                )

        if not acc_rows:
            print(
                "[WARN] No accuracy rows were computed; skipping accuracy outputs. "
                "This usually indicates missing/empty *_result.json files."
            )
            return

        _write_csv(out_acc_csv, acc_rows, list(acc_rows[0].keys()))
        print(f"[OK] Wrote {out_acc_csv}")

        if args.plot:
            # Build heatmap matrix (mean accuracy across quizzes).
            model_names = sorted(
                {row["model"] for row in acc_rows if row.get("cutoff") == min_cutoff}
                | {row["model"] for row in acc_rows}
            )
            if not model_names:
                print("[WARN] No models found for accuracy heatmap; skipping PNG output.")
                return
            mean_by_model_cutoff: dict[tuple[str, int], float] = {}
            for row in acc_rows:
                try:
                    model = str(row["model"])
                    cutoff = int(row["cutoff"])
                    mean_val = float(row["mean_accuracy"])
                except Exception:
                    continue
                mean_by_model_cutoff[(model, cutoff)] = mean_val

            def _sort_key(name: str) -> tuple[float, str]:
                v = mean_by_model_cutoff.get((name, min_cutoff), float("nan"))
                if isinstance(v, float) and math.isnan(v):
                    return (float("inf"), name)
                return (-v, name)

            model_names = sorted(model_names, key=_sort_key)
            values: list[list[float]] = []
            for m in model_names:
                row_vals: list[float] = []
                for c in cutoffs:
                    row_vals.append(mean_by_model_cutoff.get((m, c), float("nan")))
                values.append(row_vals)

            _save_accuracy_heatmap(
                out_acc_png,
                cutoffs=cutoffs,
                model_names=model_names,
                values=values,
                title_label=label,
                require_matplotlib=bool(args.require_plot),
            )
            if out_acc_png.exists():
                print(f"[OK] Wrote {out_acc_png}")


if __name__ == "__main__":
    main()
