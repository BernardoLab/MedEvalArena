#!/usr/bin/env python3
import argparse, os, json, math, glob, csv, fnmatch, sys
from collections import defaultdict
from pathlib import Path
import colorcet as cc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES, filter_by_judge


def load_extra_model_results(path: str) -> list[dict]:
    """
    Load manually specified per-generator model accuracies from a JSON file.

    Expected schema (all accuracies as fractions in [0, 1]), where each
    row represents the accuracy of a model on all questions for a given
    quiz generator (aggregated across runs/seeds as desired):

    {
      "results": [
        {
          "quiz_generator": "<generator name>",
          "model": "<answer model name to report>",
          "accuracy": 0.83,
          "n_items": 150        # optional, ignored by this loader
        },
        ...
      ]
    }

    The `quiz_generator` should match the generator name used in this
    benchmark (e.g., 'claude-opus-4-5-20251101', 'gpt-5.1-2025-11-13').
    It will be canonicalized in the same way as other generators via
    --generator_family_mode.
    """
    if not path:
        return []
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read extra model results JSON '{path}': {exc}")
        return []

    results = data.get("results")
    if not isinstance(results, list):
        print(
            f"[WARN] Extra model results JSON '{path}' "
            "does not contain a top-level 'results' list."
        )
        return []

    out: list[dict] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        generator_raw = row.get("quiz_generator")
        model = row.get("model")
        acc = row.get("accuracy")
        if not isinstance(generator_raw, str) or not isinstance(model, str):
            continue
        try:
            acc_f = float(acc)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "generator_raw": generator_raw,
                "model": model,
                "accuracy": acc_f,
            }
        )

    return out


def load_quiz_model_acc(run_dir: str):
    """Return dict model->acc for a single quiz run directory."""
    out = {}
    for p in glob.glob(os.path.join(run_dir, "*_summary.json")):
        base = os.path.basename(p)
        # Ignore judge summaries here; they don't contain model accuracies
        # and they reuse the same `model` field, which can overwrite the
        # true taker-model accuracy with 0.0 when `acc` is missing.
        if base.endswith("_judge_summary.json") or base == "summary_all_judges.json":
            continue
        with open(p, "r") as f:
            s = json.load(f)
        model = s["model"]
        out[model] = float(s.get("acc", 0.0))
    return out


def get_num_items_for_run(run_dir: str) -> int:
    """
    Return the number of items in a quiz run, based on any *_summary.json file.
    """
    # 1) Prefer quiz-taker summaries (they contain taker accuracy + n_items).
    for p in glob.glob(os.path.join(run_dir, "*_summary.json")):
        base = os.path.basename(p)
        # Prefer per-model summaries; judge summaries have identical n_items
        # but are conceptually separate from taker accuracy summaries.
        if base.endswith("_judge_summary.json") or base == "summary_all_judges.json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        n_items = s.get("n_items")
        try:
            return int(n_items or 0)
        except (TypeError, ValueError):
            return 0

    # 2) Fallback for "judge-only" runs (before quizbench.run_eval): use any
    # per-judge summary's n_items.
    for p in sorted(glob.glob(os.path.join(run_dir, "*_judge_summary.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        if not isinstance(s, dict):
            continue
        n_items = s.get("n_items")
        try:
            return int(n_items or 0)
        except (TypeError, ValueError):
            return 0

    # 3) Fallback: summary_all_judges.json is a list of judge summaries.
    all_judges_path = os.path.join(run_dir, "summary_all_judges.json")
    if os.path.exists(all_judges_path):
        try:
            with open(all_judges_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = None
        if isinstance(rows, list) and rows:
            first = rows[0]
            if isinstance(first, dict):
                n_items = first.get("n_items")
                try:
                    return int(n_items or 0)
                except (TypeError, ValueError):
                    return 0
    return 0


def print_validity_summary(
    valid_counts: dict[str, int],
    total_counts: dict[str, int],
    *,
    label: str = "Validity summary (judge-filtered)",
) -> None:
    """
    Print per-generator valid/total counts and percentages.

    Useful when running aggregation before quiz-taker eval outputs exist, since
    judge summaries still provide n_items and judge_result files provide the set
    of passing question_ids.
    """
    generators = sorted(set(total_counts) | set(valid_counts))
    rows: list[tuple[str, int, int, float]] = []
    for gen in generators:
        total = int(total_counts.get(gen, 0) or 0)
        valid = int(valid_counts.get(gen, 0) or 0)
        if total <= 0:
            continue
        pct = (valid / total) * 100.0
        rows.append((gen, total, valid, pct))

    if not rows:
        print("[WARN] No validity counts available to summarize.")
        return

    total_all = int(sum(t for _, t, _, _ in rows))
    valid_all = int(sum(v for _, _, v, _ in rows))
    pct_all = (valid_all / total_all) * 100.0 if total_all > 0 else float("nan")

    print(f"\n[INFO] {label}")
    for gen, total, valid, pct in rows:
        print(f"  {gen}: valid={valid} / total={total} ({pct:.1f}%)")
    print(f"  TOTAL: valid={valid_all} / total={total_all} ({pct_all:.1f}%)\n")


def load_quiz_model_acc_filtered(
    run_dir: str,
    judge_models: list[str],
    min_medical_score: int | None,
    require_logical_valid: bool,
    logical_mode: str = "all",
):
    """
    Return dict model->acc for a single quiz run directory, after filtering
    questions by ensemble judge decisions when available.

    If no judge_result files are present for any of judge_models in this run,
    falls back to unfiltered accuracies from *_summary.json and uses the total
    number of items as the valid question count.

    Returns (model->acc mapping, n_valid_questions_for_run).
    """
    allowed_qids = filter_by_judge(
        Path(run_dir),
        judge_models,
        min_medical_score=min_medical_score,
        require_logical_valid=require_logical_valid,
        logical_mode=logical_mode,
    )

    if allowed_qids is None:
        acc_map = load_quiz_model_acc(run_dir)
        return acc_map, get_num_items_for_run(run_dir)

    out: dict[str, float] = {}

    for result_path in glob.glob(os.path.join(run_dir, "*_result.json")):
        base = os.path.basename(result_path)
        if base.endswith("_judge_result.json"):
            continue

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            continue

        if not isinstance(rows, list):
            continue

        corr = 0
        wrong = 0

        base = os.path.basename(result_path)

        for row in rows:
            qid = row.get("question_id")
            if qid not in allowed_qids:
                continue

            gold = str(row.get("answer", "")).strip()
            if not gold:
                continue

            pred = row.get("pred")
            if pred is None:
                continue

            if pred == gold:
                corr += 1
            else:
                wrong += 1

        total = corr + wrong
        if total <= 0:
            continue

        acc = corr / total
        if base.endswith("_result.json"):
            model = base[: -len("_result.json")]
        else:
            model = base

        out[model] = acc

    return out, len(allowed_qids)


def mean_sem(xs):
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(xs) / n
    if n == 1:
        return (mean, float("nan"))
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sem = (var ** 0.5) / (n ** 0.5)
    return (mean, sem)


def _select_manifest_paths_for_batch_tag(runs_root: str, quiz_batch_tag: str) -> list[str]:
    """
    Find quizbench manifests matching a given batch tag.

    For aggregation, we treat the batch tag as a prefix and include *all*
    manifests matching `quizbench_manifest_<TAG>*.json` under `runs_root`
    (recursively). This allows iterative workflows to accumulate multiple
    manifests (e.g., `..._Jan2026.json`, `..._Jan2026v2.json`, timestamped
    suffixes, etc.) while still producing union totals without double-counting
    quiz_ids.

    Note: Judge scripts often select the newest manifest for execution, but
    aggregation typically wants the union of all manifests in the collection.
    """
    tag = (quiz_batch_tag or "").strip()
    if not tag:
        return []

    pattern = f"quizbench_manifest_{tag}*.json"
    candidates = glob.glob(os.path.join(runs_root, "**", pattern), recursive=True)
    if not candidates:
        return []
    return sorted(set(candidates))


def _select_legacy_manifest_paths(runs_root: str) -> list[str]:
    """
    Find legacy, untagged quizbench manifests under runs_root.

    Historically, QuizBench wrote `quizbench_manifest.json` without a batch tag.
    In newer workflows, tagged manifests (e.g., `quizbench_manifest_Jan2026.json`)
    can coexist with these legacy manifests in the same runs tree.
    """
    candidates: list[str] = []

    # Some older workflows wrote a single manifest at the runs root.
    root_manifest = os.path.join(runs_root, "quizbench_manifest.json")
    if os.path.exists(root_manifest):
        candidates.append(root_manifest)

    # More commonly, the legacy manifest lives in each generator directory.
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


def load_quizbench_manifest_generators(runs_root: str, *, quiz_batch_tag: str | None = None):
    """
    Return mapping quiz_id -> generator_model from any quizbench_manifest*.json under runs_root.
    """
    if quiz_batch_tag:
        manifest_paths = _select_manifest_paths_for_batch_tag(runs_root, quiz_batch_tag)
        # Also include legacy manifests created without a batch tag so totals and
        # validity counts reflect the union of legacy + tagged quiz sets.
        manifest_paths.extend(_select_legacy_manifest_paths(runs_root))
        manifest_paths = sorted(set(manifest_paths))
        print("    [DEBUG]:", manifest_paths)
    else:
        manifest_paths = glob.glob(
            os.path.join(runs_root, "**", "quizbench_manifest*.json"), recursive=True
        )
    mapping: dict[str, str] = {}

    for manifest_path in manifest_paths:
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            continue

        quizzes = manifest.get("quizzes") or []
        for q in quizzes:
            quiz_id = q.get("quiz_id")
            gen = q.get("generator_model") or q.get("generator")
            if isinstance(quiz_id, str) and isinstance(gen, str):
                existing = mapping.get(quiz_id)
                if existing and existing != gen:
                    print(
                        f"[WARN] Conflicting generator for quiz_id={quiz_id!r} "
                        f"across manifests: {existing!r} vs {gen!r}; keeping {existing!r}."
                    )
                    continue
                mapping[quiz_id] = gen

    print("    [DEBUG]:", mapping)
    return mapping


def infer_generator_from_run_dir(run_dir: str, quiz_generators: dict[str, str]) -> str:
    """
    Infer the generator model string for a given run directory.

    Preference order:
    1) quizbench_manifest.json mapping (quiz_id -> generator_model)
    2) Parse from run_dir basename, which is expected to match quiz_id created by
       quizbench.run_gen_quiz.build_quiz_id, i.e.:
           <timestamp>_<sanitized_generator_model>_seed<seed>
    """
    base = os.path.basename(run_dir.rstrip(os.sep))

    # 1) Manifest mapping
    gen = quiz_generators.get(base)
    if isinstance(gen, str) and gen:
        return gen

    # 2) Heuristic from naming convention
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].startswith("seed"):
        gen_name = "_".join(parts[1:-1])
        if gen_name:
            return gen_name

    return "unknown"


def canonicalize_generator_model(name: str, mode: str = "family") -> str:
    """
    Collapse generator names into provider families when mode == 'family'.
    """
    raw = name or "unknown"
    if mode == "exact":
        return raw

    low = raw.lower()
    if low.startswith("gpt-") or low.startswith("openai-"):
        return "openai"
    if low.startswith("claude-") or low.startswith("anthropic"):
        return "anthropic"
    if low.startswith("gemini-") or low.startswith("google-"):
        return "gemini"
    return raw


def find_run_dirs(runs_root: str, run_dir_glob: str):
    """
    Recursively find run directories (contain *_summary.json) matching the glob.
    """
    matches = []
    for dirpath, _, _ in os.walk(runs_root):
        base = os.path.basename(dirpath.rstrip(os.sep))
        if not fnmatch.fnmatch(base, run_dir_glob):
            continue
        if glob.glob(os.path.join(dirpath, "*_summary.json")):
            matches.append(dirpath)
    return sorted(matches)


def write_csv(path: str, fieldnames: list[str], rows: list[dict]):
    """Write rows to CSV, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def color_for_value(value: float, vmin: float, vmax: float) -> str | None:
    """
    Map a value in [vmin, vmax] to a background ANSI color code.

    We use a simple 5-level red->green palette suitable for terminals that
    support standard ANSI colors:
        41: red (low accuracy)
        101: bright red
        103: yellow
        102: bright green
        42: green (high accuracy)
    """
    if math.isnan(value) or math.isnan(vmin) or math.isnan(vmax):
        return None
    if vmax <= vmin:
        return None

    value = max(vmin, min(vmax, value))
    t = (value - vmin) / (vmax - vmin)
    idx = max(0, min(4, int(round(t * 4))))
    palette = [41, 101, 103, 102, 42]
    return f"\x1b[{palette[idx]}m"


def format_heatmap_cell(
    text: str, value: float, vmin: float, vmax: float, width: int
) -> str:
    """
    Return a fixed-width string for a matrix cell with an ANSI heatmap background.
    """
    padded = text.center(width)
    color = color_for_value(value, vmin, vmax)
    if color is None:
        return padded
    reset = "\x1b[0m"
    return f"{color}{padded}{reset}"


def print_accuracy_matrix(
    scores: dict[tuple[str, str], list[float]],
    row_counts: dict[str, int] | None = None,
    restrict_answer_models_to_generators: bool = True,
) -> None:
    """
    Print a generator x answer-model accuracy matrix with a heatmap in the terminal.

    When row_counts is provided, each generator row label includes the number of
    valid questions in parentheses.
    """
    if not scores:
        return

    # Aggregate across all generators to support a summary row/column.
    aggregated_by_model: dict[str, list[float]] = defaultdict(list)
    aggregated_by_generator: dict[str, list[float]] = defaultdict(list)
    for (gen, model), arr in scores.items():
        aggregated_by_model[model].extend(arr)
        aggregated_by_generator[gen].extend(arr)

    generators = sorted({g for (g, _) in scores.keys()})
    models = sorted({m for (_, m) in scores.keys()})

    if restrict_answer_models_to_generators:
        generator_set = set(generators)
        models = [m for m in models if m in generator_set]

    # Build labels that optionally include the number of valid questions.
    generator_labels: dict[str, str] = {}
    for g in generators:
        if row_counts and g in row_counts:
            generator_labels[g] = f"{g} ({row_counts[g]})"
        else:
            generator_labels[g] = g

    # Compute mean accuracies and range for coloring.
    means: dict[tuple[str, str], float] = {}
    values: list[float] = []
    for key, arr in scores.items():
        mu, _ = mean_sem(arr)
        means[key] = mu
        if not math.isnan(mu):
            values.append(mu)

    summary_means: dict[str, float] = {}
    summary_values: list[float] = []
    for m in models:
        mu, _ = mean_sem(aggregated_by_model.get(m, []))
        summary_means[m] = mu
        if not math.isnan(mu):
            summary_values.append(mu)

    generator_means: dict[str, float] = {}
    generator_values: list[float] = []
    for g in generators:
        mu, _ = mean_sem(aggregated_by_generator.get(g, []))
        generator_means[g] = mu
        if not math.isnan(mu):
            generator_values.append(mu)

    all_scores: list[float] = []
    for arr in scores.values():
        all_scores.extend(arr)
    overall_mean, _ = mean_sem(all_scores)

    all_values = values + summary_values + generator_values
    if not math.isnan(overall_mean):
        all_values.append(overall_mean)
    if not all_values:
        return

    vmin = min(all_values)
    vmax = max(all_values)

    row_label_header = "Generator (rows ↓)"
    summary_label = "Average (all generators)"
    summary_col_label = "Average (all answer models)"
    row_label_width = max(
        14,
        len(row_label_header),
        max(len(lbl) for lbl in generator_labels.values()) + 2,
        len(summary_label) + 2,
    )
    col_width = max(9, max(len(m) for m in models + [summary_col_label]) + 2)

    print("\nGenerator x Answer Model Accuracy Matrix")
    print("Rows: Generator model (↓)")
    print("Columns: Answer model (→)\n")

    horizontal_border = (
        "+" + "-" * row_label_width + "+"
        + "+".join("-" * col_width for _ in range(len(models) + 1)) + "+"
    )
    header_cells = "|".join(m.center(col_width) for m in models + [summary_col_label])
    header = f"|{row_label_header.ljust(row_label_width)}|{header_cells}|"

    print(horizontal_border)
    print(header)
    print(horizontal_border)

    for g in generators:
        row_cells = []
        for m in models:
            mu = means.get((g, m), float("nan"))
            if math.isnan(mu):
                cell = " " * col_width
            else:
                pct = mu * 100.0 if mu <= 1.0 else mu
                label = f"{pct:5.1f}"
                cell = format_heatmap_cell(label, mu, vmin, vmax, col_width)
            row_cells.append(cell)
        summary_mu = generator_means.get(g, float("nan"))
        if math.isnan(summary_mu):
            summary_cell = " " * col_width
        else:
            pct = summary_mu * 100.0 if summary_mu <= 1.0 else summary_mu
            label = f"{pct:5.1f}"
            summary_cell = format_heatmap_cell(label, summary_mu, vmin, vmax, col_width)
        label = generator_labels.get(g, g)
        print(f"|{label.ljust(row_label_width)}|{'|'.join(row_cells + [summary_cell])}|")

    print(horizontal_border)

    # Summary row across all generators.
    summary_cells = []
    for m in models:
        mu = summary_means.get(m, float("nan"))
        if math.isnan(mu):
            cell = " " * col_width
        else:
            pct = mu * 100.0 if mu <= 1.0 else mu
            label = f"{pct:5.1f}"
            cell = format_heatmap_cell(label, mu, vmin, vmax, col_width)
        summary_cells.append(cell)
    if math.isnan(overall_mean):
        overall_cell = " " * col_width
    else:
        pct = overall_mean * 100.0 if overall_mean <= 1.0 else overall_mean
        label = f"{pct:5.1f}"
        overall_cell = format_heatmap_cell(label, overall_mean, vmin, vmax, col_width)

    print(f"|{summary_label.ljust(row_label_width)}|{'|'.join(summary_cells + [overall_cell])}|")

    print(horizontal_border)


def save_accuracy_heatmap(
    scores: dict[tuple[str, str], list[float]],
    out_path: str,
    title: str | None = None,
    row_counts: dict[str, int] | None = None,
    restrict_answer_models_to_generators: bool = True,
) -> None:
    """
    Save a matplotlib + seaborn heatmap of the generator x answer-model matrix,
    including summary averages across generators (row) and answer models (column).

    The figure is written to disk and not displayed.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib/seaborn for heatmap: {exc}")
        return

    if not scores or not out_path:
        return

    generators = sorted({g for (g, _) in scores.keys()})
    models = sorted({m for (_, m) in scores.keys()})
    if not generators or not models:
        return

    aggregated_by_model: dict[str, list[float]] = defaultdict(list)
    aggregated_by_generator: dict[str, list[float]] = defaultdict(list)
    for (gen, model), arr in scores.items():
        aggregated_by_model[model].extend(arr)
        aggregated_by_generator[gen].extend(arr)

    # Build matrix of mean accuracies, converted to percentages for readability.
    data: list[list[float]] = []
    values: list[float] = []
    summary_col_label = "Average (all answer models)"
    for g in generators:
        row: list[float] = []
        for m in models:
            arr = scores.get((g, m), [])
            if not arr:
                row.append(float("nan"))
                continue
            mu, _ = mean_sem(arr)
            if math.isnan(mu):
                row.append(float("nan"))
            else:
                pct = mu * 100.0 if mu <= 1.0 else mu
                row.append(pct)
                values.append(pct)
        row_mu, _ = mean_sem(aggregated_by_generator.get(g, []))
        if math.isnan(row_mu):
            row.append(float("nan"))
        else:
            pct = row_mu * 100.0 if row_mu <= 1.0 else row_mu
            row.append(pct)
            values.append(pct)
        data.append(row)

    # Summary row across generators.
    summary_row: list[float] = []
    for m in models:
        mu, _ = mean_sem(aggregated_by_model.get(m, []))
        if math.isnan(mu):
            summary_row.append(float("nan"))
        else:
            pct = mu * 100.0 if mu <= 1.0 else mu
            summary_row.append(pct)
            values.append(pct)
    all_scores: list[float] = []
    for arr in scores.values():
        all_scores.extend(arr)
    overall_mu, _ = mean_sem(all_scores)
    if math.isnan(overall_mu):
        summary_row.append(float("nan"))
    else:
        pct = overall_mu * 100.0 if overall_mu <= 1.0 else overall_mu
        summary_row.append(pct)
        values.append(pct)
    data.append(summary_row)
    generators.append("Average (all generators)")
    models.append(summary_col_label)

    if not values:
        return

    vmin = min(values)
    vmax = max(values)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build y-axis labels, optionally including valid question counts.
    generator_labels: list[str] = []
    for g in generators:
        if row_counts and g in row_counts:
            generator_labels.append(f"{g} ({row_counts[g]})")
        else:
            generator_labels.append(g)

    # Figure size scales with matrix size for readability.
    cell_size = 1.1
    fig_width = max(8.0, cell_size * len(models))
    fig_height = max(6.0, cell_size * len(generators))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        data,
        ax=ax,
        cmap=cc.cm['CET_R2_r'],
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.5},
        vmin=vmin,
        vmax=vmax,
        xticklabels=models,
        yticklabels=generator_labels,
        square=True,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Answer model")
    ax.set_ylabel("Generator model")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    heatmap_dpi = 300
    fig.savefig(out_path, dpi=heatmap_dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_path)

def save_validity_barplot(
    valid_counts: dict[str, int],
    total_counts: dict[str, int],
    out_path: str,
    title: str | None = None,
) -> None:
    """
    Save a bar plot showing the fraction of questions that passed validity filtering
    for each generator (y-axis formatted as percent), styled to resemble ggplot2
    theme_minimal() + the gridline tweaks in the provided R example.
    """
    import os

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter, MultipleLocator
        import colorcet as cc
    except Exception as exc:
        print(f"[WARN] Could not import plotting deps for validity barplot: {exc}")
        return

    if not out_path:
        return

    # ---- Build data ----
    generators = sorted(set(total_counts) | set(valid_counts))
    rows: list[dict[str, float]] = []
    for gen in generators:
        total = int(total_counts.get(gen, 0) or 0)
        valid = int(valid_counts.get(gen, 0) or 0)
        if total <= 0:
            continue
        frac = valid / total  # 0..1
        rows.append(
            {
                "generator_model": gen,
                "frac_valid": frac,
                "percent_valid": frac * 100.0,
                "valid": valid,
                "total": total,
            }
        )

    if not rows:
        print("[WARN] No validity counts available to plot.")
        return

    # Match your original ordering (best to worst), which also matches a typical ggplot ordering.
    rows.sort(key=lambda r: (-r["frac_valid"], r["generator_model"]))
    labels = [r["generator_model"] for r in rows]
    fracs = [float(r["frac_valid"]) for r in rows]
    percents = [float(r["percent_valid"]) for r in rows]

    # ---- Colors: discrete palette from ColorCET R2 (like ggplot scale_fill_manual) ----
    # If you want reversed, change to "CET_R2_r".
    cmap = cc.cm["CET_R2"]
    n = len(labels)
    if n <= 1:
        t_vals = [0.5]
    else:
        # Sample away from the extreme endpoints for nicer fills (manual-like).
        t_vals = [0.10 + 0.80 * (i / (n - 1)) for i in range(n)]
    bar_cols = [cmap(t) for t in t_vals]

    # ---- Figure sizing ----
    fig_width = max(6.0, 1.10 * n)
    fig_height = 5.5

    # Use rc_context so we don't permanently mutate global matplotlib defaults.
    with plt.rc_context(
        {
            "font.size": 16,         # ggplot theme_minimal(base_size=16)
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    ):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        x = list(range(n))
        ax.bar(x, fracs, width=0.82, color=bar_cols, edgecolor="none", linewidth=0)

        # ---- Labels / title ----
        ax.set_xlabel("")  # labs(x = NULL)
        ax.set_ylabel("Questions passing (%)")
        if title:
            ax.set_title(title, loc="center")  # plot.title hjust=0.5

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", va="top")

        # ---- Y scale: 0..1 with percent labels, plus ~6% top expansion ----
        ax.set_ylim(0.0, 1.06)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0, symbol=''))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        # ---- Gridlines to match the R theme tweaks ----
        ax.set_axisbelow(True)
        ax.grid(False, axis="x")  # panel.grid.major.x/minor.x blank
        ax.grid(True, axis="y", which="major", color="#B3B3B3", linewidth=0.6)
        ax.grid(True, axis="y", which="minor", color="#B3B3B3", linewidth=0.6)

        # Minimal look: remove spines + tick marks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        # ---- Value labels above bars (kept from your original, but adjusted for 0..1 axis) ----
        label_offset = 0.012
        for i, (frac, pct) in enumerate(zip(fracs, percents)):
            on_top = False
            if on_top:
                ax.text(
                    i,
                    min(frac + label_offset, 1.055),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            else:
                ax.text(
                    i,
                    0.28,
                    f"{pct:.1f}",
                    color='white',
                    ha="center",
                    va="bottom",
                    fontsize=15,
                )       

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(out_path)


# def save_validity_barplot(
#     valid_counts: dict[str, int],
#     total_counts: dict[str, int],
#     out_path: str,
#     title: str | None = None,
# ) -> None:
#     """
#     Save a seaborn barplot showing the percentage of questions that passed
#     validity filtering for each generator.
#     """
#     try:
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         from matplotlib.colors import Normalize
#     except Exception as exc:
#         print(f"[WARN] Could not import matplotlib/seaborn for validity barplot: {exc}")
#         return

#     if not out_path:
#         return

#     generators = sorted(set(total_counts) | set(valid_counts))
#     rows: list[dict[str, float]] = []
#     for gen in generators:
#         total = int(total_counts.get(gen, 0) or 0)
#         valid = int(valid_counts.get(gen, 0) or 0)
#         if total <= 0:
#             continue
#         pct = (valid / total) * 100.0
#         rows.append(
#             {
#                 "generator_model": gen,
#                 "percent_valid": pct,
#                 "valid": valid,
#                 "total": total,
#             }
#         )

#     if not rows:
#         print("[WARN] No validity counts available to plot.")
#         return

#     rows.sort(key=lambda r: (-r["percent_valid"], r["generator_model"]))
#     percents = [r["percent_valid"] for r in rows]
#     labels = [r["generator_model"] for r in rows]

#     vmin = min(percents)
#     vmax = max(percents)
#     if vmax <= vmin:
#         vmax = vmin + 1e-6
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = cc.cm["CET_R2_r"]
#     palette = [cmap(norm(p)) for p in percents]

#     fig_width = max(6.0, 1.1 * len(labels))
#     fig_height = 5.0
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     sns.barplot(
#         x=labels,
#         y=percents,
#         palette=palette,
#         edgecolor="white",
#         linewidth=0.7,
#         ax=ax,
#         errorbar=None,
#     )

#     ax.set_ylabel("Questions passing validity (%)")
#     ax.set_xlabel("Generator model")
#     if title:
#         ax.set_title(title)

#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     ymax = max(percents)
#     y_max_lim = min(105.0, ymax * 1.08 + 2.0)
#     ax.set_ylim(0, max(y_max_lim, 5.0))
#     label_offset = max(0.5, (ax.get_ylim()[1] - ymax) * 0.15)

#     for idx, pct in enumerate(percents):
#         ax.text(
#             idx,
#             pct + label_offset,
#             f"{pct:.1f}%",
#             ha="center",
#             va="bottom",
#             fontsize=9,
#         )

#     fig.tight_layout()
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="eval_results/quizbench/runs")
    ap.add_argument(
        "--quiz_batch_tag",
        type=str,
        default=None,
        help=(
            "Optional quiz batch tag (e.g., Jan2026). When set, only include "
            "quizzes listed in any quizbench_manifest_<TAG>*.json discovered "
            "under --runs_root (recursively), plus any legacy "
            "`quizbench_manifest.json` files (created without a tag) under the "
            "same --runs_root."
        ),
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="eval_results/quizbench/quizbench_summary.csv",
    )
    ap.add_argument(
        "--out_csv_by_model",
        type=str,
        default="eval_results/quizbench/quizbench_summary_by_model.csv",
        help="Also write an aggregated CSV grouped only by the answer model across generators.",
    )
    ap.add_argument(
        "--run_dir_glob",
        type=str,
        default="*",
        help=(
            "Glob pattern (joined to runs_root) to find per-quiz run dirs. "
            "Use '*' to include all."
        ),
    )
    ap.add_argument(
        "--generator_family_mode",
        choices=["family", "exact"],
        default="exact",
        help="Group generators by provider family (openai/anthropic/gemini) or keep exact names.",
    )
    ap.add_argument(
        "--out_heatmap",
        type=str,
        default="eval_results/quizbench/quizbench_accuracy_heatmap.png",
        help="Path to save a matplotlib + seaborn heatmap of the generator x answer-model accuracy.",
    )
    ap.add_argument(
        "--out_filtered_barplot",
        type=str,
        default="eval_results/quizbench/filtered_barp.png",
        help="Path to save a seaborn barplot of the percent of questions passing validity filters.",
    )
    ap.add_argument(
        "--filter_by_judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable per-question filtering based on ensemble judge decisions. "
            "When disabled, uses unfiltered per-quiz accuracies."
        ),
    )
    ap.add_argument(
        "--filter_judges",
        nargs="+",
        default=DEFAULT_ENSEMBLE_JUDGES,
        help=(
            "Ensemble of judge models to use when filtering questions. "
            "If none of these judges have results for a run, that run is left unfiltered."
        ),
    )
    ap.add_argument(
        "--filter_min_med_score",
        type=int,
        default=3,
        help=(
            "Minimum medical_accuracy_score (1–5) required for a question to be kept. "
            "Set to 1 to disable score-based exclusion."
        ),
    )
    ap.add_argument(
        "--filter_require_logical_valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require judge verdict PASS (logical_validity true) for questions to be kept. "
            "Disable to ignore logical validity when filtering."
        ),
    )
    ap.add_argument(
        "--filter_logical_mode",
        type=str,
        choices=["all", "majority"],
        default="all",
        help=(
            "How to aggregate logical validity across judges when "
            "--filter_require_logical_valid is enabled. "
            "'all' (default) requires every judge with a valid, scored "
            "output to PASS; 'majority' keeps questions where a strict "
            "majority of such judges PASS."
        ),
    )
    ap.add_argument(
        "--restrict_answer_models_to_generators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), only include answer-model columns for "
            "models that also appear as quiz generators. Use "
            "--no-restrict_answer_models_to_generators to show all "
            "answer models."
        ),
    )
    args = ap.parse_args()

    quiz_batch_tag = str(args.quiz_batch_tag).strip() if args.quiz_batch_tag else None
    tagged_quiz_ids: set[str] = set()
    legacy_quiz_ids: set[str] = set()
    if quiz_batch_tag:
        tagged_manifest_paths = _select_manifest_paths_for_batch_tag(args.runs_root, quiz_batch_tag)
        legacy_manifest_paths = _select_legacy_manifest_paths(args.runs_root)
        tagged_quiz_ids = _load_quiz_ids_from_manifest_paths(tagged_manifest_paths)
        legacy_quiz_ids = _load_quiz_ids_from_manifest_paths(legacy_manifest_paths)

    # collect quiz runs (each run dir contains per-model *_summary.json), searching recursively
    run_dirs = find_run_dirs(args.runs_root, args.run_dir_glob)
    if not run_dirs and args.run_dir_glob != "*":
        fallback_dirs = find_run_dirs(args.runs_root, "*")
        if fallback_dirs:
            print(
                f"[WARN] No run dirs matched glob '{args.run_dir_glob}'. "
                f"Falling back to '*' ({len(fallback_dirs)} dirs)."
            )
            run_dirs = fallback_dirs

    # Map quiz_id (run dir basename) -> generator model if manifest is present.
    quiz_generators = load_quizbench_manifest_generators(args.runs_root, quiz_batch_tag=quiz_batch_tag)
    if quiz_batch_tag:
        if not quiz_generators:
            raise SystemExit(
                f"[FATAL] No quizbench manifests found for batch tag {quiz_batch_tag!r} under {args.runs_root} "
                "(expected quizbench_manifest_<TAG>*.json and/or legacy quizbench_manifest.json)."
            )

        allowed_quiz_ids = set(quiz_generators.keys())
        run_dirs = [
            rd for rd in run_dirs if os.path.basename(rd.rstrip(os.sep)) in allowed_quiz_ids
        ]
        if not run_dirs:
            raise SystemExit(
                f"[FATAL] No run directories matched batch tag {quiz_batch_tag!r} under {args.runs_root}. "
                "Check that *_summary.json files exist for the quizzes in the batch manifests."
            )

    # key: (generator_model, eval_model) -> list of accuracies across quizzes
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_generators: dict[tuple[str, str], set[str]] = defaultdict(set)
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    generators_by_model: dict[str, set[str]] = defaultdict(set)
    raw_generators_by_model: dict[str, set[str]] = defaultdict(set)
    generator_valid_counts: dict[str, int] = defaultdict(int)
    generator_total_counts: dict[str, int] = defaultdict(int)
    generator_valid_counts_tagged: dict[str, int] = defaultdict(int)
    generator_total_counts_tagged: dict[str, int] = defaultdict(int)
    generator_valid_counts_legacy: dict[str, int] = defaultdict(int)
    generator_total_counts_legacy: dict[str, int] = defaultdict(int)
    for rd in run_dirs:
        quiz_id = os.path.basename(rd.rstrip(os.sep))
        in_tagged = bool(quiz_batch_tag and quiz_id in tagged_quiz_ids)
        in_legacy = bool(quiz_batch_tag and quiz_id in legacy_quiz_ids)

        generator_model_raw = infer_generator_from_run_dir(rd, quiz_generators)
        generator_model = canonicalize_generator_model(
            generator_model_raw, mode=args.generator_family_mode
        )
        total_items = get_num_items_for_run(rd)
        generator_total_counts[generator_model] += int(total_items or 0)
        if args.filter_by_judge:
            acc_map, n_valid = load_quiz_model_acc_filtered(
                rd,
                judge_models=args.filter_judges,
                min_medical_score=args.filter_min_med_score,
                require_logical_valid=args.filter_require_logical_valid,
                logical_mode=args.filter_logical_mode,
            )
        else:
            acc_map = load_quiz_model_acc(rd)
            n_valid = total_items

        generator_valid_counts[generator_model] += int(n_valid or 0)

        if in_tagged:
            generator_total_counts_tagged[generator_model] += int(total_items or 0)
            generator_valid_counts_tagged[generator_model] += int(n_valid or 0)
        if in_legacy:
            generator_total_counts_legacy[generator_model] += int(total_items or 0)
            generator_valid_counts_legacy[generator_model] += int(n_valid or 0)

        for m, a in acc_map.items():
            scores[(generator_model, m)].append(a)
            raw_generators[(generator_model, m)].add(generator_model_raw)
            scores_by_model[m].append(a)
            generators_by_model[m].add(generator_model)
            raw_generators_by_model[m].add(generator_model_raw)

    print("    [DEBUG]:", generator_valid_counts)

    # Merge in any manually specified per-quiz model accuracies from an
    # external JSON file (e.g., ). Each row is treated
    # as a single "quiz" for the specified generator.
    extra_results = load_extra_model_results(args.extra_results_json)
    if extra_results:
        unmatched = 0
        for row in extra_results:
            generator_raw = row["generator_raw"]
            model = row["model"]
            acc = row["accuracy"]
            generator_model = canonicalize_generator_model(
                generator_raw, mode=args.generator_family_mode
            )
            # Only merge results for generators that actually appear in this run.
            if generator_model not in generator_total_counts:
                unmatched += 1
                continue

            generator_model_raw = generator_raw
            scores[(generator_model, model)].append(acc)
            raw_generators[(generator_model, model)].add(generator_model_raw)
            scores_by_model[model].append(acc)
            generators_by_model[model].add(generator_model)
            raw_generators_by_model[model].add(generator_model_raw)

        if unmatched:
            print(
                f"[WARN] Skipped {unmatched} extra result(s) whose quiz_id "
                "did not match any discovered run directory."
            )

    # Optionally restrict the displayed matrix/heatmap to answer models that also
    # appear as generators (based on the generator names used in `scores`).
    scores_for_display: dict[tuple[str, str], list[float]]
    if args.restrict_answer_models_to_generators:
        generator_models = {g for (g, _) in scores.keys()}
        scores_for_display = {
            (g, m): arr
            for (g, m), arr in scores.items()
            if m in generator_models
        }
    else:
        scores_for_display = dict(scores)

    rows = []
    for (generator_model, model), arr in sorted(scores.items()):
        mu, se = mean_sem(arr)
        rows.append(
            {
                "generator_model": generator_model,
                "raw_generator_models": ",".join(sorted(raw_generators[(generator_model, model)])),
                "model": model,
                "num_quizzes": len(arr),
                "mean_accuracy": mu,
                "sem": se,
            }
        )

    fieldnames = [
        "generator_model",
        "raw_generator_models",
        "model",
        "num_quizzes",
        "mean_accuracy",
        "sem",
    ]
    write_csv(args.out_csv, fieldnames, rows)

    if args.out_csv_by_model:
        rows_by_model = []
        for model, arr in sorted(scores_by_model.items()):
            mu, se = mean_sem(arr)
            rows_by_model.append(
                {
                    "model": model,
                    "num_quizzes": len(arr),
                    "mean_accuracy": mu,
                    "sem": se,
                    "generator_models": ",".join(sorted(generators_by_model[model])),
                    "raw_generator_models": ",".join(
                        sorted(raw_generators_by_model[model])
                    ),
                }
            )

        fieldnames_by_model = [
            "model",
            "num_quizzes",
            "mean_accuracy",
            "sem",
            "generator_models",
            "raw_generator_models",
        ]
        write_csv(args.out_csv_by_model, fieldnames_by_model, rows_by_model)

    if args.out_heatmap:
        save_accuracy_heatmap(
            scores_for_display,
            args.out_heatmap,
            title="Generator x Answer Model Accuracy",
            row_counts=generator_valid_counts,
        )

    if args.out_filtered_barplot:
        save_validity_barplot(
            generator_valid_counts,
            generator_total_counts,
            args.out_filtered_barplot,
            title="Questions Passing LLM-as Judge Validity Filters",
        )

    if args.filter_by_judge:
        print_validity_summary(
            generator_valid_counts,
            generator_total_counts,
            label="Questions passing judge filters",
        )
        if quiz_batch_tag:
            print_validity_summary(
                generator_valid_counts_tagged,
                generator_total_counts_tagged,
                label=f"Questions passing judge filters (tagged manifests: {quiz_batch_tag})",
            )
            print_validity_summary(
                generator_valid_counts_legacy,
                generator_total_counts_legacy,
                label="Questions passing judge filters (legacy manifests: quizbench_manifest.json)",
            )

    # Print a colored generator x taker matrix to stdout.
    print_accuracy_matrix(
        scores_for_display,
        row_counts=generator_valid_counts,
        restrict_answer_models_to_generators=args.restrict_answer_models_to_generators,
    )

    print(args.out_csv)
    if args.out_csv_by_model:
        print(args.out_csv_by_model)


if __name__ == "__main__":
    main()
