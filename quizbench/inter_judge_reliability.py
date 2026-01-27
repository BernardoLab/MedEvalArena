#!/usr/bin/env python3
"""
Compute inter-judge reliability from QuizBench judge_result files.
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from quizbench.aggregate_results import (
        _load_quiz_ids_from_manifest_paths,
        _select_legacy_manifest_paths,
        _select_manifest_paths_for_batch_tag,
        canonicalize_generator_model,
        find_run_dirs,
        infer_generator_from_run_dir,
        load_quizbench_manifest_generators,
    )
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "[FATAL] Could not import quizbench.aggregate_results. "
        "Ensure its dependencies are installed (e.g., colorcet)."
    ) from exc

from quizbench.aggregate_judges import DEFAULT_ENSEMBLE_JUDGES
from quizbench.judge_utils import (
    build_unit_id,
    extract_medical_score,
    extract_verdict,
    is_judge_output_valid,
)


def _load_json_list(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _sanitize_model_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")
    return slug or "model"


def _is_judge_generator(generator_model: str, judge_models: list[str]) -> bool:
    gen = (generator_model or "").strip()
    if not gen:
        return False
    judge_set = {m.strip() for m in judge_models if str(m).strip()}
    if gen in judge_set:
        return True
    gen_slug = _sanitize_model_name(gen)
    judge_slug_set = {_sanitize_model_name(m) for m in judge_set}
    return gen_slug in judge_slug_set


def _nominal_distance(a: Any, b: Any) -> float:
    return 0.0 if a == b else 1.0


def _interval_distance(a: Any, b: Any) -> float:
    return (float(a) - float(b)) ** 2


def krippendorff_alpha(
    units: list[list[Any]],
    distance: Callable[[Any, Any], float],
) -> float:
    if not units:
        return float("nan")

    coincidence: dict[Any, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    for ratings in units:
        counts = Counter(ratings)
        m = sum(counts.values())
        if m < 2:
            continue
        denom = m - 1
        for val_i, n_i in counts.items():
            for val_j, n_j in counts.items():
                if val_i == val_j:
                    contrib = n_i * (n_i - 1)
                else:
                    contrib = n_i * n_j
                coincidence[val_i][val_j] += contrib / denom

    row_sums = {val: sum(row.values()) for val, row in coincidence.items()}
    n_total = sum(row_sums.values())
    if n_total <= 1:
        return float("nan")

    do = 0.0
    for val_i, row in coincidence.items():
        for val_j, count in row.items():
            do += count * distance(val_i, val_j)
    do /= n_total

    de = 0.0
    for val_i, n_i in row_sums.items():
        for val_j, n_j in row_sums.items():
            de += n_i * n_j * distance(val_i, val_j)
    de /= (n_total * (n_total - 1))

    if de == 0.0:
        return 1.0 if do == 0.0 else float("nan")
    return 1.0 - (do / de)


def _looc_consensus_pass_fail(other_verdicts: list[str]) -> str | None:
    counts = Counter(other_verdicts)
    n_pass = counts.get("PASS", 0)
    n_fail = counts.get("FAIL", 0)
    if n_pass > n_fail:
        return "PASS"
    if n_fail > n_pass:
        return "FAIL"
    return None


def _looc_consensus_score(other_scores: list[int]) -> int | None:
    if not other_scores:
        return None
    return int(statistics.median_low(sorted(other_scores)))


def _binary_metrics_from_pairs(pairs: list[tuple[str, str]]) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for judge_label, consensus_label in pairs:
        if judge_label == "PASS" and consensus_label == "PASS":
            tp += 1
        elif judge_label == "FAIL" and consensus_label == "FAIL":
            tn += 1
        elif judge_label == "PASS" and consensus_label == "FAIL":
            fp += 1
        elif judge_label == "FAIL" and consensus_label == "PASS":
            fn += 1

    n = tp + tn + fp + fn
    if n == 0:
        return {
            "n_used": 0.0,
            "accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
            "kappa": float("nan"),
        }

    accuracy = (tp + tn) / n
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    f1 = (
        (2 * precision * sensitivity / (precision + sensitivity))
        if _is_finite(precision) and _is_finite(sensitivity) and (precision + sensitivity) > 0
        else float("nan")
    )

    p_j_pass = (tp + fp) / n
    p_j_fail = 1.0 - p_j_pass
    p_c_pass = (tp + fn) / n
    p_c_fail = 1.0 - p_c_pass
    pe = p_j_pass * p_c_pass + p_j_fail * p_c_fail
    if pe >= 1.0:
        kappa = float("nan")
    else:
        kappa = (accuracy - pe) / (1.0 - pe)

    return {
        "n_used": float(n),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "kappa": kappa,
    }


def _rankdata(values: list[float]) -> list[float]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0.0 or den_y <= 0.0:
        return float("nan")
    return num / math.sqrt(den_x * den_y)


def _weighted_kappa_quadratic(pairs: list[tuple[int, int]], k: int = 5) -> float:
    if not pairs:
        return float("nan")
    matrix = [[0.0 for _ in range(k)] for _ in range(k)]
    n = 0
    for judge_score, consensus_score in pairs:
        if not (1 <= judge_score <= k and 1 <= consensus_score <= k):
            continue
        matrix[judge_score - 1][consensus_score - 1] += 1.0
        n += 1
    if n == 0:
        return float("nan")

    row_marg = [sum(row) for row in matrix]
    col_marg = [sum(matrix[r][c] for r in range(k)) for c in range(k)]

    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) / (k - 1)) ** 2
            num += weight * matrix[i][j]
            expected = (row_marg[i] * col_marg[j]) / n
            den += weight * expected

    if den == 0.0:
        return 1.0 if num == 0.0 else float("nan")
    return 1.0 - (num / den)


def _ordinal_metrics_from_pairs(
    pairs: list[tuple[int, int]],
    k: int = 5,
) -> dict[str, float]:
    n = len(pairs)
    if n == 0:
        return {
            "n_used": 0.0,
            "exact_match": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
            "spearman": float("nan"),
            "weighted_kappa_quadratic": float("nan"),
        }

    exact_match = sum(1 for j, c in pairs if j == c) / n
    mae = sum(abs(j - c) for j, c in pairs) / n
    mse = sum((j - c) ** 2 for j, c in pairs) / n

    judge_scores = [float(j) for j, _ in pairs]
    consensus_scores = [float(c) for _, c in pairs]
    spearman = _pearson_corr(_rankdata(judge_scores), _rankdata(consensus_scores))
    weighted_kappa = _weighted_kappa_quadratic(pairs, k=k)

    return {
        "n_used": float(n),
        "exact_match": exact_match,
        "mae": mae,
        "mse": mse,
        "spearman": spearman,
        "weighted_kappa_quadratic": weighted_kappa,
    }


def compute_looc_logical_metrics(
    ratings_by_unit: dict[str, dict[str, str]],
    judge_models: list[str],
    *,
    min_others: int = 2,
) -> dict[str, dict[str, float]]:
    metrics_by_judge: dict[str, dict[str, float]] = {}
    for judge in judge_models:
        pairs: list[tuple[str, str]] = []
        skipped_missing = 0
        skipped_insufficient = 0
        skipped_tie = 0
        judge_pass = 0
        consensus_pass = 0

        for _, by_judge in ratings_by_unit.items():
            if judge not in by_judge:
                skipped_missing += 1
                continue
            others = [v for k, v in by_judge.items() if k != judge]
            if len(others) < min_others:
                skipped_insufficient += 1
                continue
            consensus = _looc_consensus_pass_fail(others)
            if consensus is None:
                skipped_tie += 1
                continue
            judge_label = by_judge[judge]
            pairs.append((judge_label, consensus))
            if judge_label == "PASS":
                judge_pass += 1
            if consensus == "PASS":
                consensus_pass += 1

        metrics = _binary_metrics_from_pairs(pairs)
        n_used = int(metrics["n_used"])
        metrics.update(
            {
                "n_skipped_missing": float(skipped_missing),
                "n_skipped_insufficient": float(skipped_insufficient),
                "n_skipped_tie": float(skipped_tie),
                "judge_pass_rate": (judge_pass / n_used) if n_used else float("nan"),
                "consensus_pass_rate": (consensus_pass / n_used) if n_used else float("nan"),
            }
        )
        metrics_by_judge[judge] = metrics
    return metrics_by_judge


def compute_looc_score_metrics(
    ratings_by_unit: dict[str, dict[str, int]],
    judge_models: list[str],
    *,
    min_others: int = 2,
) -> dict[str, dict[str, float]]:
    metrics_by_judge: dict[str, dict[str, float]] = {}
    for judge in judge_models:
        pairs: list[tuple[int, int]] = []
        skipped_missing = 0
        skipped_insufficient = 0

        for _, by_judge in ratings_by_unit.items():
            if judge not in by_judge:
                skipped_missing += 1
                continue
            others = [v for k, v in by_judge.items() if k != judge]
            if len(others) < min_others:
                skipped_insufficient += 1
                continue
            consensus = _looc_consensus_score(others)
            if consensus is None:
                continue
            pairs.append((by_judge[judge], consensus))

        metrics = _ordinal_metrics_from_pairs(pairs, k=5)
        metrics.update(
            {
                "n_skipped_missing": float(skipped_missing),
                "n_skipped_insufficient": float(skipped_insufficient),
            }
        )
        metrics_by_judge[judge] = metrics
    return metrics_by_judge


def _count_raters_by_unit(ratings_by_unit: dict[str, dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for ratings in ratings_by_unit.values():
        counts[len(ratings)] += 1
    return counts


def _count_pass_votes_by_unit(logical_units: dict[str, dict[str, str]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for ratings in logical_units.values():
        pass_count = sum(1 for verdict in ratings.values() if verdict == "PASS")
        counts[pass_count] += 1
    return counts


def _format_count_hist(counts: Counter[int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _filter_score_units_by_logical_votes(
    logical_units: dict[str, dict[str, str]],
    score_units: dict[str, dict[str, int]],
    *,
    min_pass: int,
    min_total: int,
) -> dict[str, dict[str, int]]:
    filtered: dict[str, dict[str, int]] = {}
    for unit_id, ratings in score_units.items():
        logical_ratings = logical_units.get(unit_id, {})
        total = len(logical_ratings)
        if total < min_total:
            continue
        pass_count = sum(1 for verdict in logical_ratings.values() if verdict == "PASS")
        if pass_count >= min_pass:
            filtered[unit_id] = ratings
    return filtered


def _summarize_score_filtering(
    logical_units: dict[str, dict[str, str]],
    score_units: dict[str, dict[str, int]],
    *,
    min_pass: int,
    min_total: int,
) -> dict[str, int]:
    summary = {
        "score_units_total": len(score_units),
        "missing_logical": 0,
        "insufficient_logical": 0,
        "failed_pass": 0,
        "passed": 0,
    }
    for unit_id in score_units:
        logical_ratings = logical_units.get(unit_id)
        if not logical_ratings:
            summary["missing_logical"] += 1
            continue
        total = len(logical_ratings)
        if total < min_total:
            summary["insufficient_logical"] += 1
            continue
        pass_count = sum(1 for verdict in logical_ratings.values() if verdict == "PASS")
        if pass_count >= min_pass:
            summary["passed"] += 1
        else:
            summary["failed_pass"] += 1
    return summary


def _print_filter_debug(
    generator: str,
    logical_units: dict[str, dict[str, str]],
    score_units_raw: dict[str, dict[str, int]],
    score_units_filtered: dict[str, dict[str, int]],
    *,
    run_count: int,
    min_others: int,
    n_judges: int,
    score_logical_min_pass: int | None,
    score_logical_min_total: int | None,
    filter_summary: dict[str, int] | None,
) -> None:
    logical_raters = _count_raters_by_unit(logical_units)
    score_raters_raw = _count_raters_by_unit(score_units_raw)
    score_raters_filtered = _count_raters_by_unit(score_units_filtered)
    pass_votes = _count_pass_votes_by_unit(logical_units)

    print(f"\n[DEBUG] Filter summary for generator={generator}")
    print("  unit id format: quiz_id::question_id")
    print(f"  runs (quiz ids): {run_count}")
    print(
        "  logical verdict units: "
        f"{len(logical_units)} (units by #raters: {_format_count_hist(logical_raters)})"
    )
    print(
        "  logical PASS vote counts (PASS count: units): "
        f"{_format_count_hist(pass_votes)}"
    )
    print(
        "  score units (medical accuracy, pre-filter): "
        f"{len(score_units_raw)} (units by #raters: {_format_count_hist(score_raters_raw)})"
    )

    if score_logical_min_pass is None or score_logical_min_total is None:
        print("  logical gate for score units: disabled")
    else:
        print(
            "  logical gate for score units:"
        )
        print(
            "    rule: "
            f"PASS>={score_logical_min_pass} AND total>={score_logical_min_total}"
        )
        if filter_summary:
            print(
                "    dropped (missing logical verdicts): "
                f"{filter_summary.get('missing_logical', 0)}"
            )
            print(
                "    dropped (insufficient logical verdicts): "
                f"{filter_summary.get('insufficient_logical', 0)}"
            )
            print(
                "    dropped (PASS below threshold): "
                f"{filter_summary.get('failed_pass', 0)}"
            )
            print(f"    kept: {filter_summary.get('passed', 0)}")

    print(
        "  score units (post-filter): "
        f"{len(score_units_filtered)} (units by #raters: "
        f"{_format_count_hist(score_raters_filtered)})"
    )
    min_raters = min_others + 1
    with_min_raters = sum(
        1 for ratings in score_units_filtered.values() if len(ratings) >= min_raters
    )
    print(
        f"  score units with >= {min_raters} raters (min_others+1): "
        f"{with_min_raters}"
    )
    if n_judges:
        full_cover = sum(
            1 for ratings in score_units_filtered.values() if len(ratings) == n_judges
        )
        print(f"  score units with all {n_judges} judges: {full_cover}")


def _compute_stats(
    ratings_by_unit: dict[str, dict[str, Any]],
    n_judges: int,
    distance: Callable[[Any, Any], float],
) -> dict[str, float]:
    units: list[list[Any]] = []
    n_units_total = len(ratings_by_unit)
    n_units_2plus = 0
    n_units_complete = 0
    total_ratings = 0
    for ratings_by_judge in ratings_by_unit.values():
        m = len(ratings_by_judge)
        total_ratings += m
        if m >= 2:
            n_units_2plus += 1
            units.append(list(ratings_by_judge.values()))
        if m == n_judges:
            n_units_complete += 1
    avg_raters = (total_ratings / n_units_total) if n_units_total else 0.0

    alpha = krippendorff_alpha(units, distance)
    return {
        "n_units_total": float(n_units_total),
        "n_units_2plus": float(n_units_2plus),
        "n_units_complete": float(n_units_complete),
        "avg_raters": avg_raters,
        "alpha": alpha,
    }


def _fmt_float(value: float, width: int, precision: int = 3) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan".rjust(width)
    return f"{value:{width}.{precision}f}"


def _print_table(
    title: str,
    stats_by_generator: dict[str, dict[str, float]],
    total_stats: dict[str, float],
    generator_run_counts: dict[str, int],
    total_runs: int,
    judge_models: list[str],
) -> None:
    print(f"\n{title}")
    print(f"Judges: {', '.join(judge_models)}")

    header = ("Generator", "Runs", "Units>=2", "UnitsAll", "AvgRaters", "Alpha")
    print(
        f"{header[0]:<20} {header[1]:>5} {header[2]:>9} {header[3]:>9} "
        f"{header[4]:>10} {header[5]:>8}"
    )
    print("-" * 68)

    for generator, stats in sorted(stats_by_generator.items()):
        runs = generator_run_counts.get(generator, 0)
        print(
            f"{generator:<20} {runs:5d} {int(stats['n_units_2plus']):9d} "
            f"{int(stats['n_units_complete']):9d} "
            f"{_fmt_float(stats['avg_raters'], 10, 2)} "
            f"{_fmt_float(stats['alpha'], 8, 3)}"
        )

    print("-" * 68)
    print(
        f"{'TOTAL':<20} {total_runs:5d} {int(total_stats['n_units_2plus']):9d} "
        f"{int(total_stats['n_units_complete']):9d} "
        f"{_fmt_float(total_stats['avg_raters'], 10, 2)} "
        f"{_fmt_float(total_stats['alpha'], 8, 3)}"
    )


def _print_looc_logical_table(
    title: str,
    looc_by_generator: dict[str, dict[str, dict[str, float]]],
    total_looc: dict[str, dict[str, float]],
    judge_models: list[str],
) -> None:
    print(f"\n{title}")
    header = ("Judge", "Used", "Acc", "Sens", "Spec", "Kappa", "PassRate")
    for generator, by_judge in sorted(looc_by_generator.items()):
        print(f"\nGenerator: {generator}")
        print(
            f"{header[0]:<28} {header[1]:>6} {header[2]:>7} {header[3]:>7} "
            f"{header[4]:>7} {header[5]:>7} {header[6]:>9}"
        )
        print("-" * 74)
        for judge in judge_models:
            metrics = by_judge.get(judge, {})
            print(
                f"{judge:<28} {int(metrics.get('n_used', 0)):6d} "
                f"{_fmt_float(metrics.get('accuracy', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('sensitivity', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('specificity', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('kappa', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('judge_pass_rate', float('nan')), 9, 3)}"
            )
    print("\nTOTAL")
    print(
        f"{header[0]:<28} {header[1]:>6} {header[2]:>7} {header[3]:>7} "
        f"{header[4]:>7} {header[5]:>7} {header[6]:>9}"
    )
    print("-" * 74)
    for judge in judge_models:
        metrics = total_looc.get(judge, {})
        print(
            f"{judge:<28} {int(metrics.get('n_used', 0)):6d} "
            f"{_fmt_float(metrics.get('accuracy', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('sensitivity', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('specificity', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('kappa', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('judge_pass_rate', float('nan')), 9, 3)}"
        )


def _print_looc_score_table(
    title: str,
    looc_by_generator: dict[str, dict[str, dict[str, float]]],
    total_looc: dict[str, dict[str, float]],
    judge_models: list[str],
) -> None:
    print(f"\n{title}")
    header = ("Judge", "Used", "Exact", "MAE", "W-Kappa", "Spearman")
    for generator, by_judge in sorted(looc_by_generator.items()):
        print(f"\nGenerator: {generator}")
        print(
            f"{header[0]:<28} {header[1]:>6} {header[2]:>7} {header[3]:>7} "
            f"{header[4]:>9} {header[5]:>9}"
        )
        print("-" * 72)
        for judge in judge_models:
            metrics = by_judge.get(judge, {})
            print(
                f"{judge:<28} {int(metrics.get('n_used', 0)):6d} "
                f"{_fmt_float(metrics.get('exact_match', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('mae', float('nan')), 7, 3)} "
                f"{_fmt_float(metrics.get('weighted_kappa_quadratic', float('nan')), 9, 3)} "
                f"{_fmt_float(metrics.get('spearman', float('nan')), 9, 3)}"
            )
    print("\nTOTAL")
    print(
        f"{header[0]:<28} {header[1]:>6} {header[2]:>7} {header[3]:>7} "
        f"{header[4]:>9} {header[5]:>9}"
    )
    print("-" * 72)
    for judge in judge_models:
        metrics = total_looc.get(judge, {})
        print(
            f"{judge:<28} {int(metrics.get('n_used', 0)):6d} "
            f"{_fmt_float(metrics.get('exact_match', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('mae', float('nan')), 7, 3)} "
            f"{_fmt_float(metrics.get('weighted_kappa_quadratic', float('nan')), 9, 3)} "
            f"{_fmt_float(metrics.get('spearman', float('nan')), 9, 3)}"
        )


def _is_finite(value: float) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _get_r2_palette(n: int, sns: Any) -> list[Any]:
    n = max(1, n)
    try:
        import colorcet as cc

        cmap = cc.cm["CET_R2"]
        if n == 1:
            return [cmap(0.5)]
        return [cmap(i / (n - 1)) for i in range(n)]
    except Exception as exc:
        print(f"[WARN] Could not load colorcet r2 palette; falling back to tab10: {exc}")
        return sns.color_palette("tab10", n_colors=n)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _collect_units_for_bootstrap(
    ratings_by_unit: dict[str, dict[str, Any]],
) -> list[list[Any]]:
    units: list[list[Any]] = []
    for ratings_by_judge in ratings_by_unit.values():
        if len(ratings_by_judge) >= 2:
            units.append(list(ratings_by_judge.values()))
    return units


def _bootstrap_alpha_ci(
    units: list[list[Any]],
    distance: Callable[[Any, Any], float],
    *,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    if iters <= 0 or len(units) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n_units = len(units)
    samples: list[float] = []
    for _ in range(iters):
        draw = [units[rng.randrange(n_units)] for _ in range(n_units)]
        alpha = krippendorff_alpha(draw, distance)
        if _is_finite(alpha):
            samples.append(float(alpha))
    if not samples:
        return float("nan"), float("nan")
    samples.sort()
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _format_bar_value(value: float) -> str:
    if _is_finite(value):
        return f"{float(value):.3f}"
    return "nan"


def _bar_label_color(value: float, *, threshold: float = 0.2) -> str:
    if _is_finite(value) and float(value) < threshold:
        return "#444444"
    return "white"


def _plot_alpha_barplot(
    stats_by_generator: dict[str, dict[str, float]],
    ratings_by_generator: dict[str, dict[str, dict[str, Any]]],
    total_stats: dict[str, float],
    out_path: str,
    title: str,
    *,
    distance: Callable[[Any, Any], float],
    bootstrap_iters: int,
    bootstrap_seed: int,
    dpi: int,
) -> None:
    if not out_path:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not import matplotlib/seaborn for plot: {exc}")
        return

    rows: list[dict[str, Any]] = []
    for generator, stats in stats_by_generator.items():
        alpha = stats.get("alpha", float("nan"))
        if not _is_finite(alpha):
            continue
        units = _collect_units_for_bootstrap(ratings_by_generator.get(generator, {}))
        ci_low, ci_high = _bootstrap_alpha_ci(
            units,
            distance,
            iters=bootstrap_iters,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "generator": generator,
                "alpha": float(alpha),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if not rows:
        print(f"[WARN] No finite alpha values to plot for {out_path}.")
        return

    rows.sort(key=lambda item: item["alpha"], reverse=True)
    labels = [row["generator"] for row in rows]
    values = [row["alpha"] for row in rows]

    sns.set_theme(style="white")
    fig_height = max(2.5, 0.35 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(4.0, fig_height))
    palette = _get_r2_palette(len(labels), sns)
    y_positions = list(range(len(labels)))
    ax.barh(y_positions, values, color=palette, edgecolor="none")
    xerr_left: list[float] = []
    xerr_right: list[float] = []
    has_ci = bootstrap_iters > 0
    for row in rows:
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        if has_ci and _is_finite(ci_low) and _is_finite(ci_high):
            xerr_left.append(max(0.0, row["alpha"] - ci_low))
            xerr_right.append(max(0.0, ci_high - row["alpha"]))
        else:
            xerr_left.append(0.0)
            xerr_right.append(0.0)
    if has_ci and any(x > 0 for x in xerr_left + xerr_right):
        ax.errorbar(
            values,
            y_positions,
            xerr=[xerr_left, xerr_right],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    for y, value in zip(y_positions, values):
        if value < 0.2:
            x_pos = 0.3
        else:
            x_pos = 0.1

        ax.text(
            x_pos,
            y,
            _format_bar_value(value),
            va="center",
            ha="center",
            color=_bar_label_color(value),
        )

    overall_alpha = total_stats.get("alpha", float("nan"))
    if _is_finite(overall_alpha):
        ax.axvline(float(overall_alpha), color="#888888", linestyle="--", linewidth=1.0)
        title = f"{title} (overall={float(overall_alpha):.3f})"

    ax.set_title(title)
    ax.set_xlabel("Krippendorff alpha")
    ax.set_ylabel("")
    ax.grid(False)
    sns.despine(ax=ax)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def _bootstrap_ci_proportion(
    p_hat: float,
    n: int,
    *,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    if iters <= 0 or n <= 0 or not _is_finite(p_hat):
        return float("nan"), float("nan")
    p = max(0.0, min(1.0, float(p_hat)))

    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        samples = rng.binomial(n, p, size=int(iters)) / float(n)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        return float(lo), float(hi)
    except Exception:
        # Fallback: normal approximation (not bootstrap), used only if numpy is unavailable.
        z = 1.96
        se = math.sqrt(p * (1.0 - p) / n)
        return max(0.0, p - z * se), min(1.0, p + z * se)


def _plot_looc_logical_validity_agreement_by_judge(
    total_looc: dict[str, dict[str, float]],
    judge_models: list[str],
    out_path: str,
    *,
    min_others: int,
    bootstrap_iters: int,
    bootstrap_seed: int,
    dpi: int,
) -> None:
    if not out_path:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not import matplotlib/seaborn for plot: {exc}")
        return

    rows: list[dict[str, Any]] = []
    for judge in judge_models:
        metrics = total_looc.get(judge, {})
        acc = float(metrics.get("accuracy", float("nan")))
        n_used = int(metrics.get("n_used", 0) or 0)
        pass_rate = float(metrics.get("judge_pass_rate", float("nan")))
        if n_used <= 0 or not _is_finite(acc):
            continue
        ci_low, ci_high = _bootstrap_ci_proportion(
            acc, n_used, iters=bootstrap_iters, seed=bootstrap_seed
        )
        rows.append(
            {
                "judge": judge,
                "accuracy": acc,
                "n_used": n_used,
                "pass_rate": pass_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if not rows:
        print(f"[WARN] No LOOC accuracy values to plot for {out_path}.")
        return

    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    labels = [r["judge"] for r in rows]
    values = [r["accuracy"] for r in rows]
    y_positions = list(range(len(rows)))

    fig_height = max(2.2, 0.45 * len(rows) + 0.8)
    fig, ax = plt.subplots(figsize=(4, fig_height))
    palette = _get_r2_palette(len(rows), sns)
    ax.barh(y_positions, values, color=palette, edgecolor="none")

    xerr_left: list[float] = []
    xerr_right: list[float] = []
    has_ci = bootstrap_iters > 0
    for r in rows:
        ci_low = r["ci_low"]
        ci_high = r["ci_high"]
        if has_ci and _is_finite(ci_low) and _is_finite(ci_high):
            xerr_left.append(max(0.0, r["accuracy"] - ci_low))
            xerr_right.append(max(0.0, ci_high - r["accuracy"]))
        else:
            xerr_left.append(0.0)
            xerr_right.append(0.0)

    if has_ci and any(x > 0 for x in xerr_left + xerr_right):
        ax.errorbar(
            values,
            y_positions,
            xerr=[xerr_left, xerr_right],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for y, value in zip(y_positions, values):
        ax.text(
            0.2,
            y,
            _format_bar_value(value),
            va="center",
            ha="center",
            color=_bar_label_color(value),
        )

    ax.set_xlim(0.0, 1.06)
    ax.set_xlabel("LOOC agreement with consensus")
    title = f"LOOC consensus logical validity agreement"
    ax.set_title(title)

    # for y, r in zip(y_positions, rows):
    #     n_used = r["n_used"]
    #     pr = r["pass_rate"]
    #     pr_str = f"{pr:.3f}" if _is_finite(pr) else "nan"
    #     text = f"N={n_used}, pass={pr_str}"
    #     x = min(1.02, max(0.02, r["accuracy"] + 0.02))
    #     ax.text(x, y, text, va="center", ha="left", fontsize=9, color="#111111")

    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def _plot_looc_medical_score_agreement_by_judge(
    total_looc: dict[str, dict[str, float]],
    judge_models: list[str],
    out_path: str,
    *,
    min_others: int,
    bootstrap_iters: int,
    bootstrap_seed: int,
    dpi: int,
) -> None:
    if not out_path:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not import matplotlib/seaborn for plot: {exc}")
        return

    rows: list[dict[str, Any]] = []
    for judge in judge_models:
        metrics = total_looc.get(judge, {})
        exact_match = float(metrics.get("exact_match", float("nan")))
        n_used = int(metrics.get("n_used", 0) or 0)
        mae = float(metrics.get("mae", float("nan")))
        if n_used <= 0 or not _is_finite(exact_match):
            continue
        ci_low, ci_high = _bootstrap_ci_proportion(
            exact_match, n_used, iters=bootstrap_iters, seed=bootstrap_seed
        )
        rows.append(
            {
                "judge": judge,
                "exact_match": exact_match,
                "n_used": n_used,
                "mae": mae,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if not rows:
        print(f"[WARN] No LOOC exact-match values to plot for {out_path}.")
        return

    rows.sort(key=lambda r: r["exact_match"], reverse=True)
    labels = [r["judge"] for r in rows]
    values = [r["exact_match"] for r in rows]
    y_positions = list(range(len(rows)))

    fig_height = max(2.2, 0.45 * len(rows) + 0.8)
    fig, ax = plt.subplots(figsize=(4, fig_height))
    palette = _get_r2_palette(len(rows), sns)
    ax.barh(y_positions, values, color=palette, edgecolor="none")

    xerr_left: list[float] = []
    xerr_right: list[float] = []
    has_ci = bootstrap_iters > 0
    for r in rows:
        ci_low = r["ci_low"]
        ci_high = r["ci_high"]
        if has_ci and _is_finite(ci_low) and _is_finite(ci_high):
            xerr_left.append(max(0.0, r["exact_match"] - ci_low))
            xerr_right.append(max(0.0, ci_high - r["exact_match"]))
        else:
            xerr_left.append(0.0)
            xerr_right.append(0.0)

    if has_ci and any(x > 0 for x in xerr_left + xerr_right):
        ax.errorbar(
            values,
            y_positions,
            xerr=[xerr_left, xerr_right],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for y, value in zip(y_positions, values):
        ax.text(
            0.2,
            y,
            _format_bar_value(value),
            va="center",
            ha="center",
            color=_bar_label_color(value),
        )

    ax.set_xlim(0.0, 1.06)
    ax.set_xlabel("LOOC agreement with consensus")
    title = f"LOOC consensus medical score agreement"
    ax.set_title(title)

    # for y, r in zip(y_positions, rows):
    #     mae = r["mae"]
    #     mae_str = f"{mae:.3f}" if _is_finite(mae) else "nan"
    #     text = f"N={r['n_used']}, mae={mae_str}"
    #     x = min(1.02, max(0.02, r["exact_match"] + 0.02))
    #     ax.text(x, y, text, va="center", ha="left", fontsize=9, color="#111111")

    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute inter-judge reliability using Krippendorff alpha."
    )
    ap.add_argument("--runs_root", type=str, default="eval_results/quizbench/runs")
    ap.add_argument(
        "--generator_filter",
        choices=["judge", "all"],
        default="judge",
        help=(
            "Which quiz generators to include. 'judge' restricts to quizzes whose "
            "generator_model is in --judge_models (default). 'all' includes all generators."
        ),
    )
    ap.add_argument(
        "--quiz_batch_tag",
        type=str,
        default=None,
        help=(
            "Optional quiz batch tag (e.g., Jan2026). When set, only include "
            "quizzes listed in matching manifests under --runs_root."
        ),
    )
    ap.add_argument(
        "--run_dir_glob",
        type=str,
        default="*",
        help="Glob pattern (joined to runs_root) to find per-quiz run dirs.",
    )
    ap.add_argument(
        "--generator_family_mode",
        choices=["family", "exact"],
        default="exact",
        help="Group generators by provider family or keep exact names.",
    )
    ap.add_argument(
        "--judge_models",
        nargs="+",
        default=DEFAULT_ENSEMBLE_JUDGES,
        help="Judge models to include (default: ensemble list).",
    )
    ap.add_argument(
        "--looc_min_others",
        type=int,
        default=2,
        help="Minimum number of other judges required for LOOC consensus.",
    )
    ap.add_argument(
        "--debug_filtering",
        action="store_true",
        help="Print per-generator filtering diagnostics for logical validity and scores.",
    )
    ap.add_argument(
        "--score_logical_min_pass",
        type=int,
        default=None,
        help=(
            "If set, only include units in the medical score analysis whose logical "
            "validity has at least this many PASS verdicts across judges."
        ),
    )
    ap.add_argument(
        "--score_logical_min_total",
        type=int,
        default=None,
        help=(
            "Minimum number of logical verdicts required for the PASS filter. "
            "Defaults to len(--judge_models) when --score_logical_min_pass is set."
        ),
    )
    ap.add_argument(
        "--out_fig_logical",
        type=str,
        default="eval_results/quizbench/inter_judge_alpha_logical.png",
        help="Output path for the logical validity alpha figure (set to '' to disable).",
    )
    ap.add_argument(
        "--out_fig_score",
        type=str,
        default="eval_results/quizbench/inter_judge_alpha_score.png",
        help="Output path for the medical score alpha figure (set to '' to disable).",
    )
    ap.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="DPI for saved figures.",
    )
    ap.add_argument(
        "--out_fig_looc_logical_agreement",
        type=str,
        default="eval_results/quizbench/looc_logical_agreement_by_judge.png",
        help=(
            "Output path for the LOOC agreement-with-consensus by-judge figure "
            "(logical validity). Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--out_fig_looc_score_agreement",
        type=str,
        default="eval_results/quizbench/looc_score_agreement_by_judge.png",
        help=(
            "Output path for the LOOC agreement-with-consensus by-judge figure "
            "(medical score exact-match). Set to '' to disable."
        ),
    )
    ap.add_argument(
        "--looc_bootstrap_iters",
        type=int,
        default=5000,
        help="Bootstrap iterations for LOOC agreement CIs (set to 0 to disable).",
    )
    ap.add_argument(
        "--looc_bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for LOOC bootstrap.",
    )
    args = ap.parse_args()

    judge_models = [m.strip() for m in args.judge_models if str(m).strip()]
    if not judge_models:
        raise SystemExit("[FATAL] No judge models specified.")
    score_logical_min_pass = args.score_logical_min_pass
    score_logical_min_total = args.score_logical_min_total
    if score_logical_min_pass is None and score_logical_min_total is not None:
        raise SystemExit(
            "[FATAL] --score_logical_min_total requires --score_logical_min_pass."
        )
    if score_logical_min_pass is not None:
        if score_logical_min_total is None:
            score_logical_min_total = len(judge_models)
        if score_logical_min_pass < 1 or score_logical_min_total < 1:
            raise SystemExit(
                "[FATAL] --score_logical_min_pass and --score_logical_min_total must be >= 1."
            )
        if score_logical_min_pass > score_logical_min_total:
            raise SystemExit(
                "[FATAL] --score_logical_min_pass cannot exceed --score_logical_min_total."
            )
        if score_logical_min_total > len(judge_models):
            raise SystemExit(
                "[FATAL] --score_logical_min_total cannot exceed the number of judge models."
            )
        print(
            "[INFO] Filtering medical score analysis to units with "
            f">= {score_logical_min_pass} PASS logical verdicts "
            f"(min total verdicts: {score_logical_min_total})."
        )

    quiz_batch_tag = str(args.quiz_batch_tag).strip() if args.quiz_batch_tag else None
    if quiz_batch_tag:
        tagged_paths = _select_manifest_paths_for_batch_tag(args.runs_root, quiz_batch_tag)
        legacy_paths = _select_legacy_manifest_paths(args.runs_root)
        tagged_ids = _load_quiz_ids_from_manifest_paths(tagged_paths)
        legacy_ids = _load_quiz_ids_from_manifest_paths(legacy_paths)
        print(
            f"[INFO] quiz_batch_tag={quiz_batch_tag!r}: "
            f"{len(tagged_ids)} tagged quiz_ids, {len(legacy_ids)} legacy quiz_ids."
        )

    run_dirs = find_run_dirs(args.runs_root, args.run_dir_glob)
    if not run_dirs and args.run_dir_glob != "*":
        fallback_dirs = find_run_dirs(args.runs_root, "*")
        if fallback_dirs:
            print(
                f"[WARN] No run dirs matched glob '{args.run_dir_glob}'. "
                f"Falling back to '*' ({len(fallback_dirs)} dirs)."
            )
            run_dirs = fallback_dirs

    quiz_generators = load_quizbench_manifest_generators(
        args.runs_root, quiz_batch_tag=quiz_batch_tag
    )
    if quiz_batch_tag:
        if not quiz_generators:
            raise SystemExit(
                f"[FATAL] No quizbench manifests found for batch tag {quiz_batch_tag!r} under {args.runs_root}."
            )
        allowed_quiz_ids = set(quiz_generators.keys())
        run_dirs = [
            rd for rd in run_dirs if os.path.basename(rd.rstrip(os.sep)) in allowed_quiz_ids
        ]
        if not run_dirs:
            raise SystemExit(
                f"[FATAL] No run directories matched batch tag {quiz_batch_tag!r} under {args.runs_root}."
            )

    if not run_dirs:
        raise SystemExit("[FATAL] No run directories found.")

    if args.generator_filter == "judge":
        kept: list[str] = []
        skipped = 0
        for rd in run_dirs:
            generator_raw = infer_generator_from_run_dir(rd, quiz_generators)
            if _is_judge_generator(generator_raw, judge_models):
                kept.append(rd)
            else:
                skipped += 1
        if skipped:
            print(
                f"[INFO] generator_filter='judge': kept {len(kept)}/{len(run_dirs)} run dirs "
                "whose generator_model is one of the judge models."
            )
        run_dirs = kept
        if not run_dirs:
            raise SystemExit(
                "[FATAL] generator_filter='judge' removed all run directories. "
                "Try --generator_filter all to include all generators."
            )

    ratings_logical: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    ratings_score: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    generator_run_dirs: dict[str, set[Path]] = defaultdict(set)
    all_run_dirs: set[Path] = set()

    for rd in run_dirs:
        run_dir = Path(rd)
        quiz_id = run_dir.name
        generator_raw = infer_generator_from_run_dir(rd, quiz_generators)
        generator_model = canonicalize_generator_model(
            generator_raw, mode=args.generator_family_mode
        )
        generator_run_dirs[generator_model].add(run_dir)
        all_run_dirs.add(run_dir)

        for judge_model in judge_models:
            result_path = run_dir / f"{judge_model}_judge_result.json"
            if not result_path.exists():
                continue
            rows = _load_json_list(result_path)
            for row in rows:
                if not is_judge_output_valid(row):
                    continue
                unit_id = build_unit_id(row, quiz_id)
                if not unit_id:
                    continue

                verdict = extract_verdict(row)
                if verdict is not None:
                    ratings_logical[generator_model][unit_id][judge_model] = verdict

                score = extract_medical_score(row)
                if score is not None:
                    ratings_score[generator_model][unit_id][judge_model] = score

    if not ratings_logical and not ratings_score:
        raise SystemExit("[FATAL] No judge ratings found in the selected run directories.")

    stats_logical: dict[str, dict[str, float]] = {}
    stats_score: dict[str, dict[str, float]] = {}
    looc_logical_by_generator: dict[str, dict[str, dict[str, float]]] = {}
    looc_score_by_generator: dict[str, dict[str, dict[str, float]]] = {}
    overall_logical: dict[str, dict[str, Any]] = defaultdict(dict)
    overall_score: dict[str, dict[str, Any]] = defaultdict(dict)
    for generator in sorted(set(ratings_logical) | set(ratings_score)):
        logical_units = ratings_logical.get(generator, {})
        score_units_raw = ratings_score.get(generator, {})
        score_units = score_units_raw
        filter_summary = None
        if score_logical_min_pass is not None and score_logical_min_total is not None:
            if args.debug_filtering:
                filter_summary = _summarize_score_filtering(
                    logical_units,
                    score_units_raw,
                    min_pass=score_logical_min_pass,
                    min_total=score_logical_min_total,
                )
            score_units = _filter_score_units_by_logical_votes(
                logical_units,
                score_units_raw,
                min_pass=score_logical_min_pass,
                min_total=score_logical_min_total,
            )
            ratings_score[generator] = score_units

        looc_logical_by_generator[generator] = compute_looc_logical_metrics(
            logical_units, judge_models, min_others=args.looc_min_others
        )
        if args.debug_filtering:
            run_count = len(generator_run_dirs.get(generator, set()))
            _print_filter_debug(
                generator,
                logical_units,
                score_units_raw,
                score_units,
                run_count=run_count,
                min_others=args.looc_min_others,
                n_judges=len(judge_models),
                score_logical_min_pass=score_logical_min_pass,
                score_logical_min_total=score_logical_min_total,
                filter_summary=filter_summary,
            )
        looc_score_by_generator[generator] = compute_looc_score_metrics(
            score_units, judge_models, min_others=args.looc_min_others
        )

        for unit_id, ratings in logical_units.items():
            overall_logical[f"{generator}::{unit_id}"] = ratings
        for unit_id, ratings in score_units.items():
            overall_score[f"{generator}::{unit_id}"] = ratings

        stats_logical[generator] = _compute_stats(
            logical_units, len(judge_models), _nominal_distance
        )
        stats_score[generator] = _compute_stats(
            score_units, len(judge_models), _interval_distance
        )

    total_stats_logical = _compute_stats(
        overall_logical, len(judge_models), _nominal_distance
    )
    total_stats_score = _compute_stats(
        overall_score, len(judge_models), _interval_distance
    )
    total_looc_logical = compute_looc_logical_metrics(
        overall_logical, judge_models, min_others=args.looc_min_others
    )
    total_looc_score = compute_looc_score_metrics(
        overall_score, judge_models, min_others=args.looc_min_others
    )

    run_counts = {gen: len(dirs) for gen, dirs in generator_run_dirs.items()}
    total_runs = len(all_run_dirs)

    _print_table(
        "Krippendorff alpha for logical validity (nominal distance)",
        stats_logical,
        total_stats_logical,
        run_counts,
        total_runs,
        judge_models,
    )
    _print_table(
        "Krippendorff alpha for medical accuracy score (interval distance)",
        stats_score,
        total_stats_score,
        run_counts,
        total_runs,
        judge_models,
    )
    _print_looc_logical_table(
        "LOOC consensus metrics for logical validity",
        looc_logical_by_generator,
        total_looc_logical,
        judge_models,
    )
    _print_looc_score_table(
        "LOOC consensus metrics for medical accuracy score",
        looc_score_by_generator,
        total_looc_score,
        judge_models,
    )

    _plot_alpha_barplot(
        stats_logical,
        ratings_logical,
        total_stats_logical,
        args.out_fig_logical,
        "Logical validity alpha",
        distance=_nominal_distance,
        bootstrap_iters=args.looc_bootstrap_iters,
        bootstrap_seed=args.looc_bootstrap_seed,
        dpi=args.plot_dpi,
    )
    _plot_alpha_barplot(
        stats_score,
        ratings_score,
        total_stats_score,
        args.out_fig_score,
        "Medical accuracy score alpha",
        distance=_interval_distance,
        bootstrap_iters=args.looc_bootstrap_iters,
        bootstrap_seed=args.looc_bootstrap_seed,
        dpi=args.plot_dpi,
    )
    _plot_looc_logical_validity_agreement_by_judge(
        total_looc_logical,
        judge_models,
        args.out_fig_looc_logical_agreement,
        min_others=args.looc_min_others,
        bootstrap_iters=args.looc_bootstrap_iters,
        bootstrap_seed=args.looc_bootstrap_seed,
        dpi=args.plot_dpi,
    )
    _plot_looc_medical_score_agreement_by_judge(
        total_looc_score,
        judge_models,
        args.out_fig_looc_score_agreement,
        min_others=args.looc_min_others,
        bootstrap_iters=args.looc_bootstrap_iters,
        bootstrap_seed=args.looc_bootstrap_seed,
        dpi=args.plot_dpi,
    )


if __name__ == "__main__":
    main()
