#!/usr/bin/env python3
"""
Aggregate topic classification outputs into per-generator distributions.

This CLI collects all topics_*.json files under eval_results/quizbench/runs/<generator>/<quiz_id>/,
computes topic frequencies across all questions for each generator model, prints tidy tables,
and writes a manuscript-ready bar plot PNG.

Example:
    python quizbench/aggregate_topics_round_robin.py \\
        --runs_root eval_results/quizbench/runs \\
        --out_png eval_results/quizbench/topic_distribution.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from scipy.stats import chi2_contingency


@dataclass(frozen=True)
class TopicFile:
    path: Path
    generator_dir: str
    run_name: str


def parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def discover_topic_files(runs_root: Path) -> list[TopicFile]:
    if not runs_root.exists():
        return []
    records: list[TopicFile] = []
    for path in sorted(runs_root.rglob("topics_*.json")):
        try:
            rel_parts = path.relative_to(runs_root).parts
        except ValueError:
            rel_parts = ()
        generator_dir = rel_parts[0] if len(rel_parts) >= 1 else path.parent.name
        run_name = rel_parts[1] if len(rel_parts) >= 2 else path.parent.name
        records.append(TopicFile(path=path, generator_dir=generator_dir, run_name=run_name))
    return records


def finalize_topic_order(base: Sequence[str], observed: Iterable[str]) -> list[str]:
    order = list(base)
    seen = set(order)
    unknown_present = False
    for topic in observed:
        if topic == "unknown":
            unknown_present = True
            continue
        if topic not in seen:
            order.append(topic)
            seen.add(topic)
    if unknown_present and "unknown" not in seen:
        order.append("unknown")
    return order


def load_topic_rows(
    topic_files: Sequence[TopicFile],
    allowed_generators: set[str] | None,
    allowed_eval_models: set[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    topic_order: list[str] = []
    seen_topics: set[str] = set()

    for record in topic_files:
        try:
            with record.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not parse {record.path}: {exc}")
            continue

        generator_model = payload.get("generator_model") or record.generator_dir
        eval_model = payload.get("eval_model") or "unknown"

        if allowed_generators and generator_model not in allowed_generators:
            continue
        if allowed_eval_models and eval_model not in allowed_eval_models:
            continue

        default_topics = payload.get("default_topics") or []
        for t in default_topics:
            if t not in seen_topics:
                seen_topics.add(t)
                topic_order.append(t)

        per_question = payload.get("per_question") or []
        quiz_id = payload.get("quiz_id") or record.run_name
        for idx, item in enumerate(per_question, start=1):
            topic = item.get("topic") or "unknown"
            rows.append(
                {
                    "generator_model": generator_model,
                    "eval_model": eval_model,
                    "quiz_id": quiz_id,
                    "question_id": item.get("question_id") or f"{quiz_id}-{idx:03d}",
                    "topic": topic,
                    "source_path": str(record.path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, topic_order

    df["topic"] = df["topic"].fillna("unknown")
    topic_order = finalize_topic_order(topic_order, pd.unique(df["topic"]))
    dtype = pd.api.types.CategoricalDtype(categories=topic_order, ordered=True)
    df["topic"] = df["topic"].astype(dtype)
    return df, topic_order


def compute_topic_distribution(
    df: pd.DataFrame, topic_order: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if df.empty:
        empty = pd.DataFrame(
            columns=["generator_model", "topic", "count", "total", "proportion"]
        )
        totals = pd.Series(dtype=int, name="total")
        return empty, totals, list(topic_order)

    topic_order = finalize_topic_order(topic_order, pd.unique(df["topic"]))
    totals = df.groupby("generator_model").size().rename("total").sort_index()
    generators = list(totals.index)

    full_index = pd.MultiIndex.from_product(
        [generators, topic_order], names=["generator_model", "topic"]
    )
    counts = (
        df.groupby(["generator_model", "topic"])
        .size()
        .reindex(full_index, fill_value=0)
        .reset_index(name="count")
    )
    counts["total"] = counts["generator_model"].map(totals)
    counts["proportion"] = counts["count"] / counts["total"].replace({0: pd.NA})
    counts["topic"] = pd.Categorical(counts["topic"], categories=topic_order, ordered=True)
    return counts, totals, topic_order


def run_chi_square_test(counts: pd.DataFrame, topic_order: Sequence[str]) -> None:
    """
    Perform a chi-square test of independence on the topic (rows) x generator (cols) table.
    Null: topic distribution is identical across generators.
    """
    if counts.empty:
        print("[WARN] Skipping chi-square test: no counts available.")
        return

    table = (
        counts.pivot(index="topic", columns="generator_model", values="count")
        .reindex(topic_order)
        .fillna(0)
    )
    # Drop empty rows/cols to avoid degenerate tables.
    table = table.loc[table.sum(axis=1) > 0, :]
    table = table.loc[:, table.sum(axis=0) > 0]

    if table.shape[0] < 2 or table.shape[1] < 2:
        print("[WARN] Skipping chi-square test: need at least 2 topics and 2 generators with counts.")
        return

    chi2, p_value, dof, expected = chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    k = min(table.shape)
    cramers_v = (chi2 / (n * (k - 1))) ** 0.5 if k > 1 and n > 0 else float("nan")
    min_expected = expected.min() if expected.size else float("nan")
    print("\nChi-square test of topic distribution across generators")
    print(f"  chi2 = {chi2:.3f}, dof = {dof}, p = {p_value:.4g}")
    print(f"  Cramer's V = {cramers_v:.3f} (effect size)")
    print(f"  Min expected cell count = {min_expected:.2f}")
    if min_expected < 5:
        print("  [NOTE] Some expected counts < 5; chi-square approximation may be unreliable.")


def print_overview(totals: pd.Series, expected_questions: int | None) -> None:
    if totals.empty:
        print("[WARN] No topic records found.")
        return
    total_questions = int(totals.sum())
    print(f"[INFO] Aggregated {total_questions} questions across {len(totals)} generator(s).")
    for gen, count in totals.items():
        msg = f"  - {gen}: {int(count)} questions"
        if expected_questions and expected_questions > 0 and count != expected_questions:
            msg += f" (expected {expected_questions})"
        print(msg)


def print_topic_tables(
    counts: pd.DataFrame, totals: pd.Series, topic_order: Sequence[str]
) -> None:
    if counts.empty:
        return
    generators = list(totals.index)
    for gen in generators:
        subset = counts[counts["generator_model"] == gen][["topic", "count", "proportion"]]
        subset = subset.sort_values("topic")
        subset = subset.assign(proportion=subset["proportion"].fillna(0).round(3))
        print(f"\nTopic distribution for generator: {gen} (N={int(totals[gen])})")
        print(subset.to_string(index=False))

    wide = (
        counts.pivot(index="topic", columns="generator_model", values="proportion")
        .reindex(topic_order)
        .fillna(0)
    )
    print("\nProportion by topic (rows) and generator (columns):")
    print(wide.round(3).to_string())


def plot_topic_distribution(
    counts: pd.DataFrame, topic_order: Sequence[str], out_png: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"[WARN] Could not import seaborn/matplotlib for plotting: {exc}")
        return

    if counts.empty:
        print("[WARN] Nothing to plot; counts are empty.")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.25)

    n_topics = len(topic_order)
    fig_width = max(6.5, 0.5 * n_topics)
    fig_height = 4.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.barplot(
        data=counts,
        x="topic",
        y="proportion",
        hue="generator_model",
        order=topic_order,
        palette="colorblind",
        ci=None,
        ax=ax,
    )

    ax.set_ylabel("Proportion of questions")
    ax.set_xlabel("Topic")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Generator model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate topic classification outputs across generator runs and plot "
            "per-generator topic distributions."
        )
    )
    ap.add_argument(
        "--runs_root",
        type=str,
        default="eval_results/quizbench/runs",
        help="Root directory containing generator run folders.",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        required=True,
        help="Destination PNG path for the Seaborn/matplotlib bar plot.",
    )
    ap.add_argument(
        "--generators",
        type=str,
        default=None,
        help="Optional comma-separated generator_model filter.",
    )
    ap.add_argument(
        "--eval_models",
        type=str,
        default=None,
        help="Optional comma-separated eval_model filter.",
    )
    ap.add_argument(
        "--expected_questions",
        type=int,
        default=50,
        help="Expected number of questions per generator (for sanity warnings).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_png = Path(args.out_png)
    allowed_generators = parse_csv_arg(args.generators)
    allowed_eval_models = parse_csv_arg(args.eval_models)

    topic_files = discover_topic_files(runs_root)
    if not topic_files:
        raise SystemExit(f"[FATAL] No topics_*.json files found under {runs_root}")

    df, topic_order = load_topic_rows(
        topic_files, allowed_generators=allowed_generators, allowed_eval_models=allowed_eval_models
    )
    counts, totals, topic_order = compute_topic_distribution(df, topic_order)
    print_overview(totals, expected_questions=args.expected_questions)
    print_topic_tables(counts, totals, topic_order)
    run_chi_square_test(counts, topic_order)
    plot_topic_distribution(counts, topic_order, out_png)


if __name__ == "__main__":
    main()
