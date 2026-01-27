#!/usr/bin/env python3
"""Plot accuracy vs cost per quiz from data/acc_vs_cost.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot accuracy vs cost per quiz.")
    parser.add_argument(
        "--input",
        default="data/acc_vs_cost.csv",
        help="Path to the CSV with accuracy and cost_per_quiz columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/quizbench",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--prefix",
        default="acc_vs_cost_scatter",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("none", "pareto", "topk"),
        default="pareto",
        help="Which points to label to reduce clutter.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=5,
        help="Number of top-accuracy points to label when using topk mode.",
    )
    parser.add_argument(
        "--label-column",
        default="model",
        help="Column name to use for point labels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = {"accuracy", "cost_per_quiz"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        },
    )

    def pareto_frontier_indices(data: pd.DataFrame) -> Iterable[int]:
        sorted_df = data.sort_values("cost_per_quiz", ascending=True)
        pareto = []
        best_acc = float("-inf")
        for idx, row in sorted_df.iterrows():
            acc = float(row["accuracy"])
            if acc > best_acc + 1e-9:
                pareto.append(idx)
                best_acc = acc
        return pareto

    def label_points(ax: plt.Axes, data: pd.DataFrame) -> None:
        if args.label_mode == "none" or args.label_column not in data.columns:
            return
        if args.label_mode == "pareto":
            label_df = data.loc[list(pareto_frontier_indices(data))]
        else:
            label_df = data.nlargest(args.label_top_k, "accuracy")

        for _, row in label_df.iterrows():
            dx, dy = 6, 6
            label = str(row[args.label_column])
            if "family" in data.columns:
                label_color = family_colors.get(row["family"], "#1f2937")
            else:
                label_color = "#1f2937"
            text = ax.annotate(
                label,
                (row["cost_per_quiz"], row["accuracy"]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=10,
                color=label_color,
            )
            text.set_path_effects(
                [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
            )

    families = sorted(df["family"].dropna().unique()) if "family" in df.columns else []
    family_palette = sns.color_palette("colorblind", n_colors=len(families))
    family_colors = dict(zip(families, family_palette))

    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "P",
        "X",
        "h",
        "H",
        "d",
        "p",
        "8",
        "1",
        "2",
        "3",
        "4",
        "+",
        "x",
        "|",
        "_",
        "*",

    ]

    models = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
    if models and len(models) > len(markers):
        raise ValueError(
            f"Not enough unique markers for {len(models)} models. "
            f"Add more markers or reduce the number of models."
        )
    model_markers = {model: markers[i] for i, model in enumerate(models)}

    def plot_scatter(log_x: bool, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        scatter_kwargs = dict(
            data=df,
            x="cost_per_quiz",
            y="accuracy",
            s=55,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=2,
            ax=ax,
        )
        if "family" in df.columns and "model" in df.columns:
            scatter_kwargs["hue"] = "family"
            scatter_kwargs["palette"] = family_colors
            scatter_kwargs["style"] = "model"
            scatter_kwargs["markers"] = model_markers
            scatter_kwargs["legend"] = False
        elif "family" in df.columns:
            scatter_kwargs["hue"] = "family"
            scatter_kwargs["palette"] = family_colors
        else:
            scatter_kwargs["color"] = "#2563eb"
        sns.scatterplot(**scatter_kwargs)

        pareto_idx = list(pareto_frontier_indices(df))
        pareto_line = None
        if pareto_idx:
            pareto_df = df.loc[pareto_idx].sort_values("cost_per_quiz", ascending=True)
            (pareto_line,) = ax.step(
                pareto_df["cost_per_quiz"],
                pareto_df["accuracy"],
                where="post",
                color="#9ca3af",
                linestyle="--",
                linewidth=1.5,
                label="Pareto frontier",
                zorder=1,
            )
        if "family" in df.columns and "model" in df.columns:
            star_models = [model for model, marker in model_markers.items() if marker == "*"]
            if star_models:
                star_df = df[df["model"].isin(star_models)]
                colors = star_df["family"].map(family_colors).fillna("#2563eb")
                ax.scatter(
                    star_df["cost_per_quiz"],
                    star_df["accuracy"],
                    s=55 * 1.5,
                    marker="*",
                    c=colors,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.9,
                    zorder=3,
                )
        ax.set_title("Accuracy vs Cost per Quiz")
        ax.set_xlabel("Cost per Quiz (USD)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(top=100)
        ax.margins(0.08)
        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Cost per Quiz (USD, log scale)")
        label_points(ax, df)
        if "family" in df.columns and "model" in df.columns:
            legend_handles = []
            for model in models:
                family = df.loc[df["model"] == model, "family"].iloc[0]
                color = family_colors.get(family, "#2563eb")
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        marker=model_markers[model],
                        color="none",
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markersize=7,
                        linewidth=0,
                        label=model,
                    )
                )
            if pareto_line is not None:
                legend_handles.append(pareto_line)
            ax.legend(
                handles=legend_handles,
                title="Model",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
                fontsize=7,
                title_fontsize=8,
            )
        elif "family" in df.columns:
            ax.legend(
                title="Family",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )
        elif pareto_line is not None:
            ax.legend(handles=[pareto_line], frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        if args.show:
            plt.show()
        plt.close(fig)

    plot_scatter(False, output_dir / f"{args.prefix}.png")
    plot_scatter(True, output_dir / f"{args.prefix}_logx.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
