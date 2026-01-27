#!/usr/bin/env python3
"""
Generate a static `index.html` MedEvalArena page based on:
  https://github.com/kmranrg/academic-project-page-template.

This script is intended to consume the CSV produced by:
  bash aggregate_filtered_results.sh

Specifically, it expects the `OUT_CSV_BY_MODEL` output (default:
`/tmp/agg_majority_by_model.csv`), which contains per-answer-model mean accuracy
aggregated across quizzes after the filtering rules in
`aggregate_filtered_results.sh`.

The output is a single HTML file suitable for GitHub Pages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html as _html
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COST_CSV_PATH = REPO_ROOT / "data" / "cost.csv"

DEFAULT_PUBLICATION_AUTHORS: list[dict[str, Any]] = [
    {"name": "Preethi Prem", "affiliations": [1]},
    {"name": "Kie Shidara", "affiliations": [2]},
    {"name": "Vikasini Kuppa", "affiliations": [3]},
    {"name": "Feng Liu", "affiliations": [4]},
    {"name": "Ahmed Alaa", "affiliations": [5]},
    {
        "name": "Danilo Bernardo",
       "url": "mailto:dbernardoj@gmail.com",
        "affiliations": [2],
        "corresponding": True,
    },
]

DEFAULT_PUBLICATION_AFFILIATIONS: list[dict[str, Any]] = [
    {
        "id": 1,
        "text": (
            "Carle Illinois College of Medicine, University of Illinois Urbana-Champaign, "
            "Urbana, IL"
        ),
    },
    {
        "id": 2,
        "text": (
            "Weill Institute of Neurology and Neurosciences, University of California, "
            "San Francisco, San Francisco, CA"
        ),
    },
    {"id": 3, "text": "University of California, Riverside, Riverside, CA"},
    {
        "id": 4,
        "text": (
            "Department of Systems Engineering, Stevens Institute of Technology, Hoboken, NJ"
        ),
    },
    {
        "id": 5,
        "text": (
            "Department of EECS, University of California Berkeley, Berkeley, CA"
        ),
    },
]


def _default_csv_by_model_path(quiz_batch_tag: str | None) -> Path:
    tag_suffix = f"_{quiz_batch_tag}" if quiz_batch_tag else ""
    return Path(f"/tmp/agg_majority_by_model{tag_suffix}.csv")


def _parse_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _parse_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_accuracy_by_model_csv(csv_path: Path) -> list[dict[str, Any]]:
    """
    Load `--out_csv_by_model` from `quizbench/aggregate_results.py`.

    Expected columns:
      - model (str)
      - num_quizzes (int)
      - mean_accuracy (float in [0, 1])
      - sem (float in [0, 1])
      - generator_models (comma-separated str)
      - raw_generator_models (comma-separated str)
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")

        required = {"model", "mean_accuracy"}
        missing = required - set(reader.fieldnames)
        if missing:
            cols = ", ".join(reader.fieldnames)
            raise ValueError(
                f"CSV missing required columns {sorted(missing)}. "
                f"Found columns: {cols}"
            )

        rows: list[dict[str, Any]] = []
        for raw in reader:
            model = (raw.get("model") or "").strip()
            if not model:
                continue

            mean_accuracy = _parse_float(raw.get("mean_accuracy"))
            sem = _parse_float(raw.get("sem"))
            num_quizzes = _parse_int(raw.get("num_quizzes"))
            generator_models = (raw.get("generator_models") or "").strip()
            raw_generator_models = (raw.get("raw_generator_models") or "").strip()

            generator_count: int | None = None
            if generator_models:
                generator_count = len([s for s in generator_models.split(",") if s.strip()])

            rows.append(
                {
                    "model": model,
                    "num_quizzes": num_quizzes,
                    "mean_accuracy": mean_accuracy,
                    "sem": sem,
                    "generator_count": generator_count,
                    "generator_models": generator_models,
                    "raw_generator_models": raw_generator_models,
                }
            )

    # Default ordering: mean_accuracy desc, then model name.
    rows.sort(
        key=lambda r: (
            -(r["mean_accuracy"] if isinstance(r.get("mean_accuracy"), (int, float)) else -1.0),
            r["model"].lower(),
        )
    )
    return rows


def load_cost_by_model_csv(csv_path: Path) -> dict[str, float]:
    """
    Load cost-per-quiz metadata by model.

    Expected columns:
      - model (str)
      - cost_per_quiz (float)
    """
    if not csv_path.exists():
        return {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        required = {"model", "cost_per_quiz"}
        missing = required - set(reader.fieldnames)
        if missing:
            return {}

        costs: dict[str, float] = {}
        for raw in reader:
            model = (raw.get("model") or "").strip()
            if not model:
                continue
            cost = _parse_float(raw.get("cost_per_quiz"))
            if cost is None:
                continue
            costs[model] = cost

    return costs


def _json_for_inline_script(obj: Any) -> str:
    """
    Produce JSON that is safe to embed directly inside a <script> tag.
    """
    s = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return s.replace("<", "\\u003c")


def _h(s: str) -> str:
    """HTML-escape a string for safe insertion into an HTML attribute or node."""
    return _html.escape(s, quote=True)


def _render_publication_authors(meta: dict[str, Any]) -> str:
    authors_raw = meta.get("authors")
    if not isinstance(authors_raw, list):
        return ""

    authors: list[dict[str, Any]] = [a for a in authors_raw if isinstance(a, dict)]
    author_nodes: list[str] = []
    corresponding_count = 0

    for author in authors:
        name = str(author.get("name") or "").strip()
        if not name:
            continue

        url = str(author.get("url") or "").strip()
        corresponding = bool(author.get("corresponding"))
        if corresponding:
            corresponding_count += 1

        aff_ids: list[int] = []
        for raw_aff_id in author.get("affiliations") or []:
            aff_id = _parse_int(raw_aff_id)
            if aff_id is not None:
                aff_ids.append(aff_id)

        sup_text = ", ".join(str(n) for n in aff_ids)
        if corresponding:
            sup_text = f"{sup_text}*" if sup_text else "*"

        name_html = _h(name)
        if url:
            extra = ' target="_blank" rel="noopener"' if url.startswith("http") else ""
            name_html = f'<a href="{_h(url)}"{extra}>{name_html}</a>'

        sup_html = f"<sup>{_h(sup_text)}</sup>" if sup_text else ""
        author_nodes.append(f"{name_html}{sup_html}")

    if not author_nodes:
        return ""

    author_spans: list[str] = []
    for i, node in enumerate(author_nodes):
        comma = "," if i < len(author_nodes) - 1 else ""
        author_spans.append(f'<span class="author-block">{node}{comma}</span>')

    affiliations_raw = meta.get("affiliations")
    affiliations: list[dict[str, Any]] = (
        [a for a in affiliations_raw if isinstance(a, dict)]
        if isinstance(affiliations_raw, list)
        else []
    )

    affiliation_nodes: list[str] = []
    for aff in affiliations:
        aff_id = _parse_int(aff.get("id"))
        text = str(aff.get("text") or "").strip()
        if aff_id is None or not text:
            continue
        affiliation_nodes.append(f"<sup>{aff_id}</sup>{_h(text)}")

    affiliation_spans: list[str] = []
    for i, node in enumerate(affiliation_nodes):
        comma = "," if i < len(affiliation_nodes) - 1 else ""
        affiliation_spans.append(f'<span class="author-block">{node}{comma}</span>')

    corresponding_html = ""
    if corresponding_count:
        note = str(meta.get("corresponding_note") or "").strip()
        if not note:
            note = "Corresponding author" if corresponding_count == 1 else "Corresponding authors"
        corresponding_html = f"""
          <div class="is-size-5 publication-authors">
            <span class="author-block"><sup>*</sup>{_h(note)}</span>
          </div>
        """.rstrip()

    affiliations_html = ""
    if affiliation_spans:
        affiliations_html = f"""
          <div class="is-size-5 publication-authors">
            {' '.join(affiliation_spans)}
          </div>
        """.rstrip()

    return f"""
          <div class="is-size-5 publication-authors">
            {' '.join(author_spans)}
          </div>
{affiliations_html}
{corresponding_html}
    """.rstrip()


def render_index_html(*, payload: dict[str, Any]) -> str:
    """
    Render a Bulma-based academic-project-style page, inspired by OlympicArena and
        https://github.com/kmranrg/academic-project-page-template.
    """
    meta = payload.get("meta", {}) or {}
    data_json = _json_for_inline_script(payload)

    title = str(meta.get("title") or "MedEvalArena")
    title_html = _h(title)

    description = str(meta.get("description") or title)
    keywords = str(meta.get("keywords") or "LLMs, Evaluation, Medicine")
    emoji = str(meta.get("emoji") or "🏥")
    subtitle = str(meta.get("subtitle") or "")

    home_url = str(meta.get("home_url") or "")
    paper_url = str(meta.get("paper_url") or "")
    code_url = str(meta.get("code_url") or "")
    data_url = str(meta.get("data_url") or "")
    submit_url = str(meta.get("submit_url") or "")
    twitter_url = str(meta.get("twitter_url") or "")

    contact_email = str(meta.get("contact_email") or "")
    issues_url = str(meta.get("issues_url") or "")
    bibtex = str(meta.get("bibtex") or "")

    def _btn(label: str, url: str, icon_html: str) -> str:
        u = _h(url)
        extra = 'target="_blank" rel="noopener"' if url.startswith("http") else ""
        return f"""
            <span class="link-block">
              <a href="{u}" class="external-link button is-normal is-rounded is-dark" {extra}>
                <span class="icon">{icon_html}</span>
                <span>{_h(label)}</span>
              </a>
            </span>
        """.rstrip()

    buttons: list[str] = []
    if paper_url:
        buttons.append(_btn("Paper", paper_url, '<i class="fas fa-file-pdf"></i>'))
    if code_url:
        buttons.append(_btn("Code", code_url, '<i class="fab fa-github"></i>'))
    if data_url:
        buttons.append(_btn("Dataset", data_url, '<span style="font-size:18px">🤗</span>'))
    buttons.append(_btn("Leaderboard", "#leaderboard", '<i class="fas fa-chart-bar"></i>'))
    buttons.append(_btn("Tables", "#tables", '<span style="font-size:18px">🏆</span>'))
    if submit_url:
        buttons.append(_btn("Submit", submit_url, '<span style="font-size:18px">📤</span>'))
    if twitter_url:
        buttons.append(_btn("Twitter", twitter_url, '<i class="fab fa-twitter"></i>'))

    subtitle_html = _h(subtitle) if subtitle else ""

    contact_lines: list[str] = []
    if contact_email:
        contact_lines.append(
            f'Email: <a href="mailto:{_h(contact_email)}">{_h(contact_email)}</a>'
        )
    if issues_url:
        contact_lines.append(
            f'GitHub Issues: <a href="{_h(issues_url)}" target="_blank" rel="noopener">{_h(issues_url)}</a>'
        )
    contact_html = "<br/>".join(contact_lines) if contact_lines else (
        "For questions, please open a GitHub issue on the repository."
    )

    bibtex_section = ""
    if bibtex.strip():
        bibtex_section = f"""
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title is-3">BibTeX</h2>
    <pre><code>{_h(bibtex.strip())}</code></pre>
  </div>
</section>
""".rstrip()

    publication_authors_html = _render_publication_authors(meta)

    generated_at = _h(str(meta.get("generated_at") or ""))
    source_csv = _h(str(meta.get("source_csv") or ""))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="description" content="{_h(description)}">
  <meta name="keywords" content="{_h(keywords)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_html}</title>

  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

  <style>
    body {{
      font-family: "Noto Sans", "Google Sans", sans-serif;
    }}
    .publication-title {{
      font-family: "Google Sans", "Noto Sans", sans-serif;
      letter-spacing: -0.02em;
      line-height: 1.15;
    }}
    .publication-subtitle {{
      margin-top: 0.6rem;
      color: #4a4a4a;
    }}
    .publication-authors {{
      margin-top: 0.35rem;
      margin-bottom: 0.35rem;
    }}
    .author-block {{
      display: inline-block;
      margin: 0 0.2rem;
    }}
    .author-block a {{
      color: inherit;
    }}

    /* Results table tweaks (in spirit of OlympicArena). */
    #results-table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 0.75rem;
      border: 1px solid #ddd;
      font-size: 14px;
    }}
    #results-table th, #results-table td {{
      padding: 10px 12px;
      vertical-align: middle;
    }}
    #results-table thead th {{
      background-color: #f2f2f2;
      border-bottom: 2px solid #ddd;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }}
    #results-table tbody tr:hover {{
      background-color: rgba(117, 209, 215, 0.08);
    }}
    .rank-badge {{
      display: inline-flex;
      min-width: 2.1rem;
      height: 2.1rem;
      align-items: center;
      justify-content: center;
      border-radius: 0.6rem;
      border: 1px solid #ddd;
      background: #fff;
      font-weight: 600;
      color: #363636;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    .centered-footer {{
      text-align: center;
    }}
    .sticky-controls {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: white;
      padding-top: 0.5rem;
      padding-bottom: 0.5rem;
    }}

    /* Leaderboard bar plot */
    :root {{
      --leaderboard-track: #e9eef4;
      --leaderboard-bg: #f7f9fc;
      --leaderboard-text: #2f2f2f;
    }}
    .leaderboard-card {{
      background: var(--leaderboard-bg);
      border: 1px solid #e2e7ef;
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 10px 28px rgba(16, 24, 40, 0.06);
    }}
    .leaderboard-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
      align-items: stretch;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .leaderboard-legend {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.85rem;
      color: #5f6b7a;
      margin-bottom: 0.75rem;
    }}
    .leaderboard-dot {{
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 999px;
      background: #3a86ff;
      display: inline-block;
    }}
    .leaderboard-plot {{
      --leader-bar-height: 170px;
      --leader-label-space: 2.4rem;
      --leader-axis-space: 2.6rem;
      --leader-gridline-color: rgba(47, 47, 47, 0.18);
      --leader-gridlines: none;
      --leader-gridline-positions: 0 0;
      position: relative;
      display: grid;
      grid-template-columns: repeat(var(--leader-count, 1), minmax(22px, 1fr));
      align-items: end;
      gap: 0.5rem;
      min-height: calc(var(--leader-bar-height) + var(--leader-label-space) + 1.2rem);
      padding: 0.25rem 0.25rem var(--leader-label-space) var(--leader-axis-space);
    }}
    .leaderboard-plot::before {{
      content: "";
      position: absolute;
      left: var(--leader-axis-space);
      right: 0;
      bottom: var(--leader-label-space);
      height: var(--leader-bar-height);
      background-image: var(--leader-gridlines);
      background-size: 100% 1px;
      background-position: var(--leader-gridline-positions);
      background-repeat: no-repeat;
      pointer-events: none;
    }}
    .leaderboard-plot--accuracy {{
      --leader-gridlines:
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color));
      --leader-gridline-positions: 0 100%, 0 75%, 0 50%, 0 25%, 0 0;
    }}
    .leaderboard-plot--accuracy .leader-bar-label {{
      align-items: flex-end;
      padding-bottom: 0.35rem;
    }}
    .leader-bar-label.leader-bar-label--bottom {{
      align-items: flex-end;
      padding-bottom: 0.35rem;
    }}
    .leader-bar-label.leader-bar-label--above {{
      inset: auto;
      left: 50%;
      bottom: calc(var(--leader-fill-height, 0%) + 0.25rem);
      transform: translateX(-50%);
      width: auto;
      height: auto;
      color: #4a4a4a;
      text-shadow: none;
      align-items: center;
      justify-content: center;
    }}
    .leaderboard-plot--cost {{
      --leader-gridlines:
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color)),
        linear-gradient(to right, var(--leader-gridline-color), var(--leader-gridline-color));
      --leader-gridline-positions: 0 100%, 0 75%, 0 50%, 0 25%, 0 0;
    }}
    .leader-y-axis {{
      position: absolute;
      left: 0;
      bottom: var(--leader-label-space);
      height: var(--leader-bar-height);
      width: var(--leader-axis-space);
      pointer-events: none;
      z-index: 2;
    }}
    .leader-y-tick {{
      position: absolute;
      right: 0;
      padding-right: 0.35rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: #5f6b7a;
      line-height: 1;
      white-space: nowrap;
      background: var(--leaderboard-bg);
    }}
    .leader-col {{
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      z-index: 1;
    }}
    .leader-label-anchor {{
      position: absolute;
      left: 50%;
      bottom: -0.5rem;
      width: 0;
      height: 0;
    }}
    .leader-label-shift {{
      display: inline-block;
      transform: translateX(-100%);
    }}
    .leader-label {{
      display: inline-block;
      font-weight: 600;
      color: var(--leaderboard-text);
      font-size: 0.78rem;
      white-space: nowrap;
      line-height: 1.1;
      transform: rotate(-65deg);
      transform-origin: top right;
    }}
    .leader-bar-vertical {{
      position: relative;
      height: var(--leader-bar-height);
      width: 55%;
      border-radius: 12px 12px 0 0;
      background-color: transparent;
      overflow: hidden;
      display: flex;
      align-items: flex-end;
    }}
    .leader-bar-fill {{
      position: relative;
      width: 100%;
      border-radius: 12px 12px 0 0;
      transition: height 0.6s ease;
      z-index: 1;
    }}
    .leader-bar-label {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: 700;
      color: #fff;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
      pointer-events: none;
      z-index: 2;
    }}
    .leaderboard-scatter-svg {{
      width: 100%;
      height: 280px;
      display: block;
      overflow: visible;
    }}
    .leaderboard-scatter-wrap {{
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      align-items: stretch;
    }}
    .leaderboard-scatter-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem 1rem;
      align-items: center;
      justify-content: center;
      color: #5f6b7a;
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .leaderboard-scatter-legend-label {{
      font-weight: 700;
      color: #1f2937;
      margin-right: 0.3rem;
    }}
    .leaderboard-scatter-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .leaderboard-scatter-legend-icon {{
      width: 18px;
      height: 18px;
      display: block;
    }}
    .leaderboard-scatter-legend-line {{
      width: 22px;
      height: 10px;
      display: block;
    }}
    .leaderboard-scatter-gridline {{
      stroke: rgba(47, 47, 47, 0.18);
      stroke-width: 1;
      shape-rendering: crispEdges;
    }}
    .leaderboard-scatter-axis {{
      stroke: rgba(47, 47, 47, 0.3);
      stroke-width: 1.2;
      shape-rendering: crispEdges;
    }}
    .leaderboard-scatter-tick {{
      fill: #5f6b7a;
      font-size: 14px;
      font-weight: 600;
    }}
    .leaderboard-scatter-axis-label {{
      fill: #5f6b7a;
      font-size: 15px;
      font-weight: 700;
    }}
    .leaderboard-scatter-pareto {{
      stroke: #9ca3af;
      stroke-width: 1.5;
      stroke-dasharray: 6 4;
      fill: none;
      stroke-linejoin: round;
      stroke-linecap: round;
    }}
    .leaderboard-scatter-point {{
      stroke: #fff;
      stroke-width: 2;
    }}
    .leaderboard-scatter-point--pareto {{
      stroke: #fff;
      stroke-width: 2;
    }}
    @media (max-width: 768px) {{
      .leaderboard-grid {{
        grid-template-columns: 1fr;
      }}
      .leaderboard-plot {{
        --leader-bar-height: 150px;
        --leader-label-space: 2.1rem;
        --leader-axis-space: 2.3rem;
        grid-template-columns: repeat(var(--leader-count, 1), minmax(20px, 1fr));
        gap: 0.4rem;
      }}
      .leader-label {{
        font-size: 0.72rem;
        transform: rotate(-30deg);
      }}
    }}
  </style>
</head>

<body>
<nav class="navbar" role="navigation" aria-label="main navigation">
  <div class="navbar-brand">
    <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false" data-target="navbarMain">
      <span aria-hidden="true"></span>
      <span aria-hidden="true"></span>
      <span aria-hidden="true"></span>
    </a>
  </div>

  <div id="navbarMain" class="navbar-menu">
    <div class="navbar-start" style="flex-grow: 1; justify-content: center;">
      {f'<a class="navbar-item" href="{_h(home_url)}"><span class="icon"><i class="fas fa-home"></i></span></a>' if home_url else ''}
      {f'<a class="navbar-item" href="{_h(code_url)}" target="_blank" rel="noopener"><span class="icon"><i class="fab fa-github"></i></span></a>' if code_url else ''}
      <a class="navbar-item" href="#leaderboard">Leaderboard</a>
      <a class="navbar-item" href="#tables">Tables</a>
    </div>
  </div>
</nav>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">{_h(emoji)} {title_html}</h1>
          {publication_authors_html}
          {f'<p class="is-size-5 publication-subtitle">{subtitle_html}</p>' if subtitle_html else ''}

          <div class="column has-text-centered" style="margin-top: 1rem;">
            <div class="publication-links">
              {''.join(buttons)}
            </div>
          </div>

        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <figure class="image" style="margin: 0 auto 0.2rem; max-width: 85%;">
          <img
            src="website_images/intro_graphic.png"
            alt="MedEvalArena introduction graphic"
            style="display: block; width: 100%; height: auto;"
          />
        </figure>
        <h2 class="title is-3">Introduction</h2>
        <div class="content has-text-justified">
          <p>
            Large Language Models have shown strong performance in medical
            question answering, but their capabilities in complex clinical
            reasoning remain difficult to characterize systematically.
            We present <strong>MedEvalArena</strong>, a dynamic evaluation
            framework designed to compare medical reasoning robustness
            across models using a symmetric, adversarial round-robin protocol.
          </p>

          <p>
            In MedEvalArena, each model generates adversarial medical quizzes
            intended to challenge the reasoning abilities of other models.
            All models are then evaluated on the full shared quiz set,
            enabling controlled and scalable comparisons beyond static
            benchmarks.
          </p>

          <p>
            Responses are assessed using an LLM-as-judge paradigm along two
            orthogonal axes, logical correctness and medical accuracy. By 
            co-evolving the evaluation data with frontier LLMs, MedEvalArena 
            provides a principled framework for evaluating medical reasoning 
            in both LLMs and humans.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-fluid">
    <div class="columns is-centered has-text-centered">
      <div class="column is-full">
        <h2 id="leaderboard" class="title is-3"><i class="fas fa-chart-bar"></i>&nbsp;Leaderboard</h2>
        <div class="leaderboard-grid">
          <div class="leaderboard-card">
            <div class="leaderboard-legend">
              <span class="leaderboard-dot"></span>
              <span>Accuracy</span>
            </div>
            <div id="leaderboard-accuracy"></div>
          </div>
          <div class="leaderboard-card">
            <div class="leaderboard-legend">
              <span class="leaderboard-dot"></span>
              <span>Cost per evaluation</span>
            </div>
            <div id="leaderboard-cost"></div>
          </div>
          <div class="leaderboard-card">
            <div class="leaderboard-legend">
              <span class="leaderboard-dot"></span>
              <span>Accuracy vs Cost per evaluation</span>
            </div>
            <div id="leaderboard-accuracy-vs-cost"></div>
          </div>
        </div>
        <div class="content has-text-justified">
          <p class="help">
            Up to Top-10 models by accuracy shown. Each evaluation contains 300 questions (50 questions generated per LLM).
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 id="tables" class="title is-3">🏟️ Results</h2>
        <div class="content has-text-justified">
          <p class="help">
            Default sort is by <b>Mean Accuracy</b> (descending). Top-3 entries are marked with 🥇🥈🥉.
          </p>
        </div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="columns is-centered">
      <div class="column is-four-fifths">
        <div class="sticky-controls">
          <div class="field is-grouped is-grouped-multiline" style="margin-bottom: 0.5rem;">
            <div class="control is-expanded">
              <input id="modelFilter" class="input" type="search" placeholder="Filter by model name…" aria-label="Filter by model name" />
            </div>
            <div class="control">
              <div class="tags has-addons">
                <span class="tag is-light">Models</span>
                <span id="modelCount" class="tag is-info">0</span>
              </div>
            </div>
            <div class="control">
              <div class="tags has-addons">
                <span class="tag is-light">Generated</span>
                <span class="tag is-light mono">{generated_at}</span>
              </div>
            </div>
          </div>
        </div>

        <div style="overflow-x: auto;">
          <table id="results-table" class="table is-striped is-hoverable is-fullwidth">
            <thead>
              <tr>
                <th data-key="rank">#</th>
                <th data-key="model">Model</th>
                <th class="has-text-right" data-key="mean_accuracy">Mean Accuracy</th>
                <th class="has-text-right" data-key="sem">SEM</th>
              </tr>
            </thead>
            <tbody id="tables-body"></tbody>
          </table>
        </div>

        <p class="help">Tip: click a column header to sort.</p>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">📬 Contact</h2>
        <div class="content has-text-justified">
          <p>{contact_html}</p>
        </div>
      </div>
    </div>
  </div>
</section>

{bibtex_section}

<footer class="footer centered-footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">
          <p>Template inspired by https://github.com/kmranrg/academic-project-page-template.</p>
        </div>
      </div>
    </div>
  </div>
</footer>

<script id="tables-data" type="application/json">{data_json}</script>

<script>
  // Bulma navbar burger toggle
  document.addEventListener('DOMContentLoaded', () => {{
    const burgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
    burgers.forEach((el) => {{
      el.addEventListener('click', () => {{
        const target = el.dataset.target;
        const menu = document.getElementById(target);
        el.classList.toggle('is-active');
        menu.classList.toggle('is-active');
      }});
    }});
  }});

  (function() {{
    const dataEl = document.getElementById('tables-data');
    const payload = JSON.parse(dataEl.textContent);
    const rowsRaw = Array.isArray(payload.rows) ? payload.rows : [];
    const costByModel = payload.cost_by_model || {{}};

    const filterEl = document.getElementById('modelFilter');
    const bodyEl = document.getElementById('tables-body');
    const countEl = document.getElementById('modelCount');
    const tableEl = document.getElementById('results-table');
    const accuracyRootEl = document.getElementById('leaderboard-accuracy');
    const costRootEl = document.getElementById('leaderboard-cost');
    const scatterRootEl = document.getElementById('leaderboard-accuracy-vs-cost');

    function isNum(x) {{
      return typeof x === 'number' && isFinite(x);
    }}

    function fmtPct(x) {{
      if (!isNum(x)) return '—';
      return (x * 100).toFixed(2) + '%';
    }}

    function fmtPctInt(x) {{
      if (!isNum(x)) return '—';
      return String(Math.round(x * 100)) + '%';
    }}

    function fmtPctIntNoSign(x) {{
      if (!isNum(x)) return '—';
      return String(Math.round(x * 100));
    }}

    function fmtInt(x) {{
      if (!isNum(x)) return '—';
      return String(Math.trunc(x));
    }}

    function fmtUsd(x) {{
      if (!isNum(x)) return '—';
      return '$' + x.toFixed(2);
    }}

    function fmtUsdInt(x) {{
      if (!isNum(x)) return '—';
      return '$' + String(Math.round(x));
    }}

    function fmtCentsNoDollar(x) {{
      if (!isNum(x)) return '—';
      return x.toFixed(2);
    }}

    function mountReact(element, component) {{
      if (!element || !window.React || !window.ReactDOM) return;
      if (typeof ReactDOM.createRoot === 'function') {{
        ReactDOM.createRoot(element).render(component);
      }} else {{
        ReactDOM.render(component, element);
      }}
    }}

    function sortByAccuracy(rows) {{
      const sorted = rows.slice();
      sorted.sort((a, b) => {{
        const na = isNum(a.mean_accuracy) ? a.mean_accuracy : -Infinity;
        const nb = isNum(b.mean_accuracy) ? b.mean_accuracy : -Infinity;
        if (na === nb) {{
          return String(a.model || '').localeCompare(String(b.model || ''));
        }}
        return nb - na;
      }});
      return sorted;
    }}

    function colorStops(idx, total) {{
      const t = total > 1 ? (idx / (total - 1)) : 0;
      const hue = 210 - (t * 80);
      return 'hsl(' + hue + ', 70%, 52%)';
    }}

    function renderLeaderboardChart(rootEl, rows, valueFn, valueFmt, emptyMsg, colorFn, options) {{
      if (!rootEl) return;
      if (!window.React || !window.ReactDOM) {{
        rootEl.textContent = 'Leaderboard chart requires React.';
        return;
      }}

      const opts = options || {{}};
      const trackClass = (typeof opts.trackClass === 'string') ? opts.trackClass : '';
      const plotClass = (typeof opts.plotClass === 'string') ? opts.plotClass : '';
      const labelFmt = (typeof opts.labelFmt === 'function') ? opts.labelFmt : valueFmt;
      const yTicks = Array.isArray(opts.yTicks) ? opts.yTicks : [];
      const minValue = isNum(opts.minValue) ? opts.minValue : 0;
      const maxValueOverride = isNum(opts.maxValue) ? opts.maxValue : null;
      const wantsAutoLabelSpace = opts.autoLabelSpace !== false;
      const labelBottomMinValue = isNum(opts.labelBottomMinValue) ? opts.labelBottomMinValue : null;

      if (!rows.length) {{
        rootEl.textContent = emptyMsg;
        return;
      }}

      const numericValues = rows.map(valueFn).filter(isNum);
      if (!numericValues.length) {{
        rootEl.textContent = emptyMsg;
        return;
      }}

      const maxValue = (maxValueOverride !== null) ? maxValueOverride : Math.max.apply(null, numericValues);
      const span = maxValue - minValue;
      const scale = span > 0 ? span : 1;
      const e = React.createElement;

      let labelSpacePx = null;
      if (wantsAutoLabelSpace && rootEl && rootEl.ownerDocument) {{
        try {{
          const doc = rootEl.ownerDocument;
          const isMobile = !!(window.matchMedia && window.matchMedia('(max-width: 768px)').matches);
          const labelAngleDeg = isNum(opts.labelAngleDeg) ? opts.labelAngleDeg : (isMobile ? -30 : -65);
          const labelSpaceMin = isNum(opts.labelSpaceMinPx) ? opts.labelSpaceMinPx : 0;
          const labelSpaceExtra = isNum(opts.labelSpaceExtraPx) ? opts.labelSpaceExtraPx : 26;

          const probe = doc.createElement('span');
          probe.className = 'leader-label mono';
          probe.style.position = 'absolute';
          probe.style.visibility = 'hidden';
          probe.style.pointerEvents = 'none';
          probe.style.whiteSpace = 'nowrap';
          probe.style.transform = 'none';
          probe.style.top = '-10000px';
          probe.style.left = '-10000px';
          doc.body.appendChild(probe);

          let maxW = 0;
          let maxH = 0;
          rows.forEach(r => {{
            probe.textContent = String(r.model || '');
            const rect = probe.getBoundingClientRect();
            if (isNum(rect.width)) maxW = Math.max(maxW, rect.width);
            if (isNum(rect.height)) maxH = Math.max(maxH, rect.height);
          }});
          probe.remove();

          const theta = Math.abs(labelAngleDeg) * Math.PI / 180;
          const drop = (maxW * Math.sin(theta)) + (maxH * Math.cos(theta));
          labelSpacePx = Math.max(labelSpaceMin, Math.ceil(drop + labelSpaceExtra));
        }} catch (err) {{
          labelSpacePx = null;
        }}
      }}

      let yAxisEl = null;
      if (yTicks.length) {{
        const tickNodes = yTicks
          .map((t, tickIdx) => {{
            if (!t || typeof t !== 'object') return null;
            const v = t.value;
            const label = t.label;
            if (!isNum(v)) return null;
            const text = (label === undefined || label === null) ? '' : String(label);
            if (!text) return null;

            const rawPos = ((v - minValue) / scale) * 100;
            const pos = Math.max(0, Math.min(100, rawPos));

            const style = {{}};
            if (pos >= 99.5) {{
              style.top = '0%';
            }} else if (pos <= 0.5) {{
              style.bottom = '0%';
            }} else {{
              style.bottom = pos.toFixed(2) + '%';
              style.transform = 'translateY(50%)';
            }}

            return e(
              'div',
              {{ className: 'leader-y-tick mono', key: 'tick-' + tickIdx, style }},
              text,
            );
          }})
          .filter(Boolean);
        if (tickNodes.length) {{
          yAxisEl = e('div', {{ className: 'leader-y-axis' }}, tickNodes);
        }}
      }}

      const nodes = rows.map((r, idx) => {{
        const value = valueFn(r);
        const hasValue = isNum(value);
        const rawHeight = hasValue ? ((value - minValue) / scale) * 100 : 0;
        const height = Math.max(0, Math.min(100, rawHeight));
        const barVarStyle = {{
          '--leader-fill-height': height.toFixed(2) + '%',
        }};
        const style = {{
          height: height.toFixed(2) + '%',
          backgroundColor: colorFn(r, idx) || '#9aa7b5',
        }};
        const barClassName = 'leader-bar-vertical' + (trackClass ? ' ' + trackClass : '');
        const isLabelBottom = hasValue && labelBottomMinValue !== null && value >= labelBottomMinValue;
        const isLabelAbove = hasValue && labelBottomMinValue !== null && value < labelBottomMinValue;
        const barLabelClassName = 'leader-bar-label mono' + (
          isLabelBottom ? ' leader-bar-label--bottom' : ''
        ) + (
          isLabelAbove ? ' leader-bar-label--above' : ''
        );
        const title = String(r.model || '') + ' - ' + valueFmt(value);
        return e(
          'div',
          {{ className: 'leader-col', key: String(r.model || 'model') + '-' + idx, title }},
          e(
            'div',
            {{ className: barClassName, style: barVarStyle }},
            e('div', {{ className: 'leader-bar-fill', style }}),
            e('div', {{ className: barLabelClassName }}, labelFmt(value)),
          ),
          e(
            'div',
            {{ className: 'leader-label-anchor' }},
            e(
              'div',
              {{ className: 'leader-label-shift' }},
              e('div', {{ className: 'leader-label mono' }}, String(r.model || '')),
            ),
          ),
        );
      }});

      const plotStyle = {{ '--leader-count': String(rows.length) }};
      if (isNum(labelSpacePx)) {{
        plotStyle['--leader-label-space'] = String(labelSpacePx) + 'px';
      }}
      const plotClassName = 'leaderboard-plot' + (plotClass ? ' ' + plotClass : '');
      const children = yAxisEl ? [yAxisEl].concat(nodes) : nodes;
      mountReact(rootEl, e('div', {{ className: plotClassName, style: plotStyle }}, children));
    }}

    function renderScatterChart(rootEl, rows, xFn, yFn, emptyMsg, colorFn, options) {{
      if (!rootEl) return;
      if (!window.React || !window.ReactDOM) {{
        rootEl.textContent = 'Scatter plot requires React.';
        return;
      }}

      const opts = options || {{}};
      const xMin = isNum(opts.xMin) ? opts.xMin : 0;
      const xMax = isNum(opts.xMax) ? opts.xMax : 1;
      const yMin = isNum(opts.yMin) ? opts.yMin : 0;
      const yMax = isNum(opts.yMax) ? opts.yMax : 1;
      const xTicks = Array.isArray(opts.xTicks) ? opts.xTicks : [];
      const yTicks = Array.isArray(opts.yTicks) ? opts.yTicks : [];
      const xLabel = (typeof opts.xLabel === 'string') ? opts.xLabel : '';
      const yLabel = (typeof opts.yLabel === 'string') ? opts.yLabel : '';

      const points = rows
        .map((r, idx) => {{
          const x = xFn(r);
          const y = yFn(r);
          return {{
            r,
            idx,
            x,
            y,
          }};
        }})
        .filter(p => isNum(p.x) && isNum(p.y));

      if (!points.length) {{
        rootEl.textContent = emptyMsg;
        return;
      }}

      const width = 1000;
      const height = 360;
      const margin = {{ left: 86, right: 22, top: 22, bottom: 74 }};
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const spanX = (xMax - xMin) || 1;
      const spanY = (yMax - yMin) || 1;

      function clamp01(t) {{
        return Math.max(0, Math.min(1, t));
      }}

      function sx(x) {{
        const t = clamp01((x - xMin) / spanX);
        return margin.left + (t * plotW);
      }}

      function sy(y) {{
        const t = clamp01((y - yMin) / spanY);
        return margin.top + ((1 - t) * plotH);
      }}

      function paretoFrontier(pointsIn) {{
        const sorted = pointsIn.slice().sort((a, b) => {{
          const dx = a.x - b.x;
          if (dx !== 0) return dx;
          const dy = b.y - a.y;
          if (dy !== 0) return dy;
          return a.idx - b.idx;
        }});
        const pareto = [];
        let bestY = -Infinity;
        sorted.forEach((p) => {{
          if (p.y > bestY + 1e-9) {{
            pareto.push(p);
            bestY = p.y;
          }}
        }});
        return pareto;
      }}

      function buildParetoPath(pointsIn) {{
        if (!pointsIn.length) return '';
        let d = '';
        let prev = pointsIn[0];
        d += 'M ' + sx(prev.x) + ' ' + sy(prev.y);
        for (let i = 1; i < pointsIn.length; i += 1) {{
          const curr = pointsIn[i];
          d += ' L ' + sx(curr.x) + ' ' + sy(prev.y);
          d += ' L ' + sx(curr.x) + ' ' + sy(curr.y);
          prev = curr;
        }}
        return d;
      }}

      function starPath(cx, cy, outerR, innerR) {{
        const spikes = 5;
        const step = Math.PI / spikes;
        let angle = -Math.PI / 2;
        let d = '';
        for (let i = 0; i < spikes * 2; i += 1) {{
          const r = (i % 2 === 0) ? outerR : innerR;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          d += (i === 0 ? 'M ' : ' L ') + x + ' ' + y;
          angle += step;
        }}
        return d + ' Z';
      }}

      function polygonPath(cx, cy, r, sides, rotation) {{
        const n = Math.max(3, sides || 3);
        const rot = isNum(rotation) ? rotation : -Math.PI / 2;
        let d = '';
        for (let i = 0; i < n; i += 1) {{
          const angle = rot + (i * 2 * Math.PI) / n;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          d += (i === 0 ? 'M ' : ' L ') + x + ' ' + y;
        }}
        return d + ' Z';
      }}

      function paretoMarkerPath(cx, cy, idx, outerR, innerR) {{
        if (idx === 0) {{
          return starPath(cx, cy, outerR, innerR);
        }}
        const sides = 3 + idx;
        return polygonPath(cx, cy, outerR, sides, -Math.PI / 2);
      }}

      const e = React.createElement;
      const nodes = [];

      // Gridlines and ticks
      xTicks.forEach((t, i) => {{
        if (!t || typeof t !== 'object') return;
        const v = t.value;
        if (!isNum(v)) return;
        const label = (t.label === undefined || t.label === null) ? '' : String(t.label);
        const x = sx(v);
        nodes.push(
          e('line', {{
            className: 'leaderboard-scatter-gridline',
            key: 'xgrid-' + i,
            x1: x,
            x2: x,
            y1: margin.top,
            y2: margin.top + plotH,
          }}),
        );
        if (label) {{
          nodes.push(
            e('text', {{
              className: 'leaderboard-scatter-tick mono',
              key: 'xlabel-' + i,
              x,
              y: margin.top + plotH + 22,
              textAnchor: 'middle',
              dominantBaseline: 'hanging',
            }}, label),
          );
        }}
      }});

      yTicks.forEach((t, i) => {{
        if (!t || typeof t !== 'object') return;
        const v = t.value;
        if (!isNum(v)) return;
        const label = (t.label === undefined || t.label === null) ? '' : String(t.label);
        const y = sy(v);
        nodes.push(
          e('line', {{
            className: 'leaderboard-scatter-gridline',
            key: 'ygrid-' + i,
            x1: margin.left,
            x2: margin.left + plotW,
            y1: y,
            y2: y,
          }}),
        );
        if (label) {{
          nodes.push(
            e('text', {{
              className: 'leaderboard-scatter-tick mono',
              key: 'ylabel-' + i,
              x: margin.left - 10,
              y,
              textAnchor: 'end',
              dominantBaseline: 'middle',
            }}, label),
          );
        }}
      }});

      // Axes
      nodes.push(
        e('line', {{
          className: 'leaderboard-scatter-axis',
          key: 'xaxis',
          x1: margin.left,
          x2: margin.left + plotW,
          y1: margin.top + plotH,
          y2: margin.top + plotH,
        }}),
      );
      nodes.push(
        e('line', {{
          className: 'leaderboard-scatter-axis',
          key: 'yaxis',
          x1: margin.left,
          x2: margin.left,
          y1: margin.top,
          y2: margin.top + plotH,
        }}),
      );

      if (xLabel) {{
        nodes.push(
          e('text', {{
            className: 'leaderboard-scatter-axis-label mono',
            key: 'xlabel-main',
            x: margin.left + (plotW / 2),
            y: height - 18,
            textAnchor: 'middle',
            dominantBaseline: 'alphabetic',
          }}, xLabel),
        );
      }}
      if (yLabel) {{
        nodes.push(
          e('text', {{
            className: 'leaderboard-scatter-axis-label mono',
            key: 'ylabel-main',
            x: 18,
            y: margin.top + (plotH / 2),
            textAnchor: 'middle',
            dominantBaseline: 'alphabetic',
            transform: 'rotate(-90 18 ' + (margin.top + (plotH / 2)) + ')',
          }}, yLabel),
        );
      }}

      const paretoPoints = paretoFrontier(points);
      const paretoIdx = new Set(paretoPoints.map(p => p.idx));
      const paretoModels = [];
      const paretoModelIndex = new Map();
      const paretoModelInfo = new Map();
      paretoPoints.forEach((p) => {{
        const model = String(p.r.model || '');
        if (!model || paretoModelIndex.has(model)) return;
        const index = paretoModels.length;
        paretoModels.push(model);
        paretoModelIndex.set(model, index);
        paretoModelInfo.set(model, {{
          index,
          row: p.r,
          color: colorFn(p.r, p.idx) || '#9aa7b5',
        }});
      }});
      if (paretoPoints.length > 1) {{
        const paretoPath = buildParetoPath(paretoPoints);
        if (paretoPath) {{
          nodes.push(
            e('path', {{
              className: 'leaderboard-scatter-pareto',
              key: 'pareto-frontier',
              d: paretoPath,
            }}),
          );
        }}
      }}

      // Points
      const paretoOuterR = 11;
      const paretoInnerR = 5.2;
      points.forEach((p) => {{
        const modelName = String(p.r.model || '');
        const title = modelName
          + '\\nCost: ' + fmtUsd(p.x)
          + '\\nAccuracy: ' + fmtPct(p.y);
        const isPareto = paretoIdx.has(p.idx);
        const fill = colorFn(p.r, p.idx) || '#9aa7b5';
        const markerIdx = isPareto ? (paretoModelIndex.get(modelName) ?? 0) : -1;
        nodes.push(
          e('g', {{ key: 'pt-' + modelName + '-' + p.idx }},
            e('title', null, title),
            isPareto
              ? e('path', {{
                  className: 'leaderboard-scatter-point--pareto',
                  d: paretoMarkerPath(sx(p.x), sy(p.y), markerIdx, paretoOuterR, paretoInnerR),
                  fill,
                }})
              : e('circle', {{
                  className: 'leaderboard-scatter-point',
                  cx: sx(p.x),
                  cy: sy(p.y),
                  r: 7,
                  fill,
                }}),
          ),
        );
      }});

      const svg = e(
        'svg',
        {{
          className: 'leaderboard-scatter-svg',
          viewBox: '0 0 ' + width + ' ' + height,
          role: 'img',
          'aria-label': 'Accuracy vs cost per evaluation scatter plot',
        }},
        nodes,
      );
      const paretoLegendItems = [];
      paretoModels.forEach((model) => {{
        const info = paretoModelInfo.get(model);
        if (!info) return;
        const color = info.color || '#9aa7b5';
        paretoLegendItems.push(
          e('div', {{ className: 'leaderboard-scatter-legend-item', key: 'pareto-' + model }},
            e(
              'svg',
              {{
                className: 'leaderboard-scatter-legend-icon',
                viewBox: '0 0 24 24',
                role: 'img',
                'aria-hidden': 'true',
              }},
              e('path', {{
                d: paretoMarkerPath(12, 12, info.index, 8, 4),
                fill: color,
                stroke: '#fff',
                strokeWidth: 2,
              }}),
            ),
            e('span', {{ className: 'mono' }}, model),
          ),
        );
      }});
      const paretoLineLegend = e(
        'div',
        {{ className: 'leaderboard-scatter-legend-item', key: 'pareto-line' }},
        e(
          'svg',
          {{
            className: 'leaderboard-scatter-legend-line',
            viewBox: '0 0 22 10',
            role: 'img',
            'aria-hidden': 'true',
          }},
          e('line', {{
            x1: 1,
            x2: 21,
            y1: 5,
            y2: 5,
            stroke: '#9ca3af',
            strokeWidth: 2,
            strokeDasharray: '6 4',
            strokeLinecap: 'round',
          }}),
        ),
        e('span', null, 'Pareto frontier'),
      );
      const legend = paretoLegendItems.length
        ? e(
            'div',
            {{ className: 'leaderboard-scatter-legend' }},
            e('span', {{ className: 'leaderboard-scatter-legend-label' }}, 'Pareto models'),
            paretoLegendItems,
            paretoLineLegend,
          )
        : null;
      mountReact(rootEl, e('div', {{ className: 'leaderboard-scatter-wrap' }}, svg, legend));
    }}

    let sortKey = 'mean_accuracy';
    let sortDir = 'desc'; // 'asc' | 'desc'

    function cmp(a, b) {{
      const dir = sortDir === 'asc' ? 1 : -1;
      const ka = a[sortKey];
      const kb = b[sortKey];

      if (typeof ka === 'string' || typeof kb === 'string') {{
        return dir * String(ka || '').localeCompare(String(kb || ''));
      }}

      const na = isNum(ka) ? ka : -Infinity;
      const nb = isNum(kb) ? kb : -Infinity;
      if (na === nb) {{
        return dir * String(a.model || '').localeCompare(String(b.model || ''));
      }}
      return dir * (na - nb);
    }}

    function applyFilterAndSort() {{
      const q = String(filterEl.value || '').trim().toLowerCase();
      const filtered = q
        ? rowsRaw.filter(r => String(r.model || '').toLowerCase().includes(q))
        : rowsRaw.slice();
      filtered.sort(cmp);
      return filtered;
    }}

    function render() {{
      const rows = applyFilterAndSort();
      countEl.textContent = String(rows.length);
      bodyEl.innerHTML = '';

      rows.forEach((r, idx) => {{
        const rank = idx + 1;
        let medal = '';
        if (rank === 1) medal = ' 🥇';
        else if (rank === 2) medal = ' 🥈';
        else if (rank === 3) medal = ' 🥉';

        const tr = document.createElement('tr');
        tr.title = r.generator_models ? ('Generators: ' + r.generator_models) : '';

        const tdRank = document.createElement('td');
        const rankSpan = document.createElement('span');
        rankSpan.className = 'rank-badge';
        rankSpan.textContent = String(rank);
        tdRank.appendChild(rankSpan);

        const tdModel = document.createElement('td');
        const modelSpan = document.createElement('span');
        modelSpan.className = 'mono';
        modelSpan.textContent = String(r.model || '');
        tdModel.appendChild(modelSpan);
        if (medal) tdModel.appendChild(document.createTextNode(medal));

        const tdMean = document.createElement('td');
        tdMean.className = 'has-text-right mono';
        tdMean.textContent = fmtPct(r.mean_accuracy);

        const tdSem = document.createElement('td');
        tdSem.className = 'has-text-right mono has-text-grey';
        tdSem.textContent = fmtPct(r.sem);

        tr.appendChild(tdRank);
        tr.appendChild(tdModel);
        tr.appendChild(tdMean);
        tr.appendChild(tdSem);
        bodyEl.appendChild(tr);
      }});
    }}

    filterEl.addEventListener('input', render);

    const headerCells = tableEl.querySelectorAll('thead th[data-key]');
    headerCells.forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.getAttribute('data-key');
        if (!key) return;

        if (sortKey === key) {{
          sortDir = (sortDir === 'asc') ? 'desc' : 'asc';
        }} else {{
          sortKey = key;
          sortDir = (key === 'model') ? 'asc' : 'desc';
        }}

        headerCells.forEach(x => x.classList.remove('has-text-info'));
        th.classList.add('has-text-info');

        render();
      }});
    }});

    const topRows = sortByAccuracy(rowsRaw).slice(0, 10);
    const colorByModel = new Map();
    topRows.forEach((r, idx) => {{
      colorByModel.set(String(r.model || ''), colorStops(idx, topRows.length));
    }});
    const topRowsByCostDesc = topRows.slice();
    topRowsByCostDesc.sort((a, b) => {{
      const ca = costByModel[String(a.model || '')];
      const cb = costByModel[String(b.model || '')];
      const na = isNum(ca) ? ca : -Infinity;
      const nb = isNum(cb) ? cb : -Infinity;
      if (na === nb) {{
        return String(a.model || '').localeCompare(String(b.model || ''));
      }}
      return nb - na;
    }});

    renderLeaderboardChart(
      accuracyRootEl,
      topRows,
      r => r.mean_accuracy,
      fmtPct,
      'No accuracy data available.',
      (r, idx) => colorByModel.get(String(r.model || '')) || colorStops(idx, topRows.length),
      {{
        minValue: 0.6,
        maxValue: 1.0,
        plotClass: 'leaderboard-plot--accuracy',
        yTicks: [
          {{ value: 1.0, label: '100%' }},
          {{ value: 0.9, label: '90%' }},
          {{ value: 0.8, label: '80%' }},
          {{ value: 0.7, label: '70%' }},
          {{ value: 0.6, label: '60%' }},
        ],
        labelFmt: fmtPctIntNoSign,
      }},
    );
    renderLeaderboardChart(
      costRootEl,
      topRowsByCostDesc,
      r => costByModel[String(r.model || '')],
      fmtUsd,
      'No cost data available.',
      (r, idx) => colorByModel.get(String(r.model || '')) || colorStops(idx, topRows.length),
      {{
        minValue: 0,
        maxValue: 10,
        plotClass: 'leaderboard-plot--cost',
        labelBottomMinValue: 2,
        yTicks: [
          {{ value: 10, label: '$10' }},
          {{ value: 0, label: '0' }},
        ],
        labelFmt: fmtCentsNoDollar,
      }},
    );
    renderScatterChart(
      scatterRootEl,
      topRows,
      r => costByModel[String(r.model || '')],
      r => r.mean_accuracy,
      'No accuracy/cost data available.',
      (r, idx) => colorByModel.get(String(r.model || '')) || colorStops(idx, topRows.length),
      {{
        xMin: 0,
        xMax: 10,
        yMin: 0.6,
        yMax: 1.0,
        xLabel: 'Cost per evaluation',
        yLabel: 'Accuracy',
        xTicks: [
          {{ value: 0, label: '$0' }},
          {{ value: 2, label: '$2' }},
          {{ value: 4, label: '$4' }},
          {{ value: 6, label: '$6' }},
          {{ value: 8, label: '$8' }},
          {{ value: 10, label: '$10' }},
        ],
        yTicks: [
          {{ value: 1.0, label: '100%' }},
          {{ value: 0.9, label: '90%' }},
          {{ value: 0.8, label: '80%' }},
          {{ value: 0.7, label: '70%' }},
          {{ value: 0.6, label: '60%' }},
        ],
      }},
    );
    render();
  }})();
</script>

<noscript>
  <section class="section">
    <div class="container is-max-desktop">
      <div class="notification is-warning">
        This page requires JavaScript enabled to render.
      </div>
    </div>
  </section>
</noscript>

</body>
</html>
"""


def build_payload(
    *,
    csv_path: Path,
    title: str,
    description: str,
    keywords: str,
    emoji: str,
    subtitle: str,
    home_url: str,
    paper_url: str,
    code_url: str,
    data_url: str,
    submit_url: str,
    twitter_url: str,
    contact_email: str,
    issues_url: str,
    bibtex: str,
) -> dict[str, Any]:
    generated_at = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    rows = load_accuracy_by_model_csv(csv_path)
    cost_by_model = load_cost_by_model_csv(DEFAULT_COST_CSV_PATH)
    return {
        "meta": {
            "title": title,
            "description": description,
            "keywords": keywords,
            "emoji": emoji,
            "subtitle": subtitle,
            "authors": DEFAULT_PUBLICATION_AUTHORS,
            "affiliations": DEFAULT_PUBLICATION_AFFILIATIONS,
            "home_url": home_url,
            "paper_url": paper_url,
            "code_url": code_url,
            "data_url": data_url,
            "submit_url": submit_url,
            "twitter_url": twitter_url,
            "contact_email": contact_email,
            "issues_url": issues_url,
            "bibtex": bibtex,
            "generated_at": generated_at,
            "source_csv": str(csv_path),
        },
        "rows": rows,
        "cost_by_model": cost_by_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate `index.html` from a QuizBench aggregation CSV."
    )
    parser.add_argument(
        "--csv-by-model",
        dest="csv_by_model",
        type=Path,
        default=None,
        help=(
            "Path to the `OUT_CSV_BY_MODEL` output from `aggregate_filtered_results.sh` "
            "(default: tmp/agg_majority_by_model[_<tag>].csv)."
        ),
    )
    parser.add_argument(
        "--quiz-batch-tag",
        dest="quiz_batch_tag",
        type=str,
        default=None,
        help="If set, uses the default /tmp suffix used by aggregate_filtered_results.sh.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=Path("index.html"),
        help="Output HTML path (default: ./index.html).",
    )

    # Page metadata / links
    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        default="MedEvalArena",
        help="Main page title (default: MedEvalArena).",
    )
    parser.add_argument(
        "--emoji",
        dest="emoji",
        type=str,
        default="🏥",
        help="Emoji shown before the title (default: 🏥).",
    )
    parser.add_argument(
        "--subtitle",
        dest="subtitle",
        type=str,
        default="",
        help="Optional subtitle line shown under the title.",
    )
    parser.add_argument(
        "--description",
        dest="description",
        type=str,
        default="",
        help="HTML meta description (defaults to --title).",
    )
    parser.add_argument(
        "--keywords",
        dest="keywords",
        type=str,
        default="LLMs, Evaluation, Medicine",
        help="HTML meta keywords (comma-separated).",
    )
    parser.add_argument(
        "--home-url",
        dest="home_url",
        type=str,
        default="https://github.com/bernardolab",
        help="Navbar home URL (optional).",
    )
    parser.add_argument(
        "--paper-url",
        dest="paper_url",
        type=str,
        default="",
        help="Paper URL for the hero buttons (optional).",
    )
    parser.add_argument(
        "--code-url",
        dest="code_url",
        type=str,
        default="https://github.com/bernardolab/MedEvalArena",
        help="Code/Repo URL for the hero buttons.",
    )
    parser.add_argument(
        "--data-url",
        dest="data_url",
        type=str,
        default="",
        help="Dataset URL for the hero buttons (optional).",
    )
    parser.add_argument(
        "--submit-url",
        dest="submit_url",
        type=str,
        default="",
        help="Submission URL for the hero buttons (optional).",
    )
    parser.add_argument(
        "--twitter-url",
        dest="twitter_url",
        type=str,
        default="",
        help="Twitter/X URL for the hero buttons (optional).",
    )
    parser.add_argument(
        "--contact-email",
        dest="contact_email",
        type=str,
        default="",
        help="Contact email shown in the Contact section (optional).",
    )
    parser.add_argument(
        "--issues-url",
        dest="issues_url",
        type=str,
        default="",
        help="GitHub Issues URL shown in the Contact section (optional).",
    )
    parser.add_argument(
        "--bibtex",
        dest="bibtex",
        type=str,
        default="",
        help="BibTeX entry to show in the BibTeX section (optional).",
    )
    parser.add_argument(
        "--bibtex-path",
        dest="bibtex_path",
        type=Path,
        default=None,
        help="Path to a .bib file (or any text file) to include verbatim in the BibTeX section.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = args.csv_by_model or _default_csv_by_model_path(args.quiz_batch_tag)

    bibtex_text = args.bibtex or ""
    if args.bibtex_path:
        try:
            bibtex_text = args.bibtex_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[ERROR] Failed to read --bibtex-path '{args.bibtex_path}': {exc}")
            return 2

    description = args.description.strip() or args.title

    try:
        payload = build_payload(
            csv_path=csv_path,
            title=args.title,
            description=description,
            keywords=args.keywords,
            emoji=args.emoji,
            subtitle=args.subtitle,
            home_url=args.home_url,
            paper_url=args.paper_url,
            code_url=args.code_url,
            data_url=args.data_url,
            submit_url=args.submit_url,
            twitter_url=args.twitter_url,
            contact_email=args.contact_email,
            issues_url=args.issues_url,
            bibtex=bibtex_text,
        )
    except FileNotFoundError:
        msg = (
            f"[ERROR] CSV not found: {csv_path}\n\n"
            "Generate it first, e.g.:\n"
            "  OUT_CSV_BY_MODEL=tmp/agg_majority_by_model.csv bash aggregate_filtered_results.sh\n"
            "then run:\n"
            "  python quizbench/generate_leaderboard_site.py --csv-by-model tmp/agg_majority_by_model.csv\n"
        )
        print(msg)
        return 2
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV '{csv_path}': {exc}")
        return 2

    if not payload.get("rows"):
        print(
            f"[WARN] Loaded 0 rows from '{csv_path}'. "
            "The generated page will have no rankings."
        )

    html = render_index_html(payload=payload)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
