"""
Shared generator prompt and spec helpers to avoid circular imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from quizbench.utils import now_utc_iso, letters


GENERATOR_SYSTEM_INSTRUCTIONS = """You are a medical board-style MCQ generator.
Produce the HARDEST realistic clinical questions (diagnosis/management) with similar structure as may be seen on subspecialty board exams.
These should be questions which 99.999 percent to 100 percent of test takers would be expected to miss.
Each item MUST have exactly 5 plausible options (A..E), one single best answer, and a short authoritative explanation.
Avoid: images, tables, "All of the above/None", ambiguous double-negatives, esoterica with no clinical relevance.
Return ONLY a single JSON object (no commentary).
"""


def _format_topics(topics: str | Sequence[str]) -> str:
    if isinstance(topics, str):
        return topics
    return ", ".join(str(t) for t in topics)


def build_json_spec(
    topics: str | Sequence[str],
    num_q: int,
    quiz_id: str,
    num_choices: int = 5,
    *,
    topic_plan: Sequence[str] | None = None,
    topic_counts: Mapping[str, int] | None = None,
) -> str:
    allowed = letters(num_choices)
    topics_str = _format_topics(topics)

    topic_constraints = ""
    if topic_plan or topic_counts:
        lines: list[str] = []
        lines.append("Topic constraints:")
        lines.append(f"- Use only these specialties/topics: {topics_str}.")
        if topic_counts:
            lines.append("- Match this topic distribution exactly:")
            for cat, n in topic_counts.items():
                lines.append(f"  - {cat}: {int(n)}")
        if topic_plan:
            lines.append("- Use this per-item topic plan in order (1-to-1 with items):")
            for idx, cat in enumerate(topic_plan, start=1):
                lines.append(f"  {idx:02d}. {cat}")
        topic_constraints = "\n" + "\n".join(lines) + "\n"

    return f"""Return a JSON object with this exact schema:

{{
  "quiz_id": "{quiz_id}",
  "created_at": "{now_utc_iso()}",
  "domain": "medicine",
  "difficulty": "extremely-hard",
  "items": [
    {{
      "question_id": "Q001",
      "question": "Clinical stem ...",
      "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
      "answer": "A single capital letter among {allowed}",
      "explanation": "2-4 sentences: why correct, why main distractor(s) are wrong."
    }}
    // ... total {num_q} items, ids incrementing as {quiz_id}-002, ... -{num_q:03d}
  ]
}}

Constraints:
- Exactly 5 options per item.
- 'answer' MUST be one of {allowed}, aligning with the correct option.
- Explanations concise and factual; no citations required; no PHI.
- 'question_id' is a simple, short, single-line identifier (for example "Q001", "Q002"); it will be overwritten downstream, so do not add prose or line breaks inside it.
{topic_constraints}- Topics mix (sampled) from: {topics_str}.
"""
