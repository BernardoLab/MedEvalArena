#!/usr/bin/env python3
"""
Helpers for turning per-category integer targets into deterministic batches.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class IntegerTargets:
    """
    Ordered integer targets loaded from a CSV.

    `categories` preserves CSV row order.
    """

    categories: List[str]
    targets_by_category: Dict[str, int]

    @property
    def total(self) -> int:
        return int(sum(self.targets_by_category.values()))


def load_integer_targets_csv(
    path: Path,
    *,
    category_col: str = "Specialty",
    count_col: str = "Number",
) -> IntegerTargets:
    """
    Load ordered integer targets from a CSV.

    The input is expected to contain columns like:
      Specialty,Number
      Pediatrics,5
      ...
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    categories: List[str] = []
    targets: Dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        if category_col not in reader.fieldnames or count_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required columns {category_col!r}/{count_col!r}: {path}"
            )

        for idx, row in enumerate(reader, start=2):
            raw_cat = (row.get(category_col) or "").strip()
            raw_num = (row.get(count_col) or "").strip()
            if not raw_cat:
                raise ValueError(f"Empty {category_col!r} at {path}:{idx}")
            if raw_cat in targets:
                raise ValueError(f"Duplicate category {raw_cat!r} at {path}:{idx}")
            try:
                num = int(raw_num)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer {count_col!r} at {path}:{idx}: {raw_num!r}") from exc
            if num < 0:
                raise ValueError(f"Negative target not allowed at {path}:{idx}: {raw_cat}={num}")
            categories.append(raw_cat)
            targets[raw_cat] = num

    return IntegerTargets(categories=categories, targets_by_category=targets)


def compute_remainders(
    *,
    targets: IntegerTargets,
    current_counts_by_category: Mapping[str, int],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Compute (remaining, overshoot, extra_current) dictionaries.

    - remaining[cat] = max(target - current, 0)
    - overshoot[cat] = max(current - target, 0)
    - extra_current includes categories present in current but missing from targets
    """
    remaining: Dict[str, int] = {}
    overshoot: Dict[str, int] = {}

    for cat in targets.categories:
        target_n = int(targets.targets_by_category.get(cat, 0))
        current_n = int(current_counts_by_category.get(cat, 0) or 0)
        if current_n < target_n:
            remaining[cat] = target_n - current_n
        elif current_n > target_n:
            overshoot[cat] = current_n - target_n

    extra_current: Dict[str, int] = {}
    for cat, n in current_counts_by_category.items():
        if cat in targets.targets_by_category:
            continue
        n_int = int(n or 0)
        if n_int > 0:
            extra_current[str(cat)] = n_int

    return remaining, overshoot, extra_current


def expand_to_desired_list(
    *,
    categories_in_order: Sequence[str],
    counts_by_category: Mapping[str, int],
) -> List[str]:
    """
    Expand counts into an ordered list with repetition.
    """
    desired: List[str] = []
    for cat in categories_in_order:
        n = int(counts_by_category.get(cat, 0) or 0)
        if n <= 0:
            continue
        desired.extend([cat] * n)
    return desired


def chunked(items: Sequence[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    out: List[List[str]] = []
    for i in range(0, len(items), chunk_size):
        out.append(list(items[i : i + chunk_size]))
    return out


def counts_for_batch(
    *,
    batch: Sequence[str],
    categories_in_order: Sequence[str],
) -> Dict[str, int]:
    """
    Convert a batch list into a compact {category: count} mapping.

    Keys are emitted in CSV order when present; unknown categories are appended
    in sorted order for determinism.
    """
    ctr = Counter(batch)
    out: Dict[str, int] = {}
    for cat in categories_in_order:
        n = int(ctr.get(cat, 0) or 0)
        if n > 0:
            out[cat] = n
    extras = sorted(k for k in ctr.keys() if k not in set(categories_in_order))
    for cat in extras:
        n = int(ctr.get(cat, 0) or 0)
        if n > 0:
            out[str(cat)] = n
    return out
