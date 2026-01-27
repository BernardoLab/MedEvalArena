#!/usr/bin/env python3
"""
Human-guided mapping from topic labels to canonical ABMS specialties.

The core use case is reconciling:
  - `topics_*.json` labels (often lowercased/shorter labels like "pediatrics"),
with:
  - `data/ABMS_specialties.csv` specialty names (title-cased, expanded, etc).

This module supports a small YAML/JSON mapping file so humans can iteratively
add overrides, while still getting robust matching via normalization.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


_SLASH_SPACING_RE = re.compile(r"\s*/\s*")


def normalize_label(value: str) -> str:
    """
    Normalize a category/topic label for robust matching.

    Behavior:
      - casefold
      - replace '&' with 'and'
      - drop punctuation (keep a-z0-9)
      - collapse whitespace
    """
    text = (value or "").strip().casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_combined_label(raw: str) -> bool:
    """
    Heuristic for multi-topic labels (used only as a last-resort fallback).
    """
    if not raw:
        return False
    if "/" in raw:
        return True
    norm = normalize_label(raw)
    return " and " in f" {norm} "


@dataclass(frozen=True)
class TopicMappingConfig:
    normalize: bool = True
    drop: frozenset[str] = frozenset({"unknown"})
    map_overrides: Mapping[str, str] | None = None  # normalized source -> canonical dest
    combined_to_misc: bool = True
    misc_category: str = "Misc"


def _load_mapping_payload(path: Path) -> Mapping[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"PyYAML is not installed, cannot read {path}. Use a .json mapping file instead."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must be a YAML mapping/object: {path}")
        return data

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Mapping file must be a JSON object: {path}")
        return data

    raise ValueError(f"Unsupported mapping file extension: {path}")


def load_topic_mapping(path: Path) -> TopicMappingConfig:
    """
    Load a mapping config from YAML/JSON.

    Supported keys:
      - normalize: bool
      - drop: list[str]
      - map: {source_label: dest_category}
      - combined_to_misc: bool
      - misc_category: str
    """
    payload = _load_mapping_payload(path)

    normalize = bool(payload.get("normalize", True))
    combined_to_misc = bool(payload.get("combined_to_misc", True))
    misc_category = str(payload.get("misc_category", "Misc")).strip() or "Misc"

    drop_list = payload.get("drop", ["unknown"])
    drop: set[str] = set()
    if isinstance(drop_list, (list, tuple, set)):
        for item in drop_list:
            if not item:
                continue
            drop.add(normalize_label(str(item)) if normalize else str(item).strip())

    raw_map = payload.get("map") or {}
    if not isinstance(raw_map, dict):
        raise ValueError("'map' must be an object/dict in the mapping file.")

    overrides: Dict[str, str] = {}
    for raw_src, raw_dst in raw_map.items():
        src = str(raw_src).strip()
        dst = str(raw_dst).strip()
        if not src or not dst:
            continue
        key = normalize_label(src) if normalize else src
        overrides[key] = dst

    return TopicMappingConfig(
        normalize=normalize,
        drop=frozenset(drop),
        map_overrides=overrides,
        combined_to_misc=combined_to_misc,
        misc_category=misc_category,
    )


def build_canonical_index(
    canonical_categories: Sequence[str],
    *,
    normalize: bool = True,
) -> Dict[str, str]:
    """
    Build a normalized lookup: normalized_category -> canonical_category.
    """
    index: Dict[str, str] = {}
    for cat in canonical_categories:
        key = normalize_label(cat) if normalize else str(cat).strip()
        if not key:
            continue
        if key in index and index[key] != cat:
            raise ValueError(
                f"Canonical categories collide under normalization: {index[key]!r} vs {cat!r}"
            )
        index[key] = str(cat)
    return index


def map_topic_label(
    raw_topic: str | None,
    *,
    canonical_categories: Sequence[str],
    canonical_index: Mapping[str, str],
    config: TopicMappingConfig | None,
    mode: str = "map",
    unmapped_policy: str = "keep",
) -> str | None:
    """
    Map a raw topic label to a canonical category (or None to drop).

    mode:
      - exact: require exact match to canonical category
      - normalize: normalized match to canonical category
      - map: normalized match, else use mapping overrides

    unmapped_policy:
      - error: raise ValueError
      - keep: return raw_topic unchanged
      - misc: return config.misc_category (or "Misc" if config is None)
      - drop: return None
    """
    if raw_topic is None:
        return None
    raw = str(raw_topic).strip()
    if not raw:
        return None
    raw_for_match = _SLASH_SPACING_RE.sub("/", raw)

    cfg = config or TopicMappingConfig()
    mode = (mode or "map").lower()
    if mode not in {"exact", "normalize", "map"}:
        raise ValueError(f"Unsupported mapping mode: {mode!r}")

    unmapped_policy = (unmapped_policy or "keep").lower()
    if unmapped_policy not in {"error", "keep", "misc", "drop"}:
        raise ValueError(f"Unsupported unmapped_policy: {unmapped_policy!r}")

    norm_raw = normalize_label(raw_for_match) if cfg.normalize else raw_for_match
    if norm_raw in cfg.drop:
        return None

    if mode == "exact":
        canonical_set = set(canonical_categories)
        if raw in canonical_set:
            return raw
        if raw_for_match != raw and raw_for_match in canonical_set:
            return raw_for_match
    else:
        if norm_raw in canonical_index:
            return canonical_index[norm_raw]

    if mode == "map":
        dst = (cfg.map_overrides or {}).get(norm_raw)
        if dst:
            return dst

    # Combined-label fallback (e.g., "hematology/oncology") → Misc for now.
    if cfg.combined_to_misc and _is_combined_label(raw_for_match):
        return cfg.misc_category

    if unmapped_policy == "drop":
        return None
    if unmapped_policy == "misc":
        return cfg.misc_category
    if unmapped_policy == "keep":
        return raw

    raise ValueError(f"Unmapped topic label: {raw!r}")


def map_topic_label_explained(
    raw_topic: str | None,
    *,
    canonical_categories: Sequence[str],
    canonical_index: Mapping[str, str],
    config: TopicMappingConfig | None,
    mode: str = "map",
    unmapped_policy: str = "keep",
) -> tuple[str | None, str]:
    """
    Same as `map_topic_label`, but also returns a short reason string.

    Reason values are stable, machine-readable strings intended for reporting:
      - none / empty / dropped
      - canonical_exact / canonical_normalized
      - override
      - combined_to_misc
      - unmapped_keep / unmapped_misc / unmapped_drop
    """
    if raw_topic is None:
        return None, "none"
    raw = str(raw_topic).strip()
    if not raw:
        return None, "empty"
    raw_for_match = _SLASH_SPACING_RE.sub("/", raw)

    cfg = config or TopicMappingConfig()
    mode = (mode or "map").lower()
    if mode not in {"exact", "normalize", "map"}:
        raise ValueError(f"Unsupported mapping mode: {mode!r}")

    unmapped_policy = (unmapped_policy or "keep").lower()
    if unmapped_policy not in {"error", "keep", "misc", "drop"}:
        raise ValueError(f"Unsupported unmapped_policy: {unmapped_policy!r}")

    norm_raw = normalize_label(raw_for_match) if cfg.normalize else raw_for_match
    if norm_raw in cfg.drop:
        return None, "dropped"

    if mode == "exact":
        canonical_set = set(canonical_categories)
        if raw in canonical_set:
            return raw, "canonical_exact"
        if raw_for_match != raw and raw_for_match in canonical_set:
            return raw_for_match, "canonical_exact"
    else:
        if norm_raw in canonical_index:
            return canonical_index[norm_raw], "canonical_normalized"

    if mode == "map":
        dst = (cfg.map_overrides or {}).get(norm_raw)
        if dst:
            return dst, "override"

    if cfg.combined_to_misc and _is_combined_label(raw_for_match):
        return cfg.misc_category, "combined_to_misc"

    if unmapped_policy == "drop":
        return None, "unmapped_drop"
    if unmapped_policy == "misc":
        return cfg.misc_category, "unmapped_misc"
    if unmapped_policy == "keep":
        return raw, "unmapped_keep"

    # unmapped_policy == "error"
    raise ValueError(f"Unmapped topic label: {raw!r}")


def validate_mapping_outputs(
    mapped_categories: Iterable[str | None],
    *,
    canonical_categories: Sequence[str],
    misc_category: str = "Misc",
) -> None:
    """
    Validate that mapped outputs are either canonical categories or Misc.
    """
    allowed = set(canonical_categories)
    for cat in mapped_categories:
        if cat is None:
            continue
        if cat == misc_category:
            continue
        if cat not in allowed:
            raise ValueError(f"Mapped category not in targets CSV: {cat!r}")
