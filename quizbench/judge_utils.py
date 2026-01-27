#!/usr/bin/env python3
"""
Utilities for normalizing judge outputs and question identifiers.
"""
from __future__ import annotations

from typing import Any, Mapping


def is_judge_output_valid(row: Mapping[str, Any]) -> bool:
    """
    Return True if a judge output should be considered valid.

    Matches quizbench/inter_judge_reliability.py behavior: only explicit False
    is treated as invalid; missing/None is allowed.
    """
    return row.get("judge_output_valid") is not False


def extract_verdict(row: Mapping[str, Any]) -> str | None:
    verdict = row.get("judge_verdict") or row.get("verdict")
    if verdict is None:
        judge_json = row.get("judge_json") or {}
        verdict = judge_json.get("verdict")
        if verdict is None:
            logical_valid = judge_json.get("logical_validity")
            if isinstance(logical_valid, bool):
                verdict = "PASS" if logical_valid else "FAIL"
    if isinstance(verdict, str):
        verdict = verdict.strip().upper()
        if verdict in {"PASS", "FAIL"}:
            return verdict
    return None


def extract_medical_score(row: Mapping[str, Any]) -> int | None:
    score = row.get("judge_medical_accuracy_score")
    if score is None:
        judge_json = row.get("judge_json") or {}
        score = judge_json.get("medical_accuracy_score")
    try:
        score_int = int(score) if score is not None else None
    except (TypeError, ValueError):
        return None
    if score_int is None or score_int < 1 or score_int > 5:
        return None
    return score_int


def build_unit_id(row: Mapping[str, Any], quiz_id: str) -> str | None:
    qid = row.get("question_id") or row.get("id")
    if not qid:
        return None
    quiz_str = str(row.get("quiz_id") or quiz_id)
    return f"{quiz_str}::{qid}"
