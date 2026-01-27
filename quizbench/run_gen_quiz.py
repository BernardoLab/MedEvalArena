#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

# Ensure package imports succeed whether run from repo root or quizbench/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quizbench.utils import ensure_dir, now_utc_iso, compact_utc_ts


def run(cmd: list, env_overrides: Dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    extra_paths = [str(ROOT_DIR)]
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    if env_overrides:
        env.update(env_overrides)

    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    if p.returncode != 0:
        raise SystemExit(f"[FATAL] {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def parse_models_csv(csv_str: str) -> List[str]:
    return [m.strip() for m in csv_str.split(",") if m.strip()]


GENERATE_QUIZ_PATH = SCRIPT_DIR / "generate_quiz.py"

def sanitize_model_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return slug or "model"


def build_quiz_id(gen_model: str, seed: int, at: datetime | None = None) -> str:
    ts_str = compact_utc_ts(at)
    return f"{ts_str}_{sanitize_model_name(gen_model)}_seed{seed}"


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"[FATAL] Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"[FATAL] Failed to parse YAML at {path}:\n{e}")
    if not isinstance(data, dict):
        raise SystemExit(f"[FATAL] Config at {path} must be a mapping/object.")
    return data


def parse_models_field(
    raw: Union[str, List[str], tuple, set, None], field_label: str, required: bool
) -> List[str]:
    """
    Normalize a models field from YAML (supports list/tuple/set or CSV string).
    """
    if raw is None:
        if required:
            raise SystemExit(
                f"[FATAL] Provide {field_label} in config (list or CSV string)."
            )
        return []
    if isinstance(raw, str):
        models = parse_models_csv(raw)
    elif isinstance(raw, (list, tuple, set)):
        models = [m.strip() for m in raw if isinstance(m, str) and m.strip()]
    else:
        raise SystemExit(
            f"[FATAL] {field_label} must be a string/CSV or a list/tuple/set of strings."
        )
    if required and not models:
        raise SystemExit(f"[FATAL] No valid models found for {field_label}.")
    return models


def coerce_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    raw = cfg.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"[FATAL] Expected integer for '{key}', got {raw!r}.")


def collect_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Collect optional environment variable overrides from config.
    - Accepts both lowercase and uppercase keys for common providers.
    - Also supports a top-level 'env' mapping for arbitrary env vars.
    """
    env: Dict[str, str] = {}

    def maybe_set(cfg_key: str, env_key: str):
        val = cfg.get(cfg_key)
        if val is not None:
            env[env_key] = str(val)

    maybe_set("openai_api_key", "OPENAI_API_KEY")
    maybe_set("anthropic_api_key", "ANTHROPIC_API_KEY")
    maybe_set("gemini_api_key", "GEMINI_API_KEY")
    maybe_set("grok_api_key", "GROK_API_KEY")
    maybe_set("grok_api_base_url", "GROK_API_BASE_URL")
    maybe_set("deepseek_api_key", "DEEPSEEK_API_KEY")
    maybe_set("openrouter_api_key", "OPENROUTER_API_KEY")

    env_section = cfg.get("env")
    if isinstance(env_section, dict):
        for k, v in env_section.items():
            if isinstance(k, str) and v is not None:
                env[k] = str(v)

    return env


def main():
    ap = argparse.ArgumentParser(
        description="Generate quizzes based on generator models specified in a YAML config."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    print(f"[INFO] Loaded config from {config_path}")

    generator_models = parse_models_field(
        cfg.get("generator_models")
        or cfg.get("generator_models_csv")
        or cfg.get("generator_model"),
        field_label="generator_models",
        required=True,
    )

    # Not used currently
    test_models = parse_models_field(
        cfg.get("test_models")
        or cfg.get("test_models_csv")
        or cfg.get("test_model"),
        field_label="test_models",
        required=False,
    )
    test_models_csv = ",".join(test_models) if test_models else None

    num_quizzes = coerce_int(cfg, "num_quizzes", 10)
    num_questions = coerce_int(cfg, "num_questions", 10)
    seed0 = coerce_int(cfg, "seed0", 123)
    quizzes_dir = str(cfg.get("quizzes_dir", "quizzes/"))
    runs_root = str(cfg.get("runs_root", "eval_uq_results/"))
    env_overrides = collect_env(cfg)

    ensure_dir(quizzes_dir)
    ensure_dir(runs_root)

    manifest = {
        "created_at": now_utc_iso(),
        "config_path": str(config_path),
        "generator_models": generator_models,
        # legacy field for compatibility (first generator)
        "generator_model": generator_models[0],
        "test_models": test_models,
        "test_models_csv": test_models_csv,
        "num_quizzes_per_generator": num_quizzes,
        "num_questions": num_questions,
        "runs_root": runs_root,
        "quizzes_dir": quizzes_dir,
        # list of generated quizzes (no eval yet)
        "quizzes": [],  # list of {quiz_id, quiz_path, generator_model, seed}
        "run_ids": [],  # kept for compatibility; will be filled by downstream eval, if desired
    }

    quizzes_info = []

    # ----------------------------------------------------
    # GENERATION ONLY: generate all quizzes from all generators
    # ----------------------------------------------------
    print("\n=== Generating quizzes from generator models ===")
    for g_idx, gen_model in enumerate(generator_models):
        print(f"\n-- Generator model: {gen_model} --")
        for k in range(num_quizzes):
            seed = seed0 + g_idx * num_quizzes + k
            quiz_id = build_quiz_id(gen_model, seed)
            expected_path = Path(quizzes_dir) / f"{quiz_id}.jsonl"
            gen_cmd = [
                sys.executable,
                str(GENERATE_QUIZ_PATH),
                "--generator_model",
                gen_model,
                "--quiz_id",
                quiz_id,
                "--num_questions",
                str(num_questions),
                "--seed",
                str(seed),
                "--out_dir",
                quizzes_dir,
            ]
            quiz_path = run(gen_cmd, env_overrides=env_overrides)  # generate_quiz.py prints path
            raw_path = Path(quiz_path)
            if expected_path.exists():
                quiz_path = str(expected_path)
            elif raw_path.exists():
                os.replace(raw_path, expected_path)
                quiz_path = str(expected_path)
            else:
                raise SystemExit(f"[FATAL] Expected quiz at {expected_path} (or {raw_path}), but none found.")

            quizzes_info.append(
                {
                    "quiz_id": quiz_id,
                    "quiz_path": quiz_path,
                    "generator_model": gen_model,
                    "seed": seed,
                }
            )
            print(f"  [OK] Generated quiz {quiz_id} at {quiz_path}")

    manifest["quizzes"] = quizzes_info

    # Save manifest (generation-only; eval happens downstream)
    manifest_path = os.path.join(runs_root, "quizbench_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    # For drop-in compatibility, still print runs_root as the machine-readable value
    print(runs_root)


if __name__ == "__main__":
    main()
