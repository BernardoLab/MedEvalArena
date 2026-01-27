#!/usr/bin/env python3
"""
Parallel driver for QuizBench evaluation.

This orchestrates multiple `python -m quizbench.run_eval` subprocesses while:
  - Limiting concurrency across different `--runs_root` directories.
  - Enforcing a fixed delay (default: 60s) before each subprocess submission.
  - Optionally printing OpenRouter key limits/remaining credits before each
    submission (via https://openrouter.ai/api/v1/key).
  - Skipping quizzes that already have eval outputs on disk (resume mode).

Important:
  - Do NOT run multiple run_eval subprocesses concurrently against the same
    `--runs_root`. run_eval updates the manifest JSON in-place, which can race.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quizbench.manifest_utils import resolve_quizbench_manifest_path
from quizbench.utils import compact_utc_ts, ensure_dir


def parse_csv_arg(val: str) -> list[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


def slugify(val: str) -> str:
    val = val.strip().replace(os.sep, "__")
    val = re.sub(r"[^A-Za-z0-9._-]+", "_", val)
    val = re.sub(r"_+", "_", val).strip("_")
    return val or "item"


def runs_root_log_label(runs_root: Path) -> str:
    """
    Produce a human-friendly label for a RUNS_ROOT directory for use in log filenames.

    Prefer the common QuizBench subset layout:
      .../quizzes_<SUBSET_TAG>/runs/<generator_model>/
    """
    try:
        parent = runs_root.parent
        grandparent = parent.parent
        if parent.name == "runs" and grandparent.name.startswith("quizzes_"):
            return f"{grandparent.name}__{runs_root.name}"
    except Exception:
        pass

    return runs_root.name


ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_BLUE = "\x1b[34m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_CYAN = "\x1b[36m"
ANSI_GRAY = "\x1b[90m"


def should_use_color(color_flag: bool | None) -> bool:
    """
    Decide whether to emit ANSI color codes.

    - If `--color/--no-color` is explicitly set, honor it.
    - Otherwise, enable only on TTY, unless NO_COLOR is set.
    """
    if color_flag is False:
        return False
    if color_flag is True:
        return True

    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") in (None, "", "dumb"):
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return sys.stdout.isatty()


_TAG_COLOR: dict[str, str] = {
    "INFO": ANSI_BLUE,
    "OK": ANSI_GREEN,
    "WARN": ANSI_YELLOW,
    "ERROR": ANSI_RED,
    "SKIP": ANSI_GRAY,
    "DRY-RUN": ANSI_MAGENTA,
    "OPENROUTER": ANSI_CYAN,
}


def format_tag(tag: str, *, use_color: bool) -> str:
    raw = f"[{tag}]"
    if not use_color:
        return raw

    color = _TAG_COLOR.get(tag, "")
    if tag == "SKIP":
        return f"{ANSI_DIM}{color}{raw}{ANSI_RESET}"
    return f"{ANSI_BOLD}{color}{raw}{ANSI_RESET}"


def short_runs_root_label(runs_root: Path) -> str:
    """
    Human-friendly RUNS_ROOT label for console logs.

    Users typically only need the final model directory name (e.g. "kimi-k2-thinking").
    """
    return runs_root.name


class SubmitRateLimiter:
    def __init__(self, *, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be >= 0")
        self._interval_seconds = float(interval_seconds)
        self._lock = asyncio.Lock()
        self._next_submit_at = 0.0

    @contextlib.asynccontextmanager
    async def submission_slot(self):
        """
        Serialize "submission" (spawning run_eval) and enforce a minimum interval
        between successive submissions.

        Callers should do any work that must happen immediately before spawning
        (e.g. rate limit checks) inside this context, but should *not* wait for the
        spawned subprocess to finish inside it.
        """
        async with self._lock:
            if self._interval_seconds > 0:
                now = time.monotonic()
                delay = max(0.0, self._next_submit_at - now)
                if delay > 0:
                    await asyncio.sleep(delay)
            try:
                yield
            finally:
                if self._interval_seconds > 0:
                    self._next_submit_at = time.monotonic() + self._interval_seconds


@dataclass(frozen=True)
class OpenRouterKeyInfo:
    status_code: int
    raw_text: str
    data: dict[str, Any] | None


async def fetch_openrouter_key_info(*, timeout_seconds: int) -> OpenRouterKeyInfo:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return OpenRouterKeyInfo(
            status_code=0,
            raw_text="OPENROUTER_API_KEY missing",
            data=None,
        )

    def _do_request() -> tuple[int, str]:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                status_code = getattr(resp, "status", None) or resp.getcode()
                raw_text = resp.read().decode("utf-8", errors="replace")
                return int(status_code), raw_text
        except urllib.error.HTTPError as exc:
            raw_text = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), raw_text
        except urllib.error.URLError as exc:
            return 0, str(exc)

    status_code, raw_text = await asyncio.to_thread(_do_request)
    parsed: dict[str, Any] | None = None
    try:
        parsed_obj = json.loads(raw_text)
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj
    except json.JSONDecodeError:
        parsed = None

    return OpenRouterKeyInfo(status_code=status_code, raw_text=raw_text, data=parsed)


def format_openrouter_limits(info: OpenRouterKeyInfo) -> str:
    if info.status_code == 0:
        return "[OPENROUTER] OPENROUTER_API_KEY missing; skipping limit check."

    if info.status_code != 200 or not isinstance(info.data, dict):
        return f"[OPENROUTER] status={info.status_code} body={info.raw_text}"

    data = info.data.get("data") if isinstance(info.data.get("data"), dict) else {}
    label = data.get("label")

    limit = data.get("limit")
    limit_remaining = data.get("limit_remaining")
    limit_reset = data.get("limit_reset")
    include_byok_in_limit = data.get("include_byok_in_limit")

    usage = data.get("usage")
    usage_daily = data.get("usage_daily")
    usage_weekly = data.get("usage_weekly")
    usage_monthly = data.get("usage_monthly")

    byok_usage = data.get("byok_usage")
    byok_usage_daily = data.get("byok_usage_daily")
    byok_usage_weekly = data.get("byok_usage_weekly")
    byok_usage_monthly = data.get("byok_usage_monthly")

    return (
        "[OPENROUTER] "
        f"label={label!r} "
        f"limit_remaining={limit_remaining} limit={limit} limit_reset={limit_reset!r} "
        f"include_byok_in_limit={include_byok_in_limit} "
        f"usage(d/w/m/all)={usage_daily}/{usage_weekly}/{usage_monthly}/{usage} "
        f"byok_usage(d/w/m/all)={byok_usage_daily}/{byok_usage_weekly}/{byok_usage_monthly}/{byok_usage}"
    )


def build_run_eval_cmd(
    *,
    runs_root: Path,
    eval_model: str,
    max_tokens: int,
    reasoning_effort: str,
    use_batch_api: bool,
    manifest_path: str | None,
    only_quiz_ids_csv: str | None,
    config_path: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "quizbench.run_eval",
        "--runs_root",
        str(runs_root),
        "--eval_model",
        eval_model,
        "--max_tokens",
        str(max_tokens),
        "--reasoning_effort",
        reasoning_effort,
    ]

    if manifest_path:
        cmd.extend(["--manifest_path", manifest_path])
    if only_quiz_ids_csv:
        cmd.extend(["--only_quiz_ids_csv", only_quiz_ids_csv])
    if config_path:
        cmd.extend(["--config", config_path])
    if use_batch_api:
        cmd.append("--use_batch_api")

    return cmd


def _load_resume_info_from_manifest(manifest_path: Path) -> tuple[list[str], int | None]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    quizzes = manifest.get("quizzes", [])
    if not isinstance(quizzes, list):
        raise SystemExit(
            f"[FATAL] Manifest has invalid 'quizzes' (expected list): {manifest_path}"
        )

    quiz_ids: list[str] = []
    for entry in quizzes:
        if not isinstance(entry, dict):
            continue
        quiz_id = entry.get("quiz_id")
        if isinstance(quiz_id, str) and quiz_id.strip():
            quiz_ids.append(quiz_id.strip())
    expected_n_items: int | None = None
    for key in ("num_questions_per_quiz", "chunk_size"):
        raw = manifest.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            expected_n_items = raw
            break
        if isinstance(raw, str):
            s = raw.strip()
            if s.isdigit():
                expected_n_items = int(s)
                break

    if expected_n_items is not None and expected_n_items <= 0:
        expected_n_items = None

    return quiz_ids, expected_n_items


def _quiz_has_eval_outputs(
    *, runs_root: Path, quiz_id: str, eval_model: str, expected_n_items: int | None
) -> bool:
    run_dir = runs_root / quiz_id
    if not run_dir.is_dir():
        return False

    summary_path = run_dir / f"{eval_model}_summary.json"
    result_path = run_dir / f"{eval_model}_result.json"

    if not (summary_path.is_file() and result_path.is_file()):
        return False
    if summary_path.stat().st_size <= 0 or result_path.stat().st_size <= 0:
        return False

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return False

    if not isinstance(summary, dict):
        return False

    required_fields = {
        "model",
        "created_at",
        "total_corr",
        "total_wrong",
        "acc",
        "n_items",
    }
    if not required_fields.issubset(summary.keys()):
        return False

    model_in_file = summary.get("model")
    if model_in_file is not None and str(model_in_file).strip() != eval_model:
        return False

    created_at = summary.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        return False

    total_corr = summary.get("total_corr")
    total_wrong = summary.get("total_wrong")
    n_items = summary.get("n_items")

    def _is_int(x: object) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if not (_is_int(total_corr) and _is_int(total_wrong) and _is_int(n_items)):
        return False
    if total_corr < 0 or total_wrong < 0 or n_items < 0:
        return False
    if total_corr + total_wrong != n_items:
        return False

    if expected_n_items is not None and n_items != expected_n_items:
        return False

    acc = summary.get("acc")
    if isinstance(acc, bool) or not isinstance(acc, (int, float)):
        return False
    if acc < 0.0 or acc > 1.0:
        return False
    if n_items > 0:
        expected_acc = total_corr / float(n_items)
        if abs(float(acc) - expected_acc) > 1e-6:
            return False

    return True


def _compute_missing_quiz_ids(
    *,
    runs_root: Path,
    eval_model: str,
    manifest_path: str | None,
) -> tuple[list[str], int, Path]:
    resolved_manifest = resolve_quizbench_manifest_path(
        runs_root, manifest_path=manifest_path
    )
    quiz_ids, expected_n_items = _load_resume_info_from_manifest(resolved_manifest)
    missing = [
        quiz_id
        for quiz_id in quiz_ids
        if not _quiz_has_eval_outputs(
            runs_root=runs_root,
            quiz_id=quiz_id,
            eval_model=eval_model,
            expected_n_items=expected_n_items,
        )
    ]
    return missing, len(quiz_ids), resolved_manifest


async def run_one(
    *,
    runs_root: Path,
    eval_model: str,
    max_tokens: int,
    reasoning_effort: str,
    use_batch_api: bool,
    manifest_path: str | None,
    config_path: str | None,
    logs_dir: Path,
    limiter: SubmitRateLimiter,
    submit_interval_seconds: float,
    dry_run: bool,
    openrouter_check: bool,
    openrouter_timeout_seconds: int,
    openrouter_print_raw: bool,
    skip_existing: bool,
    use_color: bool,
) -> int:
    rr_label = short_runs_root_label(runs_root)
    only_quiz_ids_csv: str | None = None
    if skip_existing:
        missing, total, resolved_manifest = _compute_missing_quiz_ids(
            runs_root=runs_root, eval_model=eval_model, manifest_path=manifest_path
        )
        done = total - len(missing)
        if total == 0:
            print(
                f"{format_tag('SKIP', use_color=use_color)} "
                f"No quizzes found in manifest; runs_root={rr_label} eval_model={eval_model}"
            )
            return 0
        if not missing:
            print(
                f"{format_tag('SKIP', use_color=use_color)} "
                "All quizzes already have eval outputs; "
                f"runs_root={rr_label} eval_model={eval_model} total={total}"
            )
            return 0

        only_quiz_ids_csv = ",".join(missing)
        print(
            f"{format_tag('INFO', use_color=use_color)} Resume scan: "
            f"runs_root={rr_label} eval_model={eval_model} "
            f"done={done}/{total} remaining={len(missing)} manifest={resolved_manifest.name}"
        )

    cmd = build_run_eval_cmd(
        runs_root=runs_root,
        eval_model=eval_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        use_batch_api=use_batch_api,
        manifest_path=manifest_path,
        only_quiz_ids_csv=only_quiz_ids_csv,
        config_path=config_path,
    )

    log_name = f"{compact_utc_ts()}__{slugify(runs_root_log_label(runs_root))}__{slugify(eval_model)}.log"
    log_path = logs_dir / log_name

    if dry_run:
        if submit_interval_seconds > 0:
            print(
                f"{format_tag('DRY-RUN', use_color=use_color)} "
                f"Would wait {int(submit_interval_seconds)}s before submitting."
            )
        print(f"{format_tag('DRY-RUN', use_color=use_color)} Command:")
        print(" ".join([shlex_quote(x) for x in cmd]))
        print(f"{format_tag('DRY-RUN', use_color=use_color)} Log: {log_path}")
        return 0

    ensure_dir(str(logs_dir))

    with open(log_path, "w", encoding="utf-8") as f:
        async with limiter.submission_slot():
            if openrouter_check:
                info = await fetch_openrouter_key_info(
                    timeout_seconds=openrouter_timeout_seconds
                )
                print(format_openrouter_limits(info))
                if openrouter_print_raw:
                    print(info.status_code)
                    print(info.raw_text)

            print(
                f"{format_tag('INFO', use_color=use_color)} Starting: "
                f"runs_root={rr_label} eval_model={eval_model} log={log_path}"
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=f, stderr=asyncio.subprocess.STDOUT
            )
        try:
            rc = await proc.wait()
            if rc == 0:
                print(
                    f"{format_tag('OK', use_color=use_color)} Completed: "
                    f"runs_root={rr_label} eval_model={eval_model}"
                )
            else:
                print(
                    f"{format_tag('ERROR', use_color=use_color)} run_eval failed (rc={rc}) "
                    f"runs_root={rr_label} eval_model={eval_model}"
                )
            return rc
        except asyncio.CancelledError:
            proc.terminate()
            await proc.wait()
            raise


def shlex_quote(s: str) -> str:
    """
    Small, dependency-free shlex.quote equivalent (handles typical args well).
    """
    if s == "":
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_/@%+=:,.-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


async def run_for_eval_model(
    *,
    eval_model: str,
    runs_roots: list[Path],
    max_parallel: int,
    limiter: SubmitRateLimiter,
    submit_interval_seconds: float,
    max_tokens: int,
    reasoning_effort: str,
    use_batch_api: bool,
    manifest_path: str | None,
    config_path: str | None,
    logs_dir: Path,
    dry_run: bool,
    openrouter_check: bool,
    openrouter_timeout_seconds: int,
    openrouter_print_raw: bool,
    stop_event: asyncio.Event,
    skip_existing: bool,
    use_color: bool,
) -> list[tuple[Path, int]]:
    q: asyncio.Queue[Path | None] = asyncio.Queue()
    for rr in runs_roots:
        q.put_nowait(rr)
    for _ in range(max_parallel):
        q.put_nowait(None)

    results: list[tuple[Path, int]] = []
    results_lock = asyncio.Lock()

    async def _worker(worker_id: int) -> None:
        while True:
            rr = await q.get()
            if rr is None:
                return
            if stop_event.is_set():
                return
            rc = await run_one(
                runs_root=rr,
                eval_model=eval_model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                use_batch_api=use_batch_api,
                manifest_path=manifest_path,
                config_path=config_path,
                logs_dir=logs_dir,
                limiter=limiter,
                submit_interval_seconds=submit_interval_seconds,
                dry_run=dry_run,
                openrouter_check=openrouter_check,
                openrouter_timeout_seconds=openrouter_timeout_seconds,
                openrouter_print_raw=openrouter_print_raw,
                skip_existing=skip_existing,
                use_color=use_color,
            )
            async with results_lock:
                results.append((rr, rc))

    workers = [asyncio.create_task(_worker(i)) for i in range(max_parallel)]
    await asyncio.gather(*workers)
    return results


def discover_runs_roots(*, runs_root_base: Path) -> list[Path]:
    if not runs_root_base.exists():
        raise SystemExit(f"[FATAL] runs_root_base does not exist: {runs_root_base}")
    if not runs_root_base.is_dir():
        raise SystemExit(f"[FATAL] runs_root_base is not a directory: {runs_root_base}")

    return sorted([p for p in runs_root_base.iterdir() if p.is_dir()], key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Parallel scheduler for `python -m quizbench.run_eval` across multiple RUNS_ROOTs.\n"
            "Starts subprocesses with a fixed delay between submissions (default: 60s)."
        )
    )

    ap.add_argument(
        "--subset_tag",
        type=str,
        default=None,
        help=(
            "Optional subset tag (e.g. ABMS20260101). If set, defaults "
            "--runs_root_base to eval_results/quizbench/quizzes_<SUBSET_TAG>/runs "
            "and --manifest_path to quizbench_manifest_<SUBSET_TAG>.json."
        ),
    )
    ap.add_argument(
        "--runs_root_base",
        type=str,
        default=None,
        help=(
            "Directory whose immediate subdirectories are treated as RUNS_ROOTs. "
            "Ignored if --runs_roots_csv is provided."
        ),
    )
    ap.add_argument(
        "--runs_roots_csv",
        type=str,
        default=None,
        help="Comma-separated explicit RUNS_ROOT paths (overrides --runs_root_base).",
    )
    ap.add_argument(
        "--eval_models_csv",
        type=str,
        default=None,
        help="Comma-separated eval models to run.",
    )
    ap.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Max number of concurrent run_eval subprocesses (across different RUNS_ROOTs).",
    )
    ap.add_argument(
        "--submit_interval_seconds",
        type=float,
        default=60.0,
        help="Delay enforced before each run_eval subprocess submission.",
    )
    ap.add_argument(
        "--use_batch_api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --use_batch_api through to quizbench.run_eval.",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Pass --max_tokens through to quizbench.run_eval.",
    )
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Pass --reasoning_effort through to quizbench.run_eval.",
    )
    ap.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help=(
            "Manifest filename/path passed to quizbench.run_eval. "
            "Relative paths resolve under each RUNS_ROOT."
        ),
    )
    ap.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set (default), scan each RUNS_ROOT for pre-existing per-quiz eval outputs "
            f"({{RUNS_ROOT}}/{{quiz_id}}/{{eval_model}}_result.json + _summary.json) and "
            "only evaluate missing quiz_ids to resume. Use --no-skip_existing to force re-evaluation."
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path passed through to quizbench.run_eval.",
    )
    ap.add_argument(
        "--logs_dir",
        type=str,
        default="eval_results/quizbench/logs/run_eval_parallel",
        help="Directory for per-subprocess logs.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would run without starting any subprocesses.",
    )
    ap.add_argument(
        "--openrouter_rate_limit_check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Before each submission, print remaining OpenRouter key limits (requires OPENROUTER_API_KEY).",
    )
    ap.add_argument(
        "--openrouter_timeout_seconds",
        type=int,
        default=30,
        help="Timeout for the OpenRouter key check request.",
    )
    ap.add_argument(
        "--openrouter_print_raw",
        action="store_true",
        help="Also print the raw OpenRouter key endpoint response (status code + body).",
    )
    ap.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Colorize console output with ANSI escape codes. "
            "Default: auto (TTY only; disabled if NO_COLOR is set)."
        ),
    )

    args = ap.parse_args()
    use_color = should_use_color(args.color)

    if args.max_parallel <= 0:
        raise SystemExit("[FATAL] --max_parallel must be >= 1")
    if args.submit_interval_seconds < 0:
        raise SystemExit("[FATAL] --submit_interval_seconds must be >= 0")

    manifest_path = args.manifest_path
    runs_root_base = args.runs_root_base
    if args.subset_tag:
        if runs_root_base is None:
            runs_root_base = f"eval_results/quizbench/quizzes_{args.subset_tag}/runs"
        if manifest_path is None:
            manifest_path = f"quizbench_manifest_{args.subset_tag}.json"

    runs_roots: list[Path]
    if args.runs_roots_csv:
        runs_roots = [
            Path(p).expanduser().resolve() for p in parse_csv_arg(args.runs_roots_csv)
        ]
    else:
        if not runs_root_base:
            raise SystemExit(
                "[FATAL] Must provide --runs_roots_csv or --runs_root_base (or --subset_tag)."
            )
        runs_roots = discover_runs_roots(runs_root_base=Path(runs_root_base).expanduser().resolve())

    if not runs_roots:
        raise SystemExit("[FATAL] No RUNS_ROOTs found.")

    seen: set[Path] = set()
    for rr in runs_roots:
        if rr in seen:
            raise SystemExit(f"[FATAL] Duplicate RUNS_ROOT provided: {rr}")
        seen.add(rr)
        if not rr.exists():
            raise SystemExit(f"[FATAL] RUNS_ROOT does not exist: {rr}")
        if not rr.is_dir():
            raise SystemExit(f"[FATAL] RUNS_ROOT is not a directory: {rr}")

    if args.eval_models_csv:
        eval_models = parse_csv_arg(args.eval_models_csv)
    else:
        eval_models = [
            # "gpt-5.1-2025-11-13",
            # "gemini-3-pro-preview",
            # "claude-opus-4-5-20251101",
            # "grok-4-0709",
            "kimi-k2-thinking",
            "deepseek-v3.2",
        ]

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    if not args.dry_run:
        ensure_dir(str(logs_dir))

    if args.openrouter_rate_limit_check and not os.environ.get("OPENROUTER_API_KEY"):
        print(
            f"{format_tag('WARN', use_color=use_color)} "
            "--openrouter_rate_limit_check set but OPENROUTER_API_KEY is missing."
        )

    stop_event = asyncio.Event()

    async def _run_all() -> int:
        loop = asyncio.get_running_loop()

        def _handle_sigint() -> None:
            if not stop_event.is_set():
                print(
                    f"{format_tag('WARN', use_color=use_color)} "
                    "Received interrupt; stopping after current subprocesses finish."
                )
                stop_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _handle_sigint)
            loop.add_signal_handler(signal.SIGTERM, _handle_sigint)
        except NotImplementedError:
            # Signal handlers aren't available on some platforms/event loops.
            pass

        limiter = SubmitRateLimiter(interval_seconds=args.submit_interval_seconds)

        any_failures = False
        for model in eval_models:
            if stop_event.is_set():
                break
            print(
                f"{format_tag('INFO', use_color=use_color)} "
                f"=== Eval model: {model} ==="
            )
            results = await run_for_eval_model(
                eval_model=model,
                runs_roots=runs_roots,
                max_parallel=args.max_parallel,
                limiter=limiter,
                submit_interval_seconds=args.submit_interval_seconds,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                use_batch_api=args.use_batch_api,
                manifest_path=manifest_path,
                config_path=args.config,
                logs_dir=logs_dir,
                dry_run=args.dry_run,
                openrouter_check=args.openrouter_rate_limit_check,
                openrouter_timeout_seconds=args.openrouter_timeout_seconds,
                openrouter_print_raw=args.openrouter_print_raw,
                stop_event=stop_event,
                skip_existing=args.skip_existing,
                use_color=use_color,
            )
            if any(rc != 0 for _, rc in results):
                any_failures = True

        return 1 if any_failures else 0

    try:
        rc = asyncio.run(_run_all())
    except KeyboardInterrupt:
        rc = 130
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
