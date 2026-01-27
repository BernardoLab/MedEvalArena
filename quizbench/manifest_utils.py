from __future__ import annotations

from pathlib import Path


def resolve_quizbench_manifest_path(
    runs_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    pattern: str = "quizbench_manifest*.json",
) -> Path:
    """
    Resolve which QuizBench manifest JSON to use for a given runs_root.

    If `manifest_path` is provided, it must exist (relative paths are resolved
    against `runs_root`). Otherwise, the newest file matching `pattern` under
    `runs_root` (non-recursive) is selected.
    """
    runs_root_path = Path(runs_root).expanduser().resolve()

    if manifest_path is not None:
        explicit = Path(manifest_path).expanduser()
        explicit = explicit if explicit.is_absolute() else (runs_root_path / explicit)
        explicit = explicit.resolve()
        if not explicit.exists():
            raise SystemExit(f"[FATAL] Manifest not found: {explicit}")
        if not explicit.is_file():
            raise SystemExit(f"[FATAL] Manifest is not a file: {explicit}")
        return explicit

    if not runs_root_path.exists():
        raise SystemExit(f"[FATAL] runs_root does not exist: {runs_root_path}")
    if not runs_root_path.is_dir():
        raise SystemExit(f"[FATAL] runs_root is not a directory: {runs_root_path}")

    candidates = [p for p in runs_root_path.glob(pattern) if p.is_file()]
    if not candidates:
        raise SystemExit(
            f"[FATAL] Could not find any manifests under {runs_root_path} "
            f"matching {pattern!r}."
        )

    # Pick newest by mtime; tie-break by name for determinism.
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))

