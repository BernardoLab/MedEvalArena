#!/usr/bin/env bash
set -euo pipefail

# Find JSON files using legacy absolute paths and rewrite them to be repo-relative.
OLD_RUNS_PREFIX='/Users/dbernardo/Documents/pyres/MedEvalArena/eval_results/'
NEW_RUNS_PREFIX='eval_results/'
OLD_ABMS_PATH='/Users/dbernardo/Documents/pyres/MedEvalArena/data/ABMS_specialties.csv'
NEW_ABMS_PATH='data/ABMS_specialties.csv'
OLD_TOPIC_MAP_PATH='/Users/dbernardo/Documents/pyres/MedEvalArena/data/topic_to_abms.yaml'
NEW_TOPIC_MAP_PATH='data/topic_to_abms.yaml'
DRY_RUN=0
EXAMPLE_LIMIT=5
RESPECT_IGNORE=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/rewrite_runs_root.sh [--dry-run[=N]] [--help]
  --dry-run       Preview only; show up to 5 example replacements (or N if provided).
  --respect-ignore  Honor .gitignore and other ignore files.
  --help          Show this help message.
USAGE
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --dry-run=*)
      DRY_RUN=1
      EXAMPLE_LIMIT="${1#*=}"
      ;;
    --respect-ignore)
      RESPECT_IGNORE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n\n" "$1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if ! [[ "${EXAMPLE_LIMIT}" =~ ^[0-9]+$ ]]; then
  printf "EXAMPLE_LIMIT must be a non-negative integer (got %s)\n" "${EXAMPLE_LIMIT}" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  printf "ripgrep (rg) is required to run this script.\n" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  printf "uv is required to run the replacement step (install via https://docs.astral.sh/uv/).\n" >&2
  exit 1
fi

if (( RESPECT_IGNORE )); then
  RG_IGNORE_ARGS=()
else
  RG_IGNORE_ARGS=(--no-ignore --hidden)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mapfile -t target_files < <(
  rg --files-with-matches --fixed-strings --glob '*.json' \
    "${RG_IGNORE_ARGS[@]}" \
    -e "${OLD_RUNS_PREFIX}" \
    -e "${OLD_ABMS_PATH}" \
    -e "${OLD_TOPIC_MAP_PATH}" \
    . || true
)

if (( ${#target_files[@]} == 0 )); then
  printf "No JSON files contained the target absolute paths.\n"
  exit 0
fi

UV_NO_SYNC=1 uv run python - \
  "${OLD_RUNS_PREFIX}" "${NEW_RUNS_PREFIX}" \
  "${OLD_ABMS_PATH}" "${NEW_ABMS_PATH}" \
  "${OLD_TOPIC_MAP_PATH}" "${NEW_TOPIC_MAP_PATH}" \
  "${DRY_RUN}" "${EXAMPLE_LIMIT}" \
  "${target_files[@]}" <<'PY'
import sys
from pathlib import Path

(
    old_runs_prefix,
    new_runs_prefix,
    old_abms_path,
    new_abms_path,
    old_topic_map_path,
    new_topic_map_path,
    dry_run_flag,
    example_limit_raw,
    *files,
) = sys.argv[1:]
dry_run = dry_run_flag == "1"
example_limit = int(example_limit_raw)

replacements = [
    (old_runs_prefix, new_runs_prefix),
    (old_abms_path, new_abms_path),
    (old_topic_map_path, new_topic_map_path),
]

examples = []
files_with_changes = 0
total_replacements = 0

for file_path in files:
    path = Path(file_path)
    text = path.read_text()
    if not any(old in text for old, _ in replacements):
        continue

    matches = sum(text.count(old) for old, _ in replacements)
    total_replacements += matches
    files_with_changes += 1

    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)

    if dry_run:
        if example_limit and len(examples) < example_limit:
            for line in text.splitlines():
                if any(old in line for old, _ in replacements):
                    updated_line = line
                    for old, new in replacements:
                        updated_line = updated_line.replace(old, new)
                    examples.append(f"{path}: {line.strip()} -> {updated_line.strip()}")
                    if len(examples) >= example_limit:
                        break
        continue

    if text != updated:
        path.write_text(updated)
        print(f"Updated {path}")

if dry_run:
    print(f"Dry run: would update {files_with_changes} files with {total_replacements} replacements.")
    if examples:
        print("Example replacements:")
        for example in examples:
            print(f"  {example}")
PY
