#!/usr/bin/env python3
"""
Deprecated Anthropic batch generation shim.

This entrypoint now routes to quizbench/run_batch_gen_quiz.py, which handles
provider dispatch (including Anthropic Message Batches) directly.
"""

from __future__ import annotations

from quizbench import run_batch_gen_quiz


def main() -> None:
    print(
        "[WARN] run_batch_gen_quiz_anthropic.py is deprecated and now routes to "
        "quizbench/run_batch_gen_quiz.py. Please update scripts to call the unified entrypoint."
    )
    run_batch_gen_quiz.main()


if __name__ == "__main__":
    main()
