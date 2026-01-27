import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

# Provide a minimal stub so importing the transport works even if the
# anthropic package is not installed in the local environment.
if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = SimpleNamespace(Anthropic=SimpleNamespace)
if "openai" not in sys.modules:
    _OpenAIStub = type("OpenAI", (), {})  # minimal placeholder for imports
    sys.modules["openai"] = SimpleNamespace(OpenAI=_OpenAIStub, AsyncOpenAI=_OpenAIStub)
if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(safe_load=lambda *args, **kwargs: None)

from quizbench.anthropic_message_batches import generate_quizzes_via_anthropic_batch
from quizbench.batch_generate_quiz import QuizRequest


class _FakeBatch:
    def __init__(self, batch_id: str, *, processing_status: str = "ended", request_counts=None):
        self.id = batch_id
        self.processing_status = processing_status
        self.request_counts = request_counts or {}


class _FakeEntry:
    def __init__(self, custom_id: str, message_text: str):
        self.custom_id = custom_id
        self.result = SimpleNamespace(type="succeeded", message=SimpleNamespace(content=message_text))


class _FakeBatches:
    def __init__(self, entries):
        self._entries = entries

    def create(self, requests):
        return _FakeBatch("batch-test", request_counts={"submitted": len(requests)})

    def retrieve(self, batch_id):
        return _FakeBatch(batch_id, processing_status="ended")

    def results(self, batch_id):
        return list(self._entries)


class _FakeAnthropicClient:
    def __init__(self, entries):
        self.messages = SimpleNamespace(batches=_FakeBatches(entries))


class AnthropicBatchSmokeTest(TestCase):
    def test_generate_quizzes_with_stubbed_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "quizzes"
            batch_input = Path(tmpdir) / "batch_inputs" / "input.jsonl"

            req = QuizRequest(
                quiz_id="claude_test_seed0",
                seed=0,
                num_questions=2,
                num_choices=5,
                topics_csv="cardiology,neurology",
                out_dir=str(out_dir),
            )

            quiz_payload = {
                "items": [
                    {
                        "question": "Q1",
                        "options": ["A", "B", "C", "D", "E"],
                        "answer": "A",
                        "explanation": "Because A.",
                    },
                    {
                        "question": "Q2",
                        "options": ["A", "B", "C", "D", "E"],
                        "answer": "B",
                        "explanation": "Because B.",
                    },
                ]
            }
            fake_entries = [_FakeEntry(f"quiz__{req.quiz_id}", json.dumps(quiz_payload))]
            fake_client = _FakeAnthropicClient(fake_entries)

            with mock.patch(
                "quizbench.anthropic_message_batches.get_anthropic_client",
                return_value=fake_client,
            ):
                results, batch_id = generate_quizzes_via_anthropic_batch(
                    "claude-stub-1",
                    [req],
                    batch_input_path=str(batch_input),
                    poll_interval=0,
                    temperature=0.1,
                    max_output_tokens=128,
                    overwrite=True,
                )

            self.assertEqual(batch_id, "batch-test")
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertEqual(res.status, "ok")
            self.assertTrue(res.quiz_path)
            quiz_path = Path(res.quiz_path)
            self.assertTrue(quiz_path.is_file())

            lines = quiz_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(batch_input.is_file())


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    import unittest

    unittest.main()
