import os
import time
from typing import Any, Dict, List

import anthropic

# How often to poll Message Batch status (in seconds)
BATCH_POLL_INTERVAL_SECONDS = 60


def get_anthropic_client():
    """
    Returns an Anthropic client configured for Message Batches.
    Relies on the ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Anthropic API key is not set. Please export ANTHROPIC_API_KEY before running."
        )
    return anthropic.Anthropic(api_key=api_key)


def create_message_batch(client: anthropic.Anthropic, requests: List[Dict[str, Any]]):
    """
    Creates a Message Batch with the given list of requests.
    """
    print(f"Creating Message Batch with {len(requests)} requests...")
    message_batch = client.messages.batches.create(requests=requests)
    print(f"Created Message Batch with id: {message_batch.id}")
    print(f"Initial processing_status: {message_batch.processing_status}")
    return message_batch


def wait_for_batch_completion(
    client: anthropic.Anthropic,
    message_batch,
    poll_interval: int = BATCH_POLL_INTERVAL_SECONDS,
):
    """
    Polls the Message Batch until processing_status == 'ended'.
    """
    batch_id = message_batch.id
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        status = message_batch.processing_status
        counts = getattr(message_batch, "request_counts", None)
        print(f"Batch {batch_id} status: {status}. Request counts: {counts}")
        if status == "ended":
            break
        time.sleep(poll_interval)

    print(f"Batch {batch_id} reached terminal status: {message_batch.processing_status}")
    return message_batch


def collect_batch_results(
    client: anthropic.Anthropic,
    message_batch,
) -> Dict[str, Any]:
    """
    Streams and collects all results for a completed Message Batch.
    Returns a mapping custom_id -> MessageBatchIndividualResponse.
    """
    batch_id = message_batch.id
    print(f"Collecting results for Message Batch {batch_id}...")
    combined_results: Dict[str, Any] = {}

    for entry in client.messages.batches.results(batch_id):
        combined_results[entry.custom_id] = entry

    print(f"Collected results for {len(combined_results)} requests.")
    return combined_results


def extract_text_from_anthropic_message(message) -> str:
    """
    Given an Anthropic Message object, extract the assistant text content
    in a robust way.
    """
    content = getattr(message, "content", "")

    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_val = getattr(block, "text", None)
                if text_val is None and isinstance(block, dict):
                    text_val = block.get("text", "")
                if text_val:
                    text_parts.append(text_val)
        if text_parts:
            return "".join(text_parts)
        return "".join(str(b) for b in content)

    if isinstance(content, str):
        return content

    return str(content)
