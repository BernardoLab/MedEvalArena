#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: scripts/evaluate_from_batch_api.py
Version: 2, (uses Responses API for newer OpenAI models)
Description: Assess LLMs on mARC / MMLU-Pro using the Batch API.

This script is a Batch-API compatible variant of quizbench/evaluate_from_api.py.

Supports:
  - Standard evaluation (one response per question)
  - Uncertainty quantification: 10 runs per question in a single Batch job
    (so your .jsonl file has 10 * N requests, each with age jitter applied).

For models that support Batch via:
  * OpenAI's Batch API using the `/v1/responses` endpoint, or
  * Google Gemini Batch API through the OpenAI compatibility layer
    (for models like `gemini-2.5-flash`), which still uses `/v1/chat/completions`
    for Batch as of this script.

For Gemini models, this script uses:
  * The OpenAI Python SDK (pointed at the Gemini OpenAI-compat endpoint)
    for creating and polling Batch jobs, and
  * The Google GenAI SDK (`google-genai`) for uploading and downloading
    batch input/output files, as required by the Gemini Batch API.
"""

# Import libraries
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# -----------------------------
# API keys / configuration
# -----------------------------

# You can either:
#   1. Set these constants directly, OR
#   2. Set the corresponding environment variables (recommended).
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""  # or set env var GEMINI_API_KEY

# Base URL for the Gemini OpenAI-compatibility endpoint.
# Can be overridden with the GEMINI_API_BASE_URL environment variable.
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Models that this script will run via the Batch API, using either:
#   - OpenAI's Batch API `/v1/responses` (for OpenAI models), or
#   - Gemini's OpenAI-compat Batch API `/v1/chat/completions` (for Gemini models).
BATCH_COMPATIBLE_MODELS = {
    # OpenAI models
    "gpt-5-nano-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gpt-5-pro-2025-10-06",

    # Gemini models (via OpenAI compatibility layer and Gemini Batch API)
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3-pro-preview"
}

# How often to poll Batch job status (in seconds)
BATCH_POLL_INTERVAL_SECONDS = 60
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_MEDARC_PATH = ROOT_DIR / "data_medARCv1" / "test_medARCv1.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_results"


# -----------------------------
# Data loading & preprocessing
# -----------------------------

def fix_options_column(example, debug=False):
    # Check if 'options' is a string and try to parse it as a list
    if isinstance(example.get('options'), str):
        if debug:
            print("Debugging question:", example.get('question_id', 'Unknown'))
            print("Question text:", example.get('question', 'No question text available'))
            print("Options string before parsing:", example['options'])

        # Replace various smart quotes
        cleaned_options = (
            example['options']
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )

        try:
            example['options'] = json.loads(cleaned_options)
        except json.JSONDecodeError as e:
            if debug:
                print("JSONDecodeError encountered!")
                print("Error message:", e)
                print("Problematic options string:", cleaned_options)
            raise e
    return example


def preprocess(test_df):
    """
    - Removes "N/A" options.
    - Groups examples by 'category', returning a dict: {category: [examples]}.
    """
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)

    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def load_mmlu_pro():
    """
    Loads MMLU-Pro dev set and medARC test set, then preprocesses into:
      test_df: {category: [examples]}
      dev_df:  {category: [examples with COT]}
    """
    dev_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    dev_df = preprocess(dev_dataset["validation"])

    test_dataset = load_dataset("csv", data_files={"test": str(DATA_MEDARC_PATH)})
    test_dataset = test_dataset.map(fix_options_column)

    # Align features with dev/test split
    test_dataset.features = dev_dataset["test"].features
    test_df = preprocess(test_dataset["test"])

    return test_df, dev_df


# -----------------------------
# Prompt formatting utilities
# -----------------------------

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    # Patterns for answer extraction
    patterns = [
        r"answer is \(?([A-J])\)?",                               # "answer is (D)" or "answer is D"
        r"[aA]nswer:\s*([A-J])",                                  # "Answer: D" or "answer: D"
        r"best course of action to perform immediately is"
        r" [\(\[]([A-J])[\)\]]",                                  # "best course... is (D)" or "[D]"
        r"best course of action.*[\(\[]([A-J])[\)\]]",            # "best course of action... (D)" or "[D]" in any sentence
        r"(?=.*\bbest\b)(?=.*\baction\b).*[\(\[]([A-J])[\)\]]",   # Sentence with "best" and "action" followed by (D) or [D]
    ]

    # Try each pattern sequentially
    for i, pattern in enumerate(patterns, start=1):
        match = re.search(pattern, text)
        if match:
            print(f"Pattern {i} matched: {pattern}")
            return match.group(1)
        else:
            print(f"{i} attempt failed\n" + text)

    # If no patterns matched
    return None


def process_age_strings(text):
    """
    Finds age-related substrings in the input text, adds a random decimal
    between 0 and 0.04 to the age, and reformats the substring to "XX.XXX year-old".
    """
    age_pattern = re.compile(
        r'\b(\d+)'                     # Capture the numeric age
        r'(?:\s*[-\s]?)'               # Optional separator (space or hyphen)
        r'(year-old|year old|yo)\b',   # Capture the age descriptor
        re.IGNORECASE
    )

    def add_decimal(match):
        age = int(match.group(1))
        decimal = random.uniform(0, 0.04)
        new_age = age + decimal
        new_age_str = f"{new_age:.3f}"
        return f"{new_age_str} year-old"

    modified_text = age_pattern.sub(add_decimal, text)
    return modified_text


# -----------------------------
# Batch API helpers
# -----------------------------

def _is_gemini_model(model_name: str) -> bool:
    """
    Returns True if the model name corresponds to a Gemini model that should
    be called via the OpenAI compatibility endpoint.
    """
    return model_name.startswith("gemini-")


def _openai_model_supports_sampling_params(model_name: str) -> bool:
    """
    Heuristic: GPT-5 series no longer support temperature/top_p style sampling
    parameters; for those, we omit such fields in the request body.
    """
    return not model_name.startswith("gpt-5")


def get_openai_client(model_name: str):
    """
    Returns an OpenAI client configured for Batch API operations.

    - For OpenAI models, this talks to api.openai.com (or a compatible proxy).
    - For Gemini models (model_name starting with 'gemini-'), this talks to
      Google's Gemini API via the OpenAI compatibility endpoint.
    """
    # Gemini models go through the Gemini OpenAI-compat endpoint.
    if _is_gemini_model(model_name):
        print(f"Using OpenAI endpoint for Gemini model: {model_name}")
        api_key = os.environ.get("GEMINI_API_KEY", "") or GEMINI_API_KEY
        base_url = os.environ.get("GEMINI_API_BASE_URL", GEMINI_API_BASE_URL)
        if not api_key:
            raise RuntimeError(
                "Gemini API key is not set. Please either:\n"
                "  * export GEMINI_API_KEY='YOUR_KEY_HERE', or\n"
                "  * set the GEMINI_API_KEY constant at the top of this script."
            )
        # Use OpenAI SDK against Gemini's OpenAI-compatible endpoint.
        return OpenAI(api_key=api_key, base_url=base_url)

    # Default: use OpenAI's own endpoint.
    api_key = os.environ.get("OPENAI_API_KEY", "") or OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is not set. Please either:\n"
            "  * export OPENAI_API_KEY='YOUR_KEY_HERE', or\n"
            "  * set the OPENAI_API_KEY constant at the top of this script."
        )

    # Allow overriding the base URL for OpenAI-compatible proxies if needed.
    openai_base_url = os.environ.get("OPENAI_API_BASE_URL", "").strip()
    if openai_base_url:
        return OpenAI(api_key=api_key, base_url=openai_base_url)
    return OpenAI(api_key=api_key)


def get_gemini_file_client():
    """
    Returns a google-genai Client for file upload/download when using
    Gemini via the OpenAI compatibility layer.

    Requires the `google-genai` package:
        pip install google-genai
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is required to use Gemini Batch with this script.\n"
            "Install it with:\n"
            "    pip install google-genai\n"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "Gemini API key is not set. Please either:\n"
            "  * export GEMINI_API_KEY='YOUR_KEY_HERE', or\n"
            "  * set the GEMINI_API_KEY constant at the top of this script."
        )

    # Regular GenAI client for uploads & downloads (Gemini Developer API).
    return genai.Client(api_key=api_key)


def build_batch_input_file(subject, test_questions, cot_examples_dict,
                           model_name, batch_input_path, reasoning_effort):
    """
    Builds a JSONL batch input file for a single subject (one request per question).

    For OpenAI models:
      - Uses the Responses API via `/v1/responses`.

    For Gemini models:
      - Uses Chat Completions via `/v1/chat/completions` on the
        Gemini OpenAI-compat endpoint (as documented by Google).

    Returns:
        custom_id_to_question: dict mapping custom_id -> question_dict
    """
    custom_id_to_question = {}
    is_gemini_model = _is_gemini_model(model_name)

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for question in tqdm(test_questions, desc=f"Preparing batch input for {subject}"):
            q_id = question["question_id"]
            category = question["category"]  # usually same as subject
            options = question["options"]

            # Apply age randomization without mutating the original object
            question_text = process_age_strings(question["question"])

            # Few-shot Chain-of-Thought examples for this category
            cot_examples = cot_examples_dict[category]

            prompt_intro = (
                "The following are multiple choice questions (with answers) about {}. "
                "Think step by step and then output the answer in the format of "
                "\"The answer is (X)\" at the end.\n\n".format(category)
            )

            cot_prompt = ""
            for ex in cot_examples:
                cot_prompt += format_example(ex["question"], ex["options"], ex["cot_content"])

            input_text = format_example(question_text, options)
            user_content = prompt_intro + cot_prompt + input_text

            if is_gemini_model:
                # Gemini Batch still uses /v1/chat/completions
                messages = [
                    {"role": "user", "content": user_content}
                ]
                body = {
                    "model": model_name,
                    "messages": messages,
                    "top_p": 1,
                    "reasoning_effort": reasoning_effort,
                }
                url = "/v1/chat/completions"
            else:
                # OpenAI Batch via the Responses API
                body = {
                    "model": model_name,
                    # For this eval, we use single-turn text input.
                    "input": user_content,
                }
                if reasoning_effort:
                    body["reasoning"] = {"effort": reasoning_effort}

                if _openai_model_supports_sampling_params(model_name):
                    body["top_p"] = 1

                url = "/v1/responses"

            custom_id = f"{subject}__{q_id}"
            line_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": url,
                "body": body,
            }

            f.write(json.dumps(line_obj) + "\n")
            custom_id_to_question[custom_id] = question

    return custom_id_to_question


def build_batch_input_file_uq(
    subject,
    test_questions,
    cot_examples_dict,
    model_name,
    batch_input_path,
    reasoning_effort,
    num_runs=10,
):
    """
    Builds a JSONL batch input file for uncertainty quantification.

    For each question, we create `num_runs` separate requests, each with
    age jitter applied, so the file has len(test_questions) * num_runs lines.

    For OpenAI models:
      - Uses the Responses API via `/v1/responses`.

    For Gemini models:
      - Uses Chat Completions via `/v1/chat/completions`.

    Returns:
        custom_id_to_meta: dict custom_id -> {"run": run_idx, "question": question_dict}
    """
    custom_id_to_meta = {}
    is_gemini_model = _is_gemini_model(model_name)

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for run_idx in range(num_runs):
            desc = f"Preparing batch input for {subject} (run {run_idx})"
            for question in tqdm(test_questions, desc=desc, leave=False):
                q_id = question["question_id"]
                category = question["category"]
                options = question["options"]

                # Apply age randomization anew for each run
                question_text = process_age_strings(question["question"])

                cot_examples = cot_examples_dict[category]

                prompt_intro = (
                    "The following are multiple choice questions (with answers) about {}. "
                    "Think step by step and then output the answer in the format of "
                    "\"The answer is (X)\" at the end.\n\n".format(category)
                )

                cot_prompt = ""
                for ex in cot_examples:
                    cot_prompt += format_example(ex["question"], ex["options"], ex["cot_content"])

                input_text = format_example(question_text, options)
                user_content = prompt_intro + cot_prompt + input_text

                if is_gemini_model:
                    messages = [
                        {"role": "user", "content": user_content}
                    ]
                    body = {
                        "model": model_name,
                        "messages": messages,
                        "top_p": 1,
                        "reasoning_effort": reasoning_effort,
                    }
                    url = "/v1/chat/completions"
                else:
                    body = {
                        "model": model_name,
                        "input": user_content,
                    }
                    if reasoning_effort:
                        body["reasoning"] = {"effort": reasoning_effort}
                    if _openai_model_supports_sampling_params(model_name):
                        body["top_p"] = 1
                    url = "/v1/responses"

                custom_id = f"{subject}__run{run_idx}__{q_id}"
                line_obj = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": url,
                    "body": body,
                }

                f.write(json.dumps(line_obj) + "\n")
                custom_id_to_meta[custom_id] = {"run": run_idx, "question": question}

    return custom_id_to_meta


def submit_batch_job(openai_client, batch_input_path, *,
                     is_gemini_model=False, gemini_file_client=None):
    """
    Uploads the JSONL file as a batch file and creates a Batch job.

    For OpenAI models:
      - Uses the OpenAI SDK `files.create` and `batches.create`,
        with endpoint set to `/v1/responses`.

    For Gemini models (via OpenAI compatibility):
      - Uses google-genai's `files.upload` to upload the JSONL file.
      - Passes the resulting file name/id to `openai_client.batches.create`
        as `input_file_id`, with endpoint `/v1/chat/completions`, as
        documented for Gemini Batch OpenAI compatibility.
    """
    print(f"Uploading batch input file: {batch_input_path}")

    # Endpoint differs for OpenAI vs Gemini
    if is_gemini_model:
        endpoint = "/v1/chat/completions"
    else:
        endpoint = "/v1/responses"

    if is_gemini_model:
        if gemini_file_client is None:
            raise RuntimeError(
                "Gemini file client is required when submitting a Gemini Batch job. "
                "This should be created via get_gemini_file_client()."
            )

        # Upload the JSONL file in OpenAI input format, using regular genai SDK
        display_name = os.path.basename(batch_input_path)
        upload_config = {
            "display_name": display_name,
            "mime_type": "jsonl",
        }

        uploaded_file = gemini_file_client.files.upload(
            file=batch_input_path,
            config=upload_config,
        )

        # google-genai Files.upload can return either a File object or a name string.
        input_file_id = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "file", None) or uploaded_file
        print(f"Uploaded batch input file to Gemini Files API with id/name: {input_file_id}")

        # Create batch via OpenAI compatibility endpoint
        batch = openai_client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={
                "description": f"MedEvalArena eval job for {os.path.basename(batch_input_path)}"
            },
        )
    else:
        # OpenAI-native path: use OpenAI Files API directly
        with open(batch_input_path, "rb") as f:
            batch_input_file = openai_client.files.create(
                file=f,
                purpose="batch",
            )

        print(f"Created batch input file with id: {batch_input_file.id}")

        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={
                "description": f"MedEvalArena eval job for {os.path.basename(batch_input_path)}"
            },
        )

    print(f"Created batch with id: {batch.id}")
    print(f"Initial batch status: {batch.status}")
    return batch


def wait_for_batch_completion(openai_client, batch, poll_interval=BATCH_POLL_INTERVAL_SECONDS):
    """
    Polls the Batch job until it reaches a terminal state.
    Uses the OpenAI (or OpenAI-compatible) client for /v1/batches retrieval.
    """
    terminal_states = {"completed", "failed", "cancelled", "expired"}

    while batch.status not in terminal_states:
        print(f"Batch {batch.id} status: {batch.status}. Sleeping {poll_interval}s...")
        time.sleep(poll_interval)
        batch = openai_client.batches.retrieve(batch.id)

    print(f"Batch {batch.id} reached terminal status: {batch.status}")
    if getattr(batch, "errors", None):
        print("Batch reported errors:", batch.errors)

    return batch


def _parse_batch_file(openai_client, file_id, *,
                      is_gemini_model=False, gemini_file_client=None):
    """
    Helper: given a file_id (output_file_id or error_file_id), fetches and parses
    the JSONL into a dict custom_id -> line_dict.

    For OpenAI models:
      - Uses openai_client.files.content(file_id) and reads `.text`.

    For Gemini models:
      - Uses google-genai `files.download(file=file_id)` and decodes bytes
        into UTF-8 text, as recommended by the Gemini Batch API docs.
    """
    results = {}
    if not file_id:
        return results

    if is_gemini_model:
        if gemini_file_client is None:
            raise RuntimeError(
                "Gemini file client is required when downloading Gemini Batch results."
            )
        print(f"Downloading batch file via Gemini Files API: {file_id}")
        # google-genai Files.download returns raw bytes.
        file_bytes = gemini_file_client.files.download(file=file_id)
        content = file_bytes.decode("utf-8")
    else:
        print(f"Downloading batch file: {file_id}")
        file_response = openai_client.files.content(file_id)
        # OpenAI Python SDK exposes `.text` on the response object for file content.
        content = file_response.text

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        custom_id = data["custom_id"]
        results[custom_id] = data

    return results


def collect_batch_results(openai_client, batch, *,
                          is_gemini_model=False, gemini_file_client=None):
    """
    Retrieves and merges results from the batch's output and error files.

    Returns:
        combined_results: dict custom_id -> result_line (includes either response or error)

    For Gemini models, uses google-genai for downloading files, since the
    OpenAI compatibility layer does not yet support /v1/files upload/download.
    """
    output_results = _parse_batch_file(
        openai_client,
        getattr(batch, "output_file_id", None),
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
    )
    error_results = _parse_batch_file(
        openai_client,
        getattr(batch, "error_file_id", None),
        is_gemini_model=is_gemini_model,
        gemini_file_client=gemini_file_client,
    )

    combined_results = {}
    combined_results.update(output_results)
    combined_results.update(error_results)
    return combined_results


def _extract_text_from_body(body):
    """
    Given a Batch output `body` object, extract the assistant text content in a robust way.

    Supports both:
      - OpenAI Responses API (`object: "response"` with `output` / `output_text`), and
      - Chat Completions (`choices[0].message.content`).
    """
    try:
        if not isinstance(body, dict):
            return str(body)

        # --- Responses API shape ---
        # Prefer the convenience field if present.
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Otherwise, parse the `output` list.
        output_items = body.get("output")
        if isinstance(output_items, list):
            text_chunks = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                # Messages contain a list of content parts.
                if item_type == "message":
                    content_list = item.get("content", [])
                    if isinstance(content_list, list):
                        for part in content_list:
                            if not isinstance(part, dict):
                                continue
                            part_type = part.get("type")
                            if part_type in ("output_text", "text", "input_text"):
                                text = part.get("text")
                                if isinstance(text, str):
                                    text_chunks.append(text)
                # Some providers may emit bare output_text items at the top level.
                elif item_type in ("output_text", "text") and isinstance(item.get("text"), str):
                    text_chunks.append(item["text"])

            if text_chunks:
                return "".join(text_chunks)

        # --- Chat Completions shape (for Gemini / legacy) ---
        if "choices" in body:
            choice = body["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")

            # Handle possible list-of-parts content (for multimodal/text-part responses)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            if not isinstance(content, str):
                content = str(content)
            return content

        # Fallback to stringified JSON if we can't interpret the structure.
        return json.dumps(body)[:2000]
    except Exception as e:
        return f"Error parsing response body: {e}. Raw body (truncated): {json.dumps(body)[:2000]}"


# -----------------------------
# Result building utilities
# -----------------------------

def build_results_for_subject(subject, custom_id_to_question, batch_results):
    """
    Combines Batch API outputs with the original questions into per-question result records.

    Returns:
        results: list of dicts, each containing the original question fields plus:
                - 'pred': model's extracted answer letter (A-J or None)
                - 'model_outputs': raw model output text
    """
    results = []

    for custom_id, question in custom_id_to_question.items():
        line = batch_results.get(custom_id)
        if line is None:
            model_output = "Error: no result returned for this request."
            pred = None
        else:
            if line.get("error"):
                err = line["error"]
                code = err.get("code", "unknown_code")
                msg = err.get("message", "no error message provided")
                model_output = f"Batch error ({code}): {msg}"
                pred = None
            else:
                # Successful response
                body = line["response"]["body"]
                content = _extract_text_from_body(body)
                model_output = content.replace("**", "")
                pred = extract_answer(model_output)

        res_entry = dict(question)  # copy to avoid mutating original
        res_entry["pred"] = pred
        res_entry["model_outputs"] = model_output
        if "category" not in res_entry:
            res_entry["category"] = subject
        results.append(res_entry)

    return results


def build_results_for_subject_uq(subject, custom_id_to_meta, batch_results, num_runs=10):
    """
    Builds per-run results for uncertainty quantification.

    Returns:
        per_run_results: dict run_idx -> [result_dicts]
    """
    per_run_results = {run_idx: [] for run_idx in range(num_runs)}

    for custom_id, meta in custom_id_to_meta.items():
        run_idx = meta["run"]
        question = meta["question"]
        line = batch_results.get(custom_id)

        if line is None:
            model_output = "Error: no result returned for this request."
            pred = None
        else:
            if line.get("error"):
                err = line["error"]
                code = err.get("code", "unknown_code")
                msg = err.get("message", "no error message provided")
                model_output = f"Batch error ({code}): {msg}"
                pred = None
            else:
                body = line["response"]["body"]
                content = _extract_text_from_body(body)
                model_output = content.replace("**", "")
                pred = extract_answer(model_output)

        res_entry = dict(question)
        res_entry["pred"] = pred
        res_entry["model_outputs"] = model_output
        if "category" not in res_entry:
            res_entry["category"] = subject

        per_run_results[run_idx].append(res_entry)

    return per_run_results


def build_category_record(results):
    """
    Builds the per-category correctness record, mirroring the logic from the original
    update_result() helper in quizbench/evaluate_from_api.py.
    """
    category_record = {}
    for each in results:
        category = each["category"]
        if category not in category_record:
            category_record[category] = {"corr": 0.0, "wrong": 0.0}

        if not each.get("pred"):
            # No prediction — do a deterministic random guess as in the original script
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                category_record[category]["corr"] += 1
            else:
                category_record[category]["wrong"] += 1
        elif each["pred"] == each["answer"]:
            category_record[category]["corr"] += 1
        else:
            category_record[category]["wrong"] += 1

    return category_record


# -----------------------------
# Result saving utilities
# -----------------------------

def save_res(res, output_res_path):
    """
    Deduplicates by question_id (if necessary) and writes full results JSON.
    """
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    """
    Writes per-category and total accuracy summary JSON.
    """
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


# -----------------------------
# Main evaluation routines
# -----------------------------

def evaluate_standard(subjects, model_name, output_dir, args):
    """
    Standard evaluation: one completion/response per question (no UQ).

    For OpenAI models, this uses the Responses API via Batch.
    For Gemini models, this uses Chat Completions via the Gemini Batch API.
    """
    if model_name not in BATCH_COMPATIBLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not configured for Batch API in "
            "evaluate_from_batch_api.py. Use quizbench/evaluate_from_api.py for this model, "
            "or add it to BATCH_COMPATIBLE_MODELS if it supports Batch."
        )

    is_gemini_model = _is_gemini_model(model_name)

    # OpenAI client for either OpenAI or Gemini (via OpenAI compatibility layer)
    openai_client = get_openai_client(model_name)
    gemini_file_client = get_gemini_file_client() if is_gemini_model else None

    test_df, dev_df = load_mmlu_pro()

    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]

        if args.num_tokens is not None:
            num_tokens_str = '_' + str(args.num_tokens)
        else:
            num_tokens_str = ''

        effort_str = f"_{args.reasoning_effort}" if args.reasoning_effort else ""
        model_tag = model_name + effort_str

        subject_output_dir = os.path.join(output_dir, model_tag)
        os.makedirs(subject_output_dir, exist_ok=True)

        # Use model_tag instead of model_name in filenames
        output_res_path = os.path.join(
            subject_output_dir,
            model_tag + num_tokens_str + '_' + subject + "_result.json",
        )
        output_summary_path = os.path.join(
            subject_output_dir,
            model_tag + num_tokens_str + '_' + subject + "_summary.json",
        )
        batch_input_path = os.path.join(
            subject_output_dir,
            model_tag + num_tokens_str + '_' + subject + "_batch_input.jsonl",
        )

        print(f"\n=== Subject: {subject} ===")
        print(f"Number of questions: {len(test_data)}")
        print(f"Result file:   {output_res_path}")
        print(f"Summary file:  {output_summary_path}")
        print(f"Batch input:   {batch_input_path}")

        # 1. Build the batch input JSONL
        custom_id_to_question = build_batch_input_file(
            subject=subject,
            test_questions=test_data,
            cot_examples_dict=dev_df,
            model_name=model_name,
            batch_input_path=batch_input_path,
            reasoning_effort=args.reasoning_effort,
        )

        if not custom_id_to_question:
            print(f"No questions to evaluate for subject {subject}. Skipping.")
            continue

        # 2. Submit the Batch job
        batch = submit_batch_job(
            openai_client,
            batch_input_path,
            is_gemini_model=is_gemini_model,
            gemini_file_client=gemini_file_client,
        )

        # 3. Wait until the batch completes (or otherwise finishes)
        batch = wait_for_batch_completion(openai_client, batch)

        # 4. Collect results from the Batch output & error files
        batch_results = collect_batch_results(
            openai_client,
            batch,
            is_gemini_model=is_gemini_model,
            gemini_file_client=gemini_file_client,
        )

        # 5. Combine model outputs with questions
        subject_results = build_results_for_subject(subject, custom_id_to_question, batch_results)

        # 6. Write per-question results
        save_res(subject_results, output_res_path)

        # 7. Compute and write accuracy summary
        category_record = build_category_record(subject_results)
        save_summary(category_record, output_summary_path)

        print(
            f"Subject '{subject}' evaluation complete.\n"
            f"  -> Results saved to:  {output_res_path}\n"
            f"  -> Summary saved to:  {output_summary_path}"
        )


def evaluate_uncertainty(subjects, model_name, base_output_dir, args, num_runs=10):
    """
    Uncertainty quantification evaluation.

    For each subject:
      - Builds a single Batch input JSONL with num_runs * num_questions requests
        (each question repeated num_runs times with age jitter).
      - Runs one Batch job.
      - Splits the results into per-run directories:
           base_output_dir/model_name/run_0/
           base_output_dir/model_name/run_1/
           ...
           base_output_dir/model_name/run_9/
        using the same filenames as the original quizbench/evaluate_from_api.py.
    """
    if model_name not in BATCH_COMPATIBLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not configured for Batch API in "
            "evaluate_from_batch_api.py. Use quizbench/evaluate_from_api.py for this model, "
            "or add it to BATCH_COMPATIBLE_MODELS if it supports Batch."
        )

    is_gemini_model = _is_gemini_model(model_name)

    openai_client = get_openai_client(model_name)
    gemini_file_client = get_gemini_file_client() if is_gemini_model else None

    test_df, dev_df = load_mmlu_pro()

    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    effort_str = f"_{args.reasoning_effort}" if args.reasoning_effort else ""
    model_tag = model_name + effort_str

    model_root = os.path.join(base_output_dir, model_tag)
    os.makedirs(model_root, exist_ok=True)

    # Pre-create run directories to mirror original layout
    run_dirs = {}
    for n in range(num_runs):
        run_dir = os.path.join(model_root, f"run_{n}")
        os.makedirs(run_dir, exist_ok=True)
        run_dirs[n] = run_dir

    for subject in subjects:
        test_data = test_df[subject]

        if args.num_tokens is not None:
            num_tokens_str = '_' + str(args.num_tokens)
        else:
            num_tokens_str = ''

        batch_input_path = os.path.join(
            model_root,
            f"{model_tag}{num_tokens_str}_{subject}_uq_batch_input.jsonl",
        )

        print(f"\n=== Subject: {subject} (uncertainty quantification) ===")
        print(f"Questions per run: {len(test_data)}")
        print(f"Number of runs:    {num_runs}")
        print(f"Total requests:    {len(test_data) * num_runs}")
        print(f"Batch input:       {batch_input_path}")

        # 1. Build the batch input JSONL with 10x questions
        custom_id_to_meta = build_batch_input_file_uq(
            subject=subject,
            test_questions=test_data,
            cot_examples_dict=dev_df,
            model_name=model_name,
            batch_input_path=batch_input_path,
            num_runs=num_runs,
            reasoning_effort=args.reasoning_effort,
        )

        if not custom_id_to_meta:
            print(f"No questions to evaluate for subject {subject}. Skipping.")
            continue

        # 2. Submit the Batch job
        batch = submit_batch_job(
            openai_client,
            batch_input_path,
            is_gemini_model=is_gemini_model,
            gemini_file_client=gemini_file_client,
        )

        # 3. Wait until the batch completes
        batch = wait_for_batch_completion(openai_client, batch)

        # 4. Collect results
        batch_results = collect_batch_results(
            openai_client,
            batch,
            is_gemini_model=is_gemini_model,
            gemini_file_client=gemini_file_client,
        )

        # 5. Build per-run results
        per_run_results = build_results_for_subject_uq(
            subject, custom_id_to_meta, batch_results, num_runs=num_runs
        )

        # 6. Save per-run outputs & summaries to run_0 ... run_9
        for run_idx in range(num_runs):
            run_dir = run_dirs[run_idx]
            output_res_path = os.path.join(
                run_dir,
                model_tag + num_tokens_str + '_' + subject + "_result.json",
            )
            output_summary_path = os.path.join(
                run_dir,
                model_tag + num_tokens_str + '_' + subject + "_summary.json",
            )

            results_this_run = per_run_results[run_idx]
            save_res(results_this_run, output_res_path)
            category_record = build_category_record(results_this_run)
            save_summary(category_record, output_summary_path)

            print(
                f"Subject '{subject}', run {run_idx} complete.\n"
                f"  -> Results:  {output_res_path}\n"
                f"  -> Summary:  {output_summary_path}"
            )


def evaluate(subjects, model_name, output_dir, args):
    """
    Backwards-compatible wrapper for non-UQ (standard) evaluation.
    """
    return evaluate_standard(subjects, model_name, output_dir, args)


def resume_uncertainty_from_completed_gemini_batch(
    batch_id,
    subjects,
    model_name,
    base_output_dir,
    args,
    num_runs=None,
):
    """
    Resume the uncertainty-quantification evaluation for a *completed* Gemini
    Batch API job.

    This is intended for Gemini models invoked via the OpenAI compatibility
    endpoint, which still uses `/v1/chat/completions` for Batch.

    Args:
        batch_id: The Gemini Batch job id, e.g. 'batches/xyz...'.
        subjects: Optional list of subject names to restrict processing to.
                  If empty, all subjects present in the batch will be used.
        model_name: The model that produced the batch (must be a Gemini model).
        base_output_dir: Root directory where eval results should be written.
        args: argparse.Namespace with at least `reasoning_effort` and `num_tokens`.
        num_runs: Number of UQ runs per question. If None, it will be inferred
                  from the custom_id pattern in the batch output.
    """
    from pprint import pprint  # optional, just for nicer error printing

    if model_name not in BATCH_COMPATIBLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not configured for Batch API in "
            "evaluate_from_batch_api.py. Use quizbench/evaluate_from_api.py for this model, "
            "or add it to BATCH_COMPATIBLE_MODELS if it supports Batch."
        )

    if not _is_gemini_model(model_name):
        raise ValueError(
            "resume_uncertainty_from_completed_gemini_batch is intended only for "
            "Gemini models invoked via the OpenAI compatibility endpoint."
        )

    # 1. Clients
    openai_client = get_openai_client(model_name)
    gemini_file_client = get_gemini_file_client()

    # 2. Reload datasets so we can recover the structured questions
    test_df, _ = load_mmlu_pro()

    # 3. Fetch the completed batch
    print(f"Retrieving Gemini Batch job: {batch_id}")
    batch = openai_client.batches.retrieve(batch_id)
    print(f"Batch {batch_id} status: {batch.status}")

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch_id} is not completed. Current status: {batch.status}"
        )

    if getattr(batch, "errors", None):
        print("Batch reported errors:")
        pprint(batch.errors)

    # 4. Download & parse batch outputs (and any error lines)
    batch_results = collect_batch_results(
        openai_client,
        batch,
        is_gemini_model=True,
        gemini_file_client=gemini_file_client,
    )

    if not batch_results:
        raise RuntimeError(
            f"Batch {batch_id} has no output or error lines. "
            "Nothing to resume."
        )

    # 5. Inspect custom_ids to infer subjects and runs
    def _parse_custom_id(custom_id):
        """
        Expected format from build_batch_input_file_uq:
            '<subject>__run<run_idx>__<question_id>'
        """
        m = re.match(r"(.+)__run(\d+)__(.+)", custom_id)
        if not m:
            raise ValueError(
                f"Unexpected custom_id format: {custom_id!r}. "
                "Expected '<subject>__run<run_idx>__<question_id>'."
            )
        subject_name, run_str, q_id = m.group(1), m.group(2), m.group(3)
        return subject_name, int(run_str), q_id

    subjects_in_batch = set()
    max_run_idx = -1

    # Build a global index: subject -> {question_id -> question_dict}
    question_index = {
        subj: {q["question_id"]: q for q in questions}
        for subj, questions in test_df.items()
    }

    # First pass: discover which subjects & run indices are present
    for custom_id in batch_results.keys():
        try:
            subj, run_idx, q_id = _parse_custom_id(custom_id)
        except ValueError as e:
            print(e)
            continue
        subjects_in_batch.add(subj)
        if run_idx > max_run_idx:
            max_run_idx = run_idx

    if not subjects_in_batch:
        raise RuntimeError(
            "Could not infer any subjects from batch custom_ids. "
            "Check that the batch was created with build_batch_input_file_uq."
        )

    # If the caller passed an explicit subject filter, apply it.
    if subjects:
        subjects_to_process = [s for s in subjects if s in subjects_in_batch]
        missing = [s for s in subjects if s not in subjects_in_batch]
        if missing:
            print(
                "Warning: the following requested subjects were not found in the "
                f"batch custom_ids and will be ignored: {missing}"
            )
    else:
        subjects_to_process = sorted(subjects_in_batch)

    if not subjects_to_process:
        raise RuntimeError(
            "No overlapping subjects between requested subjects and batch contents."
        )

    # Infer num_runs if needed
    if num_runs is None:
        if max_run_idx < 0:
            raise RuntimeError("Could not infer num_runs from batch custom_ids.")
        num_runs = max_run_idx + 1

    print(f"Subjects in batch: {sorted(subjects_in_batch)}")
    print(f"Subjects to process: {subjects_to_process}")
    print(f"Inferred num_runs: {num_runs}")

    # 6. Reconstruct custom_id_to_meta per subject
    subject_to_custom_meta = {subj: {} for subj in subjects_to_process}

    for custom_id in batch_results.keys():
        try:
            subj, run_idx, q_id = _parse_custom_id(custom_id)
        except ValueError:
            continue

        if subj not in subjects_to_process:
            continue

        q_by_id = question_index.get(subj)
        if q_by_id is None:
            print(f"Warning: subject {subj} not found in test_df. Skipping.")
            continue

        question = q_by_id.get(q_id)
        if question is None:
            print(
                f"Warning: question_id {q_id!r} for subject {subj!r} "
                "not found in loaded test set. Skipping."
            )
            continue

        subject_to_custom_meta[subj][custom_id] = {
            "run": run_idx,
            "question": question,
        }

    # 7. Prepare output directory layout (same as evaluate_uncertainty)
    effort_str = f"_{args.reasoning_effort}" if args.reasoning_effort else ""
    model_tag = model_name + effort_str

    model_root = os.path.join(base_output_dir, model_tag)
    os.makedirs(model_root, exist_ok=True)

    run_dirs = {}
    for n in range(num_runs):
        run_dir = os.path.join(model_root, f"run_{n}")
        os.makedirs(run_dir, exist_ok=True)
        run_dirs[n] = run_dir

    if args.num_tokens is not None:
        num_tokens_str = '_' + str(args.num_tokens)
    else:
        num_tokens_str = ''

    # 8. For each subject, rebuild per-run results & write files
    for subject in subjects_to_process:
        custom_id_to_meta = subject_to_custom_meta[subject]
        if not custom_id_to_meta:
            print(f"No batch lines for subject {subject}. Skipping.")
            continue

        print(f"\n=== Resuming subject: {subject} (uncertainty quantification) ===")
        print(f"Total lines for subject: {len(custom_id_to_meta)}")

        per_run_results = build_results_for_subject_uq(
            subject,
            custom_id_to_meta,
            batch_results,
            num_runs=num_runs,
        )

        for run_idx in range(num_runs):
            run_dir = run_dirs[run_idx]
            output_res_path = os.path.join(
                run_dir,
                model_tag + num_tokens_str + '_' + subject + "_result.json",
            )
            output_summary_path = os.path.join(
                run_dir,
                model_tag + num_tokens_str + '_' + subject + "_summary.json",
            )

            results_this_run = per_run_results[run_idx]
            save_res(results_this_run, output_res_path)
            category_record = build_category_record(results_this_run)
            save_summary(category_record, output_summary_path)

            print(
                f"Subject '{subject}', run {run_idx} resumed.\n"
                f"  -> Results:  {output_res_path}\n"
                f"  -> Summary:  {output_summary_path}"
            )

    print(f"\nResume from batch {batch_id} complete.")


# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="gpt-4",
        choices=[
            # Batch-compatible OpenAI & Gemini models
            "gpt-5-nano", "gpt-5", "gpt-5.1-2025-11-13",
            "gpt-5-pro-2025-10-06",
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",

            # V1 nonreasoning models (many of these are *not* batch compatible;
            # using them here will raise unless you add them to BATCH_COMPATIBLE_MODELS)
            "gpt-4", "gpt-4o", "gpt-4o-mini",
            # The rest are kept for compatibility with quizbench/evaluate_from_api.py,
            # but this script will raise if you select a non-Batch model.
            "o1-preview", "o1-mini", "o1-2024-12-17",
            "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct",
            "Llama-3.3-8B-Instruct", "Llama-3.3-70B-Instruct", "llama3.1-405b-instruct-fp8",
            "Mistral-7B-Instruct-v0.3",
            "medalpaca-13b",
            "Meditron3-8B", "Meditron3-70B",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "claude-3-opus-20240229",
            "gemini-1.5-flash-8b",
            "claude-3-sonnet-20240229",
        ],
    )
    parser.add_argument("--reasoning_effort", "-r", type=str, default="high")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--uncertainty_quantification", "-uq", action="store_true", default=True)
    parser.add_argument("--num_tokens", "-nt", type=int, default=None)
    parser.add_argument(
        "--resume_gemini_batch_id",
        type=str,
        default=None,
        help=(
            "If set (and --uncertainty_quantification is also passed), skip creating "
            "a new Gemini Batch job and instead download results from the given "
            "completed Gemini batch id (e.g. 'batches/n0ax7qfaj9bdjtm3gjwxpct31tda5u17oxmj')."
        ),
    )

    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")

    if args.uncertainty_quantification:
        if args.resume_gemini_batch_id:
            # Resume from an already-completed Gemini Batch job.
            resume_uncertainty_from_completed_gemini_batch(
                batch_id=args.resume_gemini_batch_id,
                subjects=assigned_subjects,
                model_name=args.model_name,
                base_output_dir=args.output_dir,
                args=args,
                # num_runs=None lets the function infer this from custom_ids.
            )
        else:
            # Single Batch per subject containing 10x questions,
            # then results are split into run_0 ... run_9 directories.
            evaluate_uncertainty(assigned_subjects, args.model_name, args.output_dir, args)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        evaluate(assigned_subjects, args.model_name, args.output_dir, args)
