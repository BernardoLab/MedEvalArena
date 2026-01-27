import json
import logging
import os
import requests

from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# OpenAI
from openai import OpenAI as _OpenAI
# Anthropic
import anthropic
from anthropic import transform_schema

# Google
from quizbench.utils_grok import call_grok_sync, is_grok_model
from pydantic import BaseModel, ValidationError, Field
from typing import Literal
import sys

logger = logging.getLogger("quizbench.clients")


# --- Helpers ---
class JudgeExtraction(BaseModel):
    analysis: str = Field(
        description="Detailed analysis of the medical reasoning and logical validity."
    )
    medical_accuracy_score: Literal["1", "2", "3", "4", "5"] = Field(
        description="Integer (1–5) indicating medical accuracy."
    )
    logical_validity: bool = Field(
        description="True if the reasoning is logically sound."
    )
    logical_false_reason: Literal["C", "N", "M", "U", "K", "T"] = Field(
        description=(
            "One-letter code for logical status: "
            "C=Contradiction, N=No answer defensible, "
            "M=Multiple answers defensible, U=Underspecified, "
            "K=Miskeyed, T=Used only when logical_validity is true."
        )
    )

from typing import Optional
import json


def _extract_text_from_responses_output_grok(resp) -> Optional[str]:
    """
    Best-effort extractor for OpenAI Responses API objects and ChatCompletion objects.
    Walks output/choices and returns the first text payload found.
    """
    # 1. Native Responses API: resp.output
    output = getattr(resp, "output", None)

    # 2. Fallback: dict-style "output"
    if output is None and isinstance(resp, dict):
        output = resp.get("output")

    # 3. Chat Completions API: resp.choices[*].message.content
    #    We normalize choices into the same "output" shape this helper expects:
    #    a list of items each with a "content" field.
    if output is None and hasattr(resp, "choices"):
        output = []
        for choice in getattr(resp, "choices") or []:
            # choice can be an object or dict
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is None:
                continue

            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

            if content is not None:
                output.append({"content": content})

    # 4. Still nothing? Bail out.
    if not output:
        return None

    # 5. Existing extraction logic
    for item in output:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if not content:
            continue

        # Simple string content
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned
            continue

        # List of structured content parts (Responses API or new ChatCompletion v1)
        if isinstance(content, list):
            for part in content:
                # Prefer explicit text fields
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    return text

                # Fall back to parsed objects if present
                parsed = part.get("parsed") if isinstance(part, dict) else getattr(part, "parsed", None)
                if parsed is not None:
                    if hasattr(parsed, "model_dump_json"):
                        return parsed.model_dump_json()
                    if hasattr(parsed, "dict"):
                        return json.dumps(parsed.dict())
                    try:
                        return json.dumps(parsed)
                    except Exception as exc:  # noqa: BLE001
                        # Hard-fail rather than silently skipping ambiguous parsed content
                        raise RuntimeError(
                            f"Failed to serialize parsed Responses output: {parsed!r}"
                        ) from exc

        # Handle unexpected content containers that still expose text/value
        maybe_text = getattr(content, "text", None) or getattr(content, "value", None)
        if maybe_text:
            return str(maybe_text)

    return None


def _extract_text_from_openrouter_kimi(resp_json: dict) -> Optional[str]:
    """
    Extract text content from an OpenRouter /chat/completions JSON response
    for Kimi (moonshotai) models, where the payload is a plain dict with
    choices[*].message.content.
    """
    if not isinstance(resp_json, dict):
        return None

    choices = resp_json.get("choices") or []
    if not choices:
        return None

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not content:
            continue

        # Kimi via OpenRouter commonly returns a single JSON string here.
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned
            continue

        # Fallback: list of structured parts, e.g. [{"type": "text", "text": "..."}]
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        inner_content = part.get("content")
                        if isinstance(inner_content, str):
                            parts.append(inner_content)
                        else:
                            parts.append(str(part))
                elif isinstance(part, str):
                    parts.append(part)
            joined = "".join(parts).strip()
            if joined:
                return joined

    return None



def _extract_text_from_responses_output(resp) -> Optional[str]:
    """
    Best-effort extractor for OpenAI Responses API objects.
    Walks output items and returns the first text payload found.
    """
    output = getattr(resp, "output", None)
    if not output:
        return None

    for item in output:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if not content:
            continue

        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                return cleaned
            continue

        if isinstance(content, list):
            for part in content:
                # Prefer explicit text fields
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    return text

                # Fall back to parsed objects if present
                parsed = part.get("parsed") if isinstance(part, dict) else getattr(part, "parsed", None)
                if parsed is not None:
                    if hasattr(parsed, "model_dump_json"):
                        return parsed.model_dump_json()
                    if hasattr(parsed, "dict"):
                        return json.dumps(parsed.dict())
                    try:
                        return json.dumps(parsed)
                    except Exception as exc:  # noqa: BLE001
                        # Hard-fail rather than silently skipping ambiguous parsed content
                        raise RuntimeError(
                            f"Failed to serialize parsed Responses output: {parsed!r}"
                        ) from exc

        # Handle unexpected content containers that still expose text/value
        maybe_text = getattr(content, "text", None) or getattr(content, "value", None)
        if maybe_text:
            return str(maybe_text)

    return None

def _raise_if_missing(var: str) -> None:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")


def _build_openrouter_input_from_prompt(prompt: str) -> list[dict]:
    """
    Build a minimal OpenRouter `input` payload for a single
    text-only user prompt, matching the Responses API schema.
    """
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }
    ]


# --- LLM call dispatcher ---
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_llm(
    model_name: str,
    prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    openrouter: bool = False,
    judge_mode: bool = False,
):
    """
    Unified text-only call.

    Returns a tuple (text, reasoning_trace) where:
      - text is the primary string content used by callers.
      - reasoning_trace is a provider-specific raw payload when available
        (DeepSeek/Kimi via OpenRouter), otherwise None.
    """
    mn = model_name.strip()

    if is_grok_model(mn):
        logger.info(
            "Calling Grok model '%s' (max_tokens=%d, temperature=%.3f).",
            mn,
            max_tokens,
            temperature,
        )

        if judge_mode:
            # Using async single call (not pseudo-batched)
            client = _OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
            resp = client.chat.completions.create(
                model=mn,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _extract_text_from_responses_output_grok(resp)
            if text:
                return text, None
            raise RuntimeError(f"Responses API returned no usable text content: {resp!r}")

        text = call_grok_sync(
            model_name=mn,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text, None

    # OpenAI gpt-5 family (Responses API)
    if mn.startswith("gpt-5"):
        _raise_if_missing("OPENAI_API_KEY")
        logger.info(
            "Calling OpenAI Responses API for model '%s' (max_tokens=%d, temperature=1.000).",
            mn,
            max_tokens,
        )
        client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if judge_mode:
            resp = client.responses.parse(
                model=mn,
                input=[{"role": "user", "content": prompt}],
                temperature=1,  # Responses API requires temperature=1 for gpt-5 family
                max_output_tokens=max_tokens,
                top_p=1,
                text_format=JudgeExtraction,
            )

            text = _extract_text_from_responses_output(resp)
            if text:
                return text, None

            # As a last resort, stringify the whole payload so the caller can attempt to parse
            return str(resp), None
        
        # Eval mode
        resp = client.responses.create(**req)
        if resp.output_text:
            return resp.output_text, None

        # If we get here, treat it as a hard failure
        raise RuntimeError(f"Responses API returned no usable text content: {resp!r}")

    # OpenAI chat models (gpt-4/4o/etc.)
    if mn.startswith("gpt-"):
        _raise_if_missing("OPENAI_API_KEY")
        logger.info(
            "Calling OpenAI model '%s' (max_tokens=%d, temperature=%.3f).",
            mn,
            max_tokens,
            temperature,
        )
        client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if model_name == "gpt-4o":
            resp = client.chat.completions.create(
                model=mn,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
            )
        else:
            resp = client.chat.completions.create(
                model=mn,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,  # gpt-5 family only supports temperature 1
                max_completion_tokens=max_tokens,
                top_p=1,
            )
        print(repr(resp))
        return resp.choices[0].message.content, None

    # Anthropic
    if mn.startswith("claude-3") or mn.startswith("claude-opus"):
        _raise_if_missing("ANTHROPIC_API_KEY")
        logger.info(
            "Calling Anthropic model '%s' (max_tokens=%d, temperature=%.3f).",
            mn,
            max_tokens,
            temperature,
        )
        a = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        if judge_mode:
            from pydantic import BaseModel  # type: ignore[redefined-outer-name]
            from anthropic import transform_schema  # type: ignore[unused-import]

            response = a.beta.messages.create(
                model="claude-sonnet-4-5", #mn,
                betas=["structured-outputs-2025-11-13"],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                output_format={
                    "type": "json_schema",
                    "schema": transform_schema(JudgeExtraction),
                }
            )

            # parsed = JudgeExtraction.model_validate_json(response.text)
            return response.content[0].text, None

        msg = a.messages.create(
            model=mn,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text, None

    # Google Gemini
    if mn.startswith("gemini-"):
        from google import genai  
        from google.genai import types

        _raise_if_missing("GOOGLE_API_KEY")
        logger.info(
            "Calling Gemini model '%s' (max_tokens=%d, temperature=%.3f).",
            mn,
            max_tokens,
            temperature,
        )

        if judge_mode:
            try:
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # type: ignore[name-defined]
                response = client.models.generate_content(
                    model=mn,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "top_p": 1,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                        "response_schema": JudgeExtraction.model_json_schema(),
                    },
                )

                parsed = JudgeExtraction.model_validate_json(response.text)
                return parsed.model_dump_json(), None

            except Exception as e:  # noqa: BLE001
                logger.error(f"Gemini API call failed: {e}")
                try:
                    if hasattr(response, "prompt_feedback"):
                        logger.error(f"Feedback: {response.prompt_feedback}")
                    if hasattr(response, "candidates"):
                        logger.error(f"Finish Reason: {response.candidates[0].finish_reason}")
                except Exception:  # noqa: BLE001
                    pass
                raise

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=mn,
            contents=prompt,
            config={
                "temperature": temperature,
                "top_p": 1,
                "max_output_tokens": max_tokens,
            },
            safety_settings=None,
        )
        return response.text, None

    # DeepSeek or Kimi (optionally via OpenRouter)
    mn_lower = mn.lower()
    if mn_lower.startswith("deepseek-") or mn_lower.startswith("kimi-"):
        print(f"  [INFO]: client.py detected {mn_lower}")

        mn_fam = mn_lower.split("-")[0]

        if mn_fam == "kimi":
            mn_fam = "moonshotai"


        if openrouter:
            print("  [INFO]: Using Openrouter")
            _raise_if_missing("OPENROUTER_API_KEY")
            logger.info(
                "Calling %s model via OpenRouter '%s' (max_tokens=%d, temperature=%.3f).",
                mn_fam,
                mn,
                max_tokens,
                temperature,
            )

            api_key = os.getenv("OPENROUTER_API_KEY")

            # Judge mode: request structured JSON with potential reasoning traces.
            if judge_mode:
                print("[Info] Using Openrouter with Judge mode")

                # if mn_lower.startswith("kimi-") or mn_lower.startswith("deepseek-"):
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": f"{mn_fam}/{mn}",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "response_format": {
                            "type": "json_schema", 
                            "json_schema": {
                                "name": "judge_extraction",
                                "schema": JudgeExtraction.schema(),
                                "strict": True,
                            }
                        },
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 1,
                    },
                )

                resp_json = response.json()

                print("[INFO] Debug trace:")
                print(resp_json)

                text = _extract_text_from_openrouter_kimi(resp_json)


                if text:
                    print(text)
                    return text, resp_json

                print("[JUDGE] Error extracting response.")
                print("[INFO] Debug trace:")
                print(json.dumps(resp_json, indent=4))
                raise RuntimeError("Failed to extract text from OpenRouter DeepSeek/Kimi response.")

            print(f"[INFO] Using model {mn} via OpenRouter")
            # Non-judge mode over OpenRouter: plain chat completion.
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": f"{mn_fam}/{mn}",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 1,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter {mn_lower} call failed "
                    f"(status={response.status_code}): {response.text[:1000]}"
                )

            result = response.json()

            try:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        else:
                            parts.append(str(part))
                    content = "".join(parts)
                if not isinstance(content, str):
                    content = str(content)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Unexpected OpenRouter response format: {result}"
                ) from exc

            return content, result

        # DeepSeek via native API
        if mn_lower.startswith("deepseek-"):




            if openrouter:
                print(f"[INFO] Using model {mn} via OpenRouter")
                # Non-judge mode over OpenRouter: plain chat completion.
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": f"{mn_fam}/{mn}",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 1,
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter {mn_lower} call failed "
                        f"(status={response.status_code}): {response.text[:1000]}"
                    )

                result = response.json()

                try:
                    choice = result["choices"][0]
                    message = choice.get("message") or {}
                    content = message.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            else:
                                parts.append(str(part))
                        content = "".join(parts)
                    if not isinstance(content, str):
                        content = str(content)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Unexpected OpenRouter response format: {result}"
                    ) from exc

                return content, result


            # Legacy Deepseek API
            _raise_if_missing("DEEPSEEK_API_KEY")
            logger.info(
                "Calling DeepSeek model '%s' (max_tokens=%d, temperature=%.3f).",
                mn,
                max_tokens,
                temperature,
            )
            deepseek_model_name = "deepseek-reasoner"

            api_key = os.getenv("DEEPSEEK_API_KEY")

            client = _OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=10)

            resp = client.chat.completions.create(
                model=deepseek_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
            )

            print(resp)
            return resp.choices[0].message.content, None

        # Placeholder other models
        raise NotImplementedError(
            f"Model '{mn}' is not wired correctly. Add provider support in quizbench/clients.py."
        )   

    # Placeholder for other providers
    raise NotImplementedError(
        f"Model '{mn}' is not wired. Add provider support in quizbench/clients.py."
    )
