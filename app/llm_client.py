"""
Centralized LLM client with Groq fallback.

Primary:  Google Gemini 2.5 Flash (via google-genai SDK)
Fallback: Groq llama-3.3-70b-versatile (via httpx, zero extra deps)

Usage:
    from app.llm_client import generate_with_fallback, get_gemini_client

    # For LLM generation (auto-falls back to Groq on Gemini failure):
    response_text = generate_with_fallback(prompt, system_instruction=..., response_schema=...)

    # For embedding (no fallback needed, Gemini-only):
    client = get_gemini_client()
    client.models.embed_content(...)
"""

import json
import logging
import os
from pathlib import Path

import httpx
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client (lazy singleton)
# ---------------------------------------------------------------------------
_gemini_client = None


def get_gemini_client() -> genai.Client:
    """Return the shared google-genai Client, creating it on first call."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[1] / ".env")
                api_key = os.environ.get("GOOGLE_API_KEY")
            except ImportError:
                pass
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Groq fallback (uses httpx — already in requirements.txt)
# ---------------------------------------------------------------------------
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_MODEL2 = "qwen/qwen3-32b"


def _call_groq(
    prompt: str,
    system_instruction: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """
    Call Groq's OpenAI-compatible API and return the raw assistant
    content string.  If a response_schema is supplied we ask the model
    to output JSON matching that schema via the system prompt.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set — "
            "cannot fall back to Groq."
        )

    messages: list[dict] = []

    # Build system message
    sys_parts: list[str] = []
    if system_instruction:
        sys_parts.append(system_instruction)
    if response_schema:
        sys_parts.append(
            "You MUST respond with valid JSON only, matching this schema:\n"
            + json.dumps(response_schema, indent=2)
        )
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": _GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    if response_schema:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=60.0) as http_client:
        resp = http_client.post(
            _GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _call_groq2(
    prompt: str,
    system_instruction: str | None = None,
    response_schema: dict | None = None,
) -> str:
    """
    Call Groq's OpenAI-compatible API and return the raw assistant
    content string.  If a response_schema is supplied we ask the model
    to output JSON matching that schema via the system prompt.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set — "
            "cannot fall back to Groq."
        )

    messages: list[dict] = []

    # Build system message
    sys_parts: list[str] = []
    if system_instruction:
        sys_parts.append(system_instruction)
    if response_schema:
        sys_parts.append(
            "You MUST respond with valid JSON only, matching this schema:\n"
            + json.dumps(response_schema, indent=2)
        )
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": _GROQ_MODEL2,
        "messages": messages,
        "temperature": 0.7,
    }
    if response_schema:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=60.0) as http_client:
        resp = http_client.post(
            _GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]



# ---------------------------------------------------------------------------
# Public API: generate_with_fallback
# ---------------------------------------------------------------------------

def generate_with_fallback(
    prompt: str,
    *,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    gemini_model: str = "gemini-2.5-flash",
) -> str:
    """
    Try Gemini first; on failure fall back to Groq model 1, then Groq model 2.

    Returns the raw text content of the LLM response (typically JSON when
    a response_schema is provided).
    """
    gemini_err = None
    groq1_err = None

    # --- Attempt 1: Gemini ---
    try:
        client = get_gemini_client()
        config_kwargs: dict = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_schema:
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=config,
        )
        logger.info("Gemini (%s) responded successfully.", gemini_model)
        return response.text

    except Exception as e:
        gemini_err = e
        logger.warning(
            "Gemini (%s) failed: %s — falling back to Groq (%s).",
            gemini_model, e, _GROQ_MODEL,
        )

    # --- Attempt 2: Groq model 1 ---
    try:
        result = _call_groq(
            prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
        )
        logger.info("Groq (%s) responded successfully (fallback 1).", _GROQ_MODEL)
        return result

    except Exception as e:
        groq1_err = e
        logger.warning(
            "Groq (%s) also failed: %s — falling back to Groq (%s).",
            _GROQ_MODEL, e, _GROQ_MODEL2,
        )

    # --- Attempt 3: Groq model 2 ---
    try:
        result = _call_groq2(
            prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
        )
        logger.info("Groq (%s) responded successfully (fallback 2).", _GROQ_MODEL2)
        return result

    except Exception as groq2_err:
        logger.error(
            "All LLMs failed. Gemini: %s | Groq1: %s | Groq2: %s",
            gemini_err, groq1_err, groq2_err,
        )
        raise gemini_err from groq2_err
