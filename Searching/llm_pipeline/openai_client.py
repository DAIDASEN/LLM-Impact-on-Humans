from __future__ import annotations

import os
import random
import time
from typing import Any, Optional

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError


def get_client() -> OpenAI:
    # OpenAI SDK reads OPENAI_API_KEY automatically; we keep it explicit in docs only.
    # Provide a conservative default timeout; can be overridden by env.
    timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "120"))
    # Disable SDK auto-retries; we do our own to avoid surprises.
    try:
        return OpenAI(timeout=timeout_s, max_retries=0)
    except TypeError:
        # Older SDK signature fallback
        return OpenAI(timeout=timeout_s)


def get_model_name() -> str:
    # Let user override; default to a GPT-5 family model name.
    return os.getenv("OPENAI_MODEL", "gpt-5")


def call_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    response_format: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
    reasoning_effort: str = "minimal",
    max_retries: int = 6,
) -> str:
    """
    Uses Responses API.
    We set reasoning effort to minimal by default (per your requirement).
    """
    client = get_client()
    model = model or get_model_name()

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    # Note: Some models may ignore 'reasoning' if unsupported; safe to pass.
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            return resp.output_text
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            # Exponential backoff with jitter
            base = min(60.0, 2.0 ** attempt)
            sleep_s = base + random.uniform(0.0, 1.0)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err



