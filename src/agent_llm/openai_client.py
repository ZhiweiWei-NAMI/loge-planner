"""Minimal OpenAI-compatible chat client for the decomposition call.

Only the ``openai`` SDK is used (already a core dependency).  The client is
deliberately strict: transport errors, auth errors, and empty completions raise
:class:`LLMRequestError`.  There is no canned/offline answer path -- a failed
call can never be mistaken for a decomposition result.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class LLMRequestError(RuntimeError):
    """The remote chat completion could not be obtained."""


@dataclass
class LLMCallConfig:
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout_s: float = 120.0
    max_retries: int = 3
    json_mode: str = "auto"  # "auto" | "on" | "off"
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallResult:
    text: str
    model: str
    base_url: str
    latency_s: float
    attempts: int
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    json_mode_used: bool = False
    json_mode_fallback: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "latency_s": round(self.latency_s, 4),
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "json_mode_used": self.json_mode_used,
            "json_mode_fallback": self.json_mode_fallback,
            "response_chars": len(self.text),
        }


def resolve_api_key(cfg: LLMCallConfig) -> str:
    """Resolve the API key from the config or the configured environment variable."""
    key = (cfg.api_key or "").strip()
    if key:
        return key
    key = (os.environ.get(cfg.api_key_env) or "").strip()
    if key:
        return key
    raise LLMRequestError(
        f"no API key: set the {cfg.api_key_env} environment variable "
        f"(or pass --api-key-env/--api-key). "
        f"Never commit keys; .env is git-ignored."
    )


_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    """Chat-completions client with bounded retries and no silent fallbacks."""

    def __init__(self, cfg: LLMCallConfig):
        self.cfg = cfg
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMRequestError(
                "the 'openai' package is required for cloud decomposition "
                "(pip install 'openai>=1.72')"
            ) from exc
        self._openai = openai
        self.api_key = resolve_api_key(cfg)
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            max_retries=0,  # retries are handled here so attempts are auditable
        )

    def _build_kwargs(self, messages: List[Dict[str, str]], json_mode: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.cfg.extra_body:
            kwargs.update(self.cfg.extra_body)
        return kwargs

    def chat(self, messages: List[Dict[str, str]]) -> LLMCallResult:
        mode = self.cfg.json_mode
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"json_mode must be auto|on|off, got {mode!r}")
        json_mode = mode in ("auto", "on")
        json_fallback = False

        last_error: Optional[Exception] = None
        started = time.perf_counter()
        attempts = 0
        while attempts < max(1, self.cfg.max_retries + 1):
            attempts += 1
            try:
                response = self._client.chat.completions.create(
                    **self._build_kwargs(messages, json_mode)
                )
                text = _extract_text(response)
                usage = {}
                if getattr(response, "usage", None) is not None:
                    try:
                        usage = response.usage.model_dump()
                    except AttributeError:  # pragma: no cover - older SDKs
                        usage = dict(response.usage)
                choice = response.choices[0] if response.choices else None
                return LLMCallResult(
                    text=text,
                    model=getattr(response, "model", self.cfg.model),
                    base_url=self.cfg.base_url,
                    latency_s=time.perf_counter() - started,
                    attempts=attempts,
                    finish_reason=getattr(choice, "finish_reason", None),
                    usage=usage,
                    json_mode_used=json_mode,
                    json_mode_fallback=json_fallback,
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = exc
                status = _status_code(exc)
                # Endpoint rejects response_format: retry once without it.
                if json_mode and status == 400 and mode == "auto":
                    json_mode = False
                    json_fallback = True
                    attempts -= 1  # this retry does not consume the budget
                    continue
                if not _is_retryable(exc, status) or attempts > self.cfg.max_retries:
                    break
                time.sleep(min(2.0 ** (attempts - 1), 30.0))

        raise LLMRequestError(
            f"chat completion failed after {attempts} attempt(s) "
            f"(model={self.cfg.model}, base_url={self.cfg.base_url}): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


def _extract_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise LLMRequestError("response contains no choices")
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError("response message has empty content")
    return content


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _is_retryable(exc: Exception, status: Optional[int]) -> bool:
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"):
        return True
    if status is not None:
        return status in _RETRYABLE_STATUS
    # Unknown transport-level failure: retry, since it carries no result.
    return name in ("TimeoutError", "ConnectionError")
