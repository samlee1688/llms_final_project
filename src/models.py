"""
Unified LLM client supporting OpenAI and Anthropic backends.

Usage:
    client = LLMClient(provider="openai", model="gpt-4o")
    response = client.generate(system_prompt, user_prompt)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
from dataclasses import dataclass, field
import threading
import time
from typing import Optional

import openai
import anthropic

import config

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


def _openai_prefers_max_completion_tokens(model_name: str) -> bool:
    """Return True for OpenAI model families that expect max_completion_tokens."""
    model = (model_name or "").strip().lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_should_swap_token_param(exc: Exception, used_param: str) -> bool:
    """Detect the common OpenAI 400 for wrong token parameter name."""
    text = str(exc).lower()
    if "unsupported parameter" not in text:
        return False
    if used_param == "max_tokens":
        return "max_completion_tokens" in text
    if used_param == "max_completion_tokens":
        return "max_tokens" in text
    return False


class _SQLiteLLMCache:
    """Thread-safe SQLite-backed prompt/response cache."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[LLMResponse]:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT model, response_text, usage_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        model, response_text, usage_json = row
        usage = {}
        try:
            usage = json.loads(usage_json)
        except Exception:
            usage = {}
        usage["cache_hit"] = True
        return LLMResponse(text=response_text, model=model, usage=usage)

    def set(
        self,
        cache_key: str,
        provider: str,
        model: str,
        response_text: str,
        usage: dict,
    ) -> None:
        payload = json.dumps(usage or {}, sort_keys=True)
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                    (cache_key, provider, model, response_text, usage_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, provider, model, response_text, payload, time.time()),
            )
            conn.commit()


class _ProviderLimiter:
    """Simple cross-thread provider limiter with dynamic cooldown."""

    def __init__(self, min_interval_seconds: float = 0.0):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._cooldown_until = 0.0

    def before_call(self) -> None:
        while True:
            wait_for = 0.0
            with self._lock:
                now = time.time()
                gap_wait = max(0.0, (self._last_call + self.min_interval_seconds) - now)
                cooldown_wait = max(0.0, self._cooldown_until - now)
                wait_for = max(gap_wait, cooldown_wait)
                if wait_for <= 0.0:
                    self._last_call = now
                    return
            if wait_for > 0.0:
                time.sleep(wait_for)

    def on_rate_limit(self, attempt: int) -> None:
        with self._lock:
            now = time.time()
            penalty = min(30.0, (2.0 ** min(attempt, 4)))
            self._cooldown_until = max(self._cooldown_until, now + penalty)

    def on_success(self) -> None:
        with self._lock:
            self._cooldown_until = 0.0


@dataclass
class LLMResponse:
    """Standardised wrapper around a single LLM completion."""

    text: str
    model: str
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class LLMClient:
    """Thin wrapper that exposes a single ``generate`` method for both
    OpenAI and Anthropic chat-completion APIs."""

    SUPPORTED_PROVIDERS = ("openai", "anthropic")
    _limiter_lock = threading.Lock()
    _provider_limiters: dict[str, _ProviderLimiter] = {}

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_cache: Optional[bool] = None,
        cache_path: Optional[str] = None,
        max_retries: Optional[int] = None,
        min_interval_seconds: Optional[float] = None,
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from {self.SUPPORTED_PROVIDERS}"
            )

        self.provider = provider
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "6"))
        self.backoff_base_seconds = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.5"))

        resolved_cache = enable_cache
        if resolved_cache is None:
            resolved_cache = _bool_env("LLM_CACHE_ENABLED", True)
        self.enable_cache = bool(resolved_cache)
        self.cache_path = cache_path or os.getenv("LLM_CACHE_PATH", ".cache/llm_cache.sqlite")
        self.cache = _SQLiteLLMCache(self.cache_path) if self.enable_cache else None

        if min_interval_seconds is None:
            min_interval_seconds = float(os.getenv("LLM_MIN_INTERVAL_MS", "0")) / 1000.0
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.provider_prompt_cache = _bool_env("LLM_PROVIDER_PROMPT_CACHE", True)

        with self._limiter_lock:
            limiter = self._provider_limiters.get(provider)
            if limiter is None:
                limiter = _ProviderLimiter(self.min_interval_seconds)
                self._provider_limiters[provider] = limiter
            elif self.min_interval_seconds > limiter.min_interval_seconds:
                limiter.min_interval_seconds = self.min_interval_seconds
            self._limiter = limiter

        if provider == "openai":
            self.model = model or config.OPENAI_MODEL
            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.model = model or config.ANTHROPIC_MODEL
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a single system + user message pair and return the response."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        cache_key = self._cache_key(system_prompt, user_prompt, temp, tokens)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._limiter.before_call()
                if self.provider == "openai":
                    out = self._generate_openai(system_prompt, user_prompt, temp, tokens)
                else:
                    out = self._generate_anthropic(system_prompt, user_prompt, temp, tokens)
                self._limiter.on_success()
                if self.cache is not None:
                    self.cache.set(
                        cache_key=cache_key,
                        provider=self.provider,
                        model=out.model,
                        response_text=out.text,
                        usage=out.usage,
                    )
                return out
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt >= self.max_retries:
                    raise
                self._limiter.on_rate_limit(attempt + 1)
                delay = self.backoff_base_seconds * (2 ** attempt) + random.uniform(0.0, 0.5)
                logger.warning(
                    "Rate limit hit for %s/%s; retrying in %.2fs (attempt %d/%d)",
                    self.provider,
                    self.model,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM generation failed without an exception")

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        token_param = "max_completion_tokens" if _openai_prefers_max_completion_tokens(self.model) else "max_tokens"
        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            token_param: max_tokens,
        }
        try:
            response = self._client.chat.completions.create(**req)
        except openai.BadRequestError as exc:
            if not _openai_should_swap_token_param(exc, token_param):
                raise
            alt_param = "max_tokens" if token_param == "max_completion_tokens" else "max_completion_tokens"
            logger.warning(
                "OpenAI model %s rejected %s; retrying once with %s",
                self.model,
                token_param,
                alt_param,
            )
            req.pop(token_param, None)
            req[alt_param] = max_tokens
            response = self._client.chat.completions.create(**req)
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content.strip(),
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )

    def _generate_anthropic(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        if self.provider_prompt_cache:
            try:
                response = self._client.messages.create(
                    model=self.model,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.debug("Anthropic prompt cache hint unsupported; using normal request (%s)", exc)
                response = self._client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        else:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        text = response.content[0].text.strip()
        return LLMResponse(
            text=text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
