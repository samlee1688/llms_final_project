"""
Unified LLM client supporting OpenAI and Anthropic backends.

Usage:
    client = LLMClient(provider="openai", model="gpt-4o")
    response = client.generate(system_prompt, user_prompt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import openai
import anthropic

import config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardised wrapper around a single LLM completion."""

    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMClient:
    """Thin wrapper that exposes a single ``generate`` method for both
    OpenAI and Anthropic chat-completion APIs."""

    SUPPORTED_PROVIDERS = ("openai", "anthropic")

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from {self.SUPPORTED_PROVIDERS}"
            )

        self.provider = provider
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS

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

        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, temp, tokens)
        return self._generate_anthropic(system_prompt, user_prompt, temp, tokens)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
