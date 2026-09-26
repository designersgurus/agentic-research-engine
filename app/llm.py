"""Thin, provider-agnostic LLM client with budget enforcement.

Providers: openai, anthropic, mock. Mock mode lets the whole pipeline run
deterministically with zero API keys (used by the public demo and tests).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Optional

import httpx

from .config import Settings
from .guardrails import Budget, estimate_tokens


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.provider = settings.resolved_llm_provider

    @property
    def model_name(self) -> str:
        return {
            "openai": self.s.openai_model,
            "anthropic": self.s.anthropic_model,
        }.get(self.provider, "mock")

    async def complete(
        self,
        system: str,
        user: str,
        *,
        budget: Budget,
        max_tokens: int = 800,
        json_mode: bool = False,
        mock: Optional[Callable[[], str]] = None,
    ) -> str:
        """Run one completion. Raises BudgetExceeded *before* calling if it would break a cap."""
        est = estimate_tokens(system) + estimate_tokens(user)
        reservation = budget.reserve_llm(est, max_tokens)
        in_tok, out_tok = est, 0
        try:
            if self.provider == "mock":
                if mock is None:
                    raise LLMError("mock provider needs a mock callable")
                text = mock()
                out_tok = estimate_tokens(text)
            elif self.provider == "openai":
                text, in_tok, out_tok = await self._openai(system, user, max_tokens, json_mode)
            elif self.provider == "anthropic":
                text, in_tok, out_tok = await self._anthropic(system, user, max_tokens, json_mode)
            else:
                raise LLMError(f"unknown provider {self.provider}")
            return text
        finally:
            budget.settle_llm(reservation, in_tok, out_tok)

    async def _openai(self, system: str, user: str, max_tokens: int, json_mode: bool):
        body: dict[str, Any] = {
            "model": self.s.openai_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
            "temperature": 0.2,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.s.llm_timeout_s) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
                json=body,
            )
        if r.status_code >= 400:
            raise LLMError(f"OpenAI error {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    async def _anthropic(self, system: str, user: str, max_tokens: int, json_mode: bool):
        if not self.s.anthropic_model:
            raise LLMError("ANTHROPIC_MODEL is not set")
        if json_mode:
            system += "\nRespond with a single valid JSON object and nothing else."
        async with httpx.AsyncClient(timeout=self.s.llm_timeout_s) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.s.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.s.anthropic_model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
        if r.status_code >= 400:
            raise LLMError(f"Anthropic error {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
