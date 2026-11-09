"""LLM client abstraction.

The executor and hypothesis generator depend on this tiny protocol instead of
LiteLLM directly, which is what makes the loop testable without network calls.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

# USD per 1M tokens. Peak cache-miss prices, used when LiteLLM has no entry
# for the model (which would otherwise report every call as free).
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
}


def lookup_price(model: str) -> tuple[float | None, float | None]:
    """Return ``(input, output)`` USD per 1M tokens for a model name."""
    name = model.split("/", 1)[1] if "/" in model else model
    return MODEL_PRICING_PER_1M.get(name, (None, None))


class LLMToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] | None = None


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class LiteLLMClient:
    """Thin wrapper over ``litellm.acompletion`` (provider normalisation only)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        api_base: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.seed = seed
        table_input, table_output = lookup_price(model)
        self.input_price_per_1m = (
            input_price_per_1m if input_price_per_1m is not None else table_input
        )
        self.output_price_per_1m = (
            output_price_per_1m if output_price_per_1m is not None else table_output
        )
        self.extra = kwargs

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.input_price_per_1m is None and self.output_price_per_1m is None:
            return None
        return (
            prompt_tokens * (self.input_price_per_1m or 0.0)
            + completion_tokens * (self.output_price_per_1m or 0.0)
        ) / 1_000_000

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from litellm import acompletion

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.extra,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = await acompletion(**kwargs)
        message = response.choices[0].message
        tool_calls: list[LLMToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            raw_arguments = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"__raw__": raw_arguments}
            tool_calls.append(
                LLMToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        if cost is None:
            try:  # fall back to LiteLLM's cost map when we have no price
                from litellm import completion_cost

                cost = float(completion_cost(completion_response=response) or 0.0)
            except Exception:  # pragma: no cover - depends on provider metadata
                cost = 0.0

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
