"""PydanticAI executor adapter (M2-8) — validates the executor contract.

The v0.1 executor is a hand-rolled asyncio tool-call loop over LiteLLM. This
adapter implements the same ``run(task, harness) -> Trajectory`` contract on
top of ``pydantic-ai``'s Agent, proving the contract is framework-independent:

- harness tools are registered verbatim via ``Tool.from_schema`` (R4: tool
  JSON schemas are not rewritten by the framework);
- the same ``validate_tool_call`` rule set applies, so attribution signals
  (WRONG_TOOL / MISSING_PARAM / EXTRA_PARAM) stay identical;
- tool execution still goes through SQLToolRuntime / ApiToolRuntime, so S1
  snapshots and cost accounting semantics are preserved.

Known limitation: pydantic-ai drives token accounting itself and provides no
per-call cost, so ``cost_usd`` is 0.0 here; ``few_shot_examples`` are not
replayed as message history (they stay a LiteLLM-executor feature).
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from harnessloop.models import (
    HarnessState,
    Step,
    Task,
    ToolCall,
    Trajectory,
)
from harnessloop.runtime.api_runtime import ApiToolRuntime
from harnessloop.runtime.litellm_executor import initial_messages
from harnessloop.runtime.tools import SQLToolRuntime, validate_tool_call

logger = logging.getLogger(__name__)


class PydanticAIExecutor:
    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        tool_max_rows: int = 8,
    ) -> None:
        try:
            from pydantic_ai import Agent  # noqa: F401
            from pydantic_ai.tools import Tool  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PydanticAIExecutor 需要 pydantic-ai（pip install pydantic-ai）"
            ) from exc
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.tool_max_rows = tool_max_rows

    async def run(self, task: Task, harness: HarnessState) -> Trajectory:
        started = time.perf_counter()
        try:
            return await self._run_inner(task, harness, started)
        except ImportError:
            raise
        except Exception as exc:
            return Trajectory(
                task_id=task.task_id,
                final_answer=None,
                final_state=None,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_inner(
        self, task: Task, harness: HarnessState, started: float
    ) -> Trajectory:
        from pydantic_ai import Agent
        from pydantic_ai.tools import Tool

        runtime: ApiToolRuntime | SQLToolRuntime | None
        if task.kind == "api":
            runtime = ApiToolRuntime(task.initial_state)
        elif task.database:
            runtime = SQLToolRuntime(task.database, max_rows=self.tool_max_rows)
        else:
            runtime = None

        tools = [
            Tool.from_schema(
                self._tool_function(runtime, tool.name),
                name=tool.name,
                description=tool.description,
                json_schema=tool.parameters,
            )
            for tool in harness.tools
        ]

        messages = initial_messages(task, harness)
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(
            m["content"] for m in reversed(messages) if m["role"] == "user"
        )
        agent = Agent(self._model_instance(), instructions=system, tools=tools)
        result = await agent.run(user)

        steps = self._map_messages(result.all_messages(), harness)
        prompt_tokens, completion_tokens = self._tokens(result)
        return Trajectory(
            task_id=task.task_id,
            steps=steps,
            final_answer=result.output if isinstance(result.output, str) else str(result.output),
            final_state=runtime.snapshot() if isinstance(runtime, ApiToolRuntime) else None,
            tokens={
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            cost_usd=0.0,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    # -- pieces --------------------------------------------------------------- #

    def _model_instance(self) -> Any:
        from pydantic_ai.models import KnownModelName, Model  # noqa: F401

        if self.api_base or self.api_key:
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(
                base_url=self.api_base, api_key=self.api_key
            )
            return OpenAIModel(self.model, provider=provider)
        return self.model

    @staticmethod
    def _tool_function(runtime: Any, name: str):
        async def invoke(**kwargs: Any) -> Any:
            if runtime is None:
                return {"error": "当前任务没有可用的工具运行时"}
            result = runtime.execute(name, kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        return invoke

    def _map_messages(self, messages: list[Any], harness: HarnessState) -> list[Step]:
        steps: list[Step] = []
        for message in messages:
            part_type = type(message).__name__
            if part_type == "ModelResponse":
                step = Step(index=len(steps))
                for part in message.parts:
                    kind = type(part).__name__
                    if kind == "TextPart":
                        step.model_output += getattr(part, "content", "") or ""
                    elif kind == "ToolCallPart":
                        call = ToolCall(
                            call_id=getattr(part, "tool_call_id", None),
                            name=part.tool_name,
                            arguments=self._as_dict(part.args),
                        )
                        call.issues = validate_tool_call(harness, call, len(steps))
                        step.tool_calls.append(call)
                steps.append(step)
            elif part_type == "ModelRequest" and steps:
                for part in message.parts:
                    if type(part).__name__ == "ToolReturnPart":
                        steps[-1].observation = part.content
        return steps

    @staticmethod
    def _as_dict(args: Any) -> dict[str, Any]:
        if isinstance(args, dict):
            return args
        try:
            import json

            return json.loads(args) if isinstance(args, str) else dict(args or {})
        except Exception:
            return {}

    @staticmethod
    def _tokens(result: Any) -> tuple[int, int]:
        usage = result.usage  # pydantic-ai 2.x: property, not a method
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return int(input_tokens), int(output_tokens)
