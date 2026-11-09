"""v0.1 executor: LiteLLM + a plain asyncio tool-call loop.

No graph framework, no hidden state — the loop is intentionally readable so the
optimisation semantics stay transparent.
"""

from __future__ import annotations

import json
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from harnessloop.llm import LLMClient, LLMResponse
from harnessloop.models import (
    AttributionLabel,
    HarnessState,
    Step,
    Task,
    ToolCall,
    Trajectory,
)
from harnessloop.runtime.api_runtime import ApiToolRuntime, serialize_state
from harnessloop.runtime.tools import SQLToolRuntime, openai_tools, validate_tool_call


def initial_messages(task: Task, harness: HarnessState) -> list[dict[str, Any]]:
    system = harness.system_prompt
    if harness.output_format:
        system = f"{system}\n\n输出格式要求：\n{harness.output_format}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for example in harness.few_shot_examples:
        messages.append({"role": "user", "content": example.input})
        messages.append({"role": "assistant", "content": example.output})

    if task.kind == "api":
        parts = [
            task.question,
            f"当前 API 环境状态（S0）：\n{serialize_state(task.initial_state or {})}",
            "请通过工具调用完成任务；每步之后检查工具返回，确认状态已按预期变更，"
            "全部完成后再给出最终答复。",
        ]
        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    parts = [task.question]
    if task.db_schema:
        parts.append(f"数据库 schema：\n{task.db_schema}")
    if task.database:
        parts.append(
            f"目标数据库：{Path(task.database).name}。"
            "不确定表结构时可调用 list_tables / describe_table，"
            "写完 SQL 后可调用 run_sql 自检。"
        )
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": [
            {
                "id": call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for index, call in enumerate(response.tool_calls)
        ],
    }


def _tool_message(call: Any, observation: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id or "",
        "name": call.name,
        "content": json.dumps(observation, ensure_ascii=False, default=str),
    }


class LiteLLMExecutor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        max_steps: int = 8,
        tool_max_rows: int = 50,
    ) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.tool_max_rows = tool_max_rows

    async def _runtime_for(self, task: Task, stack: AsyncExitStack) -> Any:
        """Build the tool runtime for one task.

        Executors that own a scoped transport (MCP stdio) override this and use
        ``stack`` so the transport is closed in the same task that opened it.
        """
        if task.kind == "api":
            return ApiToolRuntime(task.initial_state)
        if task.database:
            return SQLToolRuntime(task.database, max_rows=self.tool_max_rows)
        return None

    async def run(self, task: Task, harness: HarnessState) -> Trajectory:
        async with AsyncExitStack() as stack:
            runtime = await self._runtime_for(task, stack)
            return await self._run_loop(task, harness, runtime)

    async def _run_loop(
        self, task: Task, harness: HarnessState, runtime: Any
    ) -> Trajectory:
        started = time.perf_counter()
        messages = initial_messages(task, harness)
        steps: list[Step] = []
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0
        final_answer: str | None = None
        error: str | None = None

        try:
            for index in range(self.max_steps):
                step_started = time.perf_counter()
                response = await self.llm.complete(
                    messages=messages,
                    tools=openai_tools(harness) or None,
                )
                prompt_tokens += response.prompt_tokens
                completion_tokens += response.completion_tokens
                cost_usd += response.cost_usd

                step = Step(
                    index=index,
                    model_output=response.content or "",
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )

                if not response.tool_calls:
                    final_answer = response.content
                else:
                    messages.append(_assistant_message(response))
                    for call in response.tool_calls:
                        issues = validate_tool_call(harness, call, index)
                        blocking = any(
                            issue.label == AttributionLabel.MISSING_PARAM
                            for issue in issues
                        )
                        if blocking:
                            observation: Any = {
                                "error": "工具参数校验失败，请补齐必填参数后重试",
                                "issues": [issue.evidence for issue in issues],
                            }
                        elif runtime is None:
                            observation = {"error": "当前任务没有可用的工具运行时"}
                        else:
                            observation = await runtime.execute(call.name, call.arguments)

                        step.tool_calls.append(
                            ToolCall(
                                call_id=call.id,
                                name=call.name,
                                arguments=call.arguments,
                                issues=issues,
                            )
                        )
                        step.observation = observation
                        messages.append(_tool_message(call, observation))

                step.latency_ms = int((time.perf_counter() - step_started) * 1000)
                steps.append(step)
                if final_answer is not None:
                    break
        except Exception as exc:  # pragma: no cover - provider/network failures
            error = f"{type(exc).__name__}: {exc}"

        return Trajectory(
            task_id=task.task_id,
            steps=steps,
            final_answer=final_answer,
            final_state=runtime.snapshot() if isinstance(runtime, ApiToolRuntime) else None,
            tokens={
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            cost_usd=cost_usd,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )
