"""MCP-native executor (M3-4).

HarnessX-style agents expose their tools over the Model Context Protocol, so the
executor only needs to forward the tool calls the model already produced. This
adapter speaks MCP over stdio using the official ``mcp`` SDK:

    from harnessloop.runtime.mcp_executor import MCPExecutor

    executor = MCPExecutor(llm, command=["python", "server.py"], cwd=".")

Contract notes:

- the model still sees the *harness* tool schemas (that is what gets optimised);
  the MCP server must expose tools with the same names;
- a session is opened per task, inside the task's own event-loop task, so the
  transport is always closed by the coroutine that opened it;
- ``final_state`` is ``None``: state assertions (``kind: api``) need a runtime
  with a snapshot, which is not part of the MCP contract.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from harnessloop.llm import LLMClient
from harnessloop.models import Task
from harnessloop.runtime.litellm_executor import LiteLLMExecutor

logger = logging.getLogger(__name__)


class MCPToolRuntime:
    """Forwards harness tool calls to an initialised MCP session."""

    def __init__(self, session: Any, tools: dict[str, Any] | None = None) -> None:
        self.session = session
        self.tools = tools or {}

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.tools and name not in self.tools:
            return {
                "error": f"MCP 服务端没有工具 {name!r}",
                "available": sorted(self.tools),
            }
        result = await self.session.call_tool(name, arguments or {})
        return render_tool_result(result)

    def snapshot(self) -> None:
        return None


def render_tool_result(result: Any) -> Any:
    """Flatten an MCP ``CallToolResult`` into a JSON-friendly observation."""
    is_error = bool(
        getattr(result, "isError", False) or getattr(result, "is_error", False)
    )
    texts = [
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", None) is not None
    ]
    if len(texts) == 1:
        payload: Any = _maybe_json(texts[0])
    elif texts:
        payload = texts
    else:
        payload = getattr(result, "structuredContent", None) or {}
    return {"error": payload} if is_error else payload


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


class MCPExecutor(LiteLLMExecutor):
    """LiteLLM loop with tool calls executed against an MCP stdio server."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 30.0,
        max_steps: int = 8,
    ) -> None:
        if not command:
            raise ValueError("MCP command must not be empty")
        super().__init__(llm, max_steps=max_steps)
        self.command = command
        self.cwd = cwd
        self.env = env
        self.timeout_s = timeout_s

    async def _runtime_for(self, task: Task, stack: AsyncExitStack) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.command[0],
            args=self.command[1:],
            env=self.env,
            cwd=self.cwd,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        logger.debug(
            "MCP server %s exposes %d tools: %s",
            self.command,
            len(tools),
            sorted(tools),
        )
        return MCPToolRuntime(session, tools)
