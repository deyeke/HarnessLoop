"""M3-4: the MCP executor runs a task through a real MCP stdio server."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import conftest
import pytest

from harnessloop.llm import LLMResponse, LLMToolCall
from harnessloop.models import HarnessState, Task
from harnessloop.runtime.mcp_executor import MCPExecutor, render_tool_result

SERVER = Path(__file__).resolve().parent / "fixtures" / "fake_mcp_server.py"

HARNESS = {
    "system_prompt": "你是数据分析助手，先查表结构再写 SQL。",
    "tools": [
        {
            "name": "list_tables",
            "description": "列出数据库中的表。",
            "parameters": {
                "type": "object",
                "properties": {"database": {"type": "string"}},
                "required": ["database"],
            },
        },
        {
            "name": "run_sql",
            "description": "执行只读 SQL。",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "sql": {"type": "string"},
                },
                "required": ["database", "sql"],
            },
        },
    ],
}

GOLD_SQL = "SELECT COUNT(*) AS n FROM users"


class MCPFakeLLM:
    """Calls ``run_sql`` over MCP, then answers with the same SQL."""

    def __init__(self, database: Path) -> None:
        self.database = database

    async def complete(self, *, messages, tools=None, response_format=None):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if tool_messages:
            return LLMResponse(content=f"```sql\n{GOLD_SQL}\n```")
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    name="run_sql",
                    arguments={"database": str(self.database), "sql": GOLD_SQL},
                )
            ]
        )


def test_mcp_executor_runs_tools_over_stdio(tmp_path: Path) -> None:
    db = tmp_path / "mcp.db"
    conn = sqlite3.connect(db)
    conn.executescript(conftest.SCHEMA)
    conn.close()

    task = Task(
        task_id="mcp_count_users",
        question="一共有多少用户？",
        database=str(db),
        gold_sql=GOLD_SQL,
    )
    llm = MCPFakeLLM(db)
    executor = MCPExecutor(
        llm,
        command=[sys.executable, str(SERVER)],
        cwd=str(SERVER.parent),
        timeout_s=30.0,
    )
    trajectory = asyncio.run(executor.run(task, HarnessState.model_validate(HARNESS)))

    assert trajectory.error is None
    assert trajectory.steps[0].tool_calls[0].name == "run_sql"
    assert GOLD_SQL in str(trajectory.final_answer)
    assert trajectory.final_state is None
    assert trajectory.steps[0].observation["row_count"] == 1


def test_unknown_mcp_tool_returns_error_observation() -> None:
    class Result:
        isError, is_error = False, False
        content: list = []

    assert render_tool_result(Result()) == {}


def test_mcp_profile_validation_and_executor_selection(tmp_path: Path, api_env) -> None:
    from pydantic import ValidationError

    from harnessloop.config import ExecutorSettings, LoopConfig
    from harnessloop.loop import LoopRunner

    with pytest.raises(ValidationError):
        ExecutorSettings(kind="mcp")
    settings = ExecutorSettings(kind="mcp", mcp={"command": [sys.executable, "server.py"]})
    assert settings.mcp is not None and settings.mcp.command[0] == sys.executable

    config = LoopConfig.model_validate(
        {
            "dataset": {"path": str(api_env.dataset), "kind": "api"},
            "harness": {"path": str(api_env.harness)},
            "executor": {
                "kind": "mcp",
                "mcp": {"command": [sys.executable, str(SERVER)]},
            },
            "report": {"out_dir": str(tmp_path / "runs")},
        }
    )
    runner = LoopRunner.build(config, llm=MCPFakeLLM(tmp_path), run_id="mcp_build")
    assert isinstance(runner.executor, MCPExecutor)
    runner.trace_store.close()
