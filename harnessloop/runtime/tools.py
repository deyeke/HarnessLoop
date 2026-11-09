"""Tool runtime and argument validation.

Executing a tool call is deliberately separate from validating it: validation
issues become attribution signals, while execution failures become observations
the model can recover from.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from harnessloop.llm import LLMToolCall
from harnessloop.models import AttributionLabel, HarnessState, ToolIssue
from harnessloop.sqlutil import extract_sql, read_only_statement


def openai_tools(harness: HarnessState) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in harness.tools
    ]


def validate_tool_call(
    harness: HarnessState, call: LLMToolCall, step_index: int
) -> list[ToolIssue]:
    """Check a tool call against the harness-declared schema."""
    tool = next((item for item in harness.tools if item.name == call.name), None)
    if tool is None:
        return [
            ToolIssue(
                label=AttributionLabel.WRONG_TOOL,
                evidence=f"第 {step_index} 步调用了未声明的工具 {call.name!r}",
                tool_name=call.name,
            )
        ]

    issues: list[ToolIssue] = []
    required = list(tool.parameters.get("required") or [])
    for param in required:
        if param not in call.arguments:
            issues.append(
                ToolIssue(
                    label=AttributionLabel.MISSING_PARAM,
                    evidence=f"第 {step_index} 步调用 {call.name} 缺少必填参数 {param}",
                    tool_name=call.name,
                    param=param,
                )
            )

    properties = tool.parameters.get("properties") or {}
    if properties and tool.parameters.get("additionalProperties") is not True:
        for key in call.arguments:
            if key not in properties:
                issues.append(
                    ToolIssue(
                        label=AttributionLabel.EXTRA_PARAM,
                        evidence=f"第 {step_index} 步调用 {call.name} 传入了未声明的参数 {key}",
                        tool_name=call.name,
                        param=key,
                    )
                )
    return issues


def _json_row(row: Any) -> list[Any]:
    values: list[Any] = []
    for value in row:
        if isinstance(value, bytes):
            values.append(value.decode("utf-8", errors="replace"))
        else:
            values.append(value)
    return values


class SQLToolRuntime:
    """v0.1 tool implementations for SQL tasks (read-only)."""

    def __init__(self, database: str, max_rows: int = 50) -> None:
        self.database = database
        self.max_rows = max_rows

    @asynccontextmanager
    async def _connect(self):
        conn = await aiosqlite.connect(self.database)
        try:
            await conn.execute("PRAGMA query_only = ON")
            yield conn
        finally:
            await conn.close()

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            if name == "run_sql":
                return await self._run_sql(str(arguments.get("sql", "")))
            if name == "list_tables":
                return await self._list_tables()
            if name == "describe_table":
                return await self._describe_table(str(arguments.get("table", "")))
        except Exception as exc:  # surfaced to the model as an observation
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"error": f"工具 {name} 在 v0.1 中没有实现"}

    async def _run_sql(self, sql: str) -> dict[str, Any]:
        extracted = extract_sql(sql) or ""
        statement = read_only_statement(extracted)
        query = f"SELECT * FROM ({statement}) LIMIT {self.max_rows + 1}"
        async with self._connect() as conn:
            cursor = await conn.execute(query)
            rows = await cursor.fetchmany(self.max_rows + 1)
            columns = [description[0] for description in (cursor.description or [])]
        truncated = len(rows) > self.max_rows
        return {
            "columns": columns,
            "rows": [_json_row(row) for row in rows[: self.max_rows]],
            "truncated": truncated,
        }

    async def _list_tables(self) -> dict[str, Any]:
        async with self._connect() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            rows = await cursor.fetchall()
        return {"tables": [row[0] for row in rows]}

    async def _describe_table(self, table: str) -> dict[str, Any]:
        if not table:
            return {"error": "缺少参数 table"}
        async with self._connect() as conn:
            cursor = await conn.execute(f'PRAGMA table_info("{table}")')
            rows = await cursor.fetchall()
        if not rows:
            return {"error": f"表 {table} 不存在"}
        return {
            "table": table,
            "columns": [
                {"name": row[1], "type": row[2], "notnull": bool(row[3]), "pk": bool(row[5])}
                for row in rows
            ],
        }
