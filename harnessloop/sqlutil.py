"""Shared SQL helpers: extraction, read-only validation, execution."""

from __future__ import annotations

import re
from typing import Any

import aiosqlite
import sqlglot
from pydantic import BaseModel
from sqlglot import exp

_FENCED_SQL = re.compile(r"```(?:sql)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)


class SQLParseError(ValueError):
    """The text cannot be parsed as SQL at all."""


class SQLFormatError(ValueError):
    """The text is SQL but violates the harness output contract."""


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False


def extract_sql(text: str | None) -> str | None:
    """Pull a SQL statement out of a model answer (fenced block preferred)."""
    if not text or not text.strip():
        return None
    match = _FENCED_SQL.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    return candidate or None


def read_only_statement(sql: str) -> str:
    """Validate that ``sql`` is exactly one read-only SELECT/UNION.

    Returns the normalised SQLite statement.
    """
    if not sql or not sql.strip():
        raise SQLFormatError("空 SQL")
    try:
        statements = [stmt for stmt in sqlglot.parse(sql, read="sqlite") if stmt is not None]
    except Exception as exc:  # sqlglot raises several parser exception types
        raise SQLParseError(str(exc).splitlines()[0]) from exc
    if not statements:
        raise SQLFormatError("空 SQL")
    if len(statements) != 1:
        raise SQLFormatError(f"包含 {len(statements)} 条语句，只允许一条")
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
        raise SQLFormatError(f"只允许只读 SELECT，实际为 {statement.key.upper()}")
    return statement.sql(dialect="sqlite")


async def execute_query(database: str, sql: str, max_rows: int = 1000) -> QueryResult:
    async with aiosqlite.connect(database) as conn:
        await conn.execute("PRAGMA query_only = ON")
        cursor = await conn.execute(sql)
        rows = await cursor.fetchmany(max_rows + 1)
        columns = [description[0] for description in (cursor.description or [])]
    truncated = len(rows) > max_rows
    return QueryResult(
        columns=columns,
        rows=[list(row) for row in rows[:max_rows]],
        truncated=truncated,
    )
