"""A tiny MCP server used by the executor integration test (stdio transport)."""

from __future__ import annotations

import sqlite3

from mcp.server.mcpserver import MCPServer

server = MCPServer("fake-sql")


@server.tool()
def list_tables(database: str) -> list[str]:
    """List table names in a SQLite database."""
    conn = sqlite3.connect(database)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


@server.tool()
def run_sql(database: str, sql: str) -> dict:
    """Run a read-only SQL statement and return the rows."""
    if not sql.strip().lower().startswith("select"):
        raise ValueError("only SELECT statements are allowed")
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    return {"rows": rows, "row_count": len(rows)}


if __name__ == "__main__":
    server.run()
