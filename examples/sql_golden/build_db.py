"""Build the demo SQLite database used by this example.

Usage:
    python examples/sql_golden/build_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATABASE = BASE / "toy.db"
SCHEMA = BASE / "schema.sql"


def main() -> None:
    if DATABASE.exists():
        DATABASE.unlink()
    connection = sqlite3.connect(DATABASE)
    try:
        connection.executescript(SCHEMA.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()
    print(f"created {DATABASE}")


if __name__ == "__main__":
    main()
