"""Dataset loading for SQL (v0.1) and API (v0.2) tasks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harnessloop.models import Task


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(text)
        records = payload["tasks"] if isinstance(payload, dict) else payload
    else:
        raise ValueError(f"unsupported dataset format: {path.suffix} (use .jsonl or .json)")
    if not isinstance(records, list):
        raise ValueError("dataset must be a list of task objects")
    return records


def _schema_from_db(database: str) -> str:
    conn = sqlite3.connect(f"file:{Path(database).as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        lines: list[str] = []
        for table in tables:
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            rendered = ", ".join(
                f"{column[1]} {column[2]}".strip() for column in columns
            )
            lines.append(f"{table}({rendered})")
    finally:
        conn.close()
    return "\n".join(lines)


def load_tasks(path: str | Path) -> list[Task]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    base = dataset_path.parent
    tasks: list[Task] = []
    for record in _read_records(dataset_path):
        task = Task.model_validate(record)
        if task.kind == "sql":
            if not task.database:
                raise ValueError(f"{task.task_id}: SQL task needs a 'database' field")
            database = Path(task.database).expanduser()
            if not database.is_absolute():
                database = base / database
            database = database.resolve()
            if not database.exists():
                raise FileNotFoundError(f"{task.task_id}: database not found: {database}")
            task.database = str(database)
            if not task.db_schema:
                task.db_schema = _schema_from_db(task.database)
        else:  # api
            if not task.initial_state:
                raise ValueError(f"{task.task_id}: API task needs an 'initial_state' object")
            assertions = (task.expected or {}).get("assertions")
            if not assertions:
                raise ValueError(
                    f"{task.task_id}: API task needs expected.assertions (state assertions)"
                )
        tasks.append(task)

    ids = [task.task_id for task in tasks]
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate task ids: {sorted(duplicates)}")
    return tasks
