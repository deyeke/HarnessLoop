"""Trace storage (M3-3): JSONL sidecars (default) or one SQLite database.

The filesystem layout stays the default because it is diff-friendly and needs no
tooling. The SQLite backend exists for runs whose per-iteration trajectory file
grows past the point where re-parsing it on every read is wasteful — see
``experiments/trace_storage/RESULTS.md`` for the measured break-even.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from harnessloop.models import CheckResult, Trajectory
from harnessloop.store.workspace import RunDir, _to_jsonable


class TraceStore(Protocol):
    def write_iteration(
        self,
        iteration: int,
        trajectories: list[Trajectory],
        results: list[CheckResult],
        *,
        prefix: str = "",
    ) -> None: ...

    def read_trajectories(self, iteration: int) -> list[Trajectory]: ...

    def read_results(self, iteration: int) -> list[CheckResult]: ...

    def task_trajectory(self, task_id: str) -> list[Trajectory]: ...

    def close(self) -> None: ...


class JsonlTraceStore:
    """Current v0.1 layout: ``iter_<n>/trajectory.jsonl`` (+ check results)."""

    backend = "jsonl"

    def __init__(self, run_dir: RunDir) -> None:
        self.run_dir = run_dir

    def write_iteration(
        self,
        iteration: int,
        trajectories: list[Trajectory],
        results: list[CheckResult],
        *,
        prefix: str = "",
    ) -> None:
        directory = self.run_dir.iter_dir(iteration)
        self.run_dir.write_jsonl(directory / f"{prefix}trajectory.jsonl", trajectories)
        self.run_dir.write_jsonl(directory / f"{prefix}check_results.jsonl", results)

    def read_trajectories(self, iteration: int) -> list[Trajectory]:
        path = self.run_dir.iter_dir(iteration) / "trajectory.jsonl"
        if not path.exists():
            return []
        return [
            Trajectory.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def read_results(self, iteration: int) -> list[CheckResult]:
        path = self.run_dir.iter_dir(iteration) / "check_results.jsonl"
        if not path.exists():
            return []
        return [
            CheckResult.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def task_trajectory(self, task_id: str) -> list[Trajectory]:
        found: list[Trajectory] = []
        for directory in sorted(self.run_dir.path.glob("iter_*")):
            iteration = int(directory.name.split("_", 1)[1])
            found.extend(
                item for item in self.read_trajectories(iteration) if item.task_id == task_id
            )
        return found

    def close(self) -> None:  # nothing to release
        return None


class SQLiteTraceStore:
    """One ``traces.db`` per run with indexed lookups by iteration and task."""

    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                iteration INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                outcome TEXT,
                cost_usd REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                duration_ms INTEGER,
                payload TEXT NOT NULL,
                PRIMARY KEY (iteration, task_id)
            );
            CREATE TABLE IF NOT EXISTS check_results (
                iteration INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                predicted_sql TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY (iteration, task_id)
            );
            CREATE INDEX IF NOT EXISTS idx_trajectories_task
                ON trajectories (task_id);
            CREATE INDEX IF NOT EXISTS idx_results_passed
                ON check_results (iteration, passed);
            """
        )
        self._conn.commit()

    def write_iteration(
        self,
        iteration: int,
        trajectories: list[Trajectory],
        results: list[CheckResult],
        *,
        prefix: str = "",
    ) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO trajectories "
                "(iteration, task_id, outcome, cost_usd, prompt_tokens, "
                " completion_tokens, duration_ms, payload) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        iteration,
                        item.task_id,
                        item.outcome,
                        item.cost_usd,
                        item.tokens.get("prompt", 0),
                        item.tokens.get("completion", 0),
                        item.duration_ms,
                        json.dumps(_to_jsonable(item), ensure_ascii=False),
                    )
                    for item in trajectories
                ],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO check_results "
                "(iteration, task_id, passed, predicted_sql, payload) VALUES (?,?,?,?,?)",
                [
                    (
                        iteration,
                        item.task_id,
                        int(item.passed),
                        item.predicted_sql,
                        json.dumps(_to_jsonable(item), ensure_ascii=False),
                    )
                    for item in results
                ],
            )

    def read_trajectories(self, iteration: int) -> list[Trajectory]:
        rows = self._conn.execute(
            "SELECT payload FROM trajectories WHERE iteration = ? ORDER BY task_id",
            (iteration,),
        ).fetchall()
        return [Trajectory.model_validate(json.loads(row[0])) for row in rows]

    def read_results(self, iteration: int) -> list[CheckResult]:
        rows = self._conn.execute(
            "SELECT payload FROM check_results WHERE iteration = ? ORDER BY task_id",
            (iteration,),
        ).fetchall()
        return [CheckResult.model_validate(json.loads(row[0])) for row in rows]

    def task_trajectory(self, task_id: str) -> list[Trajectory]:
        rows = self._conn.execute(
            "SELECT payload FROM trajectories WHERE task_id = ? ORDER BY iteration",
            (task_id,),
        ).fetchall()
        return [Trajectory.model_validate(json.loads(row[0])) for row in rows]

    def pass_rate(self, iteration: int) -> float:
        row = self._conn.execute(
            "SELECT AVG(passed) FROM check_results WHERE iteration = ?", (iteration,)
        ).fetchone()
        return float(row[0] or 0.0)

    def close(self) -> None:
        self._conn.close()


def open_trace_store(
    run_dir: RunDir, backend: str = "jsonl", *, sqlite_name: str = "traces.db"
) -> TraceStore:
    if backend == "sqlite":
        return SQLiteTraceStore(run_dir.path / sqlite_name)
    if backend != "jsonl":
        raise ValueError(f"unknown trace store backend: {backend!r}")
    return JsonlTraceStore(run_dir)
