"""M3-3: JSONL and SQLite trace stores are interchangeable."""

from __future__ import annotations

import json
from pathlib import Path

from harnessloop.models import CheckResult, Trajectory
from harnessloop.store.trace_store import (
    JsonlTraceStore,
    SQLiteTraceStore,
    open_trace_store,
)
from harnessloop.store.workspace import RunDir


def _records() -> tuple[list[Trajectory], list[CheckResult]]:
    trajectories = [
        Trajectory(
            task_id=f"t{i}",
            outcome="pass" if i % 2 == 0 else "fail",
            tokens={"prompt": 10 * i, "completion": i},
            cost_usd=0.001 * i,
            duration_ms=100 + i,
            final_answer=f"answer {i}",
        )
        for i in range(4)
    ]
    results = [
        CheckResult(task_id=f"t{i}", passed=i % 2 == 0, predicted_sql=f"SELECT {i}")
        for i in range(4)
    ]
    return trajectories, results


def test_jsonl_store_roundtrip(tmp_path: Path) -> None:
    run_dir = RunDir.create(tmp_path / "runs", run_id="jsonl")
    store = open_trace_store(run_dir, "jsonl")
    trajectories, results = _records()
    store.write_iteration(0, trajectories, results)

    assert isinstance(store, JsonlTraceStore)
    assert (run_dir.iter_dir(0) / "trajectory.jsonl").exists()
    assert [item.task_id for item in store.read_trajectories(0)] == ["t0", "t1", "t2", "t3"]
    assert [item.passed for item in store.read_results(0)] == [True, False, True, False]

    store.write_iteration(1, trajectories[:1], results[:1])
    assert [item.task_id for item in store.task_trajectory("t0")] == ["t0", "t0"]
    store.close()


def test_sqlite_store_roundtrip_and_queries(tmp_path: Path) -> None:
    run_dir = RunDir.create(tmp_path / "runs", run_id="sqlite")
    store = open_trace_store(run_dir, "sqlite")
    trajectories, results = _records()
    store.write_iteration(0, trajectories, results)
    store.write_iteration(1, trajectories, results)

    assert isinstance(store, SQLiteTraceStore)
    assert (run_dir.path / "traces.db").exists()
    assert len(store.read_trajectories(1)) == 4
    assert store.read_results(1)[1].predicted_sql == "SELECT 1"
    assert store.pass_rate(0) == 0.5
    assert [item.outcome for item in store.task_trajectory("t2")] == ["pass", "pass"]
    assert not (run_dir.iter_dir(1) / "trajectory.jsonl").exists()
    store.close()


def test_loop_writes_sqlite_traces(tmp_path: Path, api_env) -> None:
    import asyncio

    import conftest

    from harnessloop.config import LoopConfig
    from harnessloop.loop import LoopRunner

    config = LoopConfig.model_validate(
        {
            "dataset": {"path": str(api_env.dataset), "kind": "api"},
            "harness": {"path": str(api_env.harness)},
            "loop": {"max_rounds": 0, "concurrency": 2},
            "store": {"backend": "sqlite"},
            "report": {"out_dir": str(tmp_path / "runs")},
        }
    )
    runner = LoopRunner.build(config, llm=conftest.ApiFakeLLM(), run_id="sqlite_loop")
    asyncio.run(runner.run())

    db = runner.run_dir.path / "traces.db"
    assert db.exists()
    import sqlite3

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT task_id FROM trajectories ORDER BY task_id").fetchall()
    conn.close()
    assert [row[0] for row in rows] == ["api_close_task", "api_create_high"]
    payload = json.loads(
        (runner.run_dir.iter_dir(0) / "metrics.json").read_text(encoding="utf-8")
    )
    assert payload["total"] == 2
