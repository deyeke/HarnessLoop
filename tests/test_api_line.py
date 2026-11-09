from __future__ import annotations

import pytest

from harnessloop.eval.api_checker import (
    APIChecker,
    PathError,
    evaluate_assertion,
    resolve_path,
)
from harnessloop.models import Assertion, AttributionLabel, Task, Trajectory
from harnessloop.runtime.api_runtime import ApiToolRuntime

STATE = {
    "users": [{"id": 1, "name": "Alice"}],
    "tasks": [
        {"id": 1, "title": "写周报", "priority": "high", "status": "open"},
        {"id": 2, "title": "修bug", "priority": "low", "status": "done"},
    ],
    "next_task_id": 3,
}


# -- path resolution -------------------------------------------------------- #


def test_resolve_path_dotted_and_index() -> None:
    assert resolve_path(STATE, "next_task_id") == 3
    assert resolve_path(STATE, "tasks[0].title") == "写周报"
    assert resolve_path(STATE, "tasks[1].status") == "done"


def test_resolve_path_filter() -> None:
    task = resolve_path(STATE, "tasks[title=修bug]")
    assert task["id"] == 2
    assert resolve_path(STATE, "tasks[title=修bug].priority") == "low"


def test_resolve_path_errors() -> None:
    with pytest.raises(PathError):
        resolve_path(STATE, "missing.key")
    with pytest.raises(PathError):
        resolve_path(STATE, "tasks[9]")
    with pytest.raises(PathError):
        resolve_path(STATE, "tasks[title=不存在]")
    with pytest.raises(PathError):
        resolve_path(STATE, "tasks[0].bad..segment[!]")


# -- assertions -------------------------------------------------------------- #


def test_assertion_ops() -> None:
    cases = [
        (Assertion(op="eq", path="tasks[0].status", value="open"), True),
        (Assertion(op="eq", path="tasks[0].status", value="done"), False),
        (Assertion(op="ne", path="tasks[0].status", value="done"), True),
        (Assertion(op="exists", path="tasks[0]"), True),
        (Assertion(op="exists", path="tasks[9]"), False),
        (Assertion(op="not_exists", path="tasks[9]"), True),
        (Assertion(op="contains", path="tasks[0].title", value="周报"), True),
        (Assertion(op="contains", path="users", value={"id": 1, "name": "Alice"}), True),
        (Assertion(op="contains", path="tasks", value={"id": 2}), False),
        (Assertion(op="length_eq", path="tasks", value=2), True),
        (Assertion(op="length_ge", path="tasks", value=3), False),
        (Assertion(op="length_le", path="tasks", value=2), True),
    ]
    for assertion, expected in cases:
        passed, _ = evaluate_assertion(STATE, assertion)
        assert passed is expected, assertion


def test_assertion_on_missing_path_fails() -> None:
    passed, detail = evaluate_assertion(
        STATE, Assertion(op="eq", path="tasks[5].status", value="done")
    )
    assert not passed
    assert "路径缺失" in detail


# -- tool runtime ------------------------------------------------------------ #


def test_runtime_create_update_delete() -> None:
    runtime = ApiToolRuntime({"tasks": [], "next_task_id": 1, "users": []})

    created = runtime.execute("create_task", {"title": "写周报", "priority": "high"})
    assert created["task"]["id"] == 1
    assert runtime.snapshot()["next_task_id"] == 2

    updated = runtime.execute("update_task", {"task_id": "1", "status": "done"})
    assert updated["task"]["status"] == "done"  # string id coerced to int

    deleted = runtime.execute("delete_task", {"task_id": 1})
    assert deleted["deleted"] is True
    assert runtime.snapshot()["tasks"] == []


def test_runtime_errors_surface_as_observations() -> None:
    runtime = ApiToolRuntime({"tasks": [], "next_task_id": 1, "users": []})
    assert "error" in runtime.execute("create_task", {})
    assert "error" in runtime.execute("update_task", {"task_id": 42, "status": "done"})
    assert "error" in runtime.execute("no_such_tool", {})


def test_runtime_isolates_initial_state() -> None:
    initial = {"tasks": [{"id": 1, "status": "open"}], "next_task_id": 2, "users": []}
    runtime = ApiToolRuntime(initial)
    runtime.execute("update_task", {"task_id": 1, "status": "done"})
    assert initial["tasks"][0]["status"] == "open"  # deep copy protects the dataset


# -- checker ------------------------------------------------------------------ #


def _api_task(expected: dict) -> Task:
    return Task(
        task_id="t",
        kind="api",
        question="q",
        initial_state={"tasks": [{"id": 1, "status": "open"}], "next_task_id": 2},
        expected=expected,
    )


async def test_checker_passes_when_assertions_hold() -> None:
    task = _api_task({"assertions": [{"op": "eq", "path": "tasks[0].status", "value": "done"}]})
    trajectory = Trajectory(
        task_id="t",
        final_state={"tasks": [{"id": 1, "status": "done"}], "next_task_id": 2},
    )
    result = await APIChecker().check(task, trajectory)
    assert result.passed
    assert result.state_diff is not None


async def test_checker_reports_mismatch_with_diff() -> None:
    task = _api_task({"assertions": [{"op": "eq", "path": "tasks[0].status", "value": "done"}]})
    trajectory = Trajectory(
        task_id="t",
        final_state={
            "tasks": [{"id": 1, "status": "open"}, {"id": 2, "status": "open"}],
            "next_task_id": 3,
        },
    )
    result = await APIChecker().check(task, trajectory)
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.RESULT_MISMATCH
    assert "tasks[0].status" in result.signals[0].evidence
    assert "状态差异" in result.signals[0].evidence


async def test_checker_missing_final_state_is_exec_error() -> None:
    task = _api_task({"assertions": [{"op": "exists", "path": "tasks"}]})
    result = await APIChecker().check(task, Trajectory(task_id="t", final_answer="done"))
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.EXEC_ERROR
