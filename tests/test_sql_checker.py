from __future__ import annotations

import pytest

from harnessloop.eval.dataset import load_tasks
from harnessloop.eval.sql_checker import SQLChecker
from harnessloop.models import AttributionLabel, Trajectory


@pytest.fixture
def tasks(toy_env):
    return {task.task_id: task for task in load_tasks(toy_env.dataset)}


async def test_correct_sql_passes(tasks) -> None:
    checker = SQLChecker()
    trajectory = Trajectory(
        task_id="sql_count_users",
        final_answer="```sql\nSELECT COUNT(*) FROM users\n```",
    )
    result = await checker.check(tasks["sql_count_users"], trajectory)
    assert result.passed
    assert result.signals == []


async def test_result_mismatch_includes_ast_hint(tasks) -> None:
    checker = SQLChecker()
    trajectory = Trajectory(
        task_id="sql_users_without_tasks",
        final_answer=(
            "```sql\nSELECT users.name FROM users "
            "JOIN tasks ON tasks.user_id = users.id\n```"
        ),
    )
    result = await checker.check(tasks["sql_users_without_tasks"], trajectory)
    assert not result.passed
    labels = [signal.label for signal in result.signals]
    assert AttributionLabel.RESULT_MISMATCH in labels
    evidence = result.signals[0].evidence
    assert "JOIN" in evidence or "行" in evidence


async def test_parse_failure(tasks) -> None:
    checker = SQLChecker()
    trajectory = Trajectory(
        task_id="sql_count_users",
        final_answer="```sql\nSELECT * FROM\n```",
    )
    result = await checker.check(tasks["sql_count_users"], trajectory)
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.PARSE_FAIL


async def test_multiple_statements_is_format_violation(tasks) -> None:
    checker = SQLChecker()
    trajectory = Trajectory(
        task_id="sql_count_users",
        final_answer="```sql\nSELECT 1; SELECT 2\n```",
    )
    result = await checker.check(tasks["sql_count_users"], trajectory)
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.FORMAT_VIOLATION


async def test_exec_error(tasks) -> None:
    checker = SQLChecker()
    trajectory = Trajectory(
        task_id="sql_count_users",
        final_answer="```sql\nSELECT missing_column FROM users\n```",
    )
    result = await checker.check(tasks["sql_count_users"], trajectory)
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.EXEC_ERROR


async def test_missing_final_answer(tasks) -> None:
    checker = SQLChecker()
    result = await checker.check(tasks["sql_count_users"], Trajectory(task_id="sql_count_users"))
    assert not result.passed
    assert result.signals[0].label == AttributionLabel.FORMAT_VIOLATION
