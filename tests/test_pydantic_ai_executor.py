from __future__ import annotations

import conftest

from harnessloop.models import HarnessState, Task
from harnessloop.runtime.pydantic_ai_executor import PydanticAIExecutor


def _test_model():
    from pydantic_ai.models.test import TestModel

    return TestModel(call_tools="all", seed=7)


async def test_api_task_contract(api_env) -> None:
    """Same run() contract as LiteLLMExecutor: steps, issues, S1, tokens."""
    executor = PydanticAIExecutor(_test_model())
    task = Task.model_validate(conftest.API_TASKS[0])
    harness = HarnessState.model_validate(conftest.API_HARNESS.copy())

    trajectory = await executor.run(task, harness)

    assert trajectory.task_id == task.task_id
    assert trajectory.error is None
    assert trajectory.steps, "TestModel should have produced at least one step"
    assert any(step.tool_calls for step in trajectory.steps)
    assert isinstance(trajectory.final_answer, str) and trajectory.final_answer
    assert trajectory.final_state is not None  # S1 snapshot for API tasks
    assert trajectory.tokens["total"] > 0
    assert trajectory.duration_ms is not None
    for step in trajectory.steps:
        for call in step.tool_calls:
            assert call.name
            for issue in call.issues:
                assert issue.evidence


async def test_sql_task_contract(toy_env) -> None:
    from harnessloop.eval.dataset import load_tasks
    from harnessloop.store.workspace import load_harness

    executor = PydanticAIExecutor(_test_model())
    tasks = load_tasks(toy_env.dataset)
    harness = load_harness(toy_env.harness)

    trajectory = await executor.run(tasks[0], harness)

    assert trajectory.error is None
    assert trajectory.final_state is None  # no S1 for SQL tasks
    assert isinstance(trajectory.final_answer, str)
    run_sql_calls = [
        call
        for step in trajectory.steps
        for call in step.tool_calls
        if call.name == "run_sql"
    ]
    assert run_sql_calls, "TestModel should have called run_sql"
