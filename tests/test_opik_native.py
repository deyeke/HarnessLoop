"""M4-2: opik_optimizer native adapter on top of real multi-turn runs."""

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import conftest
import pytest

from harnessloop.config import DEFAULT_HARNESS, OpikSettings
from harnessloop.eval.sql_checker import SQLChecker
from harnessloop.models import (
    Attribution,
    AttributionLabel,
    HarnessFacet,
    HarnessState,
    Task,
)
from harnessloop.optimize import opik_native
from harnessloop.optimize.opik_native import (
    OpikNativeMutator,
    _build_dataset,
    _HarnessAgent,
    harness_metric,
    native_available,
)
from harnessloop.runtime.litellm_executor import LiteLLMExecutor

pytestmark = pytest.mark.skipif(
    not native_available(), reason="opik_optimizer is not installed"
)


def _tasks(toy_env) -> list[Task]:
    return [
        Task(
            task_id=item["task_id"],
            question=item["question"],
            database=str(toy_env.database),
            gold_sql=item["gold_sql"],
        )
        for item in conftest.TASKS
    ]


def _harness() -> HarnessState:
    return HarnessState.model_validate(DEFAULT_HARNESS)


def _items(tasks: list[Task]) -> list[dict]:
    return [
        {"id": task.task_id, "task_id": task.task_id, "question": task.question}
        for task in tasks
    ]


def test_harness_metric_reads_agent_payload() -> None:
    assert harness_metric({}, '{"passed": true}') == 1.0
    assert harness_metric({}, '{"passed": false}') == 0.0
    assert harness_metric({}, "not json") == 0.0


def test_native_agent_runs_full_multi_turn_task(toy_env) -> None:
    tasks = _tasks(toy_env)
    llm = conftest.ChattyFakeLLM()
    agent = _HarnessAgent(
        tasks={task.task_id: task for task in tasks},
        base_harness=_harness(),
        executor=LiteLLMExecutor(llm),
        checker=SQLChecker(),
    )
    dataset = _build_dataset("harnessloop-test", _items(tasks))
    assert len(dataset.get_items()) == len(tasks)

    payload = json.loads(
        agent.invoke_agent(
            prompts={"harness": SimpleNamespace(system=DEFAULT_HARNESS["system_prompt"])},
            dataset_item={"task_id": "sql_users_without_tasks", "question": "x"},
        )
    )
    # The scripted LLM writes an INNER JOIN until the prompt mentions LEFT JOIN.
    assert payload["passed"] is False
    assert "JOIN" in payload["answer"].upper()
    assert llm.calls > 0


def test_native_agent_feeds_real_evaluation(toy_env) -> None:
    from opik_optimizer import ChatPrompt, MetaPromptOptimizer

    tasks = _tasks(toy_env)
    harness = _harness()
    agent = _HarnessAgent(
        tasks={task.task_id: task for task in tasks},
        base_harness=harness,
        executor=LiteLLMExecutor(conftest.ChattyFakeLLM()),
        checker=SQLChecker(),
    )
    dataset = _build_dataset("harnessloop-test", _items(tasks))
    optimizer = MetaPromptOptimizer(model="openai/gpt-4o-mini", verbose=0)
    score = optimizer.evaluate_prompt(
        prompt=ChatPrompt(name="harness", system=harness.system_prompt, user="{question}"),
        dataset=dataset,
        metric=harness_metric,
        agent=agent,
        n_samples=3,
        seed=7,
        return_evaluation_result=False,
        use_evaluate_on_dict_items=True,
    )
    assert score == pytest.approx(2 / 3)


class _MarkerMutator:
    async def propose(self, harness: HarnessState, hypotheses, attributions) -> HarnessState:
        return harness.model_copy(update={"system_prompt": "fallback prompt"})


def _attributions() -> list[Attribution]:
    return [
        Attribution(
            task_id="sql_users_without_tasks",
            label=AttributionLabel.RESULT_MISMATCH,
            evidence="x",
            harness_facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
        )
    ]


def _fake_module(optimizer_cls) -> ModuleType:
    module = ModuleType("opik_optimizer")
    module.ChatPrompt = lambda **kwargs: SimpleNamespace(**kwargs)
    module.MetaPromptOptimizer = optimizer_cls
    module.constants = SimpleNamespace(ENABLE_EVALUATE_ON_DICT_ITEMS=False)
    return module


def test_native_mutator_returns_optimized_prompt(monkeypatch, toy_env) -> None:
    class FakeOptimizer:
        seen: dict = {}

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def optimize_prompt(self, **kwargs):
            FakeOptimizer.seen = kwargs
            agent = kwargs["agent"]
            item = kwargs["dataset"].get_items()[0]
            payload = agent.invoke_agent(
                prompts={"harness": kwargs["prompt"]}, dataset_item=item
            )
            assert harness_metric(item, payload) == 0.0
            return SimpleNamespace(
                prompt=SimpleNamespace(system="优化后的 system prompt"), score=0.66
            )

    monkeypatch.setitem(sys.modules, "opik_optimizer", _fake_module(FakeOptimizer))
    monkeypatch.setattr(
        opik_native,
        "_build_dataset",
        lambda name, items: SimpleNamespace(name=name, get_items=lambda: list(items)),
    )
    tasks = _tasks(toy_env)
    mutator = OpikNativeMutator(
        tasks=tasks,
        executor=LiteLLMExecutor(conftest.ChattyFakeLLM()),
        checker=SQLChecker(),
        settings=OpikSettings(sample_size=2, max_trials=1),
        fallback=_MarkerMutator(),
        seed=3,
    )
    candidate = asyncio.run(mutator.propose(_harness(), [], _attributions()))

    assert candidate.system_prompt == "优化后的 system prompt"
    assert FakeOptimizer.seen["metric"] is harness_metric
    assert FakeOptimizer.seen["n_samples"] == 1  # only the failing task is sampled
    assert FakeOptimizer.seen["max_trials"] == 1


def test_profile_auto_selects_native_with_meta_fallback(api_env, tmp_path) -> None:
    from harnessloop.config import (
        DatasetConfig,
        HarnessConfig,
        LoopConfig,
        OptSettings,
        ReportSettings,
    )
    from harnessloop.loop import LoopRunner
    from harnessloop.optimize.opik_mutator import OpikMutator

    config = LoopConfig(
        model="gpt-4o-mini",
        dataset=DatasetConfig(path=str(api_env.dataset), kind="api"),
        harness=HarnessConfig(path=str(api_env.harness)),
        opt=OptSettings(mutator="opik", opik=OpikSettings(backend="auto")),
        report=ReportSettings(out_dir=str(tmp_path / "runs")),
    )
    runner = LoopRunner.build(config, llm=conftest.ApiFakeLLM(), run_id="native_build")
    try:
        assert isinstance(runner.mutator, OpikNativeMutator)
        assert isinstance(runner.mutator.fallback, OpikMutator)
    finally:
        runner.trace_store.close()


def test_native_mutator_falls_back_when_optimizer_raises(monkeypatch, toy_env) -> None:
    class BoomOptimizer:
        def __init__(self, **kwargs) -> None:
            pass

        def optimize_prompt(self, **kwargs):
            raise RuntimeError("opik server exploded")

    monkeypatch.setitem(sys.modules, "opik_optimizer", _fake_module(BoomOptimizer))
    monkeypatch.setattr(
        opik_native,
        "_build_dataset",
        lambda name, items: SimpleNamespace(name=name, get_items=lambda: list(items)),
    )
    mutator = OpikNativeMutator(
        tasks=_tasks(toy_env),
        executor=LiteLLMExecutor(conftest.ChattyFakeLLM()),
        checker=SQLChecker(),
        fallback=_MarkerMutator(),
        seed=3,
    )
    candidate = asyncio.run(mutator.propose(_harness(), [], _attributions()))
    assert candidate.system_prompt == "fallback prompt"
