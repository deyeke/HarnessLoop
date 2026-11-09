"""Native ``opik_optimizer`` adapter (M4-2): multi-turn agents, not single calls.

``opik_optimizer`` optimizes a :class:`ChatPrompt` against a dataset + metric.
By default it drives a single LLM call per dataset item, which cannot host a
tool-calling agent. This adapter supplies its own *agent* and *metric*:

- the agent receives the optimized prompt, rebuilds a ``HarnessState`` with that
  ``system_prompt``, and runs the full multi-turn executor + checker the loop
  uses (tool calls included);
- the agent returns a small JSON payload (``passed`` / ``answer`` / ``error``);
- the metric reads ``passed`` back and scores 1.0 / 0.0.

So the optimizer searches prompts against real end-to-end task outcomes.
Budget is bounded by ``opt.opik.max_trials`` and ``opt.opik.sample_size``; R7
still applies — any failure falls back to the meta-prompt mutator and finally
to rule templates.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from typing import Any

from harnessloop.config import OpikSettings
from harnessloop.eval.api_checker import APIChecker
from harnessloop.eval.sql_checker import SQLChecker
from harnessloop.models import (
    Attribution,
    HarnessState,
    Hypothesis,
    Task,
)
from harnessloop.optimize.mutator import Mutator
from harnessloop.runtime.litellm_executor import LiteLLMExecutor

logger = logging.getLogger(__name__)


def native_available() -> bool:
    """True when ``opik_optimizer`` can be imported in this environment."""
    if sys.modules.get("opik_optimizer") is not None:
        return True
    try:
        return importlib.util.find_spec("opik_optimizer") is not None
    except (ImportError, ValueError):
        return False


def _first_prompt(prompts: Any) -> Any:
    if isinstance(prompts, dict):
        return next(iter(prompts.values()))
    return prompts


def _prompt_system(prompt: Any) -> str:
    system = getattr(prompt, "system", None)
    if isinstance(system, str) and system.strip():
        return system
    messages = getattr(prompt, "messages", None) or []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _json_output(
    task_id: str, passed: bool, answer: Any, error: str | None
) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "passed": bool(passed),
            "answer": answer if isinstance(answer, str) else str(answer),
            "error": error,
        },
        ensure_ascii=False,
    )


def harness_metric(dataset_item: Any, llm_output: Any) -> float:
    """Score one task from the agent payload: 1.0 pass / 0.0 fail."""
    payload = llm_output
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(payload, dict):
        return 0.0
    return 1.0 if payload.get("passed") else 0.0


class _HarnessAgent:
    """opik_optimizer agent: one dataset item -> full multi-turn task run."""

    def __init__(
        self,
        *,
        tasks: dict[str, Task],
        base_harness: HarnessState,
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
    ) -> None:
        self.tasks = tasks
        self.base_harness = base_harness
        self.executor = executor
        self.checker = checker
        # Attributes the optimizer writes on agents it evaluates.
        self.trace_phase = "Prompt Optimization"
        self.optimizer: Any | None = None

    def invoke_agent(
        self,
        prompts: Any,
        dataset_item: dict[str, Any],
        allow_tool_use: bool = True,
        **_: Any,
    ) -> str:
        task_id = dataset_item.get("task_id") or dataset_item.get("id")
        task = self.tasks.get(str(task_id))
        if task is None:
            return _json_output(str(task_id), False, "", f"unknown task: {task_id}")

        system = _prompt_system(_first_prompt(prompts))
        harness = self.base_harness
        if system:
            harness = harness.model_copy(update={"system_prompt": system})

        trajectory, result = asyncio.run(self._run_one(task, harness))
        return _json_output(
            task.task_id,
            result.passed,
            trajectory.final_answer,
            trajectory.error,
        )

    async def _run_one(
        self, task: Task, harness: HarnessState
    ) -> tuple[Any, Any]:
        trajectory = await self.executor.run(task, harness)
        result = await self.checker.check(task, trajectory)
        trajectory.outcome = "pass" if result.passed else "fail"
        return trajectory, result


def _local_dataset_class(dataset_cls: type) -> type:
    """In-memory ``opik.Dataset`` subclass (no Opik server required)."""

    class LocalDataset(dataset_cls):  # type: ignore[misc, valid-type]
        def __init__(self, name: str, items: list[dict[str, Any]]) -> None:
            super().__init__(
                name=name,
                description="HarnessLoop in-memory evaluation set",
                project_name=None,
                rest_client=None,
                dataset_items_count=len(items),
            )
            self._items = [dict(item) for item in items]

        @property
        def id(self) -> str:
            return f"harnessloop:{self.name}"

        @property
        def dataset_items_count(self) -> int:
            return len(self._items)

        def get_items(
            self,
            nb_samples: int | None = None,
            filter_string: str | None = None,
            num_threads: int = 1,
            chunk_size: int = 100,
        ) -> list[dict[str, Any]]:
            items = self._items if nb_samples is None else self._items[:nb_samples]
            return [dict(item) for item in items]

        def stream_items(
            self,
            chunk_size: int = 100,
            num_threads: int = 1,
            filter_string: str | None = None,
            nb_samples: int | None = None,
        ):
            items = self.get_items(nb_samples)
            for start in range(0, len(items), chunk_size):
                yield items[start : start + chunk_size]

        def copy(self) -> Any:
            return LocalDataset(self.name, self._items)

        # Required by opik's Dataset ABC; never used on the in-memory path.
        def __internal_api__stream_items_as_dataclasses__(self, **_: Any):
            return iter(())

        def __internal_api__stream_item_chunks__(self, **_: Any):
            return iter(())

    return LocalDataset


def _build_dataset(name: str, items: list[dict[str, Any]]) -> Any:
    """Prefer a real Opik dataset; fall back to the in-memory subclass offline."""
    from opik import Dataset

    # Without an explicit Opik endpoint, don't probe localhost on every round.
    if not (os.getenv("OPIK_API_KEY") or os.getenv("OPIK_URL_OVERRIDE")):
        return _local_dataset_class(Dataset)(name, items)

    try:
        import opik

        client = opik.Opik()
        dataset = client.get_or_create_dataset(name)
        dataset.insert([dict(item) for item in items])
        return dataset
    except Exception as exc:  # no Opik server / no API key: stay local
        logger.debug("Opik dataset unavailable (%s), using in-memory dataset", exc)
        return _local_dataset_class(Dataset)(name, items)


@contextmanager
def _dict_item_evaluation():
    """Make opik_optimizer score in-memory items (no experiment backend)."""
    try:
        from opik_optimizer import constants
    except Exception:  # pragma: no cover - package missing
        yield
        return
    previous = constants.ENABLE_EVALUATE_ON_DICT_ITEMS
    constants.ENABLE_EVALUATE_ON_DICT_ITEMS = True
    try:
        yield
    finally:
        constants.ENABLE_EVALUATE_ON_DICT_ITEMS = previous


def _extract_system(result: Any) -> str:
    prompt = getattr(result, "prompt", None)
    prompt = _first_prompt(prompt)
    system = _prompt_system(prompt)
    if not system:
        raise ValueError("opik_optimizer returned an empty system prompt")
    return system


class OpikNativeMutator(Mutator):
    """Search ``system_prompt`` with ``opik_optimizer`` on real multi-turn runs."""

    def __init__(
        self,
        *,
        tasks: list[Task],
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        settings: OpikSettings | None = None,
        fallback: Mutator | None = None,
        seed: int = 0,
        model: str = "openai/gpt-4o-mini",
        model_parameters: dict[str, Any] | None = None,
        project_name: str | None = None,
    ) -> None:
        self.tasks = tasks
        self.executor = executor
        self.checker = checker
        self.settings = settings or OpikSettings()
        self.fallback = fallback
        self.seed = seed
        self.model = model
        self.model_parameters = model_parameters or {}
        self.project_name = project_name

    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        if not native_available():
            logger.warning(
                "opik_optimizer 未安装（pip install 'harnessloop[optimizers]'），"
                "回退到 MetaPrompt 变异"
            )
            return await self._fallback(harness, hypotheses, attributions)
        try:
            candidate = await asyncio.to_thread(
                self._propose_sync, harness, attributions
            )
        except Exception:
            logger.exception("opik_optimizer 原生适配失败，回退到 MetaPrompt 变异")
            candidate = None
        if candidate is None or candidate.model_dump() == harness.model_dump():
            return await self._fallback(harness, hypotheses, attributions)
        return candidate

    async def _fallback(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        if self.fallback is not None:
            return await self.fallback.propose(harness, hypotheses, attributions)
        return harness

    # -- engine call (worker thread) ----------------------------------------- #

    def _propose_sync(
        self, harness: HarnessState, attributions: list[Attribution]
    ) -> HarnessState | None:
        import opik_optimizer

        sample = self._sample_tasks(attributions)
        name = f"harnessloop-tasks-{self.seed}"
        dataset = _build_dataset(name, [self._item(task) for task in sample])
        prompt = opik_optimizer.ChatPrompt(
            name="harness",
            system=harness.system_prompt,
            user="{question}",
            model=self.model,
            model_parameters=self.model_parameters or None,
        )
        agent = _HarnessAgent(
            tasks={task.task_id: task for task in self.tasks},
            base_harness=harness,
            executor=self.executor,
            checker=self.checker,
        )
        optimizer = opik_optimizer.MetaPromptOptimizer(
            model=self.model,
            model_parameters=self.model_parameters or None,
            prompts_per_round=max(1, self.settings.candidates),
            n_threads=1,
            verbose=0,
            seed=self.seed,
            skip_perfect_score=True,
        )
        with _dict_item_evaluation():
            result = optimizer.optimize_prompt(
                prompt=prompt,
                dataset=dataset,
                metric=harness_metric,
                agent=agent,
                n_samples=len(sample),
                max_trials=max(1, self.settings.max_trials),
                project_name=self.project_name,
                allow_tool_use=True,
            )

        system = _extract_system(result)
        if system.strip() == harness.system_prompt.strip():
            return None
        logger.info(
            "opik_optimizer 选出新 system_prompt（样本 %s 条，score=%s）",
            len(sample),
            getattr(result, "score", None),
        )
        return harness.model_copy(update={"system_prompt": system})

    def _sample_tasks(self, attributions: list[Attribution]) -> list[Task]:
        failed_ids = {a.task_id for a in attributions}
        rng = random.Random(self.seed)
        pool = [t for t in self.tasks if t.task_id in failed_ids] or list(self.tasks)
        size = min(self.settings.sample_size, len(pool))
        return rng.sample(pool, size) if size < len(pool) else pool

    @staticmethod
    def _item(task: Task) -> dict[str, Any]:
        return {
            "id": task.task_id,
            "task_id": task.task_id,
            "question": task.question,
        }
