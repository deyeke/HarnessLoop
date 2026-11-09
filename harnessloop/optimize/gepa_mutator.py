"""GEPA-backed mutator (M2-4): the real ``gepa`` engine as a proposal backend.

HarnessLoop stays the outer loop (attribution -> hypothesis -> gate). Each
round, ``GEPAMutator`` runs ONE bounded reflective-mutation search seeded from
the current harness and this round's failure evidence:

- text components: ``system_prompt``, ``output_format``,
  ``tools.<name>.description`` (tool JSON schemas stay fixed — R4);
- the adapter evaluates candidates with the very same executor + checker the
  loop uses, so scores are comparable;
- ``max_metric_calls`` bounds the extra task executions per round;
- the loop's gate re-evaluates and accepts/rolls back as usual (R7: rule
  templates remain the permanent fallback whenever gepa is unavailable or the
  search fails).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from harnessloop.config import GepaSettings
from harnessloop.eval.api_checker import APIChecker
from harnessloop.eval.sql_checker import SQLChecker
from harnessloop.llm import LLMClient
from harnessloop.models import (
    Attribution,
    HarnessState,
    Hypothesis,
    Task,
)
from harnessloop.optimize.mutator import Mutator
from harnessloop.runtime.litellm_executor import LiteLLMExecutor

logger = logging.getLogger(__name__)


class GEPAMutator(Mutator):
    def __init__(
        self,
        llm: LLMClient,
        *,
        tasks: list[Task],
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        settings: GepaSettings | None = None,
        seed: int = 0,
        fallback: Mutator | None = None,
    ) -> None:
        self.llm = llm
        self.tasks = tasks
        self.executor = executor
        self.checker = checker
        self.settings = settings or GepaSettings()
        self.seed = seed
        self.fallback = fallback

    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        try:
            import gepa  # noqa: F401  # engine presence check
        except ImportError:
            logger.warning("gepa 未安装（pip install gepa），回退到规则模板变异")
            return await self._fallback_or_self(harness, hypotheses, attributions)
        try:
            return await asyncio.to_thread(
                self._optimize_sync, harness, hypotheses, attributions
            )
        except Exception:
            logger.exception("GEPA 反射变异失败，回退到规则模板变异")
            return await self._fallback_or_self(harness, hypotheses, attributions)

    async def _fallback_or_self(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        if self.fallback is not None:
            return await self.fallback.propose(harness, hypotheses, attributions)
        return harness

    # -- sync engine call (runs in a worker thread) -------------------------- #

    def _optimize_sync(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        import gepa

        seed_candidate = self._to_candidate(harness)
        # attributions only cover failed tasks, so every entry is a failure
        failed_ids = {attribution.task_id for attribution in attributions}
        trainset = [t for t in self.tasks if t.task_id in failed_ids] or list(self.tasks)
        adapter = _HarnessAdapter(
            tasks=self.tasks,
            base_harness=harness,
            executor=self.executor,
            checker=self.checker,
            hypotheses=hypotheses,
        )
        reflection_lm = _SyncReflectionBridge(self.llm)

        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=list(self.tasks),
            adapter=adapter,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=self.settings.reflection_minibatch_size,
            max_metric_calls=self.settings.max_metric_calls,
            seed=self.seed,
            display_progress_bar=False,
            raise_on_exception=False,
            logger=_EngineLogger(),
        )
        best = getattr(result, "best_candidate", None)
        if not best:
            logger.warning("GEPA 未产生候选（预算或收敛限制），本轮沿用当前 harness")
            return harness
        candidate = self._to_harness(best, harness)
        if candidate.model_dump() == harness.model_dump():
            return harness
        return candidate

    # -- component mapping ---------------------------------------------------- #

    def _to_candidate(self, harness: HarnessState) -> dict[str, str]:
        candidate = {"system_prompt": harness.system_prompt}
        if harness.output_format:
            candidate["output_format"] = harness.output_format
        for tool in harness.tools:
            candidate[f"tools.{tool.name}.description"] = tool.description
        return candidate

    def _to_harness(self, best: dict[str, str], base: HarnessState) -> HarnessState:
        data = base.model_dump()
        data["system_prompt"] = best.get("system_prompt", base.system_prompt)
        if best.get("output_format"):
            data["output_format"] = best["output_format"]
        for tool in data["tools"]:
            key = f"tools.{tool['name']}.description"
            if key in best:
                tool["description"] = best[key]
        return HarnessState.model_validate(data)


class _EngineLogger:
    """gepa's LoggerProtocol is just ``log(message)`` — route it to logging."""

    def log(self, message: str) -> None:
        logger.info("gepa: %s", message)


class _SyncReflectionBridge:
    """Adapt our async LLMClient to gepa's ``(str | list[dict]) -> str`` LM."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else list(prompt)
        )
        # gepa.optimize runs in a worker thread without a running loop, so a
        # dedicated event loop per call is safe here.
        response = asyncio.run(
            self.llm.complete(messages=messages, tools=None, response_format=None)
        )
        return response.content or ""


class _HarnessAdapter:
    """Evaluate GEPA candidates with HarnessLoop's executor + checker."""

    def __init__(
        self,
        *,
        tasks: list[Task],
        base_harness: HarnessState,
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        hypotheses: list[Hypothesis],
    ) -> None:
        self.tasks = tasks
        self.base_harness = base_harness
        self.executor = executor
        self.checker = checker
        self.hypotheses = hypotheses
        # gepa 0.1.4 accesses this attribute directly; None = use the engine's
        # default instruction-proposal flow built from our reflection_lm.
        self.propose_new_texts = None

    def evaluate(
        self,
        batch: list[Task],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        from gepa import EvaluationBatch

        harness = _candidate_to_harness(candidate, self.base_harness)
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        for task in batch:
            trajectory, result = asyncio.run(self._run_one(task, harness))
            outputs.append(trajectory.final_answer)
            scores.append(1.0 if result.passed else 0.0)
            if capture_traces:
                trajectories.append(self._record(task, trajectory, result))
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    async def _run_one(self, task: Task, harness: HarnessState) -> tuple[Any, Any]:
        trajectory = await self.executor.run(task, harness)
        result = await self.checker.check(task, trajectory)
        trajectory.outcome = "pass" if result.passed else "fail"
        return trajectory, result

    def _record(
        self, task: Task, trajectory: Any, result: Any
    ) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "question": task.question,
            "final_answer": trajectory.final_answer,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments, "issues": [
                    {"label": issue.label, "evidence": issue.evidence}
                    for issue in call.issues
                ]}
                for step in trajectory.steps
                for call in step.tool_calls
            ],
            "passed": result.passed,
            "signals": [
                {"label": signal.label, "evidence": signal.evidence}
                for signal in result.signals
            ],
            "state_diff": getattr(result, "state_diff", None),
        }

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        reflective: dict[str, list[dict[str, Any]]] = {}
        hint_lines = [
            f"- 已有假设（可参考）：{hypothesis.target}: {hypothesis.edit}"
            for hypothesis in self.hypotheses
        ]
        for component in components_to_update:
            entries: list[dict[str, Any]] = [
                entry
                for record in (eval_batch.trajectories or [])
                if not record.get("passed")
                for entry in self._entries_for_component(component, record)
            ]
            if hint_lines:
                entries.append({"Prior hypotheses": "\n".join(hint_lines)})
            reflective[component] = entries
        return reflective

    def _entries_for_component(
        self, component: str, record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if component.startswith("tools."):
            tool_name = component.split(".")[1]
            relevant = [
                issue
                for call in record.get("tool_calls", [])
                if call["name"] == tool_name
                for issue in call["issues"]
            ]
            base_signals = relevant or (
                record.get("signals", []) if not record.get("passed") else []
            )
        else:
            base_signals = record.get("signals", [])
        entries = [
            {
                "Input": record.get("question"),
                "Label": signal.get("label"),
                "Evidence": signal.get("evidence"),
                "StateDiff": record.get("state_diff"),
            }
            for signal in base_signals
        ]
        if not entries and not record.get("passed"):
            entries.append(
                {
                    "Input": record.get("question"),
                    "Label": "FAIL",
                    "Evidence": record.get("final_answer"),
                    "StateDiff": record.get("state_diff"),
                }
            )
        return entries


def _candidate_to_harness(
    candidate: dict[str, str], base: HarnessState
) -> HarnessState:
    data = base.model_dump()
    data["system_prompt"] = candidate.get("system_prompt", base.system_prompt)
    if candidate.get("output_format"):
        data["output_format"] = candidate["output_format"]
    for tool in data["tools"]:
        key = f"tools.{tool['name']}.description"
        if key in candidate:
            tool["description"] = candidate[key]
    return HarnessState.model_validate(data)
