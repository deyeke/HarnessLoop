"""Meta-prompt mutator (M2-5), ``mutator: opik``.

Strategy (MetaPrompt style, self-contained): given the round's failure
evidence, an LLM meta-prompt generates K candidate ``system_prompt`` rewrites;
each candidate is mini-evaluated on a sample of tasks with the very same
executor + checker the loop uses, and the best candidate is returned for the
loop's gate to judge on the full set.

This module is the self-contained MetaPrompt strategy from v0.2. Since v0.4
``opik`` profiles default to the *native* adapter
(:mod:`harnessloop.optimize.opik_native`), which injects a multi-turn agent and
a pass/fail metric into ``opik_optimizer`` itself (ADR-0006); this module then
serves as its fallback when that package is unavailable or the search fails.
R7 still applies: the rule-template mutator remains the permanent fallback.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from harnessloop.config import OpikSettings
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

_META_PROMPT = """\
你是 prompt 优化引擎。下面给出一个 AI Agent 当前的 system_prompt、它在评测集上的失败证据，\
以及本轮已生成的假设。请生成 {n_candidates} 个互不相同、可直接替换的 system_prompt 重写版本。

要求：
1. 保留原 prompt 中仍然正确的约束，只针对失败证据做最小而明确的修改；
2. 每个版本都要直接解决至少一条失败证据，并避免引入新的矛盾；
3. 用中文输出，风格与原 prompt 一致（规则式编号列表）。

当前 system_prompt：
<current>
{system_prompt}
</current>

失败证据（label / 证据）：
<evidence>
{evidence}
</evidence>

已有假设：
<hypotheses>
{hypotheses}
</hypotheses>

只输出 JSON：{{"candidates": [{{"new_text": "...", "rationale": "..."}}]}}\
"""


class OpikMutator(Mutator):
    def __init__(
        self,
        llm: LLMClient,
        *,
        tasks: list[Task],
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        settings: OpikSettings | None = None,
        fallback: Mutator | None = None,
        seed: int = 0,
    ) -> None:
        self.llm = llm
        self.tasks = tasks
        self.executor = executor
        self.checker = checker
        self.settings = settings or OpikSettings()
        self.fallback = fallback
        self.seed = seed

    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        try:
            candidate = await self._propose(harness, hypotheses, attributions)
        except Exception:
            logger.exception("MetaPrompt 变异失败，回退到规则模板变异")
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

    # -- proposal -------------------------------------------------------------- #

    async def _propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState | None:
        evidence = "\n".join(
            f"- [{a.label}] {a.task_id}: {a.evidence}" for a in attributions
        )
        hint_text = "\n".join(f"- {h.target}: {h.edit}" for h in hypotheses) or "（无）"
        prompt = _META_PROMPT.format(
            n_candidates=self.settings.candidates,
            system_prompt=harness.system_prompt,
            evidence=evidence or "（无）",
            hypotheses=hint_text,
        )
        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            response_format=self._response_schema(),
        )
        rewrites = self._parse_rewrites(response.content)
        if not rewrites:
            return None

        sample = self._sample_tasks(attributions)
        scored = []
        for rewrite in rewrites:
            candidate = harness.model_copy(update={"system_prompt": rewrite})
            score = await self._mini_evaluate(candidate, sample)
            scored.append((score, rewrite, candidate))
        scored.sort(key=lambda item: (-item[0], len(item[1])))

        best_score, _, best = scored[0]
        base_score = await self._mini_evaluate(harness, sample)
        # Only propose when the winner actually beats the current prompt on
        # the sample; otherwise let the fallback (rules) handle this round.
        if best_score <= base_score:
            return None
        logger.info(
            "MetaPrompt 最佳候选样本分 %s/%s（基线 %s/%s）",
            best_score, len(sample), base_score, len(sample),
        )
        return best

    def _sample_tasks(self, attributions: list[Attribution]) -> list[Task]:
        failed_ids = {a.task_id for a in attributions}
        rng = random.Random(self.seed)
        pool = [t for t in self.tasks if t.task_id in failed_ids] or list(self.tasks)
        size = min(self.settings.sample_size, len(pool))
        return rng.sample(pool, size) if size < len(pool) else pool

    async def _mini_evaluate(self, harness: HarnessState, tasks: list[Task]) -> float:
        passed = 0
        for task in tasks:
            trajectory = await self.executor.run(task, harness)
            result = await self.checker.check(task, trajectory)
            if result.passed:
                passed += 1
        return passed / len(tasks) if tasks else 0.0

    def _parse_rewrites(self, content: str | None) -> list[str]:
        if not content:
            return []
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                return []
            try:
                payload = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return []
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        rewrites: list[str] = []
        for item in candidates or []:
            text = item.get("new_text") if isinstance(item, dict) else None
            if isinstance(text, str) and text.strip():
                rewrites.append(text.strip())
        return rewrites[: self.settings.candidates]

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "metaprompt_candidates",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "new_text": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["new_text", "rationale"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["candidates"],
                    "additionalProperties": False,
                },
            },
        }
