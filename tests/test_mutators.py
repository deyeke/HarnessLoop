from __future__ import annotations

import json
import sys

import conftest

from harnessloop.config import GepaSettings, LoopConfig, OpikSettings
from harnessloop.eval.api_checker import APIChecker
from harnessloop.models import (
    Attribution,
    AttributionLabel,
    HarnessFacet,
    HarnessState,
    Task,
)
from harnessloop.optimize.gepa_mutator import GEPAMutator
from harnessloop.optimize.opik_mutator import OpikMutator
from harnessloop.optimize.rule_mutator import RuleTemplateMutator
from harnessloop.runtime.litellm_executor import LiteLLMExecutor
from harnessloop.store.workspace import load_harness


class OptimizerFakeLLM(conftest.ApiFakeLLM):
    """Extends ApiFakeLLM with gepa-reflection and metaprompt modes."""

    def __init__(self, *, reflection_prompt: str, metaprompt_reply: str) -> None:
        super().__init__()
        self.reflection_prompt = reflection_prompt
        self.metaprompt_reply = metaprompt_reply

    async def complete(self, *, messages, tools=None, response_format=None):
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        if "假设生成器" not in system and tools is None:
            if "The following are examples of different task inputs" in user:
                return self._text(self.reflection_prompt, 500, 120)
            if "prompt 优化引擎" in user:
                return self._text(self.metaprompt_reply, 300, 150)
        return await super().complete(
            messages=messages, tools=tools, response_format=response_format
        )

    @staticmethod
    def _text(content: str, prompt_tokens: int, completion_tokens: int):
        from harnessloop.llm import LLMResponse

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


GOOD_PROMPT = conftest.API_HARNESS["system_prompt"] + "\n" + conftest.UPDATE_TOOL_HINT
GEPA_REFLECTION_REPLY = f"```\n{GOOD_PROMPT}\n```"
META_REPLY = json.dumps(
    {
        "candidates": [
            {"new_text": GOOD_PROMPT, "rationale": "补齐 update_task 约束"},
            {"new_text": "无关的改写", "rationale": "无效候选"},
        ]
    },
    ensure_ascii=False,
)


def _api_tasks() -> list[Task]:
    return [Task.model_validate(record) for record in conftest.API_TASKS]


def _api_harness() -> HarnessState:
    return HarnessState.model_validate(conftest.API_HARNESS.copy())


async def test_gepa_mutator_real_engine_end_to_end(api_env, tmp_path) -> None:
    """LoopRunner + mutator=gepa converges using the real gepa engine."""
    from harnessloop.loop import LoopRunner

    llm = OptimizerFakeLLM(
        reflection_prompt=GEPA_REFLECTION_REPLY, metaprompt_reply=META_REPLY
    )
    config = _loop_config(api_env, tmp_path, mutator="gepa")
    runner = LoopRunner.build(config, llm=llm, run_id="run_gepa_e2e")
    assert isinstance(runner.mutator, GEPAMutator)

    summary = await runner.run()
    assert summary.accepted_iterations == [1]
    assert summary.final_pass_rate == 1.0
    accepted = (runner.run_dir.path / "iter_1" / "harness.json").read_text(encoding="utf-8")
    assert "update_task" in accepted


def _loop_config(api_env, out_dir, *, mutator: str):
    from harnessloop.config import (
        DatasetConfig,
        HarnessConfig,
        LoopSettings,
        OptSettings,
        ReportSettings,
    )

    return LoopConfig(
        model="gpt-4o-mini",
        dataset=DatasetConfig(path=str(api_env.dataset), kind="api"),
        harness=HarnessConfig(path=str(api_env.harness)),
        loop=LoopSettings(max_rounds=3, concurrency=2, seed=7),
        opt=OptSettings(
            max_hypotheses=2,
            mutator=mutator,
            gepa=GepaSettings(max_metric_calls=16, reflection_minibatch_size=2),
            # Pin the meta-prompt backend so this test exercises OpikMutator
            # itself; the native adapter has its own tests.
            opik=OpikSettings(candidates=2, sample_size=2, backend="metaprompt"),
        ),
        report=ReportSettings(out_dir=str(out_dir)),
    )


async def test_gepa_mutator_falls_back_without_gepa(api_env, monkeypatch) -> None:
    harness = _api_harness()
    attributions = [
        Attribution(
            task_id="api_close_task",
            label=AttributionLabel.RESULT_MISMATCH,
            evidence="状态断言失败：tasks[0].status eq 'done'",
            harness_facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
        )
    ]
    llm = OptimizerFakeLLM(
        reflection_prompt=GEPA_REFLECTION_REPLY, metaprompt_reply=META_REPLY
    )
    mutator = GEPAMutator(
        llm,
        tasks=_api_tasks(),
        executor=LiteLLMExecutor(llm),
        checker=APIChecker(),
        fallback=RuleTemplateMutator(),
    )
    monkeypatch.setitem(sys.modules, "gepa", None)  # import gepa -> ImportError
    result = await mutator.propose(harness, [], attributions)
    assert result.model_dump() != harness.model_dump()
    assert "先核对" in result.system_prompt or "语义" in result.system_prompt


async def test_opik_mutator_proposes_winning_rewrite(api_env, tmp_path) -> None:
    from harnessloop.loop import LoopRunner

    llm = OptimizerFakeLLM(
        reflection_prompt=GEPA_REFLECTION_REPLY, metaprompt_reply=META_REPLY
    )
    config = _loop_config(api_env, tmp_path / "runs", mutator="opik")
    runner = LoopRunner.build(config, llm=llm, run_id="run_opik_e2e")
    from harnessloop.optimize.opik_mutator import OpikMutator as _Opik

    assert isinstance(runner.mutator, _Opik)
    summary = await runner.run()
    assert summary.accepted_iterations == [1]
    assert summary.final_pass_rate == 1.0


async def test_opik_mutator_falls_back_when_no_candidate_wins(api_env) -> None:
    harness = _api_harness()
    attributions = [
        Attribution(
            task_id="api_close_task",
            label=AttributionLabel.RESULT_MISMATCH,
            evidence="状态断言失败",
            harness_facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
        )
    ]
    useless_reply = json.dumps(
        {"candidates": [{"new_text": harness.system_prompt, "rationale": "no change"}]},
        ensure_ascii=False,
    )
    llm = OptimizerFakeLLM(
        reflection_prompt=GEPA_REFLECTION_REPLY, metaprompt_reply=useless_reply
    )
    mutator = OpikMutator(
        llm,
        tasks=_api_tasks(),
        executor=LiteLLMExecutor(llm),
        checker=APIChecker(),
        fallback=RuleTemplateMutator(),
    )
    result = await mutator.propose(harness, [], attributions)
    # no candidate beat the baseline -> fell back to the rule mutator
    assert result.model_dump() != harness.model_dump()


def test_rule_fallback_harness_loads_roundtrip(api_env) -> None:
    harness = load_harness(str(api_env.harness))
    assert harness.system_prompt.startswith("你是任务管理助手")
