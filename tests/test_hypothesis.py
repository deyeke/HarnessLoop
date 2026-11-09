from __future__ import annotations

import json

from harnessloop.llm import LLMResponse
from harnessloop.models import (
    Attribution,
    AttributionGroup,
    AttributionLabel,
    HarnessFacet,
    HarnessState,
)
from harnessloop.optimize.hypothesis import HypothesisGenerator


class StubLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, *, messages, tools=None, response_format=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.content)


def _harness() -> HarnessState:
    return HarnessState(system_prompt="base", output_format=None)


def _groups() -> list[AttributionGroup]:
    return [
        AttributionGroup(
            label=AttributionLabel.RESULT_MISMATCH,
            count=2,
            task_ids=["t1", "t2"],
            facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            sample_evidence="JOIN 类型不同",
        )
    ]


def _attributions() -> list[Attribution]:
    return [
        Attribution(
            task_id="t1",
            label=AttributionLabel.RESULT_MISMATCH,
            evidence="JOIN 类型不同",
            harness_facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
        )
    ]


async def test_generates_validated_hypothesis() -> None:
    payload = {
        "hypotheses": [
            {
                "target": "system_prompt",
                "field_path": "does.not.exist",
                "direction": "add",
                "content": "使用 LEFT JOIN",
                "rationale_task_ids": ["t1"],
                "falsifiable_check": "子集通过率 +15pp",
            }
        ]
    }
    generator = HypothesisGenerator(StubLLM(json.dumps(payload)))
    hypotheses = await generator.generate(_harness(), _groups(), _attributions())
    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.edit.field_path == "system_prompt"  # invalid path replaced
    assert hypothesis.target == HarnessFacet.SYSTEM_PROMPT
    assert hypothesis.rationale[0].task_id == "t1"
    assert hypothesis.falsifiable_check


async def test_invalid_json_returns_empty() -> None:
    generator = HypothesisGenerator(StubLLM("not json at all"), attempts=1)
    hypotheses = await generator.generate(_harness(), _groups(), _attributions())
    assert hypotheses == []


async def test_invalid_target_is_skipped() -> None:
    payload = {"hypotheses": [{"target": "model_weights", "content": "x", "direction": "add"}]}
    generator = HypothesisGenerator(StubLLM(json.dumps(payload)))
    assert await generator.generate(_harness(), _groups(), _attributions()) == []


async def test_no_groups_skips_llm() -> None:
    llm = StubLLM("{}")
    generator = HypothesisGenerator(llm)
    assert await generator.generate(_harness(), [], []) == []
    assert llm.calls == 0
