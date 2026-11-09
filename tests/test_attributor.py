from __future__ import annotations

from harnessloop.attribution.attributor import Attributor
from harnessloop.models import (
    AttributionLabel,
    CheckResult,
    FailureSignal,
    HarnessFacet,
    HarnessState,
    ToolSchema,
)


def _harness() -> HarnessState:
    return HarnessState(
        system_prompt="base",
        tools=[
            ToolSchema(
                name="run_sql",
                description="run sql",
                parameters={
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            )
        ],
    )


def test_missing_param_maps_to_schema_field_path() -> None:
    attributor = Attributor()
    result = CheckResult(
        task_id="t1",
        passed=False,
        signals=[
            FailureSignal(
                label=AttributionLabel.MISSING_PARAM,
                evidence="第 2 步调用 run_sql 缺少必填参数 sql",
                tool_name="run_sql",
                param="sql",
            )
        ],
    )
    attributions = attributor.attribute([result], _harness())
    assert len(attributions) == 1
    attribution = attributions[0]
    assert attribution.harness_facet == HarnessFacet.SCHEMA_FIELD
    assert attribution.field_path == "tools.run_sql.parameters.required"


def test_unknown_tool_falls_back_to_tools_list() -> None:
    attributor = Attributor()
    result = CheckResult(
        task_id="t2",
        passed=False,
        signals=[
            FailureSignal(
                label=AttributionLabel.WRONG_TOOL,
                evidence="调用了未声明的工具 search",
                tool_name="search",
            )
        ],
    )
    attribution = attributor.attribute([result], _harness())[0]
    assert attribution.harness_facet == HarnessFacet.TOOL_DESCRIPTION
    assert attribution.field_path == "tools"


def test_every_field_path_is_resolvable() -> None:
    attributor = Attributor()
    labels = list(AttributionLabel)
    results = [
        CheckResult(
            task_id=f"t{index}",
            passed=False,
            signals=[FailureSignal(label=label, evidence=f"evidence {label.value}")],
        )
        for index, label in enumerate(labels)
    ]
    harness = _harness()
    for attribution in attributor.attribute(results, harness):
        assert attribution.field_path is not None
        assert harness.resolve_path(attribution.field_path)


def test_aggregate_counts_and_task_ids() -> None:
    attributor = Attributor()
    results = [
        CheckResult(
            task_id=f"t{index}",
            passed=False,
            signals=[
                FailureSignal(
                    label=AttributionLabel.RESULT_MISMATCH,
                    evidence=f"mismatch {index}",
                )
            ],
        )
        for index in range(3)
    ]
    groups = attributor.aggregate(attributor.attribute(results, _harness()))
    assert len(groups) == 1
    assert groups[0].count == 3
    assert groups[0].task_ids == ["t0", "t1", "t2"]
    assert groups[0].field_path == "system_prompt"
