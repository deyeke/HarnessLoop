"""M3 extension points: attribution rules and edit ops loaded from a profile."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import conftest
import pytest

from harnessloop.attribution.attributor import Attributor
from harnessloop.attribution.plugins import AttributionRegistry, AttributionRule
from harnessloop.config import load_config
from harnessloop.llm import LLMResponse, LLMToolCall
from harnessloop.loop import LoopRunner
from harnessloop.models import (
    AttributionLabel,
    CheckResult,
    EditOp,
    FailureSignal,
    HarnessFacet,
    HarnessState,
)
from harnessloop.optimize.plugins import EditOpPlugin, EditOpRegistry
from harnessloop.plugins import PluginError, PluginVersionError, load_plugin

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"
MARKER = "status 只能取 open 或 done"


def test_plugin_version_mismatch_is_rejected() -> None:
    registry = AttributionRegistry()
    with pytest.raises(PluginVersionError):
        registry.register(
            AttributionRule(name="future", locate=lambda signal, harness: None, api_version="2")
        )


def test_edit_op_cannot_shadow_builtin() -> None:
    registry = EditOpRegistry()
    plugin = EditOpPlugin(name="add", apply=lambda harness, edit: harness)
    with pytest.raises(PluginError):
        registry.register(plugin)


def test_example_rule_routes_to_tool_description() -> None:
    rule = load_plugin(f"{EXAMPLES / 'rules.py'}:status_assertion_to_update_tool")
    registry = AttributionRegistry()
    registry.register(rule)
    harness = HarnessState.model_validate(conftest.API_HARNESS)

    signal = FailureSignal(
        label=AttributionLabel.RESULT_MISMATCH,
        evidence="状态断言失败：tasks[0].status eq 'done'；实际 open",
    )
    located = registry.resolve(signal, harness)
    assert located == (HarnessFacet.TOOL_DESCRIPTION, "tools.update_task.description")


def test_unresolvable_rule_path_falls_back_to_default() -> None:
    registry = AttributionRegistry()
    registry.register(
        AttributionRule(
            name="broken",
            locate=lambda signal, harness: (HarnessFacet.SYSTEM_PROMPT, "no.such.path"),
            priority=1,
        )
    )
    attributor = Attributor(registry=registry)
    harness = HarnessState.model_validate(conftest.API_HARNESS)
    result = CheckResult(
        task_id="api_close_task",
        passed=False,
        signals=[
            FailureSignal(
                label=AttributionLabel.RESULT_MISMATCH,
                evidence="tasks[0].status 期望 done 实际 open",
            )
        ],
    )
    attributions = attributor.attribute([result], harness)
    assert attributions[0].field_path == "system_prompt"
    assert attributions[0].harness_facet is HarnessFacet.SYSTEM_PROMPT


def test_example_edit_op_tightens_enum() -> None:
    plugin = load_plugin(f"{EXAMPLES / 'edit_ops.py'}:set_enum")
    registry = EditOpRegistry()
    registry.register(plugin)
    harness = HarnessState.model_validate(conftest.API_HARNESS)

    edit = EditOp(
        kind="set_enum",
        target=HarnessFacet.SCHEMA_FIELD,
        field_path="tools.update_task.parameters.properties.status",
        payload={
            "tool": "update_task",
            "param": "status",
            "enum": ["open", "done"],
            "description": MARKER,
        },
    )
    updated = harness.apply_edit(edit, registry=registry)
    assert updated != harness

    status = next(
        tool for tool in updated.tools if tool.name == "update_task"
    ).parameters["properties"]["status"]
    assert status["enum"] == ["open", "done"]
    assert status["description"] == MARKER


def test_example_label_reclassifies_noop_failure() -> None:
    from harnessloop.attribution.labels import get_registry, load_label_plugins

    loaded = load_label_plugins([str(EXAMPLES / "labels.py")])
    # The module is cached, so a second load reports no new registrations.
    assert not loaded or [spec.name for spec in loaded] == ["NOOP_ACTION"]

    registry = get_registry()
    assert registry.get("NOOP_ACTION") is not None
    harness = HarnessState.model_validate(conftest.API_HARNESS)
    noop = FailureSignal(
        label=AttributionLabel.RESULT_MISMATCH,
        evidence="状态断言失败：tasks[0].status eq 'done'；实际 'open'；状态差异：无变化",
    )
    assert registry.classify(noop, harness) == "NOOP_ACTION"

    # Explicit attribution rules outrank label defaults, so use a bare rule
    # registry here to observe what the label itself claims.
    attributor = Attributor(registry=AttributionRegistry(), labels=registry)
    result = CheckResult(task_id="api_close_task", passed=False, signals=[noop])
    attribution = attributor.attribute([result], harness)[0]
    assert attribution.label == "NOOP_ACTION"
    assert attribution.harness_facet is HarnessFacet.SYSTEM_PROMPT
    assert attribution.field_path == "system_prompt"

    # A signal the classifier does not accept keeps its built-in label.
    other = FailureSignal(
        label=AttributionLabel.RESULT_MISMATCH,
        evidence="状态断言失败：tasks[0].status eq 'done'；实际 'open'；状态差异：~ tasks[0].status",
    )
    assert registry.classify(other, harness) is None
    attribution = attributor.attribute(
        [CheckResult(task_id="api_close_task", passed=False, signals=[other])], harness
    )[0]
    assert attribution.label == AttributionLabel.RESULT_MISMATCH


def test_label_registry_rejects_builtin_and_duplicate() -> None:
    from harnessloop.attribution.labels import AttributionLabelSpec, LabelRegistry

    registry = LabelRegistry()
    with pytest.raises(PluginError):
        registry.register(
            AttributionLabelSpec(
                name="RESULT_MISMATCH",
                facet=HarnessFacet.SYSTEM_PROMPT,
                field_path="system_prompt",
            )
        )
    spec = AttributionLabelSpec(
        name="NOOP_ACTION", facet=HarnessFacet.SYSTEM_PROMPT, field_path="system_prompt"
    )
    registry.register(spec)
    with pytest.raises(PluginError):
        registry.register(
            AttributionLabelSpec(
                name="NOOP_ACTION",
                facet=HarnessFacet.OUTPUT_FORMAT,
                field_path="output_format",
            )
        )
    assert registry.get("NOOP_ACTION") is spec


def test_custom_label_with_stale_path_stays_actionable() -> None:
    from harnessloop.attribution.labels import AttributionLabelSpec, LabelRegistry

    registry = LabelRegistry()
    registry.register(
        AttributionLabelSpec(
            name="STALE",
            facet=HarnessFacet.SCHEMA_FIELD,
            field_path="tools.nope.parameters",
        )
    )
    harness = HarnessState.model_validate(conftest.API_HARNESS)
    signal = FailureSignal(label="STALE", evidence="x")
    attribution = Attributor(registry=AttributionRegistry(), labels=registry).attribute(
        [CheckResult(task_id="t", passed=False, signals=[signal])], harness
    )[0]
    assert attribution.label == "STALE"
    assert attribution.harness_facet is HarnessFacet.SYSTEM_PROMPT
    assert attribution.field_path == "system_prompt"


def test_label_version_mismatch_and_broken_classifier() -> None:
    from harnessloop.attribution.labels import AttributionLabelSpec, LabelRegistry

    registry = LabelRegistry()
    with pytest.raises(PluginVersionError):
        registry.register(
            AttributionLabelSpec(
                name="FUTURE",
                facet=HarnessFacet.SYSTEM_PROMPT,
                field_path="system_prompt",
                api_version="2",
            )
        )

    def boom(signal, harness):
        raise RuntimeError("plugin bug")

    registry.register(
        AttributionLabelSpec(
            name="BROKEN",
            facet=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            classify=boom,
        )
    )
    signal = FailureSignal(label=AttributionLabel.RESULT_MISMATCH, evidence="x")
    assert registry.classify(signal, HarnessState.model_validate(conftest.API_HARNESS)) is None


def test_profile_plugins_drive_a_closed_loop(tmp_path: Path, api_env) -> None:
    config = load_config(EXAMPLES / "config.yaml")
    # Point the profile at the scripted fixture dataset, keep its plugin specs.
    config.dataset.path = str(api_env.dataset)
    config.harness.path = str(api_env.harness)
    config.report.out_dir = str(tmp_path / "runs")
    assert Path(config.plugins.attribution[0]).name == "rules.py"
    assert Path(config.plugins.attribution[0]).is_absolute()
    assert Path(config.plugins.edit_ops[0]).name == "edit_ops.py"

    runner = LoopRunner.build(config, llm=PluginFakeLLM(), run_id="plugin_loop")
    summary = asyncio.run(runner.run())

    assert summary.final_pass_rate == 1.0
    assert summary.accepted_iterations
    candidate = json.loads(
        (runner.run_dir.iter_dir(1) / "candidate_harness.json").read_text(encoding="utf-8")
    )
    status = next(
        tool for tool in candidate["tools"] if tool["name"] == "update_task"
    )["parameters"]["properties"]["status"]
    assert status["enum"] == ["open", "done"]
    # The plugin rule claimed the failure, so the attribution points at the tool.
    attributions = [
        json.loads(line)
        for line in (runner.run_dir.iter_dir(0) / "attribution.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    close = next(item for item in attributions if item["task_id"] == "api_close_task")
    assert close["field_path"] == "tools.update_task.description"


class PluginFakeLLM:
    """Fails api_close_task until the plugin's enum edit lands in the harness."""

    async def complete(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        system = next(
            (message["content"] for message in messages if message["role"] == "system"), ""
        )
        user = next(
            (
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            ),
            "",
        )
        saw_tool_result = any(message.get("role") == "tool" for message in messages)

        if "假设生成器" in system:
            payload = {
                "hypotheses": [
                    {
                        "target": "schema_field",
                        "field_path": "tools.update_task.parameters.properties.status",
                        "direction": "set_enum",
                        "payload": {
                            "tool": "update_task",
                            "param": "status",
                            "enum": ["open", "done"],
                            "description": MARKER,
                        },
                        "rationale_task_ids": ["api_close_task"],
                        "falsifiable_check": "api_close_task 的 tasks[0].status 断言通过",
                    }
                ]
            }
            return LLMResponse(content=json.dumps(payload, ensure_ascii=False))

        if "准备评审" in user:
            if saw_tool_result:
                return LLMResponse(content="已创建任务。")
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        name="create_task",
                        arguments={"title": "准备评审", "priority": "high", "assignee": "Bob"},
                    )
                ]
            )

        tightened = MARKER in json.dumps(tools or [], ensure_ascii=False)
        if tightened and not saw_tool_result:
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(name="update_task", arguments={"task_id": 1, "status": "done"})
                ]
            )
        if saw_tool_result:
            return LLMResponse(content="已处理。")
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    name="create_task",
                    arguments={"title": "写周报", "priority": "high", "assignee": "Alice"},
                )
            ]
        )
