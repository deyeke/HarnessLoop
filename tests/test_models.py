from __future__ import annotations

import pytest

from harnessloop.models import (
    EditOp,
    HarnessFacet,
    HarnessState,
    ToolSchema,
)


@pytest.fixture
def harness() -> HarnessState:
    return HarnessState(
        system_prompt="base prompt",
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
        output_format=None,
    )


def test_resolve_paths(harness: HarnessState) -> None:
    assert harness.resolve_path("system_prompt")
    assert harness.resolve_path("output_format")
    assert harness.resolve_path("tools")
    assert harness.resolve_path("tools.run_sql.description")
    assert harness.resolve_path("tools.run_sql.parameters.required")
    assert not harness.resolve_path("tools.missing.description")
    assert not harness.resolve_path("nope")


def test_add_to_system_prompt_returns_new_snapshot(harness: HarnessState) -> None:
    updated = harness.apply_edit(
        EditOp(
            kind="add",
            target=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            content="新增提示",
        )
    )
    assert "新增提示" in updated.system_prompt
    assert harness.system_prompt == "base prompt"  # input untouched


def test_append_required_parameter(harness: HarnessState) -> None:
    updated = harness.apply_edit(
        EditOp(
            kind="add",
            target=HarnessFacet.SCHEMA_FIELD,
            field_path="tools.run_sql.parameters.required",
            content="limit",
        )
    )
    assert updated.tools[0].parameters["required"] == ["sql", "limit"]


def test_edit_tool_description(harness: HarnessState) -> None:
    updated = harness.apply_edit(
        EditOp(
            kind="rewrite",
            target=HarnessFacet.TOOL_DESCRIPTION,
            field_path="tools.run_sql.description",
            content="执行只读查询",
        )
    )
    assert updated.tools[0].description == "执行只读查询"


def test_unknown_path_raises(harness: HarnessState) -> None:
    with pytest.raises(KeyError):
        harness.apply_edit(
            EditOp(
                kind="add",
                target=HarnessFacet.SYSTEM_PROMPT,
                field_path="tools.unknown.description",
                content="x",
            )
        )
