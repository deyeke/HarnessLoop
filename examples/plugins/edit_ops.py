"""Example EditOp plugin (M3-2): tighten a tool parameter's enum.

The three built-in ops can only append/replace scalars, so editing
``tools.<name>.parameters.properties.<param>.enum`` needs a custom op. Register
it from a profile and any hypothesis may use ``kind: set_enum``:

    plugins:
      edit_ops:
        - edit_ops.py
"""

from __future__ import annotations

from typing import Any

from harnessloop.models import EditOp, HarnessState
from harnessloop.optimize.plugins import edit_op


@edit_op("set_enum", description="设置某个 tool 参数的 enum 与描述")
def set_enum(harness: HarnessState, edit: EditOp) -> HarnessState:
    tool_name = edit.payload.get("tool")
    param = edit.payload.get("param")
    values = edit.payload.get("enum")
    if not isinstance(tool_name, str) or not isinstance(param, str):
        raise ValueError("set_enum requires payload.tool and payload.param")
    if not isinstance(values, list) or not values:
        raise ValueError("set_enum requires a non-empty payload.enum list")

    data: dict[str, Any] = harness.model_dump()
    for tool in data.get("tools", []):
        if tool.get("name") != tool_name:
            continue
        properties = tool.setdefault("parameters", {}).setdefault("properties", {})
        prop = properties.setdefault(param, {"type": "string"})
        prop["enum"] = list(values)
        description = edit.payload.get("description")
        if isinstance(description, str) and description:
            prop["description"] = description
        return HarnessState.model_validate(data)
    raise KeyError(f"set_enum: tool {tool_name!r} not found")
