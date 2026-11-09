"""Minimal state differential for API tasks (M2-3).

API agents have no golden result set; instead we record the environment state
before (S0) and after (S1) the agent's tool calls and diff them structurally.
The diff feeds failure evidence (APIChecker), reports and the UI.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel


class StateChange(BaseModel):
    path: str
    kind: Literal["added", "removed", "changed"]
    old: Any = None
    new: Any = None


def _render(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def diff_states(s0: dict[str, Any], s1: dict[str, Any]) -> list[StateChange]:
    """Return a flat, path-addressed list of changes between two states."""
    changes: list[StateChange] = []
    _walk(s0, s1, "", changes)
    return changes


def _walk(left: Any, right: Any, path: str, out: list[StateChange]) -> None:
    label = path or "$"
    sentinel = object()
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right), key=str):
            old = left.get(key, sentinel)
            new = right.get(key, sentinel)
            child = f"{label}.{key}" if path else str(key)
            if old is sentinel:
                out.append(StateChange(path=child, kind="added", new=new))
            elif new is sentinel:
                out.append(StateChange(path=child, kind="removed", old=old))
            else:
                _walk(old, new, child, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            old = left[index] if index < len(left) else sentinel
            new = right[index] if index < len(right) else sentinel
            child = f"{label}[{index}]"
            if old is sentinel:
                out.append(StateChange(path=child, kind="added", new=new))
            elif new is sentinel:
                out.append(StateChange(path=child, kind="removed", old=old))
            else:
                _walk(old, new, child, out)
        return
    if left != right:
        out.append(StateChange(path=label, kind="changed", old=left, new=right))


def render_state_diff(
    changes: list[StateChange], *, max_changes: int = 12, max_chars: int = 800
) -> str:
    """Render a compact human-readable diff for evidence and reports."""
    if not changes:
        return "无变化"
    lines: list[str] = []
    for change in changes[:max_changes]:
        if change.kind == "added":
            lines.append(f"+ {change.path} = {_render(change.new)}")
        elif change.kind == "removed":
            lines.append(f"- {change.path} (原值 {_render(change.old)})")
        else:
            lines.append(
                f"~ {change.path}: {_render(change.old)} -> {_render(change.new)}"
            )
    if len(changes) > max_changes:
        lines.append(f"... 共 {len(changes)} 处变化")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text
