"""Reference API tool runtime (M2-1/M2-2).

A self-contained, in-process "todo" mock API: the agent mutates an in-memory
state S0 -> S1 through tool calls, and the checker asserts against S1. No HTTP
server, no external service — the API line must stay zero-infrastructure.

Other domains (orders, billing, ...) can ship their own runtime implementing
the same ``execute(name, arguments) -> Any`` + ``snapshot()`` contract; the
executors only rely on that interface.
"""

from __future__ import annotations

import copy
import json
from typing import Any


class ApiToolRuntime:
    """Tool implementations for the built-in todo-domain mock API."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.state: dict[str, Any] = copy.deepcopy(initial_state or {})
        self.state.setdefault("tasks", [])
        self.state.setdefault("users", [])
        self.state.setdefault("next_task_id", 1)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    # -- entry point -------------------------------------------------------- #

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            if name == "list_tasks":
                return self._list_tasks(arguments)
            if name == "get_task":
                return self._get_task(arguments)
            if name == "create_task":
                return self._create_task(arguments)
            if name == "update_task":
                return self._update_task(arguments)
            if name == "delete_task":
                return self._delete_task(arguments)
            if name == "list_users":
                return {"users": copy.deepcopy(self.state["users"])}
        except Exception as exc:  # surfaced to the model as an observation
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"error": f"未知工具 {name}（可用: list_tasks/get_task/create_task/"
                        "update_task/delete_task/list_users）"}

    # -- helpers ------------------------------------------------------------ #

    def _find_task(self, task_id: Any) -> dict[str, Any] | None:
        wanted = _coerce_id(task_id)
        return next((t for t in self.state["tasks"] if t.get("id") == wanted), None)

    def _list_tasks(self, arguments: dict[str, Any]) -> dict[str, Any]:
        status = str(arguments.get("status") or "").strip()
        tasks = self.state["tasks"]
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        return {"tasks": copy.deepcopy(tasks), "count": len(tasks)}

    def _get_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = self._find_task(arguments.get("task_id"))
        if task is None:
            return {"error": f"任务不存在: task_id={arguments.get('task_id')}"}
        return {"task": copy.deepcopy(task)}

    def _create_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = str(arguments.get("title") or "").strip()
        if not title:
            return {"error": "create_task 缺少必填参数 title"}
        task = {
            "id": int(self.state["next_task_id"]),
            "title": title,
            "priority": str(arguments.get("priority") or "medium"),
            "status": str(arguments.get("status") or "open"),
            "assignee": str(arguments.get("assignee") or ""),
        }
        self.state["next_task_id"] = int(self.state["next_task_id"]) + 1
        self.state["tasks"].append(task)
        return {"task": copy.deepcopy(task)}

    def _update_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = self._find_task(arguments.get("task_id"))
        if task is None:
            return {"error": f"任务不存在: task_id={arguments.get('task_id')}"}
        updates = {
            key: str(arguments[key])
            for key in ("status", "priority", "assignee")
            if str(arguments.get(key) or "").strip()
        }
        if not updates:
            return {"error": "update_task 至少需要一个要更新的字段: status/priority/assignee"}
        task.update(updates)
        return {"task": copy.deepcopy(task)}

    def _delete_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = self._find_task(arguments.get("task_id"))
        if task is None:
            return {"error": f"任务不存在: task_id={arguments.get('task_id')}"}
        self.state["tasks"].remove(task)
        return {"deleted": True, "task_id": task["id"]}


def _coerce_id(value: Any) -> Any:
    """LLMs often send ids as strings; normalise numeric-looking values."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def serialize_state(state: dict[str, Any], max_chars: int = 1200) -> str:
    """Render S0 for the model prompt (truncated to keep prompts small)."""
    text = json.dumps(state, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "... (truncated)"
