"""Shared test fixtures: a tiny SQLite dataset + a scripted LLM.

The scripted LLM is what lets the closed loop run end-to-end with zero network
calls: it fails the "users without tasks" task until the harness mentions
LEFT JOIN.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from harnessloop.config import DEFAULT_HARNESS
from harnessloop.llm import LLMResponse, LLMToolCall

SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);
INSERT INTO users (id, name, country) VALUES
    (1, 'Alice', 'CN'),
    (2, 'Bob', 'US'),
    (3, 'Carol', 'JP');
INSERT INTO tasks (id, user_id, title, done) VALUES
    (1, 1, 'write report', 1),
    (2, 1, 'review pr', 0),
    (3, 2, 'deploy service', 1);
"""

TASKS = [
    {
        "task_id": "sql_count_users",
        "kind": "sql",
        "question": "数据库里一共有多少用户？",
        "database": "toy.db",
        "gold_sql": "SELECT COUNT(*) FROM users",
    },
    {
        "task_id": "sql_users_without_tasks",
        "kind": "sql",
        "question": "列出没有任何任务的用户姓名。",
        "database": "toy.db",
        "gold_sql": (
            "SELECT users.name FROM users "
            "LEFT JOIN tasks ON tasks.user_id = users.id "
            "WHERE tasks.id IS NULL"
        ),
    },
    {
        "task_id": "sql_done_tasks",
        "kind": "sql",
        "question": "已完成的任务有多少个？",
        "database": "toy.db",
        "gold_sql": "SELECT COUNT(*) FROM tasks WHERE done = 1",
    },
]

CORRECT_SQL = {
    "sql_count_users": "SELECT COUNT(*) FROM users",
    "sql_users_without_tasks": (
        "SELECT users.name FROM users "
        "LEFT JOIN tasks ON tasks.user_id = users.id "
        "WHERE tasks.id IS NULL"
    ),
    "sql_done_tasks": "SELECT COUNT(*) FROM tasks WHERE done = 1",
}

WRONG_SQL = {
    "sql_count_users": "SELECT COUNT(*) FROM users",
    "sql_users_without_tasks": (
        "SELECT users.name FROM users JOIN tasks ON tasks.user_id = users.id"
    ),
    "sql_done_tasks": "SELECT COUNT(*) FROM tasks WHERE done = 1",
}

LEFT_JOIN_HINT = (
    "涉及“列出全部 X（即使没有关联记录）”一类问题时，使用 LEFT JOIN，"
    "不要使用 INNER JOIN 或逗号连接，否则主表记录会被过滤掉。"
)


@pytest.fixture(scope="session")
def toy_env(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("toy_env")
    database = base / "toy.db"
    conn = sqlite3.connect(database)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

    dataset = base / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in TASKS) + "\n",
        encoding="utf-8",
    )

    harness = base / "harness.json"
    harness.write_text(
        json.dumps(DEFAULT_HARNESS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return SimpleNamespace(database=database, dataset=dataset, harness=harness, base=base)


class ChattyFakeLLM:
    """Scripted LLM for executor + hypothesis generator."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self.calls += 1
        system = next(
            (message["content"] for message in messages if message["role"] == "system"),
            "",
        )
        user = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )

        if "假设生成器" in system:
            payload = {
                "hypotheses": [
                    {
                        "target": "system_prompt",
                        "field_path": "system_prompt",
                        "direction": "add",
                        "content": LEFT_JOIN_HINT,
                        "rationale_task_ids": ["sql_users_without_tasks"],
                        "falsifiable_check": "无任务用户子集通过率提升 >= 15pp",
                    }
                ]
            }
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=400,
                completion_tokens=80,
                cost_usd=0.001,
            )

        if "没有" in user and "任务" in user:
            task_id = "sql_users_without_tasks"
        elif "已完成的任务" in user:
            task_id = "sql_done_tasks"
        else:
            task_id = "sql_count_users"

        if task_id == "sql_users_without_tasks" and "LEFT JOIN" not in system:
            sql = WRONG_SQL[task_id]
        else:
            sql = CORRECT_SQL[task_id]

        return LLMResponse(
            content=f"```sql\n{sql}\n```",
            prompt_tokens=120,
            completion_tokens=30,
            cost_usd=0.0005,
        )


@pytest.fixture
def fake_llm() -> ChattyFakeLLM:
    return ChattyFakeLLM()


# --------------------------------------------------------------------------- #
# API-line fixtures (v0.2): todo-domain mock API + scripted LLM
# --------------------------------------------------------------------------- #

API_STATE = {
    "users": [{"id": 1, "name": "Alice", "role": "admin"}],
    "tasks": [
        {
            "id": 1,
            "title": "写周报",
            "priority": "high",
            "status": "open",
            "assignee": "Alice",
        }
    ],
    "next_task_id": 2,
}

API_TASKS = [
    {
        "task_id": "api_close_task",
        "kind": "api",
        "question": "把 id 为 1 的任务状态更新为 done。",
        "initial_state": API_STATE,
        "expected": {
            "assertions": [{"op": "eq", "path": "tasks[0].status", "value": "done"}]
        },
    },
    {
        "task_id": "api_create_high",
        "kind": "api",
        "question": "为 Bob 创建一个标题为「准备评审」、优先级为 high 的任务。",
        "initial_state": {"users": [], "tasks": [], "next_task_id": 1},
        "expected": {
            "assertions": [
                {"op": "eq", "path": "tasks[title=准备评审].priority", "value": "high"},
                {"op": "eq", "path": "tasks[title=准备评审].assignee", "value": "Bob"},
                {"op": "eq", "path": "tasks[title=准备评审].status", "value": "open"},
            ]
        },
    },
]

API_HARNESS = {
    "system_prompt": (
        "你是任务管理助手。通过工具调用完成用户对任务系统的操作，完成后简要总结。"
    ),
    "tools": [
        {
            "name": "list_tasks",
            "description": "列出任务，可按 status 过滤。",
            "parameters": {
                "type": "object",
                "properties": {"status": {"type": "string", "description": "如 open/done"}},
                "required": [],
            },
        },
        {
            "name": "get_task",
            "description": "按 id 查看单个任务。",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
        {
            "name": "create_task",
            "description": "创建新任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "assignee": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        {
            "name": "update_task",
            "description": "更新任务的 status / priority / assignee。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "status": {"type": "string"},
                    "priority": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "delete_task",
            "description": "删除任务。",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
        {
            "name": "list_users",
            "description": "列出用户。",
            "parameters": {"type": "object", "properties": {}},
        },
    ],
    "few_shot_examples": [],
    "output_format": "",
}

UPDATE_TOOL_HINT = (
    "修改已有任务时必须使用 update_task 工具（传入 task_id 与新的 status），"
    "不要通过 create_task 重复创建来表达修改。"
)


@pytest.fixture(scope="session")
def api_env(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("api_env")
    dataset = base / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in API_TASKS) + "\n",
        encoding="utf-8",
    )
    harness = base / "harness.json"
    harness.write_text(
        json.dumps(API_HARNESS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return SimpleNamespace(dataset=dataset, harness=harness, base=base)


class ApiFakeLLM:
    """Scripted LLM for the API closed loop.

    api_close_task fails until the harness tells the model to use update_task;
    api_create_high always passes.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self.calls += 1
        system = next(
            (message["content"] for message in messages if message["role"] == "system"),
            "",
        )
        user = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        saw_tool_result = any(message.get("role") == "tool" for message in messages)

        if "假设生成器" in system:
            payload = {
                "hypotheses": [
                    {
                        "target": "system_prompt",
                        "field_path": "system_prompt",
                        "direction": "add",
                        "content": UPDATE_TOOL_HINT,
                        "rationale_task_ids": ["api_close_task"],
                        "falsifiable_check": "api_close_task 的 tasks[0].status 断言通过",
                    }
                ]
            }
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                prompt_tokens=350,
                completion_tokens=60,
                cost_usd=0.0008,
            )

        if "准备评审" in user:
            if saw_tool_result:
                return LLMResponse(content="已创建任务。", prompt_tokens=90, completion_tokens=12)
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        name="create_task",
                        arguments={"title": "准备评审", "priority": "high", "assignee": "Bob"},
                    )
                ],
                prompt_tokens=110,
                completion_tokens=25,
            )

        # api_close_task
        if UPDATE_TOOL_HINT in system:
            if saw_tool_result:
                return LLMResponse(content="任务状态已更新。", prompt_tokens=95, completion_tokens=12)
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(name="update_task", arguments={"task_id": "1", "status": "done"})
                ],
                prompt_tokens=115,
                completion_tokens=22,
            )
        if saw_tool_result:
            return LLMResponse(content="已处理。", prompt_tokens=90, completion_tokens=10)
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    name="create_task",
                    arguments={"title": "写周报", "priority": "high", "assignee": "Alice"},
                )
            ],
            prompt_tokens=110,
            completion_tokens=25,
        )


@pytest.fixture
def api_fake_llm() -> ApiFakeLLM:
    return ApiFakeLLM()

