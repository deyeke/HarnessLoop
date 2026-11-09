"""Hypothesis generation: one LLM call, structured JSON, validated against the
current harness so that every hypothesis points at an editable field."""

from __future__ import annotations

import json
from typing import Any

from harnessloop.llm import LLMClient
from harnessloop.models import (
    Attribution,
    AttributionGroup,
    EditOp,
    HarnessFacet,
    HarnessState,
    Hypothesis,
)

_SYSTEM_PROMPT = """你是 HarnessLoop 的假设生成器（假设生成器）。
你的唯一职责是：根据失败归因，提出 1-3 条最小、可证伪的 harness 修改假设。

允许修改的 harness 面（仅限）：
- system_prompt：系统提示词
- tool_description：工具描述
- schema_field：tool JSON Schema 的参数名 / 描述 / 枚举值 / 必填项
- output_format：输出格式约束

严禁建议：修改模型权重、做 SFT/RL、改写 Agent 执行拓扑、引入外部教师模型做任务合成。

输出必须是严格 JSON（不要 markdown 代码块，不要解释）：
{"hypotheses": [{"target": "system_prompt|tool_description|schema_field|output_format",
  "field_path": "例如 system_prompt 或 tools.run_sql.description",
  "direction": "add|remove|rewrite",
  "content": "要写入 harness 的具体文本（必须可直接使用，不要写占位符）",
  "rationale_task_ids": ["失败任务 id"],
  "falsifiable_check": "可验证的预期，例如：LEFT JOIN 类子集通过率提升 >= 15pp 且其他子集不下降"}]}
"""

_DEFAULT_PATH: dict[HarnessFacet, str] = {
    HarnessFacet.SYSTEM_PROMPT: "system_prompt",
    HarnessFacet.OUTPUT_FORMAT: "output_format",
    HarnessFacet.TOOL_DESCRIPTION: "tools",
    HarnessFacet.SCHEMA_FIELD: "tools",
    HarnessFacet.FEW_SHOT: "few_shot_examples",
}


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(cleaned[start : end + 1])


class HypothesisGenerator:
    def __init__(
        self,
        llm: LLMClient,
        *,
        max_hypotheses: int = 3,
        attempts: int = 3,
        edit_ops: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.max_hypotheses = max_hypotheses
        self.attempts = attempts
        # Extra kinds a plugin registered (M3-2); they extend the allowed set.
        self.edit_ops = edit_ops or []

    async def generate(
        self,
        harness: HarnessState,
        groups: list[AttributionGroup],
        attributions: list[Attribution],
    ) -> list[Hypothesis]:
        if not groups:
            return []

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(harness, groups)},
        ]

        for _ in range(self.attempts):
            try:
                response = await self.llm.complete(
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception:  # provider failure → let the rule mutator take over
                return []
            try:
                payload = parse_json_object(response.content)
                hypotheses = self._build(payload, harness, attributions)
            except (ValueError, json.JSONDecodeError, TypeError):
                continue
            if hypotheses:
                return hypotheses[: self.max_hypotheses]
        return []

    def _system_prompt(self) -> str:
        if not self.edit_ops:
            return _SYSTEM_PROMPT
        return (
            _SYSTEM_PROMPT
            + "\n已注册的自定义编辑操作（direction 可取这些值）："
            + ", ".join(self.edit_ops)
            + "；结构化参数放在 payload 对象中。\n"
        )

    @staticmethod
    def _user_prompt(harness: HarnessState, groups: list[AttributionGroup]) -> str:
        harness_json = json.dumps(harness.model_dump(), ensure_ascii=False, indent=2)
        if len(harness_json) > 4000:
            harness_json = harness_json[:4000] + "\n... (truncated)"
        lines = [
            "当前 harness（JSON）：",
            harness_json,
            "",
            "失败归因（按出现次数排序）：",
        ]
        for index, group in enumerate(groups, start=1):
            lines.append(
                f"{index}. label={group.label} count={group.count} "
                f"facet={group.facet.value} field_path={group.field_path}"
            )
            lines.append(f"   task_ids={group.task_ids}")
            lines.append(f"   样例证据：{group.sample_evidence}")
        lines.append("")
        lines.append("请输出 1-3 条最小可证伪假设的 JSON。")
        return "\n".join(lines)

    def _build(
        self,
        payload: dict[str, Any],
        harness: HarnessState,
        attributions: list[Attribution],
    ) -> list[Hypothesis]:
        items = payload.get("hypotheses")
        if not isinstance(items, list):
            return []

        by_task: dict[str, list[Attribution]] = {}
        for attribution in attributions:
            by_task.setdefault(attribution.task_id, []).append(attribution)

        hypotheses: list[Hypothesis] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                target = HarnessFacet(str(item.get("target", "")).strip())
            except ValueError:
                continue
            direction = str(item.get("direction", "add")).strip().lower()
            if direction not in {"add", "remove", "rewrite"} | set(self.edit_ops):
                continue
            payload = item.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            content = str(item.get("content", "")).strip()
            if not content and not payload:
                continue

            field_path = str(item.get("field_path") or "").strip()
            if not field_path or not harness.resolve_path(field_path):
                field_path = _DEFAULT_PATH[target]
            if not harness.resolve_path(field_path):
                continue

            rationale_ids = item.get("rationale_task_ids") or []
            rationale = [
                attribution
                for task_id in rationale_ids
                if isinstance(task_id, str)
                for attribution in by_task.get(task_id, [])
            ][:5]

            hypotheses.append(
                Hypothesis(
                    target=target,
                    edit=EditOp(
                        kind=direction,
                        target=target,
                        field_path=field_path,
                        content=content,
                        payload=payload,
                    ),
                    rationale=rationale,
                    falsifiable_check=str(item.get("falsifiable_check", "")).strip(),
                )
            )
        return hypotheses
