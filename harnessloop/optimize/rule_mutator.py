"""v0.1 rule-template mutator.

Applies LLM-proposed hypotheses when they resolve; otherwise falls back to a
small deterministic rule set keyed by the dominant attribution label. This keeps
the closed loop working even when hypothesis generation is unavailable.
"""

from __future__ import annotations

from collections import Counter

from harnessloop.models import (
    Attribution,
    AttributionLabel,
    EditOp,
    HarnessFacet,
    HarnessState,
    Hypothesis,
)
from harnessloop.optimize.plugins import EditOpRegistry

LEFT_JOIN_HINT = (
    "涉及“列出全部 X（即使没有关联记录）”一类问题时，使用 LEFT JOIN；"
    "不要使用逗号连接或 INNER JOIN，否则主表记录会被过滤掉。"
)
SEMANTIC_HINT = (
    "生成 SQL 前先核对三点：JOIN 类型、WHERE 过滤条件、聚合粒度是否与问题语义一致。"
)
FORMAT_HINT = (
    "最终回答只包含一条可执行的 SQLite SQL，放在 ```sql 代码块中；"
    "不要附加解释，不要输出多条语句，不要使用 SQLite 不支持的方言语法。"
)
EXECUTION_HINT = (
    "使用数据库中真实存在的表名和列名；不确定时先调用 list_tables 与 describe_table 查看结构。"
)
PARAM_HINT = (
    "调用工具时必须提供 schema 中声明的全部必填参数；缺失参数时先补齐再调用，"
    "不要传入未声明的参数。"
)
TOOL_HINT = "调用工具前先阅读 tool description，选择语义最匹配的工具，不要臆造工具名。"


class RuleTemplateMutator:
    def __init__(
        self,
        allow_fallback: bool = True,
        registry: EditOpRegistry | None = None,
    ) -> None:
        self.allow_fallback = allow_fallback
        self.registry = registry

    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        candidate = harness
        applied = 0

        for hypothesis in hypotheses:
            edit = hypothesis.edit
            if self._already_present(candidate, edit):
                continue
            try:
                updated = candidate.apply_edit(edit, registry=self.registry)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if updated != candidate:
                candidate = updated
                applied += 1

        if applied or not self.allow_fallback:
            return candidate
        return self._fallback(harness, attributions)

    # -- fallback rules ---------------------------------------------------- #

    def _fallback(self, harness: HarnessState, attributions: list[Attribution]) -> HarnessState:
        if not attributions:
            return harness

        counts = Counter(attribution.label for attribution in attributions)
        label, _ = counts.most_common(1)[0]
        evidence = " ".join(
            attribution.evidence
            for attribution in attributions
            if attribution.label == label
        )
        edit = self._rule_for(label, evidence)
        if self._already_present(harness, edit):
            return harness
        try:
            updated = harness.apply_edit(edit, registry=self.registry)
        except (KeyError, IndexError, TypeError, ValueError):
            return harness
        return updated if updated != harness else harness

    @staticmethod
    def _rule_for(label: AttributionLabel, evidence: str) -> EditOp:
        if label == AttributionLabel.RESULT_MISMATCH:
            lower = evidence.lower()
            content = (
                LEFT_JOIN_HINT
                if ("left join" in lower or "join 类型不同" in evidence or "join" in lower)
                else SEMANTIC_HINT
            )
            return EditOp(
                kind="add",
                target=HarnessFacet.SYSTEM_PROMPT,
                field_path="system_prompt",
                content=content,
            )
        if label in (AttributionLabel.PARSE_FAIL, AttributionLabel.FORMAT_VIOLATION):
            return EditOp(
                kind="rewrite",
                target=HarnessFacet.OUTPUT_FORMAT,
                field_path="output_format",
                content=FORMAT_HINT,
            )
        if label == AttributionLabel.EXEC_ERROR:
            return EditOp(
                kind="add",
                target=HarnessFacet.SYSTEM_PROMPT,
                field_path="system_prompt",
                content=EXECUTION_HINT,
            )
        if label in (AttributionLabel.MISSING_PARAM, AttributionLabel.EXTRA_PARAM):
            return EditOp(
                kind="add",
                target=HarnessFacet.SYSTEM_PROMPT,
                field_path="system_prompt",
                content=PARAM_HINT,
            )
        return EditOp(
            kind="add",
            target=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            content=TOOL_HINT,
        )

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _already_present(harness: HarnessState, edit: EditOp) -> bool:
        try:
            current = harness.get_field(edit.field_path)
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        if isinstance(current, str):
            return bool(edit.content) and edit.content in current
        if isinstance(current, list):
            return edit.content in current
        return False
