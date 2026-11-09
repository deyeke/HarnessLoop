"""Markdown iteration report (5 required sections)."""

from __future__ import annotations

from harnessloop.models import (
    AttributionGroup,
    GateDecision,
    Hypothesis,
    IterationMetrics,
)
from harnessloop.optimize.diff import HarnessChange


def _score_table(history: list[IterationMetrics]) -> str:
    lines = [
        "| Iter | Pass@1 | Δ | Prompt tokens | Completion tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    previous: IterationMetrics | None = None
    for metrics in history:
        delta = metrics.delta_pp(previous)
        delta_text = "—" if previous is None else f"{delta:+.1f}pp"
        lines.append(
            f"| {metrics.iteration} | {metrics.passed}/{metrics.total} "
            f"({metrics.pass_rate * 100:.1f}%) | {delta_text} | "
            f"{metrics.prompt_tokens} | {metrics.completion_tokens} | {metrics.cost_usd:.4f} |"
        )
        previous = metrics
    return "\n".join(lines)


def _attribution_table(groups: list[AttributionGroup], total_failures: int) -> str:
    if not groups:
        return "本轮没有失败样本。"
    lines = [
        "| Label | Count | 占比 | facet / field_path | 样例证据 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        share = group.count / total_failures * 100 if total_failures else 0.0
        evidence = group.sample_evidence.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {group.label} | {group.count} | {share:.0f}% | "
            f"{group.facet.value} / `{group.field_path}` | {evidence} |"
        )
    return "\n".join(lines)


def _hypotheses_section(hypotheses: list[Hypothesis]) -> str:
    if not hypotheses:
        return "本轮没有生成假设（将使用规则模板兜底）。"
    blocks: list[str] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        rationale = ", ".join(item.task_id for item in hypothesis.rationale) or "—"
        blocks.append(
            f"### H{index}: {hypothesis.target.value} / {hypothesis.edit.field_path}\n\n"
            f"- 方向：`{hypothesis.edit.kind}`\n"
            f"- 内容：{hypothesis.edit.content}\n"
            f"- 依据失败样本：{rationale}\n"
            f"- 可证伪条件：{hypothesis.falsifiable_check or '—'}"
        )
    return "\n\n".join(blocks)


def render_iteration_report(
    *,
    run_id: str,
    iteration: int,
    history: list[IterationMetrics],
    groups: list[AttributionGroup],
    hypotheses: list[Hypothesis],
    change: HarnessChange,
    decision: GateDecision | None,
    candidate_metrics: IterationMetrics,
    total_failures: int,
) -> str:
    if decision is None:
        decision_text = "基线迭代：未做修改。"
    else:
        verdict = "ACCEPT" if decision.accepted else "REJECT"
        decision_text = f"**{verdict}** — {decision.reason}"

    diff_text = change.diff.strip() if change.diff.strip() else "（无修改）"
    changed = ", ".join(change.changed_paths) if change.changed_paths else "—"

    return f"""# Iteration {iteration} Report — {run_id}

## 1. 分数

{_score_table(history)}

本轮评测：{candidate_metrics.passed}/{candidate_metrics.total}
（{candidate_metrics.pass_rate * 100:.1f}%），改动字段 {change.changed_fields} 个：
{changed}。

## 2. 归因分布

{_attribution_table(groups, total_failures)}

## 3. 本轮假设

{_hypotheses_section(hypotheses)}

## 4. Harness Diff

```diff
{diff_text}
```

## 5. 决策

{decision_text}
"""
