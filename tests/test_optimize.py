from __future__ import annotations

from harnessloop.models import (
    Attribution,
    AttributionLabel,
    EditOp,
    GateDecision,
    HarnessFacet,
    HarnessState,
    Hypothesis,
    IterationMetrics,
    ToolSchema,
)
from harnessloop.optimize.diff import diff_harness
from harnessloop.optimize.gate import ParetoGate, ThresholdGate
from harnessloop.optimize.rule_mutator import RuleTemplateMutator


def _harness() -> HarnessState:
    return HarnessState(
        system_prompt="base prompt",
        tools=[ToolSchema(name="run_sql", description="run sql")],
        output_format=None,
    )


def _metrics(iteration: int, passed: int, total: int = 10) -> IterationMetrics:
    return IterationMetrics(
        iteration=iteration,
        total=total,
        passed=passed,
        pass_rate=passed / total,
    )


def _attribution() -> Attribution:
    return Attribution(
        task_id="t1",
        label=AttributionLabel.RESULT_MISMATCH,
        evidence="JOIN 类型不同：实际 ['JOIN']，期望 ['LEFT JOIN']",
        harness_facet=HarnessFacet.SYSTEM_PROMPT,
        field_path="system_prompt",
    )


async def test_fallback_rule_adds_left_join_hint() -> None:
    mutator = RuleTemplateMutator()
    harness = _harness()
    updated = await mutator.propose(harness, [], [_attribution()])
    assert "LEFT JOIN" in updated.system_prompt
    assert harness.system_prompt == "base prompt"  # input not mutated


async def test_fallback_is_idempotent() -> None:
    mutator = RuleTemplateMutator()
    once = await mutator.propose(_harness(), [], [_attribution()])
    twice = await mutator.propose(once, [], [_attribution()])
    assert once == twice


async def test_hypothesis_edit_is_applied() -> None:
    mutator = RuleTemplateMutator()
    hypothesis = Hypothesis(
        target=HarnessFacet.OUTPUT_FORMAT,
        edit=EditOp(
            kind="rewrite",
            target=HarnessFacet.OUTPUT_FORMAT,
            field_path="output_format",
            content="```sql\n<query>\n```",
        ),
        rationale=[_attribution()],
    )
    updated = await mutator.propose(_harness(), [hypothesis], [_attribution()])
    assert updated.output_format == "```sql\n<query>\n```"


def test_diff_harness_reports_fields_and_diff() -> None:
    previous = _harness()
    candidate = previous.apply_edit(
        EditOp(
            kind="add",
            target=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            content="hint",
        )
    )
    change = diff_harness(previous, candidate)
    assert change.changed_fields == 1
    assert change.changed_paths == ["system_prompt"]
    assert "hint" in change.diff


def test_diff_harness_empty() -> None:
    change = diff_harness(_harness(), _harness())
    assert change.is_empty
    assert change.diff == ""


def test_threshold_gate() -> None:
    gate = ThresholdGate(min_gain_pp=2.0)
    change = diff_harness(_harness(), _harness()).model_copy()
    accepted: GateDecision = gate.decide(_metrics(0, 5), _metrics(1, 6), change)
    assert accepted.accepted
    rejected = gate.decide(_metrics(0, 5), _metrics(1, 5), change)
    assert not rejected.accepted


def test_pareto_gate_accepts_minimal_tie() -> None:
    previous = _harness()
    candidate = previous.apply_edit(
        EditOp(
            kind="add",
            target=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            content="hint",
        )
    )
    change = diff_harness(previous, candidate)
    gate = ParetoGate(min_gain_pp=2.0, max_equal_change_fields=1)
    decision = gate.decide(_metrics(0, 5), _metrics(1, 5), change)
    assert decision.accepted


def test_pareto_gate_rejects_large_tie() -> None:
    previous = _harness()
    candidate = previous.apply_edit(
        EditOp(
            kind="add",
            target=HarnessFacet.SYSTEM_PROMPT,
            field_path="system_prompt",
            content="hint",
        )
    ).apply_edit(
        EditOp(
            kind="rewrite",
            target=HarnessFacet.OUTPUT_FORMAT,
            field_path="output_format",
            content="format",
        )
    )
    change = diff_harness(previous, candidate)
    gate = ParetoGate(min_gain_pp=2.0, max_equal_change_fields=1)
    decision = gate.decide(_metrics(0, 5), _metrics(1, 5), change)
    assert not decision.accepted


def test_pareto_gate_rejects_regression() -> None:
    change = diff_harness(_harness(), _harness())
    gate = ParetoGate(min_gain_pp=2.0)
    decision = gate.decide(_metrics(0, 6), _metrics(1, 5), change)
    assert not decision.accepted
    assert "下降" in decision.reason
