"""API eval: state assertions against S1, with failure signals for attribution.

The agent has no golden result set; the task declares ``expected.assertions``
over the final environment state. Failed assertions become RESULT_MISMATCH
signals, tool-call issues stay generic (WRONG_TOOL / MISSING_PARAM /
EXTRA_PARAM), and every failure carries a rendered S0 -> S1 diff as evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harnessloop.eval.state_diff import diff_states, render_state_diff
from harnessloop.models import (
    Assertion,
    AttributionLabel,
    CheckResult,
    FailureSignal,
    Task,
    Trajectory,
)

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\]|\[([^=\[\]]+)=([^\[\]]*)\])?$")


class PathError(Exception):
    pass


def resolve_path(state: dict[str, Any], path: str) -> Any:
    """Resolve ``tasks[status=done][0].id``-style paths against a state dict."""
    node: Any = state
    for raw in path.split("."):
        segment = raw.strip()
        if not segment:
            continue
        match = _SEGMENT.match(segment)
        if not match:
            raise PathError(f"无法解析路径段 {segment!r}（path={path!r}）")
        key, index, filter_key, filter_value = match.groups()
        node = _descend_dict(node, key, path)
        if index is not None:
            node = _descend_list(node, int(index), path)
        elif filter_key is not None:
            node = _descend_filter(node, filter_key, filter_value, path)
    return node


def _descend_dict(node: Any, key: str, path: str) -> Any:
    if isinstance(node, dict):
        if key not in node:
            raise PathError(f"{path}: 缺少字段 {key!r}")
        return node[key]
    if isinstance(node, list):
        # Bare key over a list: map over elements that carry the key.
        return [item.get(key) for item in node if isinstance(item, dict) and key in item]
    raise PathError(f"{path}: {key!r} 需要对象，实际为 {type(node).__name__}")


def _descend_list(node: Any, index: int, path: str) -> Any:
    if not isinstance(node, list):
        raise PathError(f"{path}: 需要列表，实际为 {type(node).__name__}")
    if index >= len(node):
        raise PathError(f"{path}: 索引 {index} 超出范围（长度 {len(node)}）")
    return node[index]


def _descend_filter(node: Any, filter_key: str, filter_value: str, path: str) -> Any:
    if not isinstance(node, list):
        raise PathError(f"{path}: 过滤器需要列表，实际为 {type(node).__name__}")
    wanted = _coerce_scalar(filter_value)
    for item in node:
        if isinstance(item, dict) and item.get(filter_key) == wanted:
            return item
    raise PathError(f"{path}: 列表中没有 {filter_key}={filter_value!r} 的元素")


def _coerce_scalar(text: str) -> Any:
    stripped = text.strip().strip("'\"")
    try:
        return json.loads(stripped)
    except (ValueError, json.JSONDecodeError):
        return stripped


_MISSING = object()


def evaluate_assertion(
    state: dict[str, Any], assertion: Assertion
) -> tuple[bool, str]:
    """Evaluate one assertion; return (passed, evidence-of-actual)."""
    try:
        actual = resolve_path(state, assertion.path)
        missing = False
    except PathError as exc:
        actual, missing = str(exc), True

    if assertion.op == "exists":
        return (not missing), _actual_text(actual, missing)
    if assertion.op == "not_exists":
        return missing, _actual_text(actual, missing)

    if missing:
        return False, _actual_text(actual, missing)

    op = assertion.op
    if op == "eq":
        passed, detail = actual == assertion.value, f"实际值 {actual!r}"
    elif op == "ne":
        passed, detail = actual != assertion.value, f"实际值 {actual!r}"
    elif op == "contains":
        if isinstance(actual, (list, dict, str)):
            passed, detail = assertion.value in actual, f"实际值 {actual!r}"
        else:
            passed, detail = False, f"类型 {type(actual).__name__} 不支持 contains"
    elif op == "length_eq":
        passed, detail = _length(actual) == assertion.value, f"实际长度 {_length(actual)}"
    elif op == "length_ge":
        passed, detail = _length(actual) >= assertion.value, f"实际长度 {_length(actual)}"
    elif op == "length_le":
        passed, detail = _length(actual) <= assertion.value, f"实际长度 {_length(actual)}"
    else:  # pragma: no cover - Literal guards this
        passed, detail = False, f"未知断言 op {op!r}"
    return passed, detail


def _length(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return -1


def _actual_text(actual: Any, missing: bool) -> str:
    return f"路径缺失（{actual}）" if missing else f"实际值 {actual!r}"


def parse_assertions(expected: dict[str, Any] | None) -> list[Assertion]:
    if not expected:
        return []
    assertions: list[Assertion] = []
    for item in expected.get("assertions") or []:
        if isinstance(item, dict):
            try:
                assertions.append(Assertion.model_validate(item))
            except Exception:
                continue  # dataset validation should have caught this
    return assertions


class APIChecker:
    """Judge an API trajectory by the environment state it produced."""

    async def check(self, task: Task, trajectory: Trajectory) -> CheckResult:
        tool_signals = [
            FailureSignal(
                label=issue.label,
                evidence=issue.evidence,
                tool_name=issue.tool_name,
                param=issue.param,
            )
            for step in trajectory.steps
            for call in step.tool_calls
            for issue in call.issues
        ]
        signals, state_diff_text = self._evaluate(task, trajectory)
        passed = not signals
        all_signals = list(signals) if passed else list(signals) + tool_signals
        return CheckResult(
            task_id=task.task_id,
            passed=passed,
            signals=all_signals,
            state_diff=state_diff_text,
        )

    def _evaluate(
        self, task: Task, trajectory: Trajectory
    ) -> tuple[list[FailureSignal], str | None]:
        if trajectory.error and not trajectory.final_state:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.EXEC_ERROR,
                        evidence=f"执行未完成：{trajectory.error}",
                    )
                ],
                None,
            )

        s1 = trajectory.final_state
        if s1 is None:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.EXEC_ERROR,
                        evidence="工具运行时未返回最终状态（S1）",
                    )
                ],
                None,
            )

        s0 = task.initial_state or {}
        changes = diff_states(s0, s1)
        diff_text = render_state_diff(changes)

        signals: list[FailureSignal] = []

        schema = (task.expected or {}).get("response_schema")
        if schema and trajectory.final_answer is not None:
            signals.extend(self._schema_signals(schema, trajectory.final_answer))

        for assertion in parse_assertions(task.expected):
            passed, detail = evaluate_assertion(s1, assertion)
            if not passed:
                signals.append(
                    FailureSignal(
                        label=AttributionLabel.RESULT_MISMATCH,
                        evidence=(
                            f"状态断言失败：{assertion.path} {assertion.op} "
                            f"{assertion.value!r}；{detail}；状态差异：{diff_text}"
                        ),
                    )
                )
        return signals, diff_text

    def _schema_signals(self, schema: dict[str, Any], answer: Any) -> list[FailureSignal]:
        if not isinstance(answer, (dict, list)):
            return []  # textual summary answers are allowed when no JSON is required
        try:
            import jsonschema

            jsonschema.validate(answer, schema)
        except ImportError:  # pragma: no cover - optional dependency path
            return []
        except Exception as exc:
            return [
                FailureSignal(
                    label=AttributionLabel.FORMAT_VIOLATION,
                    evidence=f"最终答案不符合 response_schema：{str(exc).splitlines()[0]}",
                )
            ]
        return []
