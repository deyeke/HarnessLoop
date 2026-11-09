"""SQL eval: parse → execute → compare, with failure signals for attribution."""

from __future__ import annotations

from harnessloop.attribution.sql_rules import describe_sql_diff
from harnessloop.eval.compare import CompareOptions, compare_result_sets
from harnessloop.models import (
    AttributionLabel,
    CheckResult,
    FailureSignal,
    Task,
    Trajectory,
)
from harnessloop.sqlutil import (
    SQLFormatError,
    SQLParseError,
    execute_query,
    extract_sql,
    read_only_statement,
)


class SQLChecker:
    def __init__(self, options: CompareOptions | None = None, max_rows: int = 1000) -> None:
        self.options = options or CompareOptions()
        self.max_rows = max_rows

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
        result_signals, predicted_sql = await self._evaluate(task, trajectory)
        passed = not result_signals
        signals = result_signals if passed else result_signals + tool_signals
        return CheckResult(
            task_id=task.task_id,
            passed=passed,
            predicted_sql=predicted_sql,
            signals=signals,
        )

    async def _evaluate(
        self, task: Task, trajectory: Trajectory
    ) -> tuple[list[FailureSignal], str | None]:
        if trajectory.error and not trajectory.final_answer:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.FORMAT_VIOLATION,
                        evidence=f"执行未完成：{trajectory.error}",
                    )
                ],
                None,
            )

        raw_answer = trajectory.final_answer
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.FORMAT_VIOLATION,
                        evidence="达到最大步数仍未给出最终答案",
                    )
                ],
                None,
            )

        sql = extract_sql(raw_answer)
        if sql is None:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.FORMAT_VIOLATION,
                        evidence="最终答案中没有找到 SQL",
                    )
                ],
                None,
            )

        try:
            statement = read_only_statement(sql)
        except SQLParseError as exc:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.PARSE_FAIL,
                        evidence=f"SQL 解析失败：{exc}",
                    )
                ],
                sql,
            )
        except SQLFormatError as exc:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.FORMAT_VIOLATION,
                        evidence=f"SQL 格式不合约束：{exc}",
                    )
                ],
                sql,
            )

        if not task.database:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.EXEC_ERROR,
                        evidence="任务缺少 database，无法执行验证",
                    )
                ],
                statement,
            )

        try:
            actual = await execute_query(task.database, statement, self.max_rows)
        except Exception as exc:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.EXEC_ERROR,
                        evidence=f"SQL 执行报错：{str(exc).splitlines()[0]}",
                    )
                ],
                statement,
            )

        if not task.gold_sql:
            return [], statement

        try:
            gold = await execute_query(task.database, task.gold_sql, self.max_rows)
        except Exception as exc:
            return (
                [
                    FailureSignal(
                        label=AttributionLabel.EXEC_ERROR,
                        evidence=f"gold_sql 执行失败（数据集问题）：{str(exc).splitlines()[0]}",
                    )
                ],
                statement,
            )

        matched, message = compare_result_sets(
            actual.rows,
            gold.rows,
            self.options,
            actual.columns,
            gold.columns,
        )
        if matched:
            return [], statement

        hints = describe_sql_diff(statement, task.gold_sql)
        evidence = message + (f"；结构差异：{'；'.join(hints)}" if hints else "")
        return (
            [
                FailureSignal(
                    label=AttributionLabel.RESULT_MISMATCH,
                    evidence=evidence,
                )
            ],
            statement,
        )
