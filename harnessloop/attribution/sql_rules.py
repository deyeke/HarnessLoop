"""Structural SQL diff used as attribution evidence (sqlglot AST level)."""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def _tables(expression: exp.Expression) -> set[str]:
    return {table.name.lower() for table in expression.find_all(exp.Table) if table.name}


def _joins(expression: exp.Expression) -> list[str]:
    rendered: list[str] = []
    for join in expression.find_all(exp.Join):
        side = (join.args.get("side") or "").upper()
        kind = (join.args.get("kind") or "").upper()
        label = " ".join(part for part in (side, kind) if part)
        rendered.append(f"{label} JOIN".strip())
    return sorted(rendered)


def _aggregates(expression: exp.Expression) -> set[str]:
    names: set[str] = set()
    for node in expression.find_all(exp.AggFunc):
        names.add(node.key.upper())
    return names


def describe_sql_diff(predicted_sql: str, gold_sql: str) -> list[str]:
    """Return human-readable structural differences (max 5)."""
    try:
        predicted = sqlglot.parse_one(predicted_sql, read="sqlite")
        gold = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return []

    hints: list[str] = []

    predicted_tables = _tables(predicted)
    gold_tables = _tables(gold)
    if predicted_tables != gold_tables:
        hints.append(
            f"表引用不同：实际 {sorted(predicted_tables)}，期望 {sorted(gold_tables)}"
        )

    predicted_joins = _joins(predicted)
    gold_joins = _joins(gold)
    if predicted_joins != gold_joins:
        hints.append(f"JOIN 类型不同：实际 {predicted_joins or ['无']}，期望 {gold_joins or ['无']}")

    if bool(predicted.args.get("where")) != bool(gold.args.get("where")):
        hints.append(
            "WHERE 条件" + ("缺失" if gold.args.get("where") else "多余")
        )

    if bool(predicted.args.get("group")) != bool(gold.args.get("group")):
        hints.append("GROUP BY " + ("缺失" if gold.args.get("group") else "多余"))

    if bool(predicted.args.get("having")) != bool(gold.args.get("having")):
        hints.append("HAVING 条件" + ("缺失" if gold.args.get("having") else "多余"))

    if bool(predicted.args.get("distinct")) != bool(gold.args.get("distinct")):
        hints.append("DISTINCT " + ("缺失" if gold.args.get("distinct") else "多余"))

    predicted_aggregates = _aggregates(predicted)
    gold_aggregates = _aggregates(gold)
    if predicted_aggregates != gold_aggregates:
        hints.append(
            f"聚合函数不同：实际 {sorted(predicted_aggregates) or ['无']}，"
            f"期望 {sorted(gold_aggregates) or ['无']}"
        )

    if bool(predicted.args.get("limit")) != bool(gold.args.get("limit")):
        hints.append("LIMIT " + ("缺失" if gold.args.get("limit") else "多余"))

    return hints[:5]
