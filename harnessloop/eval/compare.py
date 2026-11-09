"""Result-set comparison.

We compare executed result sets, never SQL strings: order, NULL and float
semantics are handled explicitly.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel


class CompareOptions(BaseModel):
    order_sensitive: bool = False
    float_tolerance: float = 1e-6
    compare_column_names: bool = False
    column_order_sensitive: bool = False


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return str(left).strip() == str(right).strip()


def _rows_equal(left: list[Any], right: list[Any], tolerance: float) -> bool:
    if len(left) != len(right):
        return False
    return all(_values_equal(a, b, tolerance) for a, b in zip(left, right, strict=False))


def compare_result_sets(
    actual: list[list[Any]],
    expected: list[list[Any]],
    options: CompareOptions | None = None,
    actual_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> tuple[bool, str]:
    options = options or CompareOptions()

    if actual_columns is not None and expected_columns is not None:
        if len(actual_columns) != len(expected_columns):
            return False, f"列数不一致：实际 {len(actual_columns)}，期望 {len(expected_columns)}"
        if options.compare_column_names and not options.column_order_sensitive:
            if {c.lower() for c in actual_columns} != {c.lower() for c in expected_columns}:
                return (
                    False,
                    f"列名不一致：实际 {actual_columns}，期望 {expected_columns}",
                )
        elif options.compare_column_names and actual_columns != expected_columns:
            return False, f"列名/列序不一致：实际 {actual_columns}，期望 {expected_columns}"

    if len(actual) != len(expected):
        return False, f"行数不一致：实际 {len(actual)} 行，期望 {len(expected)} 行"

    if options.order_sensitive:
        for index, (actual_row, expected_row) in enumerate(
            zip(actual, expected, strict=False)
        ):
            if not _rows_equal(actual_row, expected_row, options.float_tolerance):
                return (
                    False,
                    f"第 {index + 1} 行不一致：实际 {actual_row}，期望 {expected_row}",
                )
        return True, "结果一致"

    unmatched = list(range(len(expected)))
    for actual_row in actual:
        for position in unmatched:
            if _rows_equal(actual_row, expected[position], options.float_tolerance):
                unmatched.remove(position)
                break
        else:
            return False, f"多出或不匹配的行：实际 {actual_row}"

    if unmatched:
        return False, f"缺少期望的行：{expected[unmatched[0]]}"
    return True, "结果一致（忽略顺序）"
