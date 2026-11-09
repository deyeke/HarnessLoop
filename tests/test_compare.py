from __future__ import annotations

from harnessloop.eval.compare import CompareOptions, compare_result_sets


def test_order_insensitive_by_default() -> None:
    matched, _ = compare_result_sets([[1], [2]], [[2], [1]])
    assert matched


def test_order_sensitive_option() -> None:
    matched, message = compare_result_sets(
        [[1], [2]], [[2], [1]], CompareOptions(order_sensitive=True)
    )
    assert not matched
    assert "第 1 行" in message


def test_float_tolerance() -> None:
    matched, _ = compare_result_sets([[0.30000000000000004]], [[0.3]])
    assert matched


def test_float_outside_tolerance() -> None:
    matched, _ = compare_result_sets([[0.31]], [[0.3]])
    assert not matched


def test_null_equality() -> None:
    assert compare_result_sets([[None]], [[None]])[0]
    assert not compare_result_sets([[None]], [[0]])[0]


def test_row_count_mismatch_message() -> None:
    matched, message = compare_result_sets([[1]], [[1], [2]])
    assert not matched
    assert "行数不一致" in message


def test_column_count_mismatch() -> None:
    matched, message = compare_result_sets(
        [[1, 2]], [[1]], actual_columns=["a", "b"], expected_columns=["a"]
    )
    assert not matched
    assert "列数不一致" in message


def test_column_name_comparison() -> None:
    options = CompareOptions(compare_column_names=True)
    matched, _ = compare_result_sets(
        [[1]], [[1]], options, actual_columns=["total"], expected_columns=["total"]
    )
    assert matched
    matched, message = compare_result_sets(
        [[1]], [[1]], options, actual_columns=["cnt"], expected_columns=["total"]
    )
    assert not matched
    assert "列名" in message
