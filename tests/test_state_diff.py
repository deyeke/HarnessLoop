from __future__ import annotations

from harnessloop.eval.state_diff import diff_states, render_state_diff


def test_diff_added_removed_changed() -> None:
    s0 = {"tasks": [{"id": 1, "status": "open"}], "next_id": 2}
    s1 = {"tasks": [{"id": 1, "status": "done"}, {"id": 2, "status": "open"}]}

    changes = diff_states(s0, s1)
    kinds = {(c.path, c.kind) for c in changes}
    assert ("tasks[0].status", "changed") in kinds
    assert ("tasks[1]", "added") in kinds
    assert ("next_id", "removed") in kinds


def test_diff_equal_states_is_empty() -> None:
    state = {"a": {"b": [1, 2, 3]}}
    assert diff_states(state, {"a": {"b": [1, 2, 3]}}) == []


def test_render_state_diff() -> None:
    changes = diff_states({"x": 1}, {"x": 2, "y": [1]})
    text = render_state_diff(changes)
    assert "~ x: 1 -> 2" in text
    assert "+ y = [1]" in text


def test_render_truncates_long_values() -> None:
    changes = diff_states({"x": "a"}, {"x": "word " * 200})
    text = render_state_diff(changes)
    assert "..." in text
