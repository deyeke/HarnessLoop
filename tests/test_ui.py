from __future__ import annotations

from pathlib import Path

import pytest

from harnessloop.ui.gradio_app import attribution_summary, build_demo, list_runs, load_run

RUNS_DIR = Path(__file__).resolve().parent.parent / "examples" / "sql_golden" / "runs"


@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="example runs not present")
def test_list_runs_finds_completed_runs() -> None:
    runs = list_runs(RUNS_DIR)
    assert "real_loop_priced" in runs


@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="example runs not present")
def test_load_run_builds_rows_curve_and_reports() -> None:
    view = load_run(RUNS_DIR, "real_loop_priced")
    assert view["run_id"] == "real_loop_priced"
    assert len(view["rows"]) == 4  # iter_0..iter_3
    assert view["curve"][0]["iteration"] == "iter_0"
    assert view["latest"]["harness"].endswith("harness.json")
    assert view["rows"][-1][6] == "ACCEPT"
    assert "##" in view["reports"]["iter_1"]
    # iter_1 was accepted, so its diff must be non-empty against iter_0
    assert view["diffs"]["iter_1"].startswith("---")
    assert view["diffs"]["iter_0"].startswith("（基线轮")
    assert view["attributions"]["iter_0"], "baseline iteration has attribution records"


def test_attribution_summary_counts_labels() -> None:
    text = attribution_summary(
        [{"label": "RESULT_MISMATCH"}, {"label": "RESULT_MISMATCH"}, {"label": "EXEC_ERROR"}]
    )
    assert "共 3 条归因" in text
    assert "RESULT_MISMATCH: 2 (67%)" in text


@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="example runs not present")
def test_build_demo_blocks() -> None:
    demo = build_demo(RUNS_DIR)
    assert type(demo).__name__ == "Blocks"


@pytest.mark.slow
@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="example runs not present")
def test_ui_smoke_launch_and_close() -> None:
    from harnessloop.ui.gradio_app import launch

    demo, local_url = launch(RUNS_DIR, port=7861)
    try:
        assert local_url.startswith("http://127.0.0.1:7861")
    finally:
        demo.close()
