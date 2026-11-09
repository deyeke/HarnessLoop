from __future__ import annotations

import json

from harnessloop.config import (
    DatasetConfig,
    HarnessConfig,
    LoopConfig,
    LoopSettings,
    OptSettings,
    ReportSettings,
)
from harnessloop.loop import LoopRunner


def _config(api_env, out_dir, *, rounds: int = 3) -> LoopConfig:
    return LoopConfig(
        model="gpt-4o-mini",
        dataset=DatasetConfig(path=str(api_env.dataset), kind="api"),
        harness=HarnessConfig(path=str(api_env.harness)),
        loop=LoopSettings(max_rounds=rounds, concurrency=2, seed=7),
        opt=OptSettings(max_hypotheses=2),
        report=ReportSettings(out_dir=str(out_dir)),
    )


async def test_api_closed_loop_converges(api_env, tmp_path, api_fake_llm) -> None:
    config = _config(api_env, tmp_path / "runs")
    runner = LoopRunner.build(config, llm=api_fake_llm, run_id="run_api_e2e")
    summary = await runner.run()

    assert summary.iterations[0].total == 2
    assert summary.iterations[0].passed == 1
    assert summary.accepted_iterations == [1]
    assert summary.final_pass_rate == 1.0

    run_dir = runner.run_dir.path

    # S1 snapshots and state diffs are recorded for API tasks
    checks = (run_dir / "iter_0" / "check_results.jsonl").read_text(encoding="utf-8")
    assert "状态差异" in checks
    trajectory_line = (run_dir / "iter_0" / "trajectory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "final_state" in trajectory_line

    attributions = (run_dir / "iter_0" / "attribution.jsonl").read_text(encoding="utf-8")
    assert "RESULT_MISMATCH" in attributions

    report = (run_dir / "iter_1" / "report.md").read_text(encoding="utf-8")
    assert "ACCEPT" in report

    accepted = json.loads((run_dir / "iter_1" / "harness.json").read_text(encoding="utf-8"))
    assert "update_task" in accepted["system_prompt"]


async def test_api_loop_reproducible(api_env, tmp_path) -> None:
    import conftest

    async def run_once(out_dir):
        config = _config(api_env, out_dir)
        runner = LoopRunner.build(config, llm=conftest.ApiFakeLLM())
        return await runner.run()

    first = await run_once(tmp_path / "a")
    second = await run_once(tmp_path / "b")
    assert [m.pass_rate for m in first.iterations] == [m.pass_rate for m in second.iterations]
