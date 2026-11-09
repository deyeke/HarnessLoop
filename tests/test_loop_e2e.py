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


def _config(toy_env, out_dir, *, rounds: int = 3, budget: float | None = None) -> LoopConfig:
    return LoopConfig(
        model="gpt-4o-mini",
        dataset=DatasetConfig(path=str(toy_env.dataset), kind="sql"),
        harness=HarnessConfig(path=str(toy_env.harness)),
        loop=LoopSettings(max_rounds=rounds, concurrency=2, seed=7, budget_usd_per_round=budget),
        opt=OptSettings(max_hypotheses=2),
        report=ReportSettings(out_dir=str(out_dir)),
    )


async def test_closed_loop_converges(toy_env, tmp_path, fake_llm) -> None:
    config = _config(toy_env, tmp_path / "runs")
    runner = LoopRunner.build(config, llm=fake_llm, run_id="run_e2e")
    summary = await runner.run()

    assert summary.iterations[0].total == 3
    assert summary.iterations[0].passed == 2
    assert summary.accepted_iterations == [1]
    assert summary.final_pass_rate == 1.0

    run_dir = runner.run_dir.path
    report = (run_dir / "iter_1" / "report.md").read_text(encoding="utf-8")
    assert "ACCEPT" in report
    assert "LEFT JOIN" in report

    accepted = json.loads((run_dir / "iter_1" / "harness.json").read_text(encoding="utf-8"))
    assert "LEFT JOIN" in accepted["system_prompt"]

    attributions = (run_dir / "iter_0" / "attribution.jsonl").read_text(encoding="utf-8")
    assert "RESULT_MISMATCH" in attributions

    latest = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["iteration"] == 1
    assert (run_dir / "iter_1" / "decision.json").exists()
    assert (run_dir / "iter_1" / "candidate_harness.json").exists()


async def test_reproducible_scores_with_fixed_seed(toy_env, tmp_path) -> None:
    first = await _run_once(toy_env, tmp_path / "a")
    second = await _run_once(toy_env, tmp_path / "b")
    assert [metrics.pass_rate for metrics in first.iterations] == [
        metrics.pass_rate for metrics in second.iterations
    ]


async def _run_once(toy_env, out_dir):
    import conftest

    config = _config(toy_env, out_dir)
    runner = LoopRunner.build(config, llm=conftest.ChattyFakeLLM())
    return await runner.run()


async def test_budget_limit_stops_after_baseline(toy_env, tmp_path, fake_llm) -> None:
    config = _config(toy_env, tmp_path / "runs", budget=0.0001)
    runner = LoopRunner.build(config, llm=fake_llm)
    summary = await runner.run()

    assert len(summary.iterations) == 1
    assert summary.accepted_iterations == []
    assert summary.stopped_reason is not None
    assert "预算" in summary.stopped_reason


async def test_rejected_candidate_keeps_accepted_harness(toy_env, tmp_path, fake_llm) -> None:
    config = _config(toy_env, tmp_path / "runs", rounds=1)
    config.opt.gate.strategy = "threshold"
    config.opt.gate.min_gain_pp = 100.0
    runner = LoopRunner.build(config, llm=fake_llm)
    summary = await runner.run()

    assert summary.accepted_iterations == []
    assert summary.final_pass_rate == 2 / 3
    run_dir = runner.run_dir.path
    report = (run_dir / "iter_1" / "report.md").read_text(encoding="utf-8")
    assert "REJECT" in report

    kept = json.loads((run_dir / "iter_1" / "harness.json").read_text(encoding="utf-8"))
    baseline = json.loads((run_dir / "iter_0" / "harness.json").read_text(encoding="utf-8"))
    assert kept == baseline

    candidate = json.loads(
        (run_dir / "iter_1" / "candidate_harness.json").read_text(encoding="utf-8")
    )
    assert "LEFT JOIN" in candidate["system_prompt"]
