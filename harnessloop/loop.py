"""The closed loop: evaluate → attribute → hypothesize → edit → re-evaluate."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from harnessloop import __version__
from harnessloop.attribution.attributor import Attributor
from harnessloop.attribution.labels import (
    get_registry as get_label_registry,
)
from harnessloop.attribution.labels import (
    load_label_plugins,
)
from harnessloop.attribution.plugins import (
    get_registry as get_attribution_registry,
)
from harnessloop.attribution.plugins import (
    load_rule_plugins,
)
from harnessloop.config import LoopConfig, load_config
from harnessloop.eval.api_checker import APIChecker
from harnessloop.eval.compare import CompareOptions
from harnessloop.eval.dataset import load_tasks
from harnessloop.eval.sql_checker import SQLChecker
from harnessloop.llm import LiteLLMClient, LLMClient
from harnessloop.models import (
    Attribution,
    CheckResult,
    GateDecision,
    HarnessState,
    IterationMetrics,
    RunSummary,
    Task,
    Trajectory,
)
from harnessloop.optimize.diff import HarnessChange, diff_harness
from harnessloop.optimize.gate import Gate, ParetoGate, ThresholdGate
from harnessloop.optimize.hypothesis import HypothesisGenerator
from harnessloop.optimize.mutator import Mutator
from harnessloop.optimize.plugins import (
    get_registry as get_edit_op_registry,
)
from harnessloop.optimize.plugins import (
    load_edit_op_plugins,
)
from harnessloop.optimize.rule_mutator import RuleTemplateMutator
from harnessloop.report.markdown import render_iteration_report
from harnessloop.runtime.litellm_executor import LiteLLMExecutor
from harnessloop.store.trace_store import TraceStore, open_trace_store
from harnessloop.store.workspace import RunDir, load_harness

logger = logging.getLogger(__name__)


class LoopRunner:
    def __init__(
        self,
        *,
        config: LoopConfig,
        tasks: list[Task],
        run_dir: RunDir,
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        attributor: Attributor,
        hypothesis_generator: HypothesisGenerator,
        mutator: Mutator,
        gate: Gate,
        initial_harness: HarnessState | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.config = config
        self.tasks = tasks
        self.run_dir = run_dir
        self.executor = executor
        self.checker = checker
        self.attributor = attributor
        self.hypothesis_generator = hypothesis_generator
        self.mutator = mutator
        self.gate = gate
        self.initial_harness = initial_harness
        self.trace_store = trace_store or open_trace_store(
            run_dir, config.store.backend, sqlite_name=config.store.sqlite_name
        )

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        config: LoopConfig,
        *,
        llm: LLMClient | None = None,
        run_id: str | None = None,
        initial_harness: HarnessState | None = None,
    ) -> LoopRunner:
        if config.dataset.kind not in ("sql", "api"):
            raise NotImplementedError(f"unsupported dataset kind: {config.dataset.kind}")

        tasks = load_tasks(config.dataset.path)

        harness = initial_harness
        if harness is None:
            if not config.harness.path:
                raise ValueError("config.harness.path is required for a full loop run")
            harness = load_harness(config.harness.path)

        pricing = {
            "input_price_per_1m": config.pricing.input_per_1m,
            "output_price_per_1m": config.pricing.output_per_1m,
        }
        client = llm or LiteLLMClient(
            config.llm_model(),
            temperature=config.temperature,
            api_base=config.api_base,
            api_key=config.api_key,
            seed=config.loop.seed,
            **pricing,
        )
        hypothesis_client = client
        if config.opt.hypothesis_model:
            hypothesis_client = llm or LiteLLMClient(
                config.llm_model(config.opt.hypothesis_model),
                temperature=0.2,
                api_base=config.api_base,
                api_key=config.api_key,
                seed=config.loop.seed,
                **pricing,
            )

        gate_settings = config.opt.gate
        gate: Gate
        if gate_settings.strategy == "threshold":
            gate = ThresholdGate(min_gain_pp=gate_settings.min_gain_pp)
        else:
            gate = ParetoGate(
                min_gain_pp=gate_settings.min_gain_pp,
                max_equal_change_fields=gate_settings.max_equal_change_fields,
            )

        load_rule_plugins(config.plugins.attribution)
        load_label_plugins(config.plugins.labels)
        load_edit_op_plugins(config.plugins.edit_ops)
        attribution_registry = get_attribution_registry()
        label_registry = get_label_registry()
        edit_op_registry = get_edit_op_registry()

        checker: SQLChecker | APIChecker
        if config.dataset.kind == "api":
            checker = APIChecker()
        else:
            checker = SQLChecker(CompareOptions(**config.compare.model_dump()))

        executor = cls._build_executor(config, client)
        rule_fallback = RuleTemplateMutator(registry=edit_op_registry)
        if config.opt.mutator == "gepa":
            from harnessloop.optimize.gepa_mutator import GEPAMutator

            mutator: Mutator = GEPAMutator(
                hypothesis_client,
                tasks=tasks,
                executor=executor,
                checker=checker,
                settings=config.opt.gepa,
                seed=config.loop.seed,
                fallback=rule_fallback,
            )
        elif config.opt.mutator == "opik":
            from harnessloop.optimize.opik_mutator import OpikMutator

            meta_mutator = OpikMutator(
                hypothesis_client,
                tasks=tasks,
                executor=executor,
                checker=checker,
                settings=config.opt.opik,
                fallback=rule_fallback,
                seed=config.loop.seed,
            )
            mutator = cls._build_opik_mutator(
                config, tasks, executor, checker, meta_mutator
            )
        else:
            mutator = rule_fallback

        return cls(
            config=config,
            tasks=tasks,
            run_dir=RunDir.create(config.report.out_dir, run_id=run_id),
            executor=executor,
            checker=checker,
            attributor=Attributor(
                registry=attribution_registry, labels=label_registry
            ),
            hypothesis_generator=HypothesisGenerator(
                hypothesis_client,
                max_hypotheses=config.opt.max_hypotheses,
                edit_ops=edit_op_registry.names(),
            ),
            mutator=mutator,
            gate=gate,
            initial_harness=harness,
        )

    @staticmethod
    def _build_executor(config: LoopConfig, client: LLMClient) -> LiteLLMExecutor:
        if config.executor.kind == "mcp":
            from harnessloop.runtime.mcp_executor import MCPExecutor

            mcp = config.executor.mcp
            if mcp is None:  # pragma: no cover - validated in config
                raise ValueError("executor.mcp is required when executor.kind=mcp")
            return MCPExecutor(
                client,
                command=mcp.command,
                cwd=mcp.cwd,
                env=mcp.env or None,
                timeout_s=mcp.timeout_s,
            )
        return LiteLLMExecutor(client)

    @staticmethod
    def _build_opik_mutator(
        config: LoopConfig,
        tasks: list[Task],
        executor: LiteLLMExecutor,
        checker: SQLChecker | APIChecker,
        fallback: Mutator,
    ) -> Mutator:
        """Wrap the meta-prompt mutator with the native adapter when available."""
        backend = config.opt.opik.backend
        if backend == "metaprompt":
            return fallback

        from harnessloop.optimize.opik_native import OpikNativeMutator, native_available

        if not native_available():
            if backend == "native":
                logger.warning(
                    "opt.opik.backend=native 但 opik_optimizer 未安装，"
                    "回退到 MetaPrompt 变异"
                )
            return fallback

        model_parameters: dict[str, str] = {}
        if config.api_key:
            model_parameters["api_key"] = config.api_key
        if config.api_base:
            model_parameters["api_base"] = config.api_base
        return OpikNativeMutator(
            tasks=tasks,
            executor=executor,
            checker=checker,
            settings=config.opt.opik,
            fallback=fallback,
            seed=config.loop.seed,
            model=config.llm_model(config.opt.opik.model),
            model_parameters=model_parameters,
        )

    # -- evaluation -------------------------------------------------------- #

    async def _evaluate(
        self, harness: HarnessState, iteration: int
    ) -> tuple[IterationMetrics, list[Trajectory], list[CheckResult]]:
        semaphore = asyncio.Semaphore(max(1, self.config.loop.concurrency))
        started = time.perf_counter()

        async def run_one(task: Task) -> tuple[Trajectory, CheckResult]:
            async with semaphore:
                trajectory = await self.executor.run(task, harness)
                result = await self.checker.check(task, trajectory)
                trajectory.outcome = "pass" if result.passed else "fail"
                return trajectory, result

        pairs = await asyncio.gather(*(run_one(task) for task in self.tasks))
        trajectories = [pair[0] for pair in pairs]
        results = [pair[1] for pair in pairs]

        passed = sum(1 for result in results if result.passed)
        total = len(results)
        metrics = IterationMetrics(
            iteration=iteration,
            total=total,
            passed=passed,
            pass_rate=(passed / total) if total else 0.0,
            prompt_tokens=sum(t.tokens.get("prompt", 0) for t in trajectories),
            completion_tokens=sum(t.tokens.get("completion", 0) for t in trajectories),
            cost_usd=sum(t.cost_usd for t in trajectories),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return metrics, trajectories, results

    # -- persistence ------------------------------------------------------- #

    def _write_meta(self) -> None:
        self.run_dir.write_json(
            "meta.json",
            {
                "run_id": self.run_dir.run_id,
                "version": __version__,
                "model": self.config.llm_model(),
                "hypothesis_model": self.config.llm_model(
                    self.config.opt.hypothesis_model
                )
                if self.config.opt.hypothesis_model
                else self.config.llm_model(),
                "seed": self.config.loop.seed,
                "dataset": self.config.dataset.path,
                "executor": self.config.executor.kind,
                "trace_store": self.config.store.backend,
                "plugins": self.config.plugins.model_dump(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task_count": len(self.tasks),
            },
        )

    def _write_artifacts(
        self,
        iteration: int,
        metrics: IterationMetrics,
        trajectories: list[Trajectory],
        results: list[CheckResult],
        prefix: str = "",
    ) -> None:
        self.trace_store.write_iteration(
            iteration, trajectories, results, prefix=prefix
        )
        directory = self.run_dir.iter_dir(iteration)
        self.run_dir.write_json(directory / f"{prefix}metrics.json", metrics)

    def _write_report(
        self,
        *,
        iteration: int,
        history: list[IterationMetrics],
        groups: list,
        hypotheses: list,
        change: HarnessChange,
        decision: GateDecision | None,
        candidate_metrics: IterationMetrics,
        total_failures: int,
    ) -> Path:
        report = render_iteration_report(
            run_id=self.run_dir.run_id,
            iteration=iteration,
            history=history,
            groups=groups,
            hypotheses=hypotheses,
            change=change,
            decision=decision,
            candidate_metrics=candidate_metrics,
            total_failures=total_failures,
        )
        return self.run_dir.write_text(
            self.run_dir.iter_dir(iteration) / "report.md", report
        )

    # -- loop -------------------------------------------------------------- #

    async def run(self) -> RunSummary:
        try:
            return await self._run()
        finally:
            self.trace_store.close()

    async def _run(self) -> RunSummary:
        if self.initial_harness is None:
            raise ValueError("initial harness is missing")

        self._write_meta()
        current = self.initial_harness

        metrics, trajectories, results = await self._evaluate(current, 0)
        self.run_dir.write_json(
            self.run_dir.iter_dir(0) / "harness.json", current, overwrite=False
        )
        self._write_artifacts(0, metrics, trajectories, results)
        history = [metrics]
        accepted: list[int] = []
        report_paths = [
            str(
                self._write_report(
                    iteration=0,
                    history=history,
                    groups=[],
                    hypotheses=[],
                    change=HarnessChange(),
                    decision=None,
                    candidate_metrics=metrics,
                    total_failures=metrics.total - metrics.passed,
                )
            )
        ]
        self.run_dir.set_latest(0, "iter_0/harness.json")

        stopped_reason: str | None = None
        if metrics.pass_rate >= 1.0:
            stopped_reason = "基线已全部通过"
        elif (
            self.config.loop.budget_usd_per_round is not None
            and metrics.cost_usd > self.config.loop.budget_usd_per_round
        ):
            stopped_reason = (
                f"基线成本 {metrics.cost_usd:.4f} USD 超过单轮预算 "
                f"{self.config.loop.budget_usd_per_round:.4f} USD"
            )

        for iteration in range(1, self.config.loop.max_rounds + 1):
            if stopped_reason:
                break

            attributions: list[Attribution] = self.attributor.attribute(results, current)
            groups = self.attributor.aggregate(attributions)
            self.run_dir.write_jsonl(
                self.run_dir.iter_dir(iteration - 1) / "attribution.jsonl",
                attributions,
            )
            if not groups:
                stopped_reason = "没有失败样本，闭环收敛"
                break

            hypotheses = await self.hypothesis_generator.generate(
                current, groups, attributions
            )
            self.run_dir.write_json(
                self.run_dir.iter_dir(iteration) / "hypotheses.json", hypotheses
            )

            candidate = await self.mutator.propose(current, hypotheses, attributions)
            change = diff_harness(current, candidate)
            self.run_dir.write_json(
                self.run_dir.iter_dir(iteration) / "candidate_harness.json",
                candidate,
                overwrite=False,
            )

            if change.is_empty:
                decision = GateDecision(
                    accepted=False, reason="候选 harness 与当前版本相同，跳过复测"
                )
                self.run_dir.write_json(
                    self.run_dir.iter_dir(iteration) / "decision.json", decision
                )
                self.run_dir.write_json(
                    self.run_dir.iter_dir(iteration) / "harness.json",
                    current,
                    overwrite=False,
                )
                stopped_reason = "变异器未产生有效修改"
                break

            candidate_metrics, candidate_trajectories, candidate_results = (
                await self._evaluate(candidate, iteration)
            )
            candidate_metrics.changed_fields = change.changed_fields
            candidate_metrics.diff_chars = len(change.diff)

            decision = self.gate.decide(metrics, candidate_metrics, change)
            self._write_artifacts(
                iteration, candidate_metrics, candidate_trajectories, candidate_results
            )
            self.run_dir.write_json(
                self.run_dir.iter_dir(iteration) / "decision.json", decision
            )

            if decision.accepted:
                current = candidate
                metrics = candidate_metrics
                results = candidate_results
                accepted.append(iteration)
                self.run_dir.write_json(
                    self.run_dir.iter_dir(iteration) / "harness.json",
                    current,
                    overwrite=False,
                )
                self.run_dir.set_latest(iteration, f"iter_{iteration}/harness.json")
            else:
                # Keep the accepted harness for the next round; the candidate is
                # already preserved separately for analysis.
                self.run_dir.write_json(
                    self.run_dir.iter_dir(iteration) / "harness.json",
                    current,
                    overwrite=False,
                )

            history.append(candidate_metrics)
            report_paths.append(
                str(
                    self._write_report(
                        iteration=iteration,
                        history=history,
                        groups=groups,
                        hypotheses=hypotheses,
                        change=change,
                        decision=decision,
                        candidate_metrics=candidate_metrics,
                        total_failures=sum(group.count for group in groups),
                    )
                )
            )

            budget = self.config.loop.budget_usd_per_round
            if budget is not None and candidate_metrics.cost_usd > budget:
                stopped_reason = (
                    f"第 {iteration} 轮成本 {candidate_metrics.cost_usd:.4f} USD "
                    f"超过预算 {budget:.4f} USD，已中止"
                )
            elif metrics.pass_rate >= 1.0:
                stopped_reason = "全部任务通过"

        return RunSummary(
            run_id=self.run_dir.run_id,
            iterations=history,
            accepted_iterations=accepted,
            final_pass_rate=metrics.pass_rate,
            stopped_reason=stopped_reason,
            report_paths=report_paths,
        )


def run_from_config(
    config_path: str | Path,
    *,
    llm: LLMClient | None = None,
    run_id: str | None = None,
) -> RunSummary:
    config = load_config(config_path)
    runner = LoopRunner.build(config, llm=llm, run_id=run_id)
    return asyncio.run(runner.run())
