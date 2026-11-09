"""Acceptance gates.

A candidate harness is only promoted when the gate says so. Rejected candidates
are kept on disk for analysis — the accepted version is never overwritten.
"""

from __future__ import annotations

from typing import Protocol

from harnessloop.models import GateDecision, IterationMetrics
from harnessloop.optimize.diff import HarnessChange


class Gate(Protocol):
    def decide(
        self,
        previous: IterationMetrics,
        candidate: IterationMetrics,
        change: HarnessChange,
    ) -> GateDecision: ...


class ThresholdGate:
    """Accept only when the score improves by at least ``min_gain_pp``."""

    def __init__(self, min_gain_pp: float = 2.0) -> None:
        self.min_gain_pp = min_gain_pp

    def decide(
        self,
        previous: IterationMetrics,
        candidate: IterationMetrics,
        change: HarnessChange,
    ) -> GateDecision:
        gain = candidate.delta_pp(previous)
        if gain >= self.min_gain_pp:
            return GateDecision(
                accepted=True,
                reason=f"分数提升 {gain:+.1f}pp，达到阈值 {self.min_gain_pp:.1f}pp",
            )
        return GateDecision(
            accepted=False,
            reason=f"分数变化 {gain:+.1f}pp，未达到阈值 {self.min_gain_pp:.1f}pp",
        )


class ParetoGate:
    """Accept score improvements; on a tie accept only a minimal edit."""

    def __init__(self, min_gain_pp: float = 2.0, max_equal_change_fields: int = 1) -> None:
        self.min_gain_pp = min_gain_pp
        self.max_equal_change_fields = max_equal_change_fields

    def decide(
        self,
        previous: IterationMetrics,
        candidate: IterationMetrics,
        change: HarnessChange,
    ) -> GateDecision:
        gain = candidate.delta_pp(previous)
        if gain >= self.min_gain_pp:
            return GateDecision(
                accepted=True,
                reason=f"分数提升 {gain:+.1f}pp，满足最低提升 {self.min_gain_pp:.1f}pp",
            )
        if gain > -1e-9 and change.changed_fields <= self.max_equal_change_fields:
            return GateDecision(
                accepted=True,
                reason=(
                    f"分数持平（{gain:+.1f}pp）且只改动 {change.changed_fields} 个字段，"
                    "按 Pareto 规则保留最小修改"
                ),
            )
        if gain > -1e-9:
            return GateDecision(
                accepted=False,
                reason=(
                    f"分数持平但改动 {change.changed_fields} 个字段，"
                    f"超过允许的 {self.max_equal_change_fields} 个"
                ),
            )
        return GateDecision(
            accepted=False,
            reason=f"分数下降 {gain:+.1f}pp，回滚候选 harness",
        )
