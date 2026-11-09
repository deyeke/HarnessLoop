"""HarnessLoop / 驭环 — harness-only self-improvement loop for agents."""

from harnessloop.models import (
    Attribution,
    AttributionLabel,
    CheckResult,
    EditOp,
    FailureSignal,
    HarnessFacet,
    HarnessState,
    Hypothesis,
    IterationMetrics,
    RunSummary,
    Task,
    ToolSchema,
    Trajectory,
)

__version__ = "0.4.0"

__all__ = [
    "Attribution",
    "AttributionLabel",
    "CheckResult",
    "EditOp",
    "FailureSignal",
    "HarnessFacet",
    "HarnessState",
    "Hypothesis",
    "IterationMetrics",
    "RunSummary",
    "Task",
    "ToolSchema",
    "Trajectory",
    "__version__",
]
