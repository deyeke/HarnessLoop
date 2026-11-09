"""Example custom attribution label (M4).

Loaded from a profile via ``plugins.labels`` — no core code is touched:

    plugins:
      labels:
        - labels.py

A built-in label answers "what failed"; a custom label can answer "what kind of
failure is this, really". Here a ``RESULT_MISMATCH`` whose state diff says the
environment did not change at all is re-labelled ``NOOP_ACTION``: the agent
answered without actually calling any mutating tool, so the fix belongs in the
system prompt ("really call the tool") rather than in a single tool description.
"""

from __future__ import annotations

from harnessloop.attribution.labels import attribution_label
from harnessloop.models import (
    AttributionLabel,
    FailureSignal,
    HarnessFacet,
    HarnessState,
)


@attribution_label(
    "NOOP_ACTION",
    facet=HarnessFacet.SYSTEM_PROMPT,
    field_path="system_prompt",
    priority=10,
    description="执行后环境状态完全没变：提示模型真正调用修改类工具",
)
def noop_action(signal: FailureSignal, harness: HarnessState) -> bool:
    return (
        signal.label == AttributionLabel.RESULT_MISMATCH
        and "状态差异：无变化" in signal.evidence
    )
