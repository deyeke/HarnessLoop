"""Example attribution rule plugin (M3-1).

Loaded from a profile via ``plugins.attribution`` — no core code is touched:

    plugins:
      attribution:
        - rules.py

The rule claims state-assertion failures on a task's ``status`` field and routes
them to ``update_task``'s description instead of the built-in default
(``system_prompt``). Anything it does not own returns ``None``, so lower-priority
rules and the built-in label mapping still apply.
"""

from __future__ import annotations

from harnessloop.attribution.plugins import attribution_rule
from harnessloop.models import (
    AttributionLabel,
    FailureSignal,
    HarnessFacet,
    HarnessState,
)


@attribution_rule(
    "status_assertion_to_update_tool",
    priority=10,
    description="状态断言失败时归因到 update_task 的工具描述",
)
def status_assertion_to_update_tool(
    signal: FailureSignal, harness: HarnessState
) -> tuple[HarnessFacet, str] | None:
    if signal.label != AttributionLabel.RESULT_MISMATCH:
        return None
    if ".status" not in signal.evidence:
        return None
    path = "tools.update_task.description"
    return (HarnessFacet.TOOL_DESCRIPTION, path) if harness.resolve_path(path) else None
