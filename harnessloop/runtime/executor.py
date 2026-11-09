"""Executor abstraction.

The single contract HarnessLoop requires from any agent runtime. Adapters for
HarnessX / Pydantic AI / Agno implement this protocol in later milestones.
"""

from __future__ import annotations

from typing import Protocol

from harnessloop.models import HarnessState, Task, Trajectory


class Executor(Protocol):
    async def run(self, task: Task, harness: HarnessState) -> Trajectory:
        """Execute one task under ``harness`` and return the full trajectory."""
        ...
