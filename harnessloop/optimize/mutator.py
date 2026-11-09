"""Mutator contract: turn hypotheses into a candidate harness."""

from __future__ import annotations

from typing import Protocol

from harnessloop.models import Attribution, HarnessState, Hypothesis


class Mutator(Protocol):
    async def propose(
        self,
        harness: HarnessState,
        hypotheses: list[Hypothesis],
        attributions: list[Attribution],
    ) -> HarnessState:
        """Return a candidate harness; never mutate the input in place."""
        ...
