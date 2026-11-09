"""Harness snapshot diff (unified diff + field-level change count)."""

from __future__ import annotations

import difflib
import json
from typing import Any

from pydantic import BaseModel

from harnessloop.models import HarnessState


class HarnessChange(BaseModel):
    changed_fields: int = 0
    added_chars: int = 0
    changed_paths: list[str] = []
    diff: str = ""

    @property
    def is_empty(self) -> bool:
        return self.changed_fields == 0


def _walk(left: Any, right: Any, path: str, out: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                out.append(child)
            else:
                _walk(left[key], right[key], child, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(path or "$")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            child = f"{path}[{index}]" if path else f"[{index}]"
            _walk(left_item, right_item, child, out)
        return
    if left != right:
        out.append(path or "$")


def diff_harness(previous: HarnessState, candidate: HarnessState) -> HarnessChange:
    previous_data = previous.model_dump()
    candidate_data = candidate.model_dump()

    changed_paths: list[str] = []
    _walk(previous_data, candidate_data, "", changed_paths)

    previous_text = json.dumps(previous_data, indent=2, ensure_ascii=False, sort_keys=True)
    candidate_text = json.dumps(candidate_data, indent=2, ensure_ascii=False, sort_keys=True)
    diff = "\n".join(
        difflib.unified_diff(
            previous_text.splitlines(),
            candidate_text.splitlines(),
            fromfile="harness/previous",
            tofile="harness/candidate",
            lineterm="",
        )
    )

    return HarnessChange(
        changed_fields=len(changed_paths),
        added_chars=len(candidate_text) - len(previous_text),
        changed_paths=changed_paths,
        diff=diff,
    )
