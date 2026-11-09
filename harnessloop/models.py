"""Core data models.

This module is the contract between every layer: executor, checker,
attributor, hypothesis generator, mutator and gate. Keep it boring and
explicit — the whole project is judged by whether the attribution chain
can point at a concrete, editable harness facet.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator

# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

_PATH_PART = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")


class ToolSchema(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Example(BaseModel):
    input: str
    output: str


class HarnessFacet(str, Enum):
    """The editable surface of a harness. Anything outside this enum is
    explicitly out of scope (no weights, no topology, no task synthesis)."""

    SYSTEM_PROMPT = "system_prompt"
    TOOL_DESCRIPTION = "tool_description"
    SCHEMA_FIELD = "schema_field"
    FEW_SHOT = "few_shot"
    OUTPUT_FORMAT = "output_format"


_BUILTIN_EDIT_KINDS = frozenset({"add", "remove", "rewrite"})


class EditOp(BaseModel):
    """One harness edit.

    ``kind`` is a built-in op (``add`` / ``remove`` / ``rewrite``) or the name of
    an edit op registered through :mod:`harnessloop.optimize.plugins`. Custom ops
    read structured arguments from ``payload``.
    """

    kind: str = "add"
    target: HarnessFacet
    field_path: str
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("kind must not be blank")
        return value


def _parse_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        match = _PATH_PART.match(part)
        if not match:
            raise ValueError(f"invalid field path: {path!r}")
        tokens.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", match.group(2)):
            tokens.append(int(index))
    return tokens


def _index(node: Any, token: str | int, path: str) -> str | int:
    """Return the key/index used to descend one level into ``node``."""
    if isinstance(node, list):
        if isinstance(token, int):
            if token >= len(node):
                raise IndexError(f"{path}: index {token} out of range")
            return token
        for position, item in enumerate(node):
            if isinstance(item, dict) and item.get("name") == token:
                return position
        raise KeyError(f"{path}: no tool named {token!r}")
    if isinstance(node, dict):
        if token not in node:
            raise KeyError(f"{path}: missing key {token!r}")
        return token
    raise TypeError(f"{path}: cannot descend into {type(node).__name__}")


class HarnessState(BaseModel):
    """An immutable-in-practice snapshot of everything HarnessLoop may edit."""

    system_prompt: str
    tools: list[ToolSchema] = Field(default_factory=list)
    few_shot_examples: list[Example] = Field(default_factory=list)
    output_format: str | None = None

    # -- path access ------------------------------------------------------- #

    def get_field(self, path: str) -> Any:
        node: Any = self.model_dump()
        for token in _parse_path(path):
            node = node[_index(node, token, path)]
        return node

    def resolve_path(self, path: str) -> bool:
        try:
            self.get_field(path)
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        return True

    # -- edits -------------------------------------------------------------- #

    def apply_edit(
        self, edit: EditOp, *, registry: Any | None = None
    ) -> HarnessState:
        """Apply one edit and return a new snapshot.

        Custom ``edit.kind`` values are dispatched to the edit-op registry
        (``registry`` wins over the process-wide one). Raises ``KeyError`` /
        ``IndexError`` / ``TypeError`` / ``ValueError`` when ``field_path``
        cannot be resolved — callers decide whether to skip the edit or fall
        back to a rule template.
        """
        if edit.kind not in _BUILTIN_EDIT_KINDS:
            # Imported lazily: optimize.plugins depends on this module.
            from harnessloop.optimize.plugins import get_registry as _get_edit_ops

            return (registry or _get_edit_ops()).apply(self, edit)

        data = self.model_dump()
        tokens = _parse_path(edit.field_path)
        parent: Any = data
        for token in tokens[:-1]:
            parent = parent[_index(parent, token, edit.field_path)]
        key = _index(parent, tokens[-1], edit.field_path)
        current = parent[key]

        if isinstance(current, list):
            if edit.kind == "add":
                if edit.content and edit.content not in current:
                    parent[key] = [*current, edit.content]
            elif edit.kind == "remove":
                parent[key] = [item for item in current if item != edit.content]
            else:  # rewrite
                parent[key] = [edit.content]
        elif current is None or isinstance(current, str):
            base = current or ""
            if edit.kind == "add":
                parent[key] = f"{base}\n\n{edit.content}".strip() if base else edit.content
            elif edit.kind == "remove":
                parent[key] = base.replace(edit.content, "").strip()
            else:  # rewrite
                parent[key] = edit.content
        else:
            raise TypeError(f"{edit.field_path}: unsupported field type {type(current).__name__}")

        return HarnessState.model_validate(data)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


class Assertion(BaseModel):
    """One state assertion evaluated against the final environment state (S1).

    ``path`` uses dotted segments; list entries support ``key[2]`` (index) and
    ``key[field=value]`` (first matching dict in a list).
    """

    op: Literal[
        "eq",
        "ne",
        "exists",
        "not_exists",
        "contains",
        "length_eq",
        "length_ge",
        "length_le",
    ] = "eq"
    path: str
    value: Any = None


class Task(BaseModel):
    task_id: str
    kind: Literal["sql", "api"] = "sql"
    question: str
    database: str | None = None
    db_schema: str | None = None
    gold_sql: str | None = None
    expected: dict[str, Any] | None = None
    tools: list[str] | None = None
    initial_state: dict[str, Any] | None = None

    @field_validator("task_id", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


# --------------------------------------------------------------------------- #
# Execution records
# --------------------------------------------------------------------------- #


def _label_value(value: Any) -> str:
    """Normalize a built-in enum member or a custom label name to a string."""
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("label must be a non-blank string")
    return value


# Labels are open-ended since M4: built-in names live in ``AttributionLabel``,
# third-party names are registered in :mod:`harnessloop.attribution.labels`.
LabelName = Annotated[str, BeforeValidator(_label_value)]


class ToolIssue(BaseModel):
    label: LabelName
    evidence: str


class ToolCall(BaseModel):
    call_id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    issues: list[ToolIssue] = Field(default_factory=list)


class Step(BaseModel):
    index: int
    model_output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observation: Any = None
    latency_ms: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Trajectory(BaseModel):
    task_id: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: Any = None
    outcome: Literal["pass", "fail"] | None = None
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int | None = None
    error: str | None = None
    # API tasks only: the environment state (S1) after the agent finished.
    final_state: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Evaluation + attribution
# --------------------------------------------------------------------------- #


class AttributionLabel(str, Enum):
    PARSE_FAIL = "PARSE_FAIL"
    EXEC_ERROR = "EXEC_ERROR"
    RESULT_MISMATCH = "RESULT_MISMATCH"
    WRONG_TOOL = "WRONG_TOOL"
    MISSING_PARAM = "MISSING_PARAM"
    EXTRA_PARAM = "EXTRA_PARAM"
    FORMAT_VIOLATION = "FORMAT_VIOLATION"


class FailureSignal(BaseModel):
    # Built-in name or a custom label registered in harnessloop.attribution.labels.
    label: LabelName
    evidence: str
    tool_name: str | None = None
    param: str | None = None


class CheckResult(BaseModel):
    task_id: str
    passed: bool
    predicted_sql: str | None = None
    signals: list[FailureSignal] = Field(default_factory=list)
    # API tasks only: rendered S0 → S1 diff for reports / UI.
    state_diff: str | None = None


class Attribution(BaseModel):
    task_id: str
    label: LabelName
    evidence: str
    harness_facet: HarnessFacet
    field_path: str | None = None


class AttributionGroup(BaseModel):
    label: LabelName
    count: int
    task_ids: list[str]
    facet: HarnessFacet
    field_path: str
    sample_evidence: str


class Hypothesis(BaseModel):
    target: HarnessFacet
    edit: EditOp
    rationale: list[Attribution] = Field(default_factory=list)
    falsifiable_check: str = ""


# --------------------------------------------------------------------------- #
# Loop bookkeeping
# --------------------------------------------------------------------------- #


class IterationMetrics(BaseModel):
    iteration: int
    total: int
    passed: int
    pass_rate: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    changed_fields: int = 0
    diff_chars: int = 0

    def delta_pp(self, previous: IterationMetrics | None) -> float:
        if previous is None:
            return 0.0
        return (self.pass_rate - previous.pass_rate) * 100.0


class GateDecision(BaseModel):
    accepted: bool
    reason: str


class RunSummary(BaseModel):
    run_id: str
    iterations: list[IterationMetrics]
    accepted_iterations: list[int]
    final_pass_rate: float
    stopped_reason: str | None = None
    report_paths: list[str] = Field(default_factory=list)


ToolIssue.model_rebuild()
