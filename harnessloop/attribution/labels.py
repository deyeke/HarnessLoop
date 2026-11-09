"""Attribution labels (M4): the seven built-ins plus third-party labels.

A label owns two things:

- a **default location** — the harness facet and ``field_path`` a failure of
  this kind should be attributed to;
- optionally a **classifier** — a predicate that re-labels an existing failure
  signal (for example: a ``RESULT_MISMATCH`` whose evidence says the state did
  not change at all is really a ``NOOP_ACTION``).

Labels are data, so a plugin can add one without touching core code:

    from harnessloop.attribution.labels import attribution_label
    from harnessloop.models import HarnessFacet

    @attribution_label(
        "NOOP_ACTION",
        facet=HarnessFacet.SYSTEM_PROMPT,
        field_path="system_prompt",
        priority=10,
        description="任务执行后状态没有变化",
    )
    def noop_action(signal, harness) -> bool:
        return "无变化" in signal.evidence

Classifiers run in priority order; the first match wins. A classifier that
raises is skipped with a warning so a broken plugin cannot kill the loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harnessloop.models import (
    AttributionLabel,
    FailureSignal,
    HarnessFacet,
    HarnessState,
)
from harnessloop.plugins import (
    API_VERSION,
    PluginError,
    load_plugins,
    require_api_version,
)

logger = logging.getLogger(__name__)

LabelClassifier = Callable[[FailureSignal, HarnessState], bool]

BUILTIN_LABELS: tuple[str, ...] = tuple(label.value for label in AttributionLabel)

_BUILTIN_SPECS: dict[str, tuple[HarnessFacet, str]] = {
    AttributionLabel.PARSE_FAIL.value: (HarnessFacet.OUTPUT_FORMAT, "output_format"),
    AttributionLabel.EXEC_ERROR.value: (HarnessFacet.SYSTEM_PROMPT, "system_prompt"),
    AttributionLabel.RESULT_MISMATCH.value: (HarnessFacet.SYSTEM_PROMPT, "system_prompt"),
    AttributionLabel.WRONG_TOOL.value: (HarnessFacet.TOOL_DESCRIPTION, "tools"),
    AttributionLabel.MISSING_PARAM.value: (HarnessFacet.SCHEMA_FIELD, "tools"),
    AttributionLabel.EXTRA_PARAM.value: (HarnessFacet.SCHEMA_FIELD, "tools"),
    AttributionLabel.FORMAT_VIOLATION.value: (HarnessFacet.OUTPUT_FORMAT, "output_format"),
}


@dataclass(frozen=True)
class AttributionLabelSpec:
    name: str
    facet: HarnessFacet
    field_path: str
    description: str = ""
    priority: int = 50
    classify: LabelClassifier | None = None
    api_version: str = API_VERSION
    builtin: bool = False


class LabelRegistry:
    def __init__(self, *, api_version: str = API_VERSION, with_builtins: bool = True) -> None:
        self.api_version = api_version
        self._labels: dict[str, AttributionLabelSpec] = {}
        if with_builtins:
            for name, (facet, path) in _BUILTIN_SPECS.items():
                self._labels[name] = AttributionLabelSpec(
                    name=name,
                    facet=facet,
                    field_path=path,
                    description=f"内置标签 {name}",
                    builtin=True,
                )

    def register(
        self, spec: AttributionLabelSpec, *, replace: bool = False
    ) -> AttributionLabelSpec:
        require_api_version(spec.name, spec.api_version, supported=self.api_version)
        if spec.name in BUILTIN_LABELS and not spec.builtin:
            raise PluginError(f"label {spec.name!r} is built-in and cannot be redefined")
        existing = self._labels.get(spec.name)
        if existing is not None and existing is not spec and not replace:
            raise PluginError(f"attribution label {spec.name!r} is already registered")
        self._labels[spec.name] = spec
        return spec

    def labels(self) -> list[AttributionLabelSpec]:
        return sorted(self._labels.values(), key=lambda spec: (spec.priority, spec.name))

    def classifiers(self) -> list[AttributionLabelSpec]:
        return [spec for spec in self.labels() if spec.classify is not None]

    def get(self, name: str) -> AttributionLabelSpec | None:
        return self._labels.get(name)

    def names(self) -> list[str]:
        return [spec.name for spec in self.labels()]

    def default_for(self, name: str) -> tuple[HarnessFacet, str] | None:
        spec = self._labels.get(name)
        if spec is None:
            return None
        return spec.facet, spec.field_path

    def classify(self, signal: FailureSignal, harness: HarnessState) -> str | None:
        """Return the first custom label whose classifier accepts the signal."""
        for spec in self.classifiers():
            try:
                matched = spec.classify(signal, harness)  # type: ignore[misc]
            except Exception as exc:  # a broken plugin must not kill the loop
                logger.warning("label classifier %s raised %s", spec.name, exc)
                continue
            if matched:
                return spec.name
        return None


_GLOBAL = LabelRegistry()


def get_registry() -> LabelRegistry:
    return _GLOBAL


def attribution_label(
    name: str,
    *,
    facet: HarnessFacet,
    field_path: str,
    priority: int = 50,
    description: str = "",
    api_version: str = API_VERSION,
    classify: LabelClassifier | None = None,
) -> Callable[[LabelClassifier | None], AttributionLabelSpec]:
    """Decorator: register a label (and its optional classifier) globally."""

    def decorator(func: LabelClassifier | None = None) -> AttributionLabelSpec:
        spec = AttributionLabelSpec(
            name=name,
            facet=facet,
            field_path=field_path,
            description=description or ((func.__doc__ or "").strip() if func else ""),
            priority=priority,
            classify=classify or func,
            api_version=api_version,
        )
        _GLOBAL.register(spec)
        return spec

    return decorator


def register_label(
    spec: AttributionLabelSpec, *, replace: bool = False
) -> AttributionLabelSpec:
    return _GLOBAL.register(spec, replace=replace)


def load_label_plugins(specs: Iterable[str]) -> list[AttributionLabelSpec]:
    """Load labels from profile specs into the global registry."""
    loaded: list[AttributionLabelSpec] = []
    for obj in load_plugins(list(specs), kind="attribution label"):
        before = set(_GLOBAL.names())
        if isinstance(obj, AttributionLabelSpec):
            loaded.append(_GLOBAL.register(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, AttributionLabelSpec):
                    loaded.append(_GLOBAL.register(item))
        elif callable(obj):
            func = obj
            loaded.append(
                _GLOBAL.register(
                    AttributionLabelSpec(
                        name=getattr(func, "__name__", "label"),
                        facet=HarnessFacet.SYSTEM_PROMPT,
                        field_path="system_prompt",
                        description=(func.__doc__ or "").strip(),
                        classify=func,
                    )
                )
            )
        # A module without attr registered itself via @attribution_label.
        for name in _GLOBAL.names():
            if name not in before:
                spec = _GLOBAL.get(name)
                if spec is not None and spec not in loaded:
                    loaded.append(spec)
    return loaded
