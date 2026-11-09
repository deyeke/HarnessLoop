"""Attribution rule plugins (M3-1).

A rule maps a failure signal onto a concrete, resolvable harness field. Rules
run in priority order *before* the built-in label mapping, so a plugin can
override the default location without touching core code:

    from harnessloop.attribution.plugins import attribution_rule

    @attribution_rule("noop_state_change", priority=10)
    def noop_state_change(signal, harness):
        if "无变化" not in signal.evidence:
            return None
        return HarnessFacet.SYSTEM_PROMPT, "system_prompt"

Return ``None`` to defer to lower-priority rules. A rule that returns an
unresolvable ``field_path`` is skipped with a warning — attribution must always
land on something editable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harnessloop.models import FailureSignal, HarnessFacet, HarnessState
from harnessloop.plugins import (
    API_VERSION,
    PluginError,
    load_plugins,
    require_api_version,
)

logger = logging.getLogger(__name__)

Located = tuple[HarnessFacet, str]
RuleLocator = Callable[[FailureSignal, HarnessState], "Located | None"]


@dataclass(frozen=True)
class AttributionRule:
    name: str
    locate: RuleLocator
    priority: int = 50
    api_version: str = API_VERSION
    description: str = ""


class AttributionRegistry:
    def __init__(self, *, api_version: str = API_VERSION) -> None:
        self.api_version = api_version
        self._rules: dict[str, AttributionRule] = {}

    def register(self, rule: AttributionRule, *, replace: bool = False) -> AttributionRule:
        require_api_version(rule.name, rule.api_version, supported=self.api_version)
        existing = self._rules.get(rule.name)
        if existing is not None and existing is not rule and not replace:
            raise PluginError(f"attribution rule {rule.name!r} is already registered")
        self._rules[rule.name] = rule
        return rule

    def rules(self) -> list[AttributionRule]:
        return sorted(self._rules.values(), key=lambda rule: (rule.priority, rule.name))

    def resolve(
        self, signal: FailureSignal, harness: HarnessState
    ) -> Located | None:
        for rule in self.rules():
            try:
                located = rule.locate(signal, harness)
            except Exception as exc:  # a broken plugin must not kill the loop
                logger.warning("attribution rule %s raised %s", rule.name, exc)
                continue
            if located is None:
                continue
            facet, path = located
            if harness.resolve_path(path):
                return facet, path
            logger.warning(
                "attribution rule %s returned unresolvable path %r", rule.name, path
            )
        return None


_GLOBAL = AttributionRegistry()


def get_registry() -> AttributionRegistry:
    return _GLOBAL


def attribution_rule(
    name: str,
    *,
    priority: int = 50,
    api_version: str = API_VERSION,
    description: str = "",
) -> Callable[[RuleLocator], AttributionRule]:
    """Decorator: register a rule into the global registry at import time."""

    def decorator(func: RuleLocator) -> AttributionRule:
        rule = AttributionRule(
            name=name,
            locate=func,
            priority=priority,
            api_version=api_version,
            description=description or (func.__doc__ or "").strip(),
        )
        _GLOBAL.register(rule)
        return rule

    return decorator


def register_rule(rule: AttributionRule, *, replace: bool = False) -> AttributionRule:
    return _GLOBAL.register(rule, replace=replace)


def load_rule_plugins(specs: Iterable[str]) -> list[AttributionRule]:
    """Load rules from profile specs into the global registry."""
    loaded: list[AttributionRule] = []
    for obj in load_plugins(list(specs), kind="attribution"):
        if isinstance(obj, AttributionRule):
            loaded.append(_GLOBAL.register(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, AttributionRule):
                    loaded.append(_GLOBAL.register(item))
        elif callable(obj):
            func = obj
            loaded.append(
                _GLOBAL.register(
                    AttributionRule(
                        name=getattr(func, "__name__", "plugin"),
                        locate=func,
                        description=(func.__doc__ or "").strip(),
                    )
                )
            )
        # a module without attr registered itself via @attribution_rule
    return loaded
