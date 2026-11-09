"""EditOp plugins (M3-2).

The three built-in kinds (``add`` / ``remove`` / ``rewrite``) can only append or
replace scalars in a harness field. Anything structured — tightening an enum,
adding a schema property, inserting a few-shot example — is expressed as a
custom op without touching core code:

    from harnessloop.optimize.plugins import edit_op

    @edit_op("set_enum", description="replace a parameter enum")
    def set_enum(harness, edit):
        data = harness.model_dump()
        ...
        return HarnessState.model_validate(data)

Hypotheses carry the op name in ``EditOp.kind`` and structured arguments in
``EditOp.payload``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harnessloop.models import EditOp, HarnessState
from harnessloop.plugins import (
    API_VERSION,
    PluginError,
    load_plugins,
    require_api_version,
)

logger = logging.getLogger(__name__)

BUILTIN_EDIT_KINDS = frozenset({"add", "remove", "rewrite"})

EditOpHandler = Callable[[HarnessState, EditOp], HarnessState]


@dataclass(frozen=True)
class EditOpPlugin:
    name: str
    apply: EditOpHandler
    api_version: str = API_VERSION
    description: str = ""


class EditOpRegistry:
    def __init__(self, *, api_version: str = API_VERSION) -> None:
        self.api_version = api_version
        self._ops: dict[str, EditOpPlugin] = {}

    def register(self, plugin: EditOpPlugin, *, replace: bool = False) -> EditOpPlugin:
        require_api_version(plugin.name, plugin.api_version, supported=self.api_version)
        if plugin.name in BUILTIN_EDIT_KINDS:
            raise PluginError(
                f"edit op {plugin.name!r} shadows a built-in kind "
                f"({', '.join(sorted(BUILTIN_EDIT_KINDS))})"
            )
        existing = self._ops.get(plugin.name)
        if existing is not None and existing is not plugin and not replace:
            raise PluginError(f"edit op {plugin.name!r} is already registered")
        self._ops[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> EditOpPlugin | None:
        return self._ops.get(name)

    def names(self) -> list[str]:
        return sorted(self._ops)

    def apply(self, harness: HarnessState, edit: EditOp) -> HarnessState:
        plugin = self.get(edit.kind)
        if plugin is None:
            known = ", ".join([*sorted(BUILTIN_EDIT_KINDS), *self.names()])
            raise KeyError(f"unknown edit op {edit.kind!r}; registered: {known}")
        updated = plugin.apply(harness, edit)
        if not isinstance(updated, HarnessState):  # pragma: no cover - plugin bug
            raise PluginError(
                f"edit op {edit.kind!r} must return HarnessState, got {type(updated).__name__}"
            )
        return updated


_GLOBAL = EditOpRegistry()


def get_registry() -> EditOpRegistry:
    return _GLOBAL


def edit_op(
    name: str,
    *,
    api_version: str = API_VERSION,
    description: str = "",
) -> Callable[[EditOpHandler], EditOpPlugin]:
    """Decorator: register a custom edit op into the global registry."""

    def decorator(func: EditOpHandler) -> EditOpPlugin:
        plugin = EditOpPlugin(
            name=name,
            apply=func,
            api_version=api_version,
            description=description or (func.__doc__ or "").strip(),
        )
        _GLOBAL.register(plugin)
        return plugin

    return decorator


def register_edit_op(plugin: EditOpPlugin, *, replace: bool = False) -> EditOpPlugin:
    return _GLOBAL.register(plugin, replace=replace)


def load_edit_op_plugins(specs: Iterable[str]) -> list[EditOpPlugin]:
    loaded: list[EditOpPlugin] = []
    for obj in load_plugins(list(specs), kind="edit-op"):
        if isinstance(obj, EditOpPlugin):
            loaded.append(_GLOBAL.register(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, EditOpPlugin):
                    loaded.append(_GLOBAL.register(item))
        elif callable(obj):
            func = obj
            loaded.append(
                _GLOBAL.register(
                    EditOpPlugin(
                        name=getattr(func, "__name__", "plugin"),
                        apply=func,
                        description=(func.__doc__ or "").strip(),
                    )
                )
            )
        # a module without attr registered itself via @edit_op
    return loaded
