"""Plugin loading and version checks shared by every extension point.

A plugin is referenced from a profile as a string:

- ``pkg.module:attr`` — attribute of an importable module;
- ``pkg.module`` — module imported for its side effects (decorator registration);
- ``path/to/file.py[:attr]`` — a standalone file, so examples need no packaging.

Every plugin declares the *major* version of the extension API it targets. The
core refuses mismatched majors instead of silently mis-dispatching.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

API_VERSION = "1"

# ``C:\path\rules.py:attr`` must not be split on the drive colon.
_FILE_SPEC = re.compile(r"^(?P<target>.+\.py)(?::(?P<attr>[A-Za-z_]\w*))?$")
_MODULE_SPEC = re.compile(r"^(?P<target>[A-Za-z_][\w.]*)(?::(?P<attr>[A-Za-z_]\w*))?$")


def split_spec(spec: str) -> tuple[str, str]:
    """Return ``(target, attr)`` for a plugin spec."""
    match = _FILE_SPEC.match(spec.strip()) or _MODULE_SPEC.match(spec.strip())
    if not match:
        raise PluginError(f"invalid plugin spec: {spec!r}")
    return match.group("target"), match.group("attr") or ""


class PluginError(RuntimeError):
    """A plugin could not be loaded or registered."""


class PluginVersionError(PluginError):
    """A plugin targets an incompatible extension-API major version."""


def require_api_version(
    component: str, version: str, *, supported: str = API_VERSION
) -> None:
    target_major = str(version).split(".")[0].strip()
    supported_major = supported.split(".")[0]
    if target_major != supported_major:
        raise PluginVersionError(
            f"plugin {component!r} targets extension API v{version}, "
            f"this build supports v{supported} (major versions must match)"
        )


def load_plugin(spec: str) -> Any:
    """Load a plugin spec and return its object (or module when no attr given)."""
    target, attr = split_spec(spec)
    if not target:
        raise PluginError(f"invalid plugin spec: {spec!r}")

    path = Path(target)
    if path.suffix == ".py":
        module = _load_file_module(path)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise PluginError(f"cannot import plugin module {target!r}: {exc}") from exc

    if not attr:
        return module
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise PluginError(f"plugin {spec!r} has no attribute {attr!r}") from exc


def load_plugins(specs: list[str], *, kind: str) -> list[Any]:
    """Load every spec, tagging failures with the extension point name."""
    loaded: list[Any] = []
    for spec in specs:
        try:
            loaded.append(load_plugin(spec))
        except PluginError as exc:
            raise PluginError(f"{kind} plugin {spec!r}: {exc}") from exc
    return loaded


def _load_file_module(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PluginError(f"plugin file not found: {resolved}")
    module_name = f"_harnessloop_plugin_{resolved.stem}_{abs(hash(str(resolved))) & 0xFFFFFF:x}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise PluginError(f"cannot build import spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise PluginError(f"plugin file {resolved} failed to import: {exc}") from exc
    return module
