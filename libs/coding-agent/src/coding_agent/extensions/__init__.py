"""sg-haiku's extension/hook system (see module docstrings in `types.py`/`loader.py`/
`runner.py`, and `/PLAN.md` for deferred scope)."""

from coding_agent.extensions.api import ExtensionAPI
from coding_agent.extensions.loader import discover_and_load_extensions
from coding_agent.extensions.runner import ExtensionRunner
from coding_agent.extensions.types import (
    AfterProviderResponseEvent,
    ContextEventResult,
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionLoadError,
    LoadExtensionsResult,
    ModelSelectEvent,
    ResourcesDiscoverCollected,
    ResourcesDiscoverEvent,
    ResourcesDiscoverResult,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionCompactEvent,
    SessionShutdownEvent,
    SessionStartEvent,
    ThinkingLevelSelectEvent,
)

__all__ = [
    "AfterProviderResponseEvent",
    "ContextEventResult",
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionLoadError",
    "ExtensionRunner",
    "LoadExtensionsResult",
    "ModelSelectEvent",
    "ResourcesDiscoverCollected",
    "ResourcesDiscoverEvent",
    "ResourcesDiscoverResult",
    "SessionBeforeCompactEvent",
    "SessionBeforeCompactResult",
    "SessionCompactEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "ThinkingLevelSelectEvent",
    "discover_and_load_extensions",
]
