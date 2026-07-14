"""Extension discovery/loading for sg-haiku's extension/hook system.

Discovery order: project-local, then global, then explicitly configured paths, deduped by
resolved path (first occurrence wins). sg-haiku extensions are plain `.py` files already
runnable on the host interpreter, so loading is just `importlib` — no transpiler, no
virtual-module bundling, no `npm install` step (this repo's own dependencies are already on
the venv path an extension's `import` statements will see).

No sandboxing: extensions run with full process privileges. Only install extensions from
sources you trust.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tomllib
from pathlib import Path
from uuid import uuid4

from coding_agent.extensions.api import ExtensionAPI
from coding_agent.extensions.types import (
    Extension,
    ExtensionLoadError,
    LoadExtensionsResult,
    SourceInfo,
)

GLOBAL_EXTENSIONS_DIR = Path.home() / ".haiku" / "extensions"
PROJECT_EXTENSIONS_DIR_NAME = Path(".haiku") / "extensions"

# Tracks venv site-packages dirs already appended to sys.path this process, so re-loading
# extensions (e.g. across multiple discover_and_load_extensions calls) never adds duplicates.
_spliced_venv_paths: set[str] = set()


async def discover_and_load_extensions(
    configured_paths: list[str] | None,
    cwd: Path,
    agent_dir: Path | None = None,
) -> LoadExtensionsResult:
    resolved_cwd = cwd.resolve()
    resolved_agent_dir = (agent_dir or Path.home() / ".haiku").resolve()

    all_paths: list[str] = []
    seen: set[Path] = set()

    def add_paths(paths: list[str]) -> None:
        for path in paths:
            resolved = Path(path)
            resolved = resolved if resolved.is_absolute() else (resolved_cwd / resolved)
            resolved = resolved.resolve()
            if resolved not in seen:
                seen.add(resolved)
                all_paths.append(path)

    # 1. Project-local extensions: cwd/.haiku/extensions/
    add_paths(discover_extensions_in_dir(resolved_cwd / PROJECT_EXTENSIONS_DIR_NAME))

    # 2. Global extensions: agent_dir/extensions/
    add_paths(discover_extensions_in_dir(resolved_agent_dir / "extensions"))

    # 3. Explicitly configured paths
    for configured in configured_paths or []:
        candidate = Path(configured)
        candidate = candidate if candidate.is_absolute() else (resolved_cwd / candidate)
        if candidate.is_dir():
            entries = resolve_extension_entries(candidate)
            if entries:
                add_paths(entries)
                continue
            add_paths(discover_extensions_in_dir(candidate))
            continue
        add_paths([str(candidate)])

    return await load_extensions(all_paths, resolved_cwd)


def discover_extensions_in_dir(directory: Path) -> list[str]:
    if not directory.exists():
        return []

    discovered: list[str] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return []

    for entry in entries:
        if entry.is_file() and entry.suffix == ".py":
            discovered.append(str(entry))
            continue
        if entry.is_dir():
            resolved_entries = resolve_extension_entries(entry)
            if resolved_entries:
                discovered.extend(resolved_entries)

    return discovered


def resolve_extension_entries(directory: Path) -> list[str] | None:
    """A `[tool.haiku] extensions = [...]` table in `pyproject.toml` wins over a plain
    `__init__.py`."""
    pyproject_path = directory / "pyproject.toml"
    if pyproject_path.exists():
        manifest_extensions = _read_haiku_manifest(pyproject_path)
        if manifest_extensions:
            entries = []
            for relative in manifest_extensions:
                resolved = (directory / relative).resolve()
                if resolved.exists():
                    entries.append(str(resolved))
            if entries:
                return entries

    init_py = directory / "__init__.py"
    if init_py.exists():
        return [str(init_py)]

    return None


def _read_haiku_manifest(pyproject_path: Path) -> list[str] | None:
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    extensions = data.get("tool", {}).get("haiku", {}).get("extensions")
    if isinstance(extensions, list) and all(isinstance(item, str) for item in extensions):
        return extensions
    return None


async def load_extensions(paths: list[str], cwd: Path) -> LoadExtensionsResult:
    extensions: list[Extension] = []
    errors: list[ExtensionLoadError] = []

    for path in paths:
        extension, error = await _load_extension(path, cwd)
        if error is not None:
            errors.append(ExtensionLoadError(path=path, error=error))
        elif extension is not None:
            extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors)


async def _load_extension(path: str, cwd: Path) -> tuple[Extension | None, str | None]:
    candidate = Path(path)
    resolved_path = candidate if candidate.is_absolute() else (cwd / candidate)
    resolved_path = resolved_path.resolve()

    try:
        factory = _load_extension_module(resolved_path)
        if factory is None:
            return None, f"Extension does not export a valid factory function: {path}"

        source_info = SourceInfo(path=path, resolved_path=str(resolved_path))
        extension = Extension(path=path, resolved_path=str(resolved_path), source_info=source_info)
        api = ExtensionAPI(extension)
        result = factory(api)
        if inspect.isawaitable(result):
            await result
        return extension, None
    except Exception as exc:  # noqa: BLE001 - per-extension isolation, doesn't abort others.
        return None, f"Failed to load extension: {exc}"


def _find_extension_venv_site_packages(entry_point: Path) -> Path | None:
    """Looks for a `.venv/` sibling in the entry point's containing extension directory
    (installed by `extensions.install.provision_venv`) and returns its site-packages dir, or
    `None` if there's no venv or it's malformed. Never raises -- a broken/missing venv just
    means the extension loads against the host's own `sys.path` only, same as before this
    feature existed."""
    venv_dir = entry_point.parent / ".venv"
    if not venv_dir.is_dir():
        return None

    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        try:
            for line in pyvenv_cfg.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "version_info" or key.strip() == "version":
                    major, minor, *_ = value.strip().split(".")
                    candidate = venv_dir / "lib" / f"python{major}.{minor}" / "site-packages"
                    if candidate.is_dir():
                        return candidate
        except OSError:
            pass

    windows_candidate = venv_dir / "Lib" / "site-packages"
    if windows_candidate.is_dir():
        return windows_candidate

    matches = sorted(venv_dir.glob("lib/python3.*/site-packages"))
    if matches:
        return matches[0]

    return None


def _splice_venv_path(entry_point: Path) -> None:
    """Appends (never prepends) the extension's venv site-packages dir to `sys.path`, if any,
    so its declared dependencies become importable when the extension module executes.
    Appending -- not prepending -- means the host's own already-resolvable dependency
    versions always win on a name collision (host's own site-packages entries are earlier in
    `sys.path`); this is deliberate ("don't break the agent"), not an oversight. Never removed
    afterward, matching the process-lifetime `sys.path` addition the hand-written
    `atlas_browser` extension already does for the same reason."""
    site_packages = _find_extension_venv_site_packages(entry_point)
    if site_packages is None:
        return
    key = str(site_packages)
    if key in _spliced_venv_paths:
        return
    _spliced_venv_paths.add(key)
    sys.path.append(key)


def _load_extension_module(resolved_path: Path):  # noqa: ANN202 - returns the raw `activate` attr
    if not resolved_path.exists():
        raise FileNotFoundError(f"Extension entry point not found: {resolved_path}")

    _splice_venv_path(resolved_path)

    module_name = f"_haiku_extension_{uuid4().hex}"
    is_package_entry = resolved_path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        resolved_path,
        submodule_search_locations=[str(resolved_path.parent)] if is_package_entry else None,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extension module: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    factory = getattr(module, "activate", None)
    if not callable(factory):
        return None
    return factory
