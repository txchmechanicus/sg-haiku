"""Extension install mechanism: fetches an extension (git URL or local path) into
`.haiku/extensions/` and, if it declares dependencies, provisions them into a per-extension
`uv` venv so they never touch the host's own venv or collide with the host's own dependency
versions.

This is dependency isolation only, not process isolation: extensions still run in-process,
with full privileges, exactly as before (see `loader.py`'s docstring) — only their pip
dependencies get a private venv. `loader.py` appends that venv's site-packages to `sys.path`
at load time (never prepends), so the host's own already-resolved dependency versions always
win on a name collision. An extension that needs a *different* version of a package the host
also directly depends on will silently get the host's version instead of its own declared
one; packages exclusive to the extension are unaffected.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from coding_agent.extensions.loader import resolve_extension_entries

MAX_INSTALL_OUTPUT_CHARS = 20_000
DEFAULT_INSTALL_TIMEOUT_SECONDS = 300


class ExtensionInstallError(Exception):
    """Raised for any install-time failure: fetch, venv creation, dependency install,
    validation."""


@dataclass(frozen=True)
class InstallResult:
    name: str
    destination: Path
    venv_created: bool
    warnings: list[str] = field(default_factory=list)


def classify_source(source: str) -> Literal["git", "local"]:
    """An existing local directory is "local"; everything else is treated as "git" (attempt
    a clone and let git itself surface a clear error for garbage input -- simplest rule, no
    upfront URL-scheme sniffing needed)."""
    if Path(source).is_dir():
        return "local"
    return "git"


def derive_extension_name(
    source: str, kind: Literal["git", "local"], explicit_name: str | None
) -> str:
    if explicit_name is not None:
        name = explicit_name
    elif kind == "local":
        name = Path(source).resolve().name
    else:
        name = source.rstrip("/")
        if name.endswith(".git"):
            name = name[: -len(".git")]
        name = name.rsplit("/", 1)[-1]
        name = name.rsplit(":", 1)[-1]  # scp-like "git@host:org/repo"

    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ExtensionInstallError(f"Could not derive a safe extension name from: {source!r}")
    return name


def resolve_destination(
    name: str, *, project: bool, cwd: Path, agent_dir: Path | None = None
) -> Path:
    if project:
        return cwd / ".haiku" / "extensions" / name
    return (agent_dir or Path.home() / ".haiku") / "extensions" / name


async def fetch_git_source(url: str, destination: Path) -> None:
    returncode, output = await _run_subprocess(
        ["git", "clone", "--depth", "1", url, str(destination)]
    )
    if returncode != 0:
        raise ExtensionInstallError(f"git clone failed:\n{output}")


def fetch_local_source(source_path: Path, destination: Path) -> None:
    if not source_path.is_dir():
        raise ExtensionInstallError(f"Local extension source is not a directory: {source_path}")
    shutil.copytree(
        source_path,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "*.pyc"),
    )


def _read_dependencies(extension_dir: Path) -> list[str]:
    """Reads the `[project] dependencies` list from `pyproject.toml`, if any -- the
    extension's own dependency manifest, matching how every other package in this repo (a
    `uv` workspace) declares dependencies. A `pyproject.toml` present only for the
    `[tool.haiku] extensions` entry-point table (no real `[project]`/deps) yields no
    dependencies."""
    pyproject_path = extension_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    dependencies = data.get("project", {}).get("dependencies")
    if isinstance(dependencies, list) and all(isinstance(item, str) for item in dependencies):
        return dependencies
    return []


def needs_venv(extension_dir: Path) -> bool:
    return bool(_read_dependencies(extension_dir))


async def provision_venv(
    extension_dir: Path, *, timeout: int = DEFAULT_INSTALL_TIMEOUT_SECONDS
) -> None:
    if shutil.which("uv") is None:
        raise ExtensionInstallError(
            "uv not found on PATH; install extension dependencies manually or install uv: "
            "https://docs.astral.sh/uv/"
        )

    dependencies = _read_dependencies(extension_dir)
    if not dependencies:
        return

    venv_dir = extension_dir / ".venv"
    # Pinned to the host's own running interpreter: the extension's code still executes
    # in-process via `exec_module`, so a venv built for a different Python minor version
    # would produce binary-incompatible compiled extension modules even though `sys.path`
    # would find them.
    returncode, output = await _run_subprocess(
        ["uv", "venv", str(venv_dir), "--python", sys.executable], timeout=timeout
    )
    if returncode != 0:
        raise ExtensionInstallError(f"uv venv failed:\n{output}")

    venv_python = str(venv_dir / "bin" / "python")
    if (venv_dir / "Scripts" / "python.exe").exists():
        venv_python = str(venv_dir / "Scripts" / "python.exe")

    # Installs the extension's own pyproject.toml-declared dependency specifiers directly
    # (not `uv pip install <extension_dir>`, which would try to build+install the extension
    # itself as a package and require a [build-system] plus a real package layout -- the
    # extension isn't installed as a package, it's exec'd in place by the loader, so only its
    # *dependencies* need to land in the venv).
    install_args = ["uv", "pip", "install", "--python", venv_python, *dependencies]
    returncode, output = await _run_subprocess(install_args, timeout=timeout)
    if returncode != 0:
        raise ExtensionInstallError(f"uv pip install failed:\n{output}")


async def _run_subprocess(
    args: list[str], *, cwd: Path | None = None, timeout: int = DEFAULT_INSTALL_TIMEOUT_SECONDS
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        raise ExtensionInstallError(
            f"Command timed out after {timeout} seconds: {' '.join(args)}"
        ) from None

    output = (stdout + stderr).decode(errors="replace")
    if len(output) > MAX_INSTALL_OUTPUT_CHARS:
        output = "...[truncated]...\n" + output[-MAX_INSTALL_OUTPUT_CHARS:]
    return process.returncode or 0, output


async def install_extension(
    source: str,
    *,
    name: str | None = None,
    project: bool = False,
    force: bool = False,
    cwd: Path,
    agent_dir: Path | None = None,
    skip_venv: bool = False,
) -> InstallResult:
    kind = classify_source(source)
    resolved_name = derive_extension_name(source, kind, name)
    destination = resolve_destination(resolved_name, project=project, cwd=cwd, agent_dir=agent_dir)

    if kind == "local":
        resolved_source = Path(source).resolve()
        resolved_destination = destination.resolve()
        collides = (
            resolved_source == resolved_destination
            or resolved_destination in resolved_source.parents
        )
        if collides:
            # Checked before any destructive step below: with --force, the collision branch
            # rmtree's `destination` before fetching -- if source and destination are the same
            # directory (or destination is an ancestor of source), that would delete the
            # source out from under itself before it's ever copied.
            raise ExtensionInstallError(
                f"Source and destination are the same directory (or destination contains the "
                f"source): {resolved_source}. Install from a copy elsewhere instead."
            )

    if destination.exists():
        if not force:
            raise ExtensionInstallError(
                f"An extension named {resolved_name!r} is already installed at {destination}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(destination)

    try:
        if kind == "git":
            await fetch_git_source(source, destination)
        else:
            fetch_local_source(Path(source), destination)

        if resolve_extension_entries(destination) is None:
            raise ExtensionInstallError(
                f"No entry point found in {destination} "
                "(expected an __init__.py or a [tool.haiku] extensions table in pyproject.toml)."
            )

        venv_created = False
        warnings: list[str] = []
        if not skip_venv and needs_venv(destination):
            await provision_venv(destination)
            venv_created = True

        return InstallResult(
            name=resolved_name,
            destination=destination,
            venv_created=venv_created,
            warnings=warnings,
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def list_installed_extensions(
    *, cwd: Path, agent_dir: Path | None = None
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    locations: list[tuple[Path, str]] = [
        (cwd / ".haiku" / "extensions", "project"),
        ((agent_dir or Path.home() / ".haiku") / "extensions", "global"),
    ]
    for directory, location in locations:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir(), key=lambda entry: entry.name):
            if not child.is_dir():
                continue
            entries.append(
                {
                    "name": child.name,
                    "location": location,
                    "has_venv": (child / ".venv").is_dir(),
                    "entry": resolve_extension_entries(child),
                }
            )
    return entries


def uninstall_extension(
    name: str, *, project: bool, cwd: Path, agent_dir: Path | None = None
) -> bool:
    destination = resolve_destination(name, project=project, cwd=cwd, agent_dir=agent_dir)
    if not destination.exists():
        return False
    shutil.rmtree(destination)
    return True
