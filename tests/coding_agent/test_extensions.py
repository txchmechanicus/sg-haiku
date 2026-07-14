from __future__ import annotations

from pathlib import Path

import pytest
from agent.core import (
    BeforeAgentStartResult,
    ProviderRequestPayload,
    ToolCallContext,
    ToolCallHookResult,
    ToolResultHookResult,
)
from agent.sessions import SessionManager
from coding_agent.cli import app
from coding_agent.extensions.loader import discover_and_load_extensions
from coding_agent.extensions.runner import ExtensionRunner
from coding_agent.extensions.types import (
    ContextEventResult,
    MessageEndEventResult,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
)
from coding_agent.tools.core import Tool, ToolRegistry
from typer.testing import CliRunner
from upstream.models import ToolCall, UserMessage
from upstream.types import AgentToolResult


def _write_extension(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _session_manager(cwd: Path) -> SessionManager:
    return SessionManager.create(cwd=cwd, write_enabled=False)


ACTIVATE_TOOL_CALL_BLOCKER = """
from coding_agent.extensions import ExtensionAPI


async def _block(call, ctx):
    if call.name == "bash" and "rm -rf" in call.arguments.get("command", ""):
        from agent.core import ToolCallHookResult
        return ToolCallHookResult(block=True, reason="blocked")
    return None


def activate(api: ExtensionAPI) -> None:
    api.on("tool_call", _block)
"""

ACTIVATE_BROKEN = """
raise RuntimeError("boom at import time")


def activate(api):
    pass
"""

ACTIVATE_NOT_CALLABLE = """
activate = "not a function"
"""


class TestLoaderDiscovery:
    @pytest.mark.asyncio
    async def test_discovers_project_then_global_then_configured(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "home" / ".haiku"
        configured_dir = tmp_path / "configured"
        cwd.mkdir(parents=True)
        (agent_dir / "extensions").mkdir(parents=True)
        configured_dir.mkdir(parents=True)

        _write_extension(
            cwd / ".haiku" / "extensions", "project_ext.py", ACTIVATE_TOOL_CALL_BLOCKER
        )
        _write_extension(
            agent_dir / "extensions", "global_ext.py", ACTIVATE_TOOL_CALL_BLOCKER
        )
        _write_extension(configured_dir, "configured_ext.py", ACTIVATE_TOOL_CALL_BLOCKER)

        result = await discover_and_load_extensions(
            [str(configured_dir)], cwd, agent_dir=agent_dir
        )

        assert result.errors == []
        loaded_names = sorted(Path(ext.path).name for ext in result.extensions)
        assert loaded_names == ["configured_ext.py", "global_ext.py", "project_ext.py"]

    @pytest.mark.asyncio
    async def test_isolates_a_broken_extension_from_the_rest(self, tmp_path: Path) -> None:
        cwd = tmp_path
        ext_dir = cwd / ".haiku" / "extensions"
        _write_extension(ext_dir, "broken.py", ACTIVATE_BROKEN)
        _write_extension(ext_dir, "good.py", ACTIVATE_TOOL_CALL_BLOCKER)

        result = await discover_and_load_extensions(None, cwd, agent_dir=tmp_path / "no-agent-dir")

        assert len(result.extensions) == 1
        assert Path(result.extensions[0].path).name == "good.py"
        assert len(result.errors) == 1
        assert "broken.py" in result.errors[0].path

    @pytest.mark.asyncio
    async def test_non_callable_activate_is_reported_as_error(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".haiku" / "extensions"
        _write_extension(ext_dir, "bad.py", ACTIVATE_NOT_CALLABLE)

        result = await discover_and_load_extensions(None, tmp_path, agent_dir=tmp_path / "none")

        assert result.extensions == []
        assert len(result.errors) == 1
        assert "does not export a valid factory function" in result.errors[0].error

    @pytest.mark.asyncio
    async def test_no_extensions_is_not_an_error(self, tmp_path: Path) -> None:
        result = await discover_and_load_extensions(None, tmp_path, agent_dir=tmp_path / "none")
        assert result.extensions == []
        assert result.errors == []


class TestRunnerDispatch:
    @pytest.mark.asyncio
    async def test_notify_is_fail_open_across_handlers(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        calls: list[str] = []

        async def failing_handler(event, ctx):
            calls.append("failing")
            raise RuntimeError("boom")

        async def ok_handler(event, ctx):
            calls.append("ok")

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["agent_start"] = [failing_handler, ok_handler]
        errors = []
        runner = ExtensionRunner(
            [ext],
            cwd=tmp_path,
            session_manager=_session_manager(tmp_path),
            error_listener=errors.append,
        )

        await runner.notify("agent_start", object())

        assert calls == ["failing", "ok"]
        assert len(errors) == 1
        assert "boom" in errors[0].error

    @pytest.mark.asyncio
    async def test_emit_message_end_chains_and_validates_role(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        first_message = UserMessage(content="first", timestamp=1)
        second_message = UserMessage(content="second", timestamp=2)

        async def handler_one(message, ctx):
            assert message.content == "original"
            return MessageEndEventResult(message=first_message)

        async def handler_two(message, ctx):
            assert message.content == "first"
            return MessageEndEventResult(message=second_message)

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["message_end"] = [handler_one, handler_two]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_message_end(UserMessage(content="original", timestamp=0))

        assert result is second_message

    @pytest.mark.asyncio
    async def test_emit_tool_call_block_short_circuits(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        calls: list[str] = []

        async def first(call, ctx):
            calls.append("first")
            return ToolCallHookResult(block=True, reason="blocked by first")

        async def second(call, ctx):
            calls.append("second")
            return None

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["tool_call"] = [first, second]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_tool_call(
            ToolCall(id="1", name="bash", arguments={"command": "echo hi"})
        )

        assert result is not None
        assert result.block is True
        assert result.reason == "blocked by first"
        assert calls == ["first"]

    @pytest.mark.asyncio
    async def test_emit_tool_result_progressively_patches_fields(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        async def set_details(call, result, is_error, ctx):
            return ToolResultHookResult(details={"a": 1})

        async def set_error(call, result, is_error, ctx):
            assert result.details == {"a": 1}
            return ToolResultHookResult(is_error=True)

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["tool_result"] = [set_details, set_error]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        call = ToolCall(id="1", name="bash", arguments={})
        original = AgentToolResult.text("ok")
        patch = await runner.emit_tool_result(call, original, False)

        assert patch is not None
        assert patch.details == {"a": 1}
        assert patch.is_error is True
        assert patch.content == original.content

    @pytest.mark.asyncio
    async def test_emit_context_pipeline_last_output_wins(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        async def add_a(messages, ctx):
            return ContextEventResult(messages=[*messages, UserMessage(content="a", timestamp=1)])

        async def add_b(messages, ctx):
            return ContextEventResult(messages=[*messages, UserMessage(content="b", timestamp=2)])

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["context"] = [add_a, add_b]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_context([UserMessage(content="start", timestamp=0)])

        assert [message.content for message in result] == ["start", "a", "b"]

    @pytest.mark.asyncio
    async def test_emit_before_provider_request_pipeline(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        async def patch_system_prompt(payload: ProviderRequestPayload, ctx):
            payload.system_prompt = payload.system_prompt + "+A"
            return payload

        async def patch_again(payload: ProviderRequestPayload, ctx):
            payload.system_prompt = payload.system_prompt + "+B"
            return payload

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["before_provider_request"] = [patch_system_prompt, patch_again]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        payload = ProviderRequestPayload(messages=[], specs=[], system_prompt="base")
        result = await runner.emit_before_provider_request(payload)

        assert result.system_prompt == "base+A+B"

    @pytest.mark.asyncio
    async def test_emit_before_agent_start_accumulates_messages_and_chains_prompt(
        self, tmp_path: Path
    ) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        async def first(prompt, system_prompt, ctx):
            return BeforeAgentStartResult(
                messages=[UserMessage(content="from-first", timestamp=1)],
                system_prompt="patched-once",
            )

        async def second(prompt, system_prompt, ctx):
            assert system_prompt == "patched-once"
            return BeforeAgentStartResult(
                messages=[UserMessage(content="from-second", timestamp=2)]
            )

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["before_agent_start"] = [first, second]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_before_agent_start("hello", "base")

        assert result is not None
        assert result.system_prompt == "patched-once"
        assert [message.content for message in result.messages] == [
            "from-first",
            "from-second",
        ]

    @pytest.mark.asyncio
    async def test_session_before_compact_cancel_short_circuits(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        calls: list[str] = []

        async def cancels(event, ctx):
            calls.append("cancels")
            return SessionBeforeCompactResult(cancel=True)

        async def never_runs(event, ctx):
            calls.append("never")
            return None

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["session_before_compact"] = [cancels, never_runs]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_session_before_compact(
            SessionBeforeCompactEvent(reason="threshold", previousSummary=None)
        )

        assert result is not None
        assert result.cancel is True
        assert calls == ["cancels"]

    def test_register_tools_merges_into_registry(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, RegisteredTool, SourceInfo

        async def handler(_args: dict, _ctx: ToolCallContext) -> tuple[AgentToolResult, bool]:
            return AgentToolResult.text("done"), False

        tool = Tool(name="extra", description="extra tool", parameters={}, handler=handler)
        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.tools["extra"] = RegisteredTool(
            tool=tool, source_info=SourceInfo(path="a", resolved_path="a")
        )
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        registry = ToolRegistry()
        runner.register_tools(registry)

        assert "extra" in registry.names()

    @pytest.mark.asyncio
    async def test_emit_resources_discover_collects_skill_paths(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, ResourcesDiscoverResult, SourceInfo

        seen_events = []

        async def handler_one(event, ctx):
            seen_events.append(event)
            return ResourcesDiscoverResult(skillPaths=["a", "b"])

        async def handler_two(event, ctx):
            return ResourcesDiscoverResult(skillPaths=["c"])

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["resources_discover"] = [handler_one, handler_two]
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))

        result = await runner.emit_resources_discover(cwd=tmp_path)

        assert [str(p) for p in result.skill_paths] == ["a", "b", "c"]
        assert seen_events[0].type == "resources_discover"
        assert seen_events[0].cwd == str(tmp_path)
        assert seen_events[0].reason == "startup"

    @pytest.mark.asyncio
    async def test_emit_resources_discover_is_fail_open(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, ResourcesDiscoverResult, SourceInfo

        async def failing(event, ctx):
            raise RuntimeError("boom")

        async def ok(event, ctx):
            return ResourcesDiscoverResult(skillPaths=["x"])

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        ext.handlers["resources_discover"] = [failing, ok]
        errors = []
        runner = ExtensionRunner(
            [ext],
            cwd=tmp_path,
            session_manager=_session_manager(tmp_path),
            error_listener=errors.append,
        )

        result = await runner.emit_resources_discover(cwd=tmp_path)

        assert [str(p) for p in result.skill_paths] == ["x"]
        assert len(errors) == 1
        assert "boom" in errors[0].error

    def test_has_handlers(self, tmp_path: Path) -> None:
        from coding_agent.extensions.types import Extension, SourceInfo

        ext = Extension(
            path="a", resolved_path="a", source_info=SourceInfo(path="a", resolved_path="a")
        )
        runner = ExtensionRunner([ext], cwd=tmp_path, session_manager=_session_manager(tmp_path))
        assert runner.has_handlers("tool_call") is False

        ext.handlers["tool_call"] = [lambda call, ctx: None]
        assert runner.has_handlers("tool_call") is True


class TestCliWiring:
    """End-to-end: a project-local `.haiku/extensions/` file actually blocks a tool call
    during a real `haiku` CLI invocation (mock provider, matching the existing CLI test
    style in test_cli_tools.py)."""

    def test_project_extension_blocks_ls_tool_call(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _write_extension(
            tmp_path / ".haiku" / "extensions",
            "block_ls.py",
            '''
from coding_agent.extensions import ExtensionAPI
from agent.core import ToolCallHookResult


async def _block(call, ctx):
    if call.name == "ls":
        return ToolCallHookResult(block=True, reason="ls is blocked in this sandbox")
    return None


def activate(api: ExtensionAPI) -> None:
    api.on("tool_call", _block)
''',
        )

        result = CliRunner().invoke(app, ["list files"])

        assert result.exit_code == 0
        assert "ls is blocked in this sandbox" in result.stdout
        assert "Tool ls returned" in result.stdout

    def test_no_extensions_present_behaves_unchanged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["list files"])

        assert result.exit_code == 0
        assert "Tool ls returned" in result.stdout
        assert "blocked" not in result.stdout

    def test_resources_discover_skill_path_is_offered_to_the_model(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A `resources_discover` handler's `skillPaths` actually reach skill discovery, so
        the contributed skill shows up in the system prompt's `<available_skills>` block."""
        monkeypatch.chdir(tmp_path)
        extra_skills_dir = tmp_path / "vendored-skills"
        (extra_skills_dir / "vendored-skill").mkdir(parents=True)
        (extra_skills_dir / "vendored-skill" / "SKILL.md").write_text(
            "---\nname: vendored-skill\ndescription: A vendored skill.\n---\nDo it.\n",
            encoding="utf-8",
        )
        marker_file = tmp_path / "marker.txt"
        _write_extension(
            tmp_path / ".haiku" / "extensions",
            "vendor_skills.py",
            f'''
from coding_agent.extensions import ExtensionAPI, ResourcesDiscoverResult


async def _discover(event, ctx):
    return ResourcesDiscoverResult(skillPaths=[{str(extra_skills_dir)!r}])


async def _check_system_prompt(prompt, system_prompt, ctx):
    with open({str(marker_file)!r}, "w") as f:
        f.write("found" if "vendored-skill" in ctx.get_system_prompt() else "missing")


def activate(api: ExtensionAPI) -> None:
    api.on("resources_discover", _discover)
    api.on("before_agent_start", _check_system_prompt)
''',
        )

        result = CliRunner().invoke(app, ["hi", "--no-tools", "--no-session"])

        assert result.exit_code == 0
        assert marker_file.read_text() == "found"

    def test_registered_tool_prompt_snippet_and_guidelines_reach_system_prompt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        marker_file = tmp_path / "marker.txt"
        _write_extension(
            tmp_path / ".haiku" / "extensions",
            "custom_tool.py",
            f'''
from coding_agent.extensions import ExtensionAPI
from coding_agent.tools.core import Tool


async def _handler(args, ctx):
    from upstream.types import AgentToolResult
    return AgentToolResult.text("ok"), False


async def _check_system_prompt(prompt, system_prompt, ctx):
    text = ctx.get_system_prompt()
    ok = (
        "Available tools:\\n  - deploy: deploys the app" in text
        and "Guidelines:\\n  - Always confirm before deploying." in text
    )
    with open({str(marker_file)!r}, "w") as f:
        f.write("found" if ok else "missing")


def activate(api: ExtensionAPI) -> None:
    api.register_tool(
        Tool(
            name="deploy",
            description="Deploys the app.",
            parameters={{}},
            handler=_handler,
            prompt_snippet="deploy: deploys the app",
            prompt_guidelines=("Always confirm before deploying.",),
        )
    )
    api.on("before_agent_start", _check_system_prompt)
''',
        )

        result = CliRunner().invoke(app, ["hi", "--no-session"])

        assert result.exit_code == 0
        assert marker_file.read_text() == "found"

    def test_after_provider_response_fires_with_duration_and_no_transport_metadata(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        marker_file = tmp_path / "marker.txt"
        _write_extension(
            tmp_path / ".haiku" / "extensions",
            "track_latency.py",
            f'''
from coding_agent.extensions import ExtensionAPI


async def _after_response(event, ctx):
    ok = event.status is None and event.headers == {{}} and event.durationMs >= 0
    with open({str(marker_file)!r}, "w") as f:
        f.write("found" if ok else "missing")


def activate(api: ExtensionAPI) -> None:
    api.on("after_provider_response", _after_response)
''',
        )

        result = CliRunner().invoke(app, ["hi", "--no-tools", "--no-session"])

        assert result.exit_code == 0
        assert marker_file.read_text() == "found"


class TestExtensionVenvSplicing:
    """Covers the loader-side half of `haiku extensions install`'s per-extension venv
    isolation: `.venv/` site-packages discovery and its (append-only, permanent) `sys.path`
    splicing in `loader._load_extension_module`."""

    def test_finds_site_packages_via_glob(self, tmp_path: Path) -> None:
        from coding_agent.extensions import loader

        entry_point = tmp_path / "my_ext" / "__init__.py"
        entry_point.parent.mkdir(parents=True)
        entry_point.write_text("", encoding="utf-8")
        site_packages = entry_point.parent / ".venv" / "lib" / "python3.11" / "site-packages"
        site_packages.mkdir(parents=True)

        found = loader._find_extension_venv_site_packages(entry_point)

        assert found == site_packages

    def test_prefers_pyvenv_cfg_version(self, tmp_path: Path) -> None:
        from coding_agent.extensions import loader

        entry_point = tmp_path / "my_ext" / "__init__.py"
        entry_point.parent.mkdir(parents=True)
        entry_point.write_text("", encoding="utf-8")
        venv_dir = entry_point.parent / ".venv"
        site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
        site_packages.mkdir(parents=True)
        (venv_dir / "pyvenv.cfg").write_text("version_info = 3.12.4\n", encoding="utf-8")

        found = loader._find_extension_venv_site_packages(entry_point)

        assert found == site_packages

    def test_no_venv_returns_none(self, tmp_path: Path) -> None:
        from coding_agent.extensions import loader

        entry_point = tmp_path / "my_ext" / "__init__.py"
        entry_point.parent.mkdir(parents=True)
        entry_point.write_text("", encoding="utf-8")

        assert loader._find_extension_venv_site_packages(entry_point) is None

    def test_venv_present_but_empty_returns_none(self, tmp_path: Path) -> None:
        from coding_agent.extensions import loader

        entry_point = tmp_path / "my_ext" / "__init__.py"
        entry_point.parent.mkdir(parents=True)
        entry_point.write_text("", encoding="utf-8")
        (entry_point.parent / ".venv").mkdir()

        assert loader._find_extension_venv_site_packages(entry_point) is None

    @pytest.mark.asyncio
    async def test_extension_can_import_its_venv_dependency(self, tmp_path: Path) -> None:
        from uuid import uuid4

        module_name = f"fake_dep_{uuid4().hex}"
        ext_dir = tmp_path / ".haiku" / "extensions" / "my_ext"
        site_packages = ext_dir / ".venv" / "lib" / "python3.11" / "site-packages"
        site_packages.mkdir(parents=True)
        (site_packages / f"{module_name}.py").write_text("VALUE = 42\n", encoding="utf-8")

        _write_extension(
            ext_dir,
            "__init__.py",
            f'''
import {module_name}
from coding_agent.extensions import ExtensionAPI


def activate(api: ExtensionAPI) -> None:
    assert {module_name}.VALUE == 42
''',
        )

        result = await discover_and_load_extensions(None, tmp_path)

        assert result.errors == []
        assert len(result.extensions) == 1

    @pytest.mark.asyncio
    async def test_host_path_wins_over_venv_path_on_name_collision(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Locks in the accepted, documented limitation: appending (not prepending) the
        venv's site-packages means the host's own already-resolvable version of a
        same-named module always wins."""
        import sys
        from uuid import uuid4

        module_name = f"collide_{uuid4().hex}"

        host_path = tmp_path / "host_path"
        host_path.mkdir()
        (host_path / f"{module_name}.py").write_text("VALUE = 'host'\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(host_path))

        ext_dir = tmp_path / ".haiku" / "extensions" / "my_ext"
        site_packages = ext_dir / ".venv" / "lib" / "python3.11" / "site-packages"
        site_packages.mkdir(parents=True)
        (site_packages / f"{module_name}.py").write_text("VALUE = 'venv'\n", encoding="utf-8")

        _write_extension(
            ext_dir,
            "__init__.py",
            f'''
import {module_name}
from coding_agent.extensions import ExtensionAPI


def activate(api: ExtensionAPI) -> None:
    assert {module_name}.VALUE == "host"
''',
        )

        result = await discover_and_load_extensions(None, tmp_path)

        assert result.errors == []
        assert len(result.extensions) == 1

        del sys.modules[module_name]
