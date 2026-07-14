from __future__ import annotations

from pathlib import Path

import pytest
from coding_agent.extensions.install import (
    ExtensionInstallError,
    classify_source,
    derive_extension_name,
    install_extension,
    list_installed_extensions,
    needs_venv,
    provision_venv,
    resolve_destination,
    uninstall_extension,
)


def _write_extension(directory: Path, *, with_deps: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").write_text(
        "def activate(api) -> None:\n    pass\n", encoding="utf-8"
    )
    if with_deps:
        (directory / "pyproject.toml").write_text(
            '[project]\nname = "my-ext"\nversion = "0.1.0"\ndependencies = ["six"]\n',
            encoding="utf-8",
        )
    return directory


class TestClassifySource:
    def test_existing_local_dir_is_local(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "my-ext"
        source_dir.mkdir()
        assert classify_source(str(source_dir)) == "local"

    def test_nonexistent_path_and_urls_are_git(self, tmp_path: Path) -> None:
        assert classify_source("https://github.com/x/y.git") == "git"
        assert classify_source("git@github.com:x/y.git") == "git"
        assert classify_source(str(tmp_path / "does-not-exist")) == "git"


class TestDeriveExtensionName:
    def test_explicit_name_wins(self) -> None:
        assert derive_extension_name("https://x/y.git", "git", "custom") == "custom"

    def test_git_url_strips_dot_git_and_trailing_slash(self) -> None:
        assert derive_extension_name("https://github.com/x/my-ext.git", "git", None) == "my-ext"
        assert derive_extension_name("https://github.com/x/my-ext/", "git", None) == "my-ext"

    def test_scp_like_git_url(self) -> None:
        assert derive_extension_name("git@github.com:x/my-ext.git", "git", None) == "my-ext"

    def test_local_dir_uses_basename(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "my-ext"
        source_dir.mkdir()
        assert derive_extension_name(str(source_dir), "local", None) == "my-ext"

    @pytest.mark.parametrize("bad_name", ["", ".", "..", "a/b", "a\\b"])
    def test_rejects_unsafe_names(self, bad_name: str) -> None:
        with pytest.raises(ExtensionInstallError):
            derive_extension_name("ignored", "git", bad_name)


class TestResolveDestination:
    def test_project_destination(self, tmp_path: Path) -> None:
        cwd = tmp_path / "project"
        assert resolve_destination("my-ext", project=True, cwd=cwd) == (
            cwd / ".haiku" / "extensions" / "my-ext"
        )

    def test_global_destination_uses_agent_dir(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "home" / ".haiku"
        assert resolve_destination(
            "my-ext", project=False, cwd=tmp_path, agent_dir=agent_dir
        ) == (agent_dir / "extensions" / "my-ext")


class TestNeedsVenv:
    def test_pyproject_with_dependencies_needs_venv(self, tmp_path: Path) -> None:
        _write_extension(tmp_path, with_deps=True)
        assert needs_venv(tmp_path) is True

    def test_pyproject_without_project_table_does_not_need_venv(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.haiku]\nextensions = ["__init__.py"]\n', encoding="utf-8"
        )
        assert needs_venv(tmp_path) is False

    def test_pyproject_with_empty_dependencies_does_not_need_venv(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "0"\ndependencies = []\n', encoding="utf-8"
        )
        assert needs_venv(tmp_path) is False

    def test_no_pyproject_does_not_need_venv(self, tmp_path: Path) -> None:
        _write_extension(tmp_path, with_deps=False)
        assert needs_venv(tmp_path) is False

    def test_malformed_pyproject_does_not_need_venv(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("not valid toml [[[", encoding="utf-8")
        assert needs_venv(tmp_path) is False


class TestProvisionVenv:
    """`provision_venv` must install the extension's declared *dependency specifiers*
    directly, not `uv pip install <extension_dir>` -- the extension is exec'd in place by the
    loader, never actually installed as a package, so treating it as one would require a
    [build-system] and a real installable package layout it doesn't have."""

    @pytest.mark.asyncio
    async def test_installs_bare_dependency_specifiers_not_the_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_extension(tmp_path, with_deps=True)
        calls: list[list[str]] = []

        async def fake_run_subprocess(args, **_kwargs):
            calls.append(args)
            return 0, ""

        monkeypatch.setattr(
            "coding_agent.extensions.install._run_subprocess", fake_run_subprocess
        )
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/uv")

        await provision_venv(tmp_path)

        venv_call, install_call = calls
        assert venv_call[:2] == ["uv", "venv"]
        assert install_call[:4] == ["uv", "pip", "install", "--python"]
        assert install_call[5:] == ["six"]
        assert str(tmp_path) not in install_call

    @pytest.mark.asyncio
    async def test_no_dependencies_skips_provisioning_entirely(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_extension(tmp_path, with_deps=False)

        async def fail_run_subprocess(*_args, **_kwargs):
            raise AssertionError("should not shell out when there are no dependencies")

        monkeypatch.setattr(
            "coding_agent.extensions.install._run_subprocess", fail_run_subprocess
        )

        await provision_venv(tmp_path)


class TestInstallExtension:
    @pytest.mark.asyncio
    async def test_installs_local_extension_without_deps(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir, with_deps=False)
        agent_dir = tmp_path / "home" / ".haiku"

        result = await install_extension(
            str(source_dir), cwd=tmp_path, agent_dir=agent_dir
        )

        assert result.name == "source"
        assert result.destination == agent_dir / "extensions" / "source"
        assert result.venv_created is False
        assert (result.destination / "__init__.py").exists()

    @pytest.mark.asyncio
    async def test_project_flag_installs_under_cwd(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir)
        cwd = tmp_path / "project"
        cwd.mkdir()

        result = await install_extension(str(source_dir), project=True, cwd=cwd)

        assert result.destination == cwd / ".haiku" / "extensions" / "source"

    @pytest.mark.asyncio
    async def test_provisions_venv_when_deps_declared(self, tmp_path: Path, monkeypatch) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir, with_deps=True)
        agent_dir = tmp_path / "home" / ".haiku"

        calls: list[Path] = []

        async def fake_provision_venv(extension_dir: Path, **_kwargs) -> None:
            calls.append(extension_dir)
            (extension_dir / ".venv").mkdir()

        monkeypatch.setattr(
            "coding_agent.extensions.install.provision_venv", fake_provision_venv
        )

        result = await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

        assert result.venv_created is True
        assert calls == [result.destination]
        assert (result.destination / ".venv").is_dir()

    @pytest.mark.asyncio
    async def test_skip_venv_flag_skips_provisioning(self, tmp_path: Path, monkeypatch) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir, with_deps=True)

        async def fail_provision_venv(*_args, **_kwargs) -> None:
            raise AssertionError("provision_venv should not be called")

        monkeypatch.setattr(
            "coding_agent.extensions.install.provision_venv", fail_provision_venv
        )

        result = await install_extension(
            str(source_dir), cwd=tmp_path, agent_dir=tmp_path / "home" / ".haiku", skip_venv=True
        )

        assert result.venv_created is False

    @pytest.mark.asyncio
    async def test_collision_without_force_raises(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir)
        agent_dir = tmp_path / "home" / ".haiku"
        await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

        with pytest.raises(ExtensionInstallError):
            await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

    @pytest.mark.asyncio
    async def test_collision_with_force_overwrites(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir)
        agent_dir = tmp_path / "home" / ".haiku"
        await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

        result = await install_extension(
            str(source_dir), cwd=tmp_path, agent_dir=agent_dir, force=True
        )

        assert result.destination.exists()

    @pytest.mark.asyncio
    async def test_no_entry_point_rolls_back(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "README.md").write_text("not an extension", encoding="utf-8")
        agent_dir = tmp_path / "home" / ".haiku"

        with pytest.raises(ExtensionInstallError):
            await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

        assert not (agent_dir / "extensions" / "source").exists()

    @pytest.mark.asyncio
    async def test_installing_from_own_destination_is_rejected(self, tmp_path: Path) -> None:
        """A --force reinstall where source IS the destination would otherwise rmtree the
        source before ever copying it -- must be rejected up front, not left to fail
        confusingly (or silently destroy the extension) partway through."""
        agent_dir = tmp_path / "home" / ".haiku"
        destination = agent_dir / "extensions" / "my-ext"
        _write_extension(destination)

        with pytest.raises(ExtensionInstallError):
            await install_extension(
                str(destination), name="my-ext", cwd=tmp_path, agent_dir=agent_dir, force=True
            )

        # Must survive untouched -- this is the whole point of the guard.
        assert (destination / "__init__.py").exists()

    @pytest.mark.asyncio
    async def test_git_source_invokes_fetch_git_source(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[tuple[str, Path]] = []

        async def fake_fetch_git_source(url: str, destination: Path) -> None:
            calls.append((url, destination))
            _write_extension(destination)

        monkeypatch.setattr(
            "coding_agent.extensions.install.fetch_git_source", fake_fetch_git_source
        )

        agent_dir = tmp_path / "home" / ".haiku"
        result = await install_extension(
            "https://github.com/someone/my-ext.git", cwd=tmp_path, agent_dir=agent_dir
        )

        assert result.name == "my-ext"
        assert calls == [("https://github.com/someone/my-ext.git", result.destination)]


class TestListAndUninstall:
    @pytest.mark.asyncio
    async def test_list_reports_project_and_global_with_venv_marker(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cwd = tmp_path / "project"
        agent_dir = tmp_path / "home" / ".haiku"
        source_dir = tmp_path / "source"
        _write_extension(source_dir, with_deps=True)

        async def fake_provision_venv(extension_dir: Path, **_kwargs) -> None:
            (extension_dir / ".venv").mkdir()

        monkeypatch.setattr(
            "coding_agent.extensions.install.provision_venv", fake_provision_venv
        )
        await install_extension(str(source_dir), project=True, cwd=cwd, agent_dir=agent_dir)
        await install_extension(str(source_dir), name="other", cwd=cwd, agent_dir=agent_dir)

        entries = list_installed_extensions(cwd=cwd, agent_dir=agent_dir)
        by_name = {entry["name"]: entry for entry in entries}
        assert by_name["source"]["location"] == "project"
        assert by_name["source"]["has_venv"] is True
        assert by_name["other"]["location"] == "global"

    def test_list_empty_returns_empty(self, tmp_path: Path) -> None:
        assert list_installed_extensions(cwd=tmp_path, agent_dir=tmp_path / "home" / ".haiku") == []

    @pytest.mark.asyncio
    async def test_uninstall_removes_installed_extension(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_extension(source_dir)
        agent_dir = tmp_path / "home" / ".haiku"
        await install_extension(str(source_dir), cwd=tmp_path, agent_dir=agent_dir)

        removed = uninstall_extension("source", project=False, cwd=tmp_path, agent_dir=agent_dir)
        assert removed is True
        assert not (agent_dir / "extensions" / "source").exists()

    def test_uninstall_nonexistent_returns_false(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "home" / ".haiku"
        removed = uninstall_extension("nope", project=False, cwd=tmp_path, agent_dir=agent_dir)
        assert removed is False
