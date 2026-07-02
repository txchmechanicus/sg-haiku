from __future__ import annotations

import json
from pathlib import Path

import pytest
from coding_agent.cli import app
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_runner(runner: CliRunner):
    """Convenience wrapper: invoke app and return result."""

    def invoke(*args: str, **kwargs) -> object:
        return runner.invoke(app, list(args), **kwargs)

    return invoke


def read_session(filename: str) -> list[dict]:
    return [json.loads(line) for line in Path(filename).read_text(encoding="utf-8").splitlines()]
