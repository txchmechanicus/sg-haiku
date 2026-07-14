from __future__ import annotations

from coding_agent.cli import compact
from coding_agent.cli.app import app
from coding_agent.cli.auth import auth_app
from coding_agent.cli.extensions import extensions_app
from coding_agent.cli.skills import skills_app

app.add_typer(auth_app, name="auth")
app.add_typer(extensions_app, name="extensions")
app.add_typer(skills_app, name="skills")
compact.register(app)

__all__ = ["app"]


if __name__ == "__main__":
    app()
