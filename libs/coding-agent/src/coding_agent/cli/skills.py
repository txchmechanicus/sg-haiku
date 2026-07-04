from __future__ import annotations

from pathlib import Path

import typer
from agent.skills import discover_skills

from coding_agent.cli.console import console

skills_app = typer.Typer(add_completion=False, no_args_is_help=True)


@skills_app.command("list")
def skills_list() -> None:
    skills, diagnostics = discover_skills(Path.cwd())
    if not skills:
        console.print("No skills found.")
    for skill in skills:
        console.print(f"{skill.name}\t{skill.description}\t{skill.file_path}")
    if diagnostics:
        console.print("\n[Skill conflicts]")
        for diagnostic in diagnostics:
            console.print(f"{diagnostic.type}: {diagnostic.message}")
