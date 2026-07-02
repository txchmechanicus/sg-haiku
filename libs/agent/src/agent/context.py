from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.core import SYSTEM_PROMPT

CONTEXT_FILE_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


@dataclass(frozen=True)
class PromptContext:
    prompt: str
    system_prompt: str
    context_files: list[Path] = field(default_factory=list)


class PromptContextBuilder:
    def __init__(
        self,
        *,
        cwd: Path,
        base_system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.cwd = cwd.resolve()
        self.base_system_prompt = base_system_prompt

    def build(
        self,
        *,
        prompt: str,
        include_context_files: bool = True,
        system_prompt: str | None = None,
        append_system_prompts: list[str] | None = None,
        prompt_template: Path | None = None,
        use_prompt_templates: bool = True,
    ) -> PromptContext:
        context_files = self.discover_context_files() if include_context_files else []
        effective_system_prompt = self._effective_system_prompt(
            context_files=context_files,
            system_prompt=system_prompt,
            append_system_prompts=append_system_prompts or [],
        )
        effective_prompt = self.apply_prompt_template(
            prompt=prompt,
            template=prompt_template,
            enabled=use_prompt_templates,
        )
        return PromptContext(
            prompt=effective_prompt,
            system_prompt=effective_system_prompt,
            context_files=context_files,
        )

    def discover_context_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[tuple[int, int]] = set()
        for name in CONTEXT_FILE_NAMES:
            path = self.cwd / name
            if not path.is_file():
                continue
            stat = path.stat()
            file_id = (stat.st_dev, stat.st_ino)
            if file_id not in seen:
                files.append(path)
                seen.add(file_id)
        return files

    def _effective_system_prompt(
        self,
        *,
        context_files: list[Path],
        system_prompt: str | None,
        append_system_prompts: list[str],
    ) -> str:
        parts = [
            self._read_value(system_prompt)
            if system_prompt is not None
            else self.base_system_prompt
        ]
        for path in context_files:
            parts.append(f"Context from {path.name}:\n{path.read_text(encoding='utf-8')}")
        for value in append_system_prompts:
            parts.append(self._read_value(value))
        return "\n\n".join(part for part in parts if part)

    def apply_prompt_template(
        self,
        *,
        prompt: str,
        template: Path | None,
        enabled: bool,
    ) -> str:
        if template is None or not enabled:
            return prompt
        if not template.exists():
            raise ValueError(f"Prompt template does not exist: {template}")
        if not template.is_file():
            raise ValueError(f"Prompt template is not a file: {template}")
        text = template.read_text(encoding="utf-8")
        if "{prompt}" in text:
            return text.replace("{prompt}", prompt)
        return f"{text.rstrip()}\n\n{prompt}"

    def _read_value(self, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return value
