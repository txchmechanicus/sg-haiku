"""`@file` attachment processing, shared by the one-shot CLI (`haiku @notes.md
@screenshot.png "..."`, `@`-prefixed argv tokens stripped by `_HaikuGroup.parse_args` in
`cli/app.py`) and interactive mode (`@`-mentions inline in a typed message, extracted by
`modes.interactive.file_mentions.extract_at_mentions`). Turns each referenced path into either
an inline text block or an image attachment on the initial user message, matching Pi's
`cli/file-processor.ts` (`processFileArguments`). Lives under `tools/` rather than `cli/` so
`modes.interactive` (which `cli/interactive.py` itself imports) can depend on it without a
circular import back into `coding_agent.cli`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from upstream.models import ImageContent

from coding_agent.tools.images import read_image_content, sniff_image_mime_type


class FileProcessingError(Exception):
    """Raised for any `@file` argument that can't be attached (missing path, unreadable)."""


@dataclass(frozen=True)
class ProcessedFiles:
    text: str
    images: list[ImageContent] = field(default_factory=list)


def process_file_arguments(file_args: list[str], cwd: Path) -> ProcessedFiles:
    text_blocks: list[str] = []
    images: list[ImageContent] = []

    for file_arg in file_args:
        resolved = Path(file_arg).expanduser()
        if not resolved.is_absolute():
            resolved = cwd / resolved
        resolved = resolved.resolve()

        if not resolved.exists():
            raise FileProcessingError(f"File not found: {file_arg}")
        if not resolved.is_file():
            raise FileProcessingError(f"Not a file: {file_arg}")
        if resolved.stat().st_size == 0:
            continue

        mime_type = sniff_image_mime_type(resolved)
        if mime_type is not None:
            data, final_mime_type = read_image_content(resolved, mime_type)
            images.append(ImageContent(data=data, mimeType=final_mime_type))
            continue

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FileProcessingError(f"Not a text file or supported image: {file_arg}") from exc
        text_blocks.append(f'<file name="{resolved}">\n{content}\n</file>')

    return ProcessedFiles(text="\n\n".join(text_blocks), images=images)
