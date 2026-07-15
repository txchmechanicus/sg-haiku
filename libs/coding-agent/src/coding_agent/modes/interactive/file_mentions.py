"""`@file` mentions in the interactive-mode prompt input: live autocomplete (mirrors the
`/`-command hint list in `session.py`) plus extraction of the mentioned paths from a submitted
message so they can be attached the same way `haiku @file.txt "..."` attaches them on the
one-shot CLI (`coding_agent.cli.file_processor`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_MENTION_MATCHES = 20

# An @-mention starts at an "@" preceded by start-of-string or whitespace (so "user@example.com"
# mid-word doesn't trigger one) and runs to the next whitespace.
_AT_MENTION_RE = re.compile(r"(?<!\S)@(\S+)")


@dataclass(frozen=True)
class FileMention:
    display: str
    insert: str
    is_dir: bool


def find_at_mention_span(text: str, cursor_position: int) -> tuple[int, int] | None:
    """Finds the (start, end) span of the `@`-token the cursor is currently positioned at the
    end of, or `None` if the cursor isn't inside an active mention -- `start` is the index of
    the `@` itself, `end` is the cursor position (not the whole token, so hints reflect only
    what's been typed so far, matching how `/`-command hints filter as you type)."""
    cursor_position = min(cursor_position, len(text))
    start = cursor_position
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    if start >= cursor_position or text[start] != "@":
        return None
    return start, cursor_position


def list_file_mentions(query: str, cwd: Path) -> list[FileMention]:
    """`query` is whatever's been typed after the `@` so far (e.g. `"src/mai"`). Lists the
    directory named by the part before the last `/` (or `cwd` if there is none), filtered to
    entries whose name starts with the part after it -- directories first, alphabetical within
    each group, capped at `MAX_FILE_MENTION_MATCHES`. Dotfiles are hidden unless the prefix
    itself starts with `.`, matching common fuzzy-finder convention."""
    if "/" in query:
        dir_part, _, prefix = query.rpartition("/")
    else:
        dir_part, prefix = "", query

    base = Path(dir_part).expanduser() if dir_part else Path()
    search_dir = base if base.is_absolute() else (cwd / base)

    try:
        entries = list(search_dir.iterdir())
    except OSError:
        return []

    show_dotfiles = prefix.startswith(".")
    prefix_lower = prefix.lower()
    matches = [
        entry
        for entry in entries
        if entry.name.lower().startswith(prefix_lower)
        and (show_dotfiles or not entry.name.startswith("."))
    ]
    matches.sort(key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    matches = matches[:MAX_FILE_MENTION_MATCHES]

    results: list[FileMention] = []
    for entry in matches:
        relative = f"{dir_part}/{entry.name}" if dir_part else entry.name
        if entry.is_dir():
            results.append(FileMention(display=f"{relative}/", insert=f"{relative}/", is_dir=True))
        else:
            results.append(FileMention(display=relative, insert=relative, is_dir=False))
    return results


def extract_at_mentions(text: str) -> list[str]:
    """Pulls every `@`-mentioned path out of free-form submitted text (unlike the one-shot
    CLI, where `@file` is a whole argv token, an interactive message keeps the mention inline
    in the sentence -- `"look at @notes.txt please"` -- so this scans for the same boundary
    `find_at_mention_span` uses instead of requiring the whole input to start with `@`)."""
    return _AT_MENTION_RE.findall(text)
