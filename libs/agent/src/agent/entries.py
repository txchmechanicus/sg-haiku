from __future__ import annotations

from dataclasses import dataclass

from upstream.models import Message


@dataclass(frozen=True)
class EntryRef:
    id: str
    message: Message
