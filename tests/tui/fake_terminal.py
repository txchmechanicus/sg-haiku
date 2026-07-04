from __future__ import annotations

import asyncio

from tui.terminal import Terminal


class FakeTerminal(Terminal):
    """Test double for `Terminal`: scripted input, captured output, no tty."""

    def __init__(self, *, width: int = 80, height: int = 24) -> None:
        self.width = width
        self.height = height
        self.written: list[str] = []
        self.raw_mode = False
        self._input: asyncio.Queue[bytes] = asyncio.Queue()

    def write(self, data: str) -> None:
        self.written.append(data)

    async def read(self) -> bytes:
        return await self._input.get()

    def get_size(self) -> tuple[int, int]:
        return self.width, self.height

    def enter_raw_mode(self) -> None:
        self.raw_mode = True

    def exit_raw_mode(self) -> None:
        self.raw_mode = False

    def feed(self, data: bytes) -> None:
        self._input.put_nowait(data)

    @property
    def output(self) -> str:
        return "".join(self.written)
