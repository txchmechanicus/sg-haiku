from __future__ import annotations

import asyncio
import os
import shutil
import sys
import termios
import tty
from abc import ABC, abstractmethod
from typing import TextIO


class Terminal(ABC):
    """Abstraction over raw terminal I/O that `TUI` renders through."""

    @abstractmethod
    def write(self, data: str) -> None:
        """Write raw data to the terminal output."""

    @abstractmethod
    async def read(self) -> bytes:
        """Wait for and return the next chunk of raw input bytes."""

    @abstractmethod
    def get_size(self) -> tuple[int, int]:
        """Return (columns, rows) of the terminal."""

    @abstractmethod
    def enter_raw_mode(self) -> None:
        """Put the terminal into raw mode and start accepting `read()` calls."""

    @abstractmethod
    def exit_raw_mode(self) -> None:
        """Restore the terminal's previous mode."""


class ProcessTerminal(Terminal):
    """Real terminal backed by the current process's stdin/stdout."""

    def __init__(self, *, input_fd: int | None = None, output: TextIO | None = None) -> None:
        self._input_fd = input_fd if input_fd is not None else sys.stdin.fileno()
        self._output = output if output is not None else sys.stdout
        self._old_settings: list | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes] | None = None

    def write(self, data: str) -> None:
        self._output.write(data)
        self._output.flush()

    def get_size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size()
        return size.columns, size.lines

    def enter_raw_mode(self) -> None:
        self._old_settings = termios.tcgetattr(self._input_fd)
        tty.setraw(self._input_fd)
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._loop.add_reader(self._input_fd, self._on_readable)

    def exit_raw_mode(self) -> None:
        if self._loop is not None:
            self._loop.remove_reader(self._input_fd)
            self._loop = None
        if self._old_settings is not None:
            termios.tcsetattr(self._input_fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
        self._queue = None

    def _on_readable(self) -> None:
        try:
            data = os.read(self._input_fd, 1024)
        except OSError:
            return
        if data and self._queue is not None:
            self._queue.put_nowait(data)

    async def read(self) -> bytes:
        if self._queue is None:
            raise RuntimeError("Terminal.read() called outside raw mode")
        return await self._queue.get()
