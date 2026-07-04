from __future__ import annotations

import os

import pytest
from tui.terminal import ProcessTerminal


def test_enter_raw_mode_rejects_non_tty() -> None:
    read_fd, write_fd = os.pipe()
    try:
        terminal = ProcessTerminal(input_fd=read_fd)
        with pytest.raises(ValueError, match="not a terminal"):
            terminal.enter_raw_mode()
    finally:
        os.close(read_fd)
        os.close(write_fd)
