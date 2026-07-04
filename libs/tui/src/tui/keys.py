from __future__ import annotations

from dataclasses import dataclass

# Names for non-printable keys parse_key can recognize.
_SIMPLE_SEQUENCES: dict[bytes, str] = {
    b"\r": "enter",
    b"\n": "enter",
    b"\x7f": "backspace",
    b"\x08": "backspace",
    b"\x03": "ctrl+c",
    b"\x04": "ctrl+d",
    b"\x1b": "escape",
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[H": "home",
    b"\x1b[F": "end",
    b"\x1bOH": "home",
    b"\x1bOF": "end",
    b"\x1b[3~": "delete",
    b"\x01": "ctrl+a",
    b"\x05": "ctrl+e",
    b"\x0b": "ctrl+k",
    b"\x15": "ctrl+u",
}


@dataclass(frozen=True)
class Key:
    name: str
    char: str | None = None


def parse_key(data: bytes) -> Key:
    """Parse a single input chunk from a terminal into a `Key`.

    Handles a minimal set of ANSI escape sequences (arrows/home/end/delete),
    the common control characters, and single printable characters. Extended
    Kitty-keyboard-protocol sequences and key-repeat/release detection are out
    of scope for this pass.
    """
    name = _SIMPLE_SEQUENCES.get(data)
    if name is not None:
        return Key(name=name)

    if len(data) == 1 and data[0] >= 0x20:
        char = data.decode("utf-8", errors="replace")
        return Key(name="char", char=char)

    if len(data) > 1:
        try:
            char = data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            if len(char) == 1 and char.isprintable():
                return Key(name="char", char=char)

    return Key(name="unknown", char=None)


def _utf8_char_width(first_byte: int) -> int:
    if first_byte & 0xE0 == 0xC0:
        return 2
    if first_byte & 0xF0 == 0xE0:
        return 3
    if first_byte & 0xF8 == 0xF0:
        return 4
    return 1


_KNOWN_SEQUENCES = sorted(_SIMPLE_SEQUENCES, key=len, reverse=True)


def split_keys(data: bytes) -> list[bytes]:
    """Split a raw terminal read into individual key chunks.

    A single `read()` can return several keystrokes at once (fast typing, a
    paste, or the OS simply coalescing buffered input) — this walks the chunk
    so each recognized escape sequence or character is dispatched as its own
    key, instead of the whole blob being handed to `parse_key` as one unit.
    """
    chunks: list[bytes] = []
    i = 0
    n = len(data)
    while i < n:
        matched_seq: bytes | None = None
        for seq in _KNOWN_SEQUENCES:
            if data.startswith(seq, i):
                matched_seq = seq
                break
        if matched_seq is not None:
            chunks.append(matched_seq)
            i += len(matched_seq)
            continue

        width = _utf8_char_width(data[i])
        chunks.append(data[i : i + width])
        i += width
    return chunks
