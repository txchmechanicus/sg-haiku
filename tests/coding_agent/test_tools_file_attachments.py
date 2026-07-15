from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from coding_agent.tools.file_attachments import (
    FileProcessingError,
    process_file_arguments,
)
from PIL import Image


def test_text_file_wraps_in_file_tag(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")

    result = process_file_arguments([str(path)], tmp_path)

    assert result.text == f'<file name="{path}">\nhello world\n</file>'
    assert result.images == []


def test_multiple_text_files_are_joined(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("first", encoding="utf-8")
    b.write_text("second", encoding="utf-8")

    result = process_file_arguments([str(a), str(b)], tmp_path)

    assert f'<file name="{a}">\nfirst\n</file>' in result.text
    assert f'<file name="{b}">\nsecond\n</file>' in result.text


def test_image_file_becomes_image_content(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), color="red").save(buffer, format="PNG")
    path = tmp_path / "pic.png"
    path.write_bytes(buffer.getvalue())

    result = process_file_arguments([str(path)], tmp_path)

    assert result.text == ""
    assert len(result.images) == 1
    assert result.images[0].mimeType == "image/png"
    assert result.images[0].data


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileProcessingError, match="File not found"):
        process_file_arguments([str(tmp_path / "nope.txt")], tmp_path)


def test_empty_file_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    result = process_file_arguments([str(path)], tmp_path)

    assert result.text == ""
    assert result.images == []


def test_relative_path_resolves_against_cwd(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("relative", encoding="utf-8")

    result = process_file_arguments(["notes.txt"], tmp_path)

    assert "relative" in result.text


def test_tilde_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes.txt").write_text("home content", encoding="utf-8")

    result = process_file_arguments(["~/notes.txt"], tmp_path)

    assert "home content" in result.text


def test_no_files_returns_empty(tmp_path: Path) -> None:
    result = process_file_arguments([], tmp_path)

    assert result.text == ""
    assert result.images == []


def test_directory_raises(tmp_path: Path) -> None:
    directory = tmp_path / "subdir"
    directory.mkdir()

    with pytest.raises(FileProcessingError, match="Not a file"):
        process_file_arguments([str(directory)], tmp_path)
