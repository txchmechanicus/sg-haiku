from __future__ import annotations

from pathlib import Path

from coding_agent.modes.interactive.file_mentions import (
    extract_at_mentions,
    find_at_mention_span,
    list_file_mentions,
)


class TestFindAtMentionSpan:
    def test_at_start_of_input(self) -> None:
        assert find_at_mention_span("@notes", 6) == (0, 6)

    def test_mid_sentence(self) -> None:
        text = "look at @notes.txt please"
        # cursor right after "@notes.txt"
        cursor = len("look at @notes.txt")
        assert find_at_mention_span(text, cursor) == (8, cursor)

    def test_no_active_mention_after_space(self) -> None:
        text = "look at @notes.txt please"
        assert find_at_mention_span(text, len(text)) is None

    def test_bare_at_with_nothing_typed_yet(self) -> None:
        assert find_at_mention_span("@", 1) == (0, 1)

    def test_no_at_token_under_cursor(self) -> None:
        assert find_at_mention_span("hello world", 5) is None

    def test_email_like_token_not_preceded_by_whitespace_is_not_a_mention(self) -> None:
        # "user@example.com" -- the "@" isn't preceded by start-of-string or whitespace at
        # the point the cursor sits right after "@example", so it must not trigger.
        text = "user@example"
        assert find_at_mention_span(text, len(text)) is None

    def test_cursor_clamped_to_text_length(self) -> None:
        assert find_at_mention_span("@notes", 999) == (0, 6)


class TestListFileMentions:
    def test_lists_directory_contents_filtered_by_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("", encoding="utf-8")
        (tmp_path / "novel.md").write_text("", encoding="utf-8")
        (tmp_path / "other.txt").write_text("", encoding="utf-8")

        matches = list_file_mentions("no", tmp_path)

        names = {match.insert for match in matches}
        assert names == {"notes.txt", "novel.md"}

    def test_directories_sort_before_files(self, tmp_path: Path) -> None:
        (tmp_path / "zdir").mkdir()
        (tmp_path / "afile.txt").write_text("", encoding="utf-8")

        matches = list_file_mentions("", tmp_path)

        assert matches[0].is_dir is True
        assert matches[0].display == "zdir/"

    def test_subdirectory_navigation(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")

        matches = list_file_mentions("src/ma", tmp_path)

        assert len(matches) == 1
        assert matches[0].insert == "src/main.py"

    def test_dotfiles_hidden_unless_prefix_starts_with_dot(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("", encoding="utf-8")

        assert [m.insert for m in list_file_mentions("", tmp_path)] == ["visible.txt"]
        assert [m.insert for m in list_file_mentions(".", tmp_path)] == [".env"]

    def test_nonexistent_directory_returns_empty(self, tmp_path: Path) -> None:
        assert list_file_mentions("does-not-exist/x", tmp_path) == []

    def test_absolute_path_query(self, tmp_path: Path) -> None:
        (tmp_path / "abs.txt").write_text("", encoding="utf-8")

        matches = list_file_mentions(f"{tmp_path}/abs", tmp_path)

        assert len(matches) == 1
        assert matches[0].insert == f"{tmp_path}/abs.txt"


class TestExtractAtMentions:
    def test_extracts_single_mention(self) -> None:
        assert extract_at_mentions("look at @notes.txt please") == ["notes.txt"]

    def test_extracts_multiple_mentions(self) -> None:
        assert extract_at_mentions("compare @a.txt and @b.txt") == ["a.txt", "b.txt"]

    def test_no_mentions(self) -> None:
        assert extract_at_mentions("just a normal message") == []

    def test_email_like_token_not_extracted(self) -> None:
        assert extract_at_mentions("email me at user@example.com") == []

    def test_mention_at_very_start(self) -> None:
        assert extract_at_mentions("@file.txt summarize this") == ["file.txt"]
