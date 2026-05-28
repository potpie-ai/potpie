"""Pure query signal extraction for context intelligence."""

from __future__ import annotations

import pytest

from domain.intelligence_signals import SignalSet, extract_signals

pytestmark = pytest.mark.unit


class TestPrExtraction:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("look at PR #42", 42),
            ("look at PR42", 42),
            ("look at pr #7", 7),
            ("see pull request #15", 15),
            ("see pull request 99", 99),
            ("ref to #128 in commit", 128),
        ],
    )
    def test_pr_patterns(self, query: str, expected: int) -> None:
        assert extract_signals(query).mentioned_pr == expected

    def test_no_pr_when_no_mention(self) -> None:
        assert extract_signals("just a regular question").mentioned_pr is None

    def test_pr_short_form_at_word_start_does_not_match_hash(self) -> None:
        # The "\B#" anchor requires no word boundary before "#", so a leading "#"
        # at the very start of a string still has a non-word boundary before it
        # (start-of-string is a non-word position) and matches.
        assert extract_signals("#5 was the bug").mentioned_pr == 5


class TestFilePathExtraction:
    def test_extracts_python_file_path(self) -> None:
        signals = extract_signals("see app/services/auth.py for details")
        assert signals.mentioned_file_paths == ["app/services/auth.py"]

    def test_extracts_multiple_file_paths(self) -> None:
        signals = extract_signals("compare src/foo.ts with src/bar.tsx")
        assert signals.mentioned_file_paths == ["src/foo.ts", "src/bar.tsx"]

    def test_dedupes_repeated_file_paths(self) -> None:
        signals = extract_signals("src/x.go src/x.go src/x.go")
        assert signals.mentioned_file_paths == ["src/x.go"]

    @pytest.mark.parametrize(
        "ext", ["py", "ts", "tsx", "js", "jsx", "go", "rs", "java", "kt", "rb",
                "cs", "cpp", "h", "hpp", "c", "sql", "md", "yaml", "yml", "toml", "json"]
    )
    def test_supported_extensions(self, ext: str) -> None:
        signals = extract_signals(f"see dir/file.{ext}")
        assert signals.mentioned_file_paths == [f"dir/file.{ext}"]

    def test_unsupported_extension_not_extracted(self) -> None:
        # Extensions outside the whitelist must not match.
        assert extract_signals("see dir/file.xyz").mentioned_file_paths == []

    def test_no_paths_when_no_slash(self) -> None:
        # Must be at least one ``foo/`` segment.
        assert extract_signals("just file.py here").mentioned_file_paths == []


class TestSymbolExtraction:
    def test_camelcase_symbols(self) -> None:
        signals = extract_signals("FooBar wraps BazQux")
        assert "FooBar" in signals.mentioned_symbols
        assert "BazQux" in signals.mentioned_symbols

    def test_snake_case_symbols(self) -> None:
        signals = extract_signals("call extract_signals to derive things")
        assert "extract_signals" in signals.mentioned_symbols

    def test_short_lowercase_words_filtered_out(self) -> None:
        # "for" / "and" / "the" / "with" are excluded even though they have len ≥ 3
        # because they're not snake_case (no underscore).
        signals = extract_signals("the and for")
        assert signals.mentioned_symbols == []

    def test_symbols_capped_at_twelve(self) -> None:
        snake = " ".join(f"sym_{i}" for i in range(20))
        signals = extract_signals(snake)
        assert len(signals.mentioned_symbols) == 12

    def test_dedupes_symbols(self) -> None:
        signals = extract_signals("foo_bar foo_bar foo_bar")
        assert signals.mentioned_symbols.count("foo_bar") == 1


class TestIntentSignals:
    @pytest.mark.parametrize(
        "query",
        ["why was this changed?", "who introduced this?", "what was the rationale",
         "when was it removed", "merged feature branch"],
    )
    def test_history_keywords_set_needs_history(self, query: str) -> None:
        assert extract_signals(query).needs_history is True

    def test_pr_mention_implies_needs_history(self) -> None:
        signals = extract_signals("look at PR #1234")
        assert signals.needs_history is True

    def test_no_history_keywords(self) -> None:
        assert extract_signals("show structure of x").needs_history is False

    def test_ownership_requires_keyword_and_path(self) -> None:
        # Ownership intent requires BOTH an ownership keyword AND a file path.
        assert extract_signals("who owns app/auth.py").needs_ownership is True

    def test_ownership_keyword_alone_not_enough(self) -> None:
        # No path → no ownership intent.
        assert extract_signals("who owns this").needs_ownership is False

    def test_ownership_path_alone_not_enough(self) -> None:
        assert extract_signals("look at app/auth.py").needs_ownership is False

    def test_navigation_intent(self) -> None:
        assert extract_signals("what does this function do").is_code_navigation is True
        assert extract_signals("show me the structure").is_code_navigation is True

    def test_navigation_suppressed_by_history(self) -> None:
        # Even if navigation keywords are present, history intent wins.
        signals = extract_signals("show me why this was changed")
        assert signals.needs_history is True
        assert signals.is_code_navigation is False


class TestRawQueryHandling:
    def test_strips_whitespace(self) -> None:
        signals = extract_signals("   hello   ")
        assert signals.raw_query == "hello"

    def test_empty_query_yields_default_signal_set(self) -> None:
        signals = extract_signals("")
        assert signals == SignalSet(raw_query="")

    def test_none_query_treated_as_empty(self) -> None:
        # The function defends against None.
        signals = extract_signals(None)  # type: ignore[arg-type]
        assert signals.raw_query == ""
        assert signals.mentioned_pr is None
