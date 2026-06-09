# ruff: noqa: S101

from pathlib import Path
import tomllib
from typing import Protocol

import pytest

from parsing_rs import build_workspace_index


class ReadonlyResult(Protocol):
    pass


def _assert_readonly_assignment(
    result_name: str,
    result: ReadonlyResult,
    assignment: str,
) -> None:
    with pytest.raises(AttributeError):
        exec(assignment, {}, {result_name: result})  # noqa: S102


def _workspace(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "auth.rs").write_text('fn login_handler() {\n    println!("ok");\n}\n')
    _ = (src / "logo.png").write_bytes(b"\x89PNG")
    _ = (tmp_path / ".gitignore").write_text("ignored.txt\n")
    _ = (tmp_path / "ignored.txt").write_text("ignore me\n")
    return tmp_path


def test_build_workspace_index_returns_counts_and_indexes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    index = build_workspace_index(str(workspace))

    assert index.file_count() == 3
    assert index.content_file_count() == 1

    files = index.search_files("auth", 10)
    assert len(files) == 1
    assert files[0].path == "src/auth.rs"


def test_build_workspace_index_search_content_content_type(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    index = build_workspace_index(str(workspace))

    results = index.search_content("login_handler", 10)
    assert results
    assert results[0].path == "src/auth.rs"
    assert results[0].line == 1
    assert "fn login_handler" in results[0].snippet
    assert results[0].score >= 1000


def test_build_workspace_index_returns_empty_on_empty_query(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    index = build_workspace_index(str(workspace))

    assert index.search_files("", 10) == []
    assert index.search_content("", 10) == []
    assert index.search_files("auth", 0) == []


def test_build_workspace_index_error_message_on_missing_repo(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(RuntimeError, match=r"workspace not found: .*does-not-exist"):
        _ = build_workspace_index(str(missing))


def test_build_workspace_index_error_message_on_file_root(tmp_path: Path) -> None:
    file_root = tmp_path / "file.txt"
    _ = file_root.write_text("not a directory\n")

    with pytest.raises(
        RuntimeError, match=r"workspace is not a directory: .*file\.txt"
    ):
        _ = build_workspace_index(str(file_root))


def test_build_workspace_index_result_fields_are_readonly_and_present(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    index = build_workspace_index(str(workspace))
    results = index.search_content("login_handler", 10)

    file_result = index.search_files("auth", 10)[0]
    assert hasattr(file_result, "path")
    assert hasattr(file_result, "score")

    content_result = results[0]
    assert hasattr(content_result, "path")
    assert hasattr(content_result, "line")
    assert hasattr(content_result, "snippet")
    assert hasattr(content_result, "score")

    _assert_readonly_assignment(
        "file_result", file_result, "file_result.path = 'mutated'"
    )
    _assert_readonly_assignment("file_result", file_result, "file_result.score = 0")
    _assert_readonly_assignment(
        "content_result", content_result, "content_result.path = 'mutated'"
    )
    _assert_readonly_assignment(
        "content_result", content_result, "content_result.line = 0"
    )
    _assert_readonly_assignment(
        "content_result", content_result, "content_result.snippet = 'mutated'"
    )
    _assert_readonly_assignment(
        "content_result", content_result, "content_result.score = 0"
    )


def test_python_package_supports_python_3_11_to_3_13() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["requires-python"] == ">=3.11,<3.14"
