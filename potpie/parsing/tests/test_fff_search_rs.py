# ruff: noqa: S101

from pathlib import Path
import tomllib

import pytest

from parsing.parsing_rs import build_workspace_index


def _workspace(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "auth.rs").write_text('fn login_handler() {\n    println!("ok");\n}\n')
    _ = (src / "logo.png").write_bytes(b"\x89PNG")
    _ = (tmp_path / ".gitignore").write_text("ignored.txt\n")
    _ = (tmp_path / "ignored.txt").write_text("ignore me\n")
    return tmp_path


def test_workspace_index_exposes_counts_and_readonly_search_results(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    index = build_workspace_index(str(workspace))

    assert index.file_count() == 3
    assert index.content_file_count() == 1

    files = index.search_files("auth", 10)
    assert len(files) == 1
    assert files[0].path == "src/auth.rs"
    assert isinstance(files[0].score, (int, float))
    contents = index.search_content("login_handler", 10)
    assert contents
    assert contents[0].path == "src/auth.rs"
    assert contents[0].line == 1
    assert "fn login_handler" in contents[0].snippet
    assert contents[0].score >= 1000
    for result, assignments in (
        (files[0], {"path": "mutated", "score": 0}),
        (contents[0], {"path": "mutated", "line": 0, "snippet": "mutated", "score": 0}),
    ):
        for field, value in assignments.items():
            with pytest.raises(AttributeError):
                setattr(result, field, value)


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


@pytest.mark.source_contract
def test_python_package_supports_python_3_12_to_3_14() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["requires-python"] == ">=3.12,<3.15"
