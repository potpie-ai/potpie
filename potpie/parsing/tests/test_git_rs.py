import subprocess
from pathlib import Path
from shutil import which

import pytest

from parsing_rs import (
    FileEntry,
    GitParseError,
    GitRefNotFoundError,
    GitRepositoryNotFoundError,
    bare_clone,
    list_files,
)


@pytest.fixture(scope="module")
def local_git_repo(tmp_path_factory):
    git_executable = which("git")
    assert git_executable is not None
    repo_dir = tmp_path_factory.mktemp("source_repo")

    (repo_dir / "README.md").write_text("# Test Project\n")
    (repo_dir / "Makefile").write_text("all:\n\techo done\n")

    src = repo_dir / "src"
    src.mkdir()
    (src / "main.rs").write_text("fn main() {}\n")

    lib = src / "lib"
    lib.mkdir()
    (lib / "utils.py").write_text("def helper(): pass\n")

    subprocess.run([git_executable, "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        [git_executable, "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [git_executable, "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run([git_executable, "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        [git_executable, "commit", "-m", "initial"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    return repo_dir


def _bare_clone_repo(local_git_repo: Path, destination: Path) -> Path:
    bare_clone(f"file://{local_git_repo}", str(destination), "HEAD", None)
    return destination


def test_bare_clone_local_repo_success(local_git_repo, tmp_path):
    dest = tmp_path / "dest"

    bare_clone(f"file://{local_git_repo}", str(dest), "HEAD", None)

    assert dest.joinpath("HEAD").exists()
    assert dest.joinpath("objects").exists()
    assert dest.joinpath("refs").exists()


def test_bare_clone_nonexistent_repo(tmp_path):
    with pytest.raises(GitRepositoryNotFoundError):
        bare_clone(f"file://{tmp_path / 'missing-repo'}", str(tmp_path / "dest"), "HEAD", None)


def test_bare_clone_bad_ref_cleans_dest(local_git_repo, tmp_path):
    dest = tmp_path / "dest"

    with pytest.raises(GitRefNotFoundError):
        bare_clone(f"file://{local_git_repo}", str(dest), "missing-ref", None)

    assert not dest.exists()


def test_list_files_returns_file_entries(local_git_repo, tmp_path):
    bare_repo = _bare_clone_repo(local_git_repo, tmp_path / "bare")

    entries = list_files(str(bare_repo), "HEAD")

    assert isinstance(entries, list)
    assert entries
    assert all(isinstance(entry, FileEntry) for entry in entries)


def test_list_files_directory_entries_included(local_git_repo, tmp_path):
    bare_repo = _bare_clone_repo(local_git_repo, tmp_path / "bare")

    entries = list_files(str(bare_repo), "HEAD")

    assert any(entry.kind == "directory" for entry in entries)


def test_list_files_field_correctness(local_git_repo, tmp_path):
    bare_repo = _bare_clone_repo(local_git_repo, tmp_path / "bare")

    entries = {entry.path: entry for entry in list_files(str(bare_repo), "HEAD")}

    readme = entries["README.md"]
    assert readme.path == "README.md"
    assert readme.name == "README.md"
    assert readme.dir == ""
    assert readme.ext == "md"
    assert readme.depth == 0
    assert readme.kind == "file"

    main_rs = entries["src/main.rs"]
    assert main_rs.path == "src/main.rs"
    assert main_rs.name == "main.rs"
    assert main_rs.dir == "src"
    assert main_rs.ext == "rs"
    assert main_rs.depth == 1
    assert main_rs.kind == "file"

    makefile = entries["Makefile"]
    assert makefile.ext == ""


def test_list_files_invalid_repo_path(tmp_path):
    with pytest.raises(GitRepositoryNotFoundError):
        list_files(str(tmp_path / "nonexistent"), "HEAD")


def test_list_files_invalid_ref(local_git_repo, tmp_path):
    bare_repo = _bare_clone_repo(local_git_repo, tmp_path / "bare")

    with pytest.raises(GitRefNotFoundError):
        list_files(str(bare_repo), "nonexistent-ref")


def test_exception_hierarchy():
    assert issubclass(GitRepositoryNotFoundError, Exception)
    assert issubclass(GitRefNotFoundError, Exception)
    assert issubclass(GitParseError, Exception)
