"""``potpie.build_info`` and the hatch hook that feeds it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from potpie import build_info


def test_describe_reports_the_distribution_and_the_engine_separately() -> None:
    info = build_info.describe()
    assert info["name"] == "potpie"
    assert isinstance(info["version"], str) and info["version"]
    assert set(info["build"]) == {"rev", "dirty", "built_at"}
    assert info["engine"]["name"] == "potpie-context-engine"
    assert isinstance(info["engine"]["version"], str) and info["engine"]["version"]


def test_human_line_shows_the_short_rev_and_the_dirty_mark() -> None:
    base = {"name": "potpie", "version": "2.0.0"}
    clean = {**base, "build": {"rev": "81da1550e39044a3f72265f5de5b07fee9af856d", "dirty": False}}
    dirty = {**base, "build": {"rev": "81da1550e39044a3f72265f5de5b07fee9af856d", "dirty": True}}
    unknown = {**base, "build": {"rev": None, "dirty": None}}
    assert build_info.human_line(clean) == "potpie 2.0.0 (81da1550e3)"
    assert build_info.human_line(dirty) == "potpie 2.0.0 (81da1550e3, dirty)"
    assert build_info.human_line(unknown) == "potpie 2.0.0 (build rev unknown)"


def test_short_rev_tolerates_anything() -> None:
    assert build_info.short_rev(None) is None
    assert build_info.short_rev("") is None
    assert build_info.short_rev(12) is None
    assert build_info.short_rev("abcdef0123456789") == "abcdef0123"


def test_build_stamp_is_a_dict_even_when_the_file_is_absent() -> None:
    assert isinstance(build_info.build_stamp(), dict)




# --- a checkout outranks the stamp -------------------------------------------


def _clear_caches() -> None:
    """Drop the per-process caches, whichever of them a test left patched."""
    for name in ("checkout_root", "build_stamp", "_checkout_dirty"):
        clear = getattr(getattr(build_info, name), "cache_clear", None)
        if clear is not None:
            clear()


def _head(cwd: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_inside_a_checkout_the_stamp_is_head_not_the_baked_file(monkeypatch, tmp_path: Path) -> None:
    """An editable install runs whatever the checkout holds now; a stamp written
    at the last `make cli-install` named a rev that no longer existed in the
    code, and `stale` compared that stale stamp against itself."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "one")
    _clear_caches()
    monkeypatch.setattr(build_info, "checkout_root", lambda: tmp_path)
    monkeypatch.setattr(build_info, "_baked_stamp", lambda: {"rev": "baked-and-stale", "dirty": False})

    stamp = build_info.build_stamp()
    assert stamp["rev"] == _head(tmp_path)
    assert stamp["source"] == build_info.STAMP_SOURCE_CHECKOUT
    assert stamp["built_at"] is None
    assert set(build_info.describe()["build"]) == {"rev", "dirty", "built_at"}
    assert build_info.describe()["build"]["dirty"] is False

    # A new commit moves the rev the next process sees (this one is cached).
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "two")
    assert build_info.build_stamp()["rev"] != _head(tmp_path)
    build_info.build_stamp.cache_clear()
    assert build_info.build_stamp()["rev"] == _head(tmp_path)

    # `dirty` is the hook's definition: modified tracked files, not untracked ones.
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")
    build_info._checkout_dirty.cache_clear()
    assert build_info.describe()["build"]["dirty"] is False
    _git(tmp_path, "add", "loose.txt")
    _git(tmp_path, "commit", "-q", "-m", "track")
    (tmp_path / "loose.txt").write_text("y", encoding="utf-8")
    build_info._checkout_dirty.cache_clear()
    assert build_info.describe()["build"]["dirty"] is True
    _clear_caches()


def test_the_rev_is_read_from_git_files_for_branches_packed_refs_and_detached_heads(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "one")
    head = _head(tmp_path)
    assert build_info._rev_from_git_files(tmp_path) == head

    _git(tmp_path, "pack-refs", "--all")
    assert not (tmp_path / ".git" / "refs" / "heads" / "main").exists()
    assert build_info._rev_from_git_files(tmp_path) == head

    _git(tmp_path, "checkout", "-q", "--detach", head)
    assert build_info._rev_from_git_files(tmp_path) == head

    # A worktree's `.git` is a file pointing at its gitdir; refs live in the common dir.
    _git(tmp_path, "checkout", "-q", "main")
    wt = tmp_path.parent / (tmp_path.name + "-wt")
    _git(tmp_path, "worktree", "add", "-q", str(wt), "-b", "side")
    assert (wt / ".git").is_file()
    assert build_info._rev_from_git_files(wt) == head

    assert build_info._rev_from_git_files(tmp_path / "nowhere") is None


def test_outside_a_checkout_the_baked_stamp_stands(monkeypatch) -> None:
    _clear_caches()
    monkeypatch.setattr(build_info, "checkout_root", lambda: None)
    monkeypatch.setattr(build_info, "_baked_stamp", lambda: {"rev": "baked", "dirty": True, "built_at": "t"})
    assert build_info.build_stamp() == {"rev": "baked", "dirty": True, "built_at": "t"}
    assert build_info.describe()["build"] == {"rev": "baked", "dirty": True, "built_at": "t"}
    _clear_caches()


def test_cli_version_is_the_distribution_version_plus_the_short_rev(monkeypatch) -> None:
    """Telemetry tagged every command `cli_version: 0.1.0` — the engine library's
    constant — so no dashboard could tell one CLI build from another."""
    version = build_info.distribution_version() or "unknown"
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": "81da1550e39044a3f72265f5de5b07fee9af856d"})
    assert build_info.cli_version() == f"{version}+81da1550e3"
    assert build_info.cli_release() == f"potpie-cli@{version}+81da1550e3"
    assert not build_info.cli_version().startswith("0.1.0")
    monkeypatch.setattr(build_info, "build_stamp", lambda: {})
    assert build_info.cli_version() == version


# --- the hook ---------------------------------------------------------------


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def test_hook_stamps_rev_and_dirty_from_a_checkout(tmp_path: Path) -> None:
    import hatch_build

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "init")

    info = hatch_build.collect_build_info(tmp_path, "2.0.0")
    assert info["version"] == "2.0.0"
    assert isinstance(info["rev"], str) and len(info["rev"]) == 40
    assert info["dirty"] is False
    assert info["built_at"].endswith("+00:00")

    # Untracked files are not dirt: uv drops an untracked `.ok` marker into
    # every checkout it builds from, and a clean rev must stamp as clean there.
    (tmp_path / ".ok").write_text("", encoding="utf-8")
    assert hatch_build.collect_build_info(tmp_path, "2.0.0")["dirty"] is False

    # A modified tracked file is.
    (tmp_path / "tracked.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "track")
    (tmp_path / "tracked.txt").write_text("b", encoding="utf-8")
    assert hatch_build.collect_build_info(tmp_path, "2.0.0")["dirty"] is True

    target = hatch_build.write_build_info(tmp_path, "2.0.0")
    assert target == tmp_path / "potpie" / "_build.json"
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["rev"] == hatch_build.collect_build_info(tmp_path, "2.0.0")["rev"]
    assert written["dirty"] is True


def test_hook_outside_git_keeps_an_existing_stamp_rather_than_nulling_it(tmp_path: Path) -> None:
    """A wheel built from an sdist has no .git; the sdist's stamp must survive."""
    import hatch_build

    assert hatch_build.collect_build_info(tmp_path, "2.0.0")["rev"] is None

    existing = tmp_path / "potpie" / "_build.json"
    existing.parent.mkdir()
    existing.write_text(json.dumps({"version": "2.0.0", "rev": "from-the-sdist"}), encoding="utf-8")
    hatch_build.write_build_info(tmp_path, "2.0.0")
    assert json.loads(existing.read_text(encoding="utf-8"))["rev"] == "from-the-sdist"

    # With nothing to preserve, nulls are still written: absence of the file
    # and "built outside git" are different facts.
    bare = tmp_path / "bare"
    bare.mkdir()
    written = hatch_build.write_build_info(bare, "2.0.0")
    assert json.loads(written.read_text(encoding="utf-8"))["rev"] is None
