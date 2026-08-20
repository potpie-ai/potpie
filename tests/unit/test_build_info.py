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
