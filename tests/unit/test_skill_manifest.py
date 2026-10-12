"""One install manifest per skills target, and the move off the three old files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, skills
from potpie.skills.catalog import catalog_by_id
from potpie.skills.targets import AgentTarget

LEGACY_PREFIXES = ("skills_", "skill_hashes_", "skill_disabled_")


@pytest.fixture(autouse=True)
def _reset_cli_output_mode():
    _common.set_json(False)
    yield
    _common.set_json(False)


@pytest.fixture()
def potpie_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "potpie"
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "in_memory")
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "harness"))
    return home


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def _skill_state_files(home: Path) -> list[str]:
    return sorted(p.name for p in home.glob("skill*.json"))


def _write_legacy(home: Path, stem: str, **files: object) -> None:
    """Write the old per-target files: ``versions=``, ``hashes=``, ``disabled=``."""
    home.mkdir(parents=True, exist_ok=True)
    prefixes = {
        "versions": "skills_",
        "hashes": "skill_hashes_",
        "disabled": "skill_disabled_",
    }
    for kind, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content)
        (home / f"{prefixes[kind]}{stem}.json").write_text(text, encoding="utf-8")


def _cli(*args: str):
    return CliRunner().invoke(skills.skills_app, list(args))


def test_a_fresh_install_keeps_one_manifest_per_target(
    potpie_home: Path, tmp_path: Path
) -> None:
    repo = _repo(tmp_path)
    assert _cli("install", "--agent", "codex").exit_code == 0
    assert _cli("install", "--agent", "codex", "--path", str(repo)).exit_code == 0
    assert _cli("remove", "potpie-graph", "--agent", "codex").exit_code == 0

    names = _skill_state_files(potpie_home)
    assert len(names) == 2
    assert "skill_manifest_codex_global.json" in names
    project = next(n for n in names if n != "skill_manifest_codex_global.json")
    assert project.startswith("skill_manifest_codex_project_repo_")
    manifest = json.loads(
        (potpie_home / "skill_manifest_codex_global.json").read_text()
    )
    records = manifest["skills"]
    skill_md = tmp_path / "harness" / ".agents" / "skills" / "potpie-cli" / "SKILL.md"
    assert records["potpie-cli"] == {
        "version": catalog_by_id()["potpie-cli"].version,
        "sha256": hashlib.sha256(skill_md.read_bytes()).hexdigest(),
    }
    # Removed by id: no longer installed, still recorded as disabled.
    assert records["potpie-graph"]["disabled"] is True
    assert "version" not in records["potpie-graph"]


@pytest.mark.parametrize("scope", ["global", "project"])
def test_the_three_old_files_fold_into_one_manifest(
    potpie_home: Path, tmp_path: Path, scope: str
) -> None:
    target = AgentTarget(
        agent="claude", scope=scope, path=_repo(tmp_path), home=potpie_home
    )
    stem = target.manifest.stem
    _write_legacy(
        potpie_home,
        stem,
        versions={"potpie-cli": "3", "potpie-graph": "6"},
        hashes={"potpie-cli": "ab" * 32},
        disabled={"potpie-graph": "disabled"},
    )

    assert target.manifest.read() == {
        "potpie-cli": {"version": "3", "sha256": "ab" * 32},
        "potpie-graph": {"version": "6", "disabled": True},
    }
    assert _skill_state_files(potpie_home) == [f"skill_manifest_{stem}.json"]
    assert target.disabled() == frozenset({"potpie-graph"})


def test_the_released_versions_only_file_migrates(
    potpie_home: Path, tmp_path: Path
) -> None:
    """2.0.1 wrote only ``skills_<agent>_global.json``: versions, no hashes."""
    target = AgentTarget(agent="claude", home=potpie_home)
    target.install(skill_id="potpie-cli", version="3")
    target.manifest.path.unlink()
    _write_legacy(potpie_home, "claude_global", versions={"potpie-cli": "2"})
    (target.skills_root / "potpie-cli" / "SKILL.md").write_text("edited")

    assert target.installed() == {"potpie-cli": "2"}
    # No recorded hash, so an edit cannot be told from an old install.
    assert target.locally_modified(skill_id="potpie-cli") is False
    assert _skill_state_files(potpie_home) == ["skill_manifest_claude_global.json"]


def test_a_corrupt_or_partial_old_file_migrates_what_is_readable(
    potpie_home: Path,
) -> None:
    target = AgentTarget(agent="codex", home=potpie_home)
    _write_legacy(
        potpie_home,
        "codex_global",
        versions="{not json",
        hashes=["not", "a", "dict"],
        disabled={"potpie-graph": "disabled"},
    )

    assert target.manifest.read() == {"potpie-graph": {"disabled": True}}
    assert target.installed() == {}
    assert _skill_state_files(potpie_home) == ["skill_manifest_codex_global.json"]


def test_migration_runs_once_and_the_manifest_wins_after(potpie_home: Path) -> None:
    target = AgentTarget(agent="codex", home=potpie_home)
    _write_legacy(potpie_home, "codex_global", versions={"potpie-cli": "3"})
    first = target.manifest.read()
    written = target.manifest.path.read_bytes()

    # An older potpie run after the move writes its own files again.
    _write_legacy(potpie_home, "codex_global", versions={"potpie-cli": "1"})

    assert target.manifest.read() == first == {"potpie-cli": {"version": "3"}}
    assert target.manifest.path.read_bytes() == written
    assert (potpie_home / "skills_codex_global.json").exists()


def test_a_corrupt_manifest_reads_as_nothing_recorded(potpie_home: Path) -> None:
    target = AgentTarget(agent="codex", home=potpie_home)
    for text in ("{not json", '["a"]', '{"skills": ["a"]}', '{"skills": {"x": 1}}'):
        target.manifest.path.parent.mkdir(parents=True, exist_ok=True)
        target.manifest.path.write_text(text, encoding="utf-8")
        assert target.manifest.read() == {}


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs a home the current user cannot write",
)
def test_an_unwritable_home_keeps_the_old_files(potpie_home: Path) -> None:
    target = AgentTarget(agent="codex", home=potpie_home)
    _write_legacy(potpie_home, "codex_global", versions={"potpie-cli": "3"})
    potpie_home.chmod(0o500)
    try:
        assert target.manifest.read() == {"potpie-cli": {"version": "3"}}
    finally:
        potpie_home.chmod(0o700)
    assert _skill_state_files(potpie_home) == ["skills_codex_global.json"]


def test_skills_status_json_is_unchanged_by_the_migration(
    potpie_home: Path, tmp_path: Path
) -> None:
    repo = _repo(tmp_path)
    where = ("--agent", "claude", "--path", str(repo))
    _cli("install", *where)
    _cli("remove", "potpie-graph", *where)
    (repo / ".claude" / "skills" / "potpie-cli" / "SKILL.md").write_text("edited")
    _common.set_json(True)
    before = json.loads(_cli("status", *where).output)
    listed = json.loads(_cli("list", *where).output)

    # Put the same state back the way earlier releases stored it.
    target = AgentTarget(agent="claude", scope="project", path=repo, home=potpie_home)
    records = target.manifest.read()
    target.manifest.path.unlink()
    _write_legacy(
        potpie_home,
        target.manifest.stem,
        versions={s: r["version"] for s, r in records.items() if "version" in r},
        hashes={s: r["sha256"] for s, r in records.items() if "sha256" in r},
        disabled={s: "disabled" for s, r in records.items() if r.get("disabled")},
    )

    assert json.loads(_cli("status", *where).output) == before
    assert json.loads(_cli("list", *where).output) == listed
    assert before["drifted"] == ["potpie-cli"]
    assert before["disabled"] == ["potpie-graph"]
    assert target.manifest.read() == records
    assert _skill_state_files(potpie_home) == [target.manifest.path.name]
