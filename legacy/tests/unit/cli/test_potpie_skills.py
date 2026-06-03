from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli import skill_catalog, skill_lock, skill_targets
from adapters.inbound.cli.skill_manager import SkillManager

pytestmark = pytest.mark.unit


def _write_skill_md(path: Path, *, name: str, description: str) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                "Body",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _make_packaged_catalog(root: Path, skill_ids: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for sid in skill_ids:
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        _write_skill_md(d / "SKILL.md", name=f"Skill {sid}", description=f"Desc {sid}")
        # Add a file that participates in hashing.
        (d / "extra.txt").write_text(f"extra {sid}\n", encoding="utf-8")
        # Add ignored hash files/dirs.
        (d / "metadata.json").write_text('{"ignored": true}\n', encoding="utf-8")
        (d / ".git").mkdir(exist_ok=True)
        (d / ".git" / "config").write_text("[core]\nignored = true\n", encoding="utf-8")
    return root


def _invoke_skills(args: list[str]) -> tuple[int, dict]:
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["--json", "skills", *args])
    stdout = result.stdout.strip()
    payload = json.loads(stdout) if stdout else {}
    return result.exit_code, payload


@pytest.fixture()
def packaged_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Provide a deterministic bundled catalog for all tests in this module.
    skills = [
        "potpie-agent-context",
        "potpie-pot-scope",
        "potpie-cli",
        "potpie-cli-troubleshooting",
        "alpha",
        "beta",
    ]
    root = _make_packaged_catalog(tmp_path / "bundled_skills", skills)
    monkeypatch.setattr(skill_catalog, "_packaged_skills_root", lambda: root)
    return root


def test_catalog_discovery_only_immediate_children(packaged_catalog: Path) -> None:
    # Nested SKILL.md must not be discovered as its own skill.
    nested = packaged_catalog / "alpha" / "nested-child"
    nested.mkdir(parents=True, exist_ok=True)
    _write_skill_md(nested / "SKILL.md", name="Nested", description="Nested")

    entries, diagnostics = skill_catalog.discover_bundled_skills()
    assert diagnostics == []
    ids = [e.id for e in entries]
    assert "alpha" in ids
    assert "nested-child" not in ids


def test_hash_path_dir_is_deterministic_and_ignores_rules(tmp_path: Path) -> None:
    root = tmp_path / "dir"
    root.mkdir()
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")

    # Ignored artifacts
    (root / "metadata.json").write_text("ignored", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ignored", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\0\1")

    h1 = skill_catalog.hash_path_dir(root)
    # Touching ignored files must not change hash.
    (root / "metadata.json").write_text("ignored but changed", encoding="utf-8")
    (root / ".git" / "HEAD").write_text("ignored2", encoding="utf-8")
    h2 = skill_catalog.hash_path_dir(root)
    assert h1 == h2

    # Changing a tracked file must change hash.
    (root / "a.txt").write_text("a2", encoding="utf-8")
    h3 = skill_catalog.hash_path_dir(root)
    assert h3 != h1


def test_lockfile_read_write_sorted_keys_and_version_checks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    lock = skill_lock.empty_lock()
    lock["skills"] = {
        "zeta": {"installedHash": "sha256:z", "templateHash": "sha256:z"},
        "alpha": {"installedHash": "sha256:a", "templateHash": "sha256:a"},
    }
    skill_lock.write_lock(repo, lock)

    raw = (repo / ".agents" / "skills-lock.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["version"] == skill_lock.LOCK_VERSION
    assert list(parsed["skills"].keys()) == ["alpha", "zeta"]

    # Unsupported version yields diagnostic and empty lock.
    (repo / ".agents" / "skills-lock.json").write_text(
        json.dumps({"version": 999, "skills": {}}), encoding="utf-8"
    )
    lock2, diag = skill_lock.read_lock(repo)
    assert lock2["version"] == skill_lock.LOCK_VERSION
    assert diag and diag["code"] == "INVALID_LOCKFILE"


def test_agent_recommendation_matrix_matches_spec() -> None:
    assert skill_targets.recommended_skill_ids("default") == [
        "potpie-agent-context",
        "potpie-pot-scope",
    ]
    assert skill_targets.recommended_skill_ids("cursor") == [
        "potpie-agent-context",
        "potpie-pot-scope",
        "potpie-cli",
        "potpie-cli-troubleshooting",
    ]
    assert skill_targets.recommended_skill_ids("claude") == [
        "potpie-agent-context",
        "potpie-pot-scope",
        "potpie-cli",
    ]
    assert skill_targets.recommended_skill_ids("codex") == [
        "potpie-agent-context",
        "potpie-pot-scope",
        "potpie-cli",
    ]


def test_cli_list_available_json_contract(packaged_catalog: Path, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code, payload = _invoke_skills(["list", "--available", "--path", str(repo)])
    assert code == 0
    assert payload["ok"] is True
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert payload["mode"] == "available"
    assert isinstance(payload["skills"], list)


def test_cli_status_install_update_remove_doctor_end_to_end(
    packaged_catalog: Path, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # status --agent cursor --json
    code, payload = _invoke_skills(["status", "--agent", "cursor", "--path", str(repo)])
    assert code == 0
    assert payload["ok"] is True
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert payload["agent"] == "cursor"
    assert payload["missing"]  # recommended set not installed yet

    # install <id> --yes
    code, payload = _invoke_skills(
        ["install", "alpha", "--yes", "--path", str(repo), "--agent", "default"]
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert [row["id"] for row in payload["installed"]] == ["alpha"]
    assert (repo / ".agents" / "skills" / "alpha" / "SKILL.md").exists()
    assert (repo / ".agents" / "skills-lock.json").exists()

    # install --agent cursor --yes (installs recommended set)
    code, payload = _invoke_skills(
        ["install", "--agent", "cursor", "--yes", "--path", str(repo)]
    )
    assert code == 0
    assert payload["ok"] is True
    installed_ids = {row["id"] for row in payload["installed"]}
    assert installed_ids.issuperset(set(skill_targets.recommended_skill_ids("cursor")))

    # doctor --json
    code, payload = _invoke_skills(["doctor", "--agent", "cursor", "--path", str(repo)])
    assert code == 0
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert "diagnostics" in payload

    # mutate the packaged template for alpha to force "outdated"
    (packaged_catalog / "alpha" / "extra.txt").write_text("changed\n", encoding="utf-8")

    # update --yes should update outdated skills (alpha included)
    code, payload = _invoke_skills(["update", "--yes", "--path", str(repo)])
    assert code == 0
    assert payload["ok"] is True
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION

    # remove <id> --yes
    code, payload = _invoke_skills(["remove", "alpha", "--yes", "--path", str(repo)])
    assert code == 0
    assert payload["ok"] is True
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert not (repo / ".agents" / "skills" / "alpha").exists()


def test_json_error_envelopes_unknown_skill_needs_yes_locally_modified_unowned(
    packaged_catalog: Path, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # UNKNOWN_SKILL
    code, payload = _invoke_skills(["install", "does-not-exist", "--yes", "--path", str(repo)])
    assert code == 1
    assert payload["ok"] is False
    assert payload["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert payload["error"]["code"] == "UNKNOWN_SKILL"

    # Install alpha once.
    code, _ = _invoke_skills(["install", "alpha", "--yes", "--path", str(repo)])
    assert code == 0

    # NEEDS_YES when overwriting without --yes in non-interactive mode.
    # (If nothing changed, install is a no-op; force an overwrite by changing
    # the bundled template hash.)
    (packaged_catalog / "alpha" / "extra.txt").write_text("template changed\n", encoding="utf-8")
    code, payload = _invoke_skills(["install", "alpha", "--path", str(repo)])
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "NEEDS_YES"

    # LOCALLY_MODIFIED_REFUSED for update of an owned locally modified skill.
    alpha_md = repo / ".agents" / "skills" / "alpha" / "SKILL.md"
    alpha_md.write_text(alpha_md.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")
    code, payload = _invoke_skills(["update", "alpha", "--yes", "--path", str(repo)])
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "LOCALLY_MODIFIED_REFUSED"

    # UNOWNED_SKILL_REFUSED: installed dir exists but lock entry missing.
    unowned_dir = repo / ".agents" / "skills" / "beta"
    unowned_dir.mkdir(parents=True, exist_ok=True)
    _write_skill_md(unowned_dir / "SKILL.md", name="Beta", description="Beta")
    lock_path = repo / ".agents" / "skills-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["skills"].pop("beta", None)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code, payload = _invoke_skills(["remove", "beta", "--yes", "--path", str(repo)])
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "UNOWNED_SKILL_REFUSED"


def test_status_computation_missing_outdated_locally_modified_and_lock_diagnostics(
    packaged_catalog: Path, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    manager = SkillManager(repo, agent="cursor")
    status = manager.status()
    assert status["ok"] is True
    assert status["schemaVersion"] == skill_catalog.SCHEMA_VERSION
    assert status["missing"]  # recommended set not installed

    # Install alpha to create lock ownership.
    _invoke_skills(["install", "alpha", "--yes", "--path", str(repo)])

    # Make it locally modified.
    alpha_md = repo / ".agents" / "skills" / "alpha" / "SKILL.md"
    alpha_md.write_text(alpha_md.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")
    status2 = SkillManager(repo, agent="default").status()
    assert any(row["id"] == "alpha" for row in status2["installed"])
    assert any(row["id"] == "alpha" for row in status2["locallyModified"])

    # Break lockfile -> diagnostics should include INVALID_LOCKFILE.
    (repo / ".agents" / "skills-lock.json").write_text("{not json", encoding="utf-8")
    status3 = SkillManager(repo, agent="default").status()
    assert any(d.get("code") == "INVALID_LOCKFILE" for d in status3.get("diagnostics", []))
