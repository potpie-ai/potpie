"""Every packaged agent template reaches the wheel.

Hatchling drops files that ``.gitignore`` matches, even tracked ones, so a
template under an ignored directory (``.claude/`` is ignored for local Claude
Code state) silently vanishes from the published package and the installer has
nothing to copy. Such a file must be listed in ``[tool.hatch.build.force-include]``.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = "potpie/cli/templates"


def _git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def tracked_templates() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    listed = _git("ls-files", TEMPLATES)
    if listed.returncode != 0:
        pytest.skip("git ls-files unavailable")
    return [line for line in listed.stdout.splitlines() if line]


def test_ignored_templates_are_force_included(tracked_templates: list[str]) -> None:
    ignored = _git(
        "check-ignore", "--no-index", "--stdin", stdin="\n".join(tracked_templates)
    )
    hidden = {line for line in ignored.stdout.splitlines() if line}
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_included = set(metadata["tool"]["hatch"]["build"]["force-include"])

    assert hidden - force_included == set()


def test_single_source_layout_is_what_ships(tracked_templates: list[str]) -> None:
    shipped = set(tracked_templates)

    assert f"{TEMPLATES}/routing/POTPIE.md" in shipped
    assert not any("/global_agent_bundle/" in path for path in shipped)
    assert not any(
        path.endswith("/SKILL.md") and "/agent_bundle/.agents/skills/" not in path
        for path in shipped
    )
