from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "potpie" / "context-engine"


def _pinned_version(requirements: list[str], package: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(package)}(?:\[[^]]+\])?==(?P<version>[^;\s]+)$",
        re.IGNORECASE,
    )
    matches = [
        match.group("version")
        for requirement in requirements
        if (match := pattern.match(requirement)) is not None
    ]
    assert len(matches) == 1, f"expected one exact {package} pin, got {matches}"
    return matches[0]


def test_documents_extra_matches_base_and_workspace_engine_version() -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    engine = tomllib.loads((ENGINE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    base_version = _pinned_version(
        root["project"]["dependencies"], "potpie-context-engine"
    )
    documents_version = _pinned_version(
        root["project"]["optional-dependencies"]["documents"],
        "potpie-context-engine",
    )

    assert documents_version == base_version == engine["project"]["version"]
