"""Source metadata contract for the final package boundary."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE_EXTRAS = {
    "embedded",
    "http",
    "postgres",
    "neo4j",
    "embeddings",
    "github",
    "reconciliation",
    "hatchet",
    "observability",
}


def test_source_metadata_declares_final_distribution_boundary() -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    engine = tomllib.loads(
        (ROOT / "potpie/context-engine/pyproject.toml").read_text(encoding="utf-8")
    )

    assert root["project"]["scripts"] == {
        "potpie": "potpie.cli.main:main",
        "potpie-daemon": "potpie.daemon.main:main",
        "potpie-mcp": "potpie.mcp.main:main",
    }
    assert "potpie-context-engine[embedded]==0.2.0" in root["project"]["dependencies"]
    assert not any("[all]" in item for item in root["project"]["dependencies"])
    assert set(root["project"]["optional-dependencies"]) == ENGINE_EXTRAS

    assert engine["project"]["version"] == "0.2.0"
    assert engine["project"]["dependencies"] == [
        "filelock>=3.18,<4",
        "pydantic>=2.0",
    ]
    assert set(engine["project"]["optional-dependencies"]) == ENGINE_EXTRAS
    assert "scripts" not in engine["project"]
    assert "hooks" not in engine.get("tool", {}).get("hatch", {}).get("build", {})
