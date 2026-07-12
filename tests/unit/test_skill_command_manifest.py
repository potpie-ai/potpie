"""The static skill command manifest stays aligned with Typer registration."""

from __future__ import annotations

import inspect

import pytest

from potpie.skills import installer
from potpie.skills.manifest import load_command_manifest
from scripts.generate_skill_command_manifest import collect_manifest

pytestmark = pytest.mark.unit


def test_static_command_manifest_matches_registered_cli() -> None:
    assert load_command_manifest() == collect_manifest()


def test_runtime_skill_validation_does_not_introspect_typer() -> None:
    source = inspect.getsource(installer)
    assert "potpie.cli.main" not in source
    assert "import typer" not in source
    assert "registered_commands" not in source


def test_every_bundled_potpie_command_matches_static_manifest() -> None:
    installer.validate_packaged_skill_command_snippets()
