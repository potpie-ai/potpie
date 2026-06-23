"""Hatch build hook: inject OAuth client IDs into the wheel at build time.

During ``hatch build`` / ``uv build``, reads ``LINEAR_CLIENT_ID`` and
``POTPIE_GITHUB_CLIENT_ID`` from the build environment and writes
``adapters/outbound/cli_auth/_build_config.py`` so installed users do not need
to configure those variables locally.

Both are public OAuth client identifiers (not secrets). Runtime env vars still
take precedence for local development overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_OUT = Path("adapters/outbound/cli_auth/_build_config.py")

_HEADER = """\
# Auto-generated at wheel build time — do not edit manually.
# Override any value at runtime by setting the corresponding environment variable.
"""


class OAuthClientIdInjectionHook(BuildHookInterface):
    """Inject public OAuth client IDs from the build environment into the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        linear_client_id = os.getenv("LINEAR_CLIENT_ID", "").strip()
        github_client_id = os.getenv("POTPIE_GITHUB_CLIENT_ID", "").strip()

        _OUT.write_text(
            _HEADER
            + f"LINEAR_CLIENT_ID = {linear_client_id!r}\n"
            + f"POTPIE_GITHUB_CLIENT_ID = {github_client_id!r}\n",
            encoding="utf-8",
        )
        build_data["artifacts"].append(str(_OUT))
