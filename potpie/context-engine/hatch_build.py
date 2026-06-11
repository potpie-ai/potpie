"""Hatch build hook: bake OAuth client IDs into the wheel at build time.

During ``hatch build`` / ``uv build``, this hook reads the following env vars
and writes their values into ``adapters/outbound/cli_auth/_build_config.py``
so that installed users never need to set them in their own environment:

    LINEAR_CLIENT_ID          – Linear OAuth app client ID (PKCE, public)
    POTPIE_GITHUB_CLIENT_ID   – GitHub OAuth app client ID (device flow, public)

Both are *public* client identifiers, not secrets. No client secrets are
embedded. The env vars still take precedence at runtime (for dev overrides).
"""

from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_OUT = Path("adapters/outbound/cli_auth/_build_config.py")

_HEADER = """\
# Auto-generated at wheel build time by hatch_build.py — do not edit manually.
# Override any value at runtime by setting the corresponding environment variable.
"""


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        linear_client_id = os.getenv("LINEAR_CLIENT_ID", "").strip()
        github_client_id = os.getenv("POTPIE_GITHUB_CLIENT_ID", "").strip()

        _OUT.write_text(
            _HEADER
            + f"LINEAR_CLIENT_ID = {linear_client_id!r}\n"
            + f"POTPIE_GITHUB_CLIENT_ID = {github_client_id!r}\n",
            encoding="utf-8",
        )
        # Ensure hatch includes the generated file in the wheel even though
        # it may not exist at the time hatch resolves the file list.
        build_data["artifacts"].append(str(_OUT))
