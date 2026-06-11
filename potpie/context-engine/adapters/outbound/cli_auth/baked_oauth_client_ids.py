"""OAuth client IDs baked into the wheel at package build time.

Values are written to ``_build_config.py`` by the Hatch build hook
(``oauth_client_id_injection_hook.py``). When running from a source checkout
without a wheel build, that module is absent and constants resolve to empty
strings; set ``LINEAR_CLIENT_ID`` / ``POTPIE_GITHUB_CLIENT_ID`` in the
environment instead.
"""

from __future__ import annotations

try:
    from adapters.outbound.cli_auth._build_config import (
        LINEAR_CLIENT_ID,
        POTPIE_GITHUB_CLIENT_ID,
    )
except ImportError:
    LINEAR_CLIENT_ID = ""
    POTPIE_GITHUB_CLIENT_ID = ""
