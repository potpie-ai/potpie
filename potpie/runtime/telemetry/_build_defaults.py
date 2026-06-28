"""Sentry defaults shipped with the root package at build time.

Values are written to ``_build_config.py`` by the root Hatch build hook. In a
source checkout, the generated module is absent and build-time constants resolve
to empty strings.
"""

from __future__ import annotations

DEFAULT_POTPIE_SENTRY_ENVIRONMENT = "prod_oss"

try:
    from potpie.runtime.telemetry._build_config import (
        POTPIE_SENTRY_DIST,
        POTPIE_SENTRY_DSN,
        POTPIE_SENTRY_ENABLED,
        POTPIE_SENTRY_ENVIRONMENT,
        POTPIE_SENTRY_RELEASE,
        POTPIE_TELEMETRY_DISABLED,
    )
except ImportError:
    POTPIE_TELEMETRY_DISABLED = ""
    POTPIE_SENTRY_ENABLED = ""
    POTPIE_SENTRY_DSN = ""
    POTPIE_SENTRY_ENVIRONMENT = ""
    POTPIE_SENTRY_RELEASE = ""
    POTPIE_SENTRY_DIST = ""
