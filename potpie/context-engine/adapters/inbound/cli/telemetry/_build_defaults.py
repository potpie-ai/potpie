"""Telemetry defaults shipped with the package at build time.

Values are written to ``_build_config.py`` by the Hatch build hook
(``oauth_client_id_injection_hook.py``). In a source checkout, the generated
module is absent and every constant resolves to an empty string.
"""

from __future__ import annotations

try:
    from adapters.inbound.cli.telemetry._build_config import (
        POTPIE_POSTHOG_API_KEY,
        POTPIE_POSTHOG_ENABLED,
        POTPIE_POSTHOG_HOST,
        POTPIE_PRODUCT_ANALYTICS_ENABLED,
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
    POTPIE_POSTHOG_ENABLED = ""
    POTPIE_PRODUCT_ANALYTICS_ENABLED = ""
    POTPIE_POSTHOG_API_KEY = ""
    POTPIE_POSTHOG_HOST = ""

