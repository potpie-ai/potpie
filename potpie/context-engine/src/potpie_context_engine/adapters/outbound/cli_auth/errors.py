"""Base exception for the CLI auth/credential subsystem."""

from __future__ import annotations


class CliAuthError(Exception):
    """Base for all expected CLI auth and credential failures.

    Provider flows (GitHub device flow, Firebase, Potpie, Atlassian), the HTTP
    transport (:class:`~potpie_context_engine.adapters.outbound.cli_auth.http.AuthHttpError`), and the
    credential store all subclass this, so a command boundary can catch the whole
    subsystem with a single ``except CliAuthError`` instead of enumerating types.
    """
