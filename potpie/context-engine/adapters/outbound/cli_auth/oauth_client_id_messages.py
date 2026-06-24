"""User-facing messages when OAuth client IDs are missing."""

from __future__ import annotations

_POTPIE_ISSUES_URL = "https://github.com/potpie-ai/potpie/issues"


def missing_linear_client_id_message() -> str:
    return (
        "Set LINEAR_CLIENT_ID in potpie/.env (see .env.template), "
        f"or report this at {_POTPIE_ISSUES_URL}."
    )


def missing_github_client_id_message() -> str:
    return (
        "Set POTPIE_GITHUB_CLIENT_ID in potpie/.env (see .env.template), "
        f"or report this at {_POTPIE_ISSUES_URL}."
    )
