"""User-facing messages when OAuth client IDs are missing."""

from __future__ import annotations

from pathlib import Path

_POTPIE_ISSUES_URL = "https://github.com/potpie-ai/potpie/issues"


def looks_like_local_development() -> bool:
    """True when the CLI is run from a Potpie source checkout or editable install."""
    if _cwd_in_potpie_source_tree():
        return True
    return _package_installed_from_source_tree()


def missing_linear_client_id_message() -> str:
    if looks_like_local_development():
        return "Set LINEAR_CLIENT_ID in potpie/.env (see .env.template)."
    return (
        "Linear login is not available in this install. "
        f"Please report this at {_POTPIE_ISSUES_URL}"
    )


def missing_github_client_id_message() -> str:
    if looks_like_local_development():
        return "Set POTPIE_GITHUB_CLIENT_ID in potpie/.env (see .env.template)."
    return (
        "GitHub login is not available in this install. "
        f"Please report this at {_POTPIE_ISSUES_URL}"
    )


def _cwd_in_potpie_source_tree() -> bool:
    cur = Path.cwd().resolve()
    for _ in range(24):
        if (cur / "potpie" / "context-engine" / "pyproject.toml").is_file():
            return True
        if (cur / "pyproject.toml").is_file() and (
            cur / "adapters" / "outbound" / "cli_auth"
        ).is_dir():
            return True
        if cur.parent == cur:
            break
        cur = cur.parent
    return False


def _package_installed_from_source_tree() -> bool:
    try:
        import adapters.outbound.cli_auth.baked_oauth_client_ids as mod

        root = Path(mod.__file__).resolve().parent
        ce_root = root.parent.parent.parent
        return (ce_root / "pyproject.toml").is_file()
    except Exception:
        return False
