"""Build local + connectivity snapshot for unified ``status`` / ``doctor``."""

from __future__ import annotations

import os

from adapters.inbound.cli.credentials_store import get_active_pot_id, get_stored_api_base_url, get_stored_api_key
from adapters.inbound.cli.env_bootstrap import load_cli_env
from adapters.inbound.cli.output import DoctorSnapshot
from adapters.inbound.cli.potpie_api_config import resolve_potpie_api_base_url, resolve_potpie_api_key
from adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient, PotpieContextApiError
from adapters.outbound.settings_env import EnvContextEngineSettings


def _database_url_env_set() -> bool:
    return bool(
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("CONTEXT_ENGINE_DATABASE_URL")
        or os.getenv("POSTGRES_SERVER")
    )


def _format_api_error(exc: PotpieContextApiError) -> str:
    import json

    d = exc.detail
    if isinstance(d, dict):
        inner = d.get("detail")
        if isinstance(inner, dict):
            return json.dumps(inner)
        if isinstance(inner, str):
            return inner
        return json.dumps(d)
    return str(d)


def _cli_client_name() -> str:
    for var in ("POTPIE_CLIENT_NAME", "CONTEXT_ENGINE_CLIENT_NAME"):
        v = os.getenv(var)
        if v and v.strip():
            return v.strip()
    return "potpie-cli"


def build_doctor_snapshot() -> DoctorSnapshot:
    """Local config flags + optional Potpie health/auth probe."""
    load_cli_env()
    s = EnvContextEngineSettings()
    cg = s.is_enabled()
    neo = bool(s.neo4j_uri())

    has_potpie_key_env = bool(os.getenv("POTPIE_API_KEY"))
    has_stored_key = bool(get_stored_api_key())
    base_env = os.getenv("POTPIE_API_URL") or os.getenv("POTPIE_BASE_URL")
    base_stored = get_stored_api_base_url()
    port = os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT")
    db = _database_url_env_set()
    gh = bool(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN"))
    maps_set = bool(
        os.getenv("CONTEXT_ENGINE_REPO_TO_POT") or os.getenv("CONTEXT_ENGINE_POTS")
    )

    health_ok: bool | None = None
    health_msg: str | None = None
    auth_ok: bool | None = None
    auth_msg: str | None = None
    try:
        c = PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
            timeout=15.0,
            client_surface="cli",
            client_name=_cli_client_name(),
        )
        code, _payload = c.get_health()
        health_ok = code == 200
        health_msg = None if health_ok else f"HTTP {code}"
        try:
            c.list_context_pots()
            auth_ok = True
        except PotpieContextApiError as exc:
            auth_ok = False
            auth_msg = f"HTTP {exc.status_code}: {_format_api_error(exc)}"
    except ValueError:
        health_ok = None
        health_msg = "skipped (no base URL / API key)"
        auth_ok = None
        auth_msg = "skipped (no base URL / API key)"
    except OSError as exc:
        health_ok = False
        health_msg = str(exc)
        auth_ok = False
        auth_msg = str(exc)

    summary: list[str] = [
        "[dim]search / ingest / pot hard-reset call Potpie POST /api/v2/context/* with X-API-Key.[/dim]",
        "[dim]Local Neo4j is not required on this machine.[/dim]",
    ]
    if health_ok is True:
        summary.append("[green]GET /health on Potpie base URL succeeded.[/green]")
    if auth_ok is True:
        summary.append("[green]Authenticated /api/v2/context probe succeeded.[/green]")
    elif auth_ok is False:
        summary.append(
            "[red]Authenticated /api/v2/context probe failed; search / ingest / MCP calls will fail.[/red]"
        )

    return DoctorSnapshot(
        context_graph_enabled=cg,
        neo4j_effective_set=neo,
        pot_maps_set=maps_set,
        active_pot_id=get_active_pot_id() or None,
        potpie_api_key_env=has_potpie_key_env,
        potpie_stored_token=has_stored_key,
        potpie_base_url=base_env or base_stored or None,
        potpie_port_hint=(
            f"http://127.0.0.1:{port}"
            if port and not (base_env or base_stored)
            else None
        ),
        database_url_set=db,
        github_token_set=gh,
        potpie_health_ok=health_ok,
        potpie_health_message=health_msg,
        potpie_auth_ok=auth_ok,
        potpie_auth_message=auth_msg,
        summary_lines=summary,
    )
