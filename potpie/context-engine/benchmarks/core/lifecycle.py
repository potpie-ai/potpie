"""Ephemeral pot lifecycle: create per scenario, reset on teardown.

Each scenario gets its own pot so state doesn't leak across scenarios.
We use the existing ``POST /api/v2/context/pots`` and ``POST /api/v2/context/reset``
endpoints; no new admin surface is needed.

Pot rows persist after the bench run (no DELETE endpoint exists today),
but ``/reset`` empties their graph + ledger. Pot rows are cheap; if they
accumulate, a periodic admin cleanup is the right fix, not a benchmark
shortcut.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from potpie_context_engine.adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)

logger = logging.getLogger(__name__)


def _load_dotenv_once() -> None:
    """Best-effort load of a .env file walking up from CWD.

    Done at import time so CLI commands pick up POTPIE_BENCH_* and
    OPENAI_API_KEY without the caller having to remember ``source .env``.
    Env vars set in the shell win over .env values.
    """
    if os.environ.get("_POTPIE_BENCH_DOTENV_LOADED") == "1":
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
    os.environ["_POTPIE_BENCH_DOTENV_LOADED"] = "1"


_load_dotenv_once()


@dataclass(frozen=True)
class EphemeralPot:
    pot_id: str
    slug: str
    repo_name: str | None


def _resolve_repo_alias() -> str:
    return os.environ.get("POTPIE_BENCH_REPO", "acme/sandbox")


def _inprocess_enabled() -> bool:
    return os.environ.get("POTPIE_BENCH_INPROCESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def make_client():
    """Return the engine client.

    With ``POTPIE_BENCH_INPROCESS=1`` (or ``run --local``) this returns the
    in-process driver — no HTTP server on :8001, no Celery worker; the bench
    builds the engine container and reconciles inline against the shared
    Postgres/Neo4j. Otherwise it returns the HTTP client.
    """
    if _inprocess_enabled():
        from benchmarks.core.local_engine import InProcessEngineClient

        return InProcessEngineClient()

    base_url = os.environ.get("POTPIE_BENCH_API_URL") or os.environ.get(
        "POTPIE_API_URL"
    )
    api_key = os.environ.get("POTPIE_BENCH_API_KEY") or os.environ.get("POTPIE_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "Set POTPIE_BENCH_API_URL and POTPIE_BENCH_API_KEY (or POTPIE_API_URL / POTPIE_API_KEY) "
            "before running benchmarks, or set POTPIE_BENCH_INPROCESS=1 to run in-process."
        )
    return PotpieContextApiClient(
        base_url,
        api_key,
        timeout=300.0,
        client_surface="benchmarks",
        client_name="ce-benchmarks",
    )


def create_ephemeral_pot(
    client: PotpieContextApiClient, *, scenario_id: str, attach_repo: bool = True
) -> EphemeralPot:
    """Create a fresh pot for a single scenario.

    The slug is namespaced with ``bench-`` and a short UUID so concurrent
    runs don't collide.
    """
    short = uuid.uuid4().hex[:8]
    # Slugs accept lowercase + digits + hyphens only (engine: 1-63 chars).
    # Truncate aggressively and normalize underscores -> hyphens.
    safe_id = scenario_id.replace("_", "-")[:40]
    slug = f"bench-{safe_id}-{short}"
    display_name = f"bench: {scenario_id} ({short})"
    primary_repo = _resolve_repo_alias() if attach_repo else None

    created = client.create_context_pot(
        slug=slug,
        display_name=display_name,
        primary_repo_name=primary_repo,
    )
    pot_id = created.get("id") or created.get("pot_id")
    if not pot_id:
        raise RuntimeError(f"create_context_pot returned no id: {created!r}")
    # The in-process driver reconciles batches explicitly (it doesn't wait on
    # a worker / flush timer), so the windowed default is fine and there's no
    # HTTP surface to PUT against — skip the ingestion-config call.
    if getattr(client, "inprocess", False):
        logger.info(
            "created ephemeral pot %s slug=%s repo=%s (in-process)",
            pot_id,
            slug,
            primary_repo,
        )
        return EphemeralPot(pot_id=str(pot_id), slug=slug, repo_name=primary_repo)
    # The server-side default is ``windowed/5min`` (waits for the 60s
    # periodic flush). Bench scenarios complete in seconds, so flip the
    # ephemeral pot to ``immediate`` so each submitted event is picked up
    # by the worker right away — otherwise every drain times out.
    try:
        import httpx

        with httpx.Client(timeout=20.0) as http:
            r = http.put(
                f"{client._base}/api/v2/context/pots/{pot_id}/ingestion-config",
                json={"mode": "immediate"},
                headers={
                    "X-API-Key": client._api_key,
                    "Content-Type": "application/json",
                },
            )
            if r.status_code >= 300:
                logger.warning(
                    "could not set ephemeral pot %s to immediate mode (HTTP %d): %s",
                    pot_id,
                    r.status_code,
                    r.text[:200],
                )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("ingestion-config set failed for pot %s: %s", pot_id, exc)
    logger.info("created ephemeral pot %s slug=%s repo=%s", pot_id, slug, primary_repo)
    return EphemeralPot(pot_id=str(pot_id), slug=slug, repo_name=primary_repo)


def reset_pot(client: PotpieContextApiClient, pot: EphemeralPot) -> None:
    """Hard-reset a pot's graph + ledger. Best-effort; logs on failure."""
    try:
        client.reset({"pot_id": pot.pot_id})
        logger.info("reset pot %s", pot.pot_id)
    except PotpieContextApiError as exc:
        logger.warning("reset failed for pot %s: %s", pot.pot_id, exc)
