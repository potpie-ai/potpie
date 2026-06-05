"""Background sandbox prewarm dispatched on ``repository.added``.

Without this, the first agent batch pays the bare-clone latency (anything
from a few seconds for a small public repo to multiple minutes for a large
private one). With it, the clone runs on a daemon thread while the agent
schedules the batch, so the agent's first ``sandbox_*`` call hits a warm
workspace.

The dispatch is fire-and-forget — failures here never block ``attach`` or
``detach`` HTTP responses; the agent's existing cold-sandbox handling
(``unknown_ref`` / empty ``list_dir`` / retry once) still covers the
worst case where the prewarm never ran.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading

logger = logging.getLogger(__name__)


def prewarm_disabled() -> bool:
    return (
        os.getenv("CONTEXT_ENGINE_DISABLE_SANDBOX_PREWARM") or ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_default_branch(
    *, repo_url: str | None, owner: str, repo: str, token: str | None
) -> str | None:
    """Ask the remote for its actual HEAD branch.

    Returns ``None`` on any failure — caller treats that as "skip prewarm".
    Output of ``git ls-remote --symref <url> HEAD`` looks like::

        ref: refs/heads/master\tHEAD
        <sha>\tHEAD

    A timeout / non-zero exit / unparseable stdout all map to ``None`` so a
    transient network blip never blocks an attach.
    """
    url = (repo_url or "").strip() or f"https://github.com/{owner}/{repo}.git"
    auth_url = url
    if token and url.startswith("https://"):
        auth_url = url.replace("https://", f"https://x-access-token:{token}@", 1)
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--symref", auth_url, "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("ref: refs/heads/"):
            head, _, _ = stripped[len("ref: refs/heads/") :].partition("\t")
            head = head.strip()
            if head:
                return head
    return None


def dispatch_pot_repo_prewarm(
    *,
    user_id: str | None,
    pot_id: str,
    owner: str,
    repo: str,
    default_branch: str | None,
    repo_url: str | None,
) -> None:
    """Fire-and-forget bare-clone prewarm + workspace acquire.

    Spawns a daemon thread so the HTTP request returns immediately. The
    thread runs its own asyncio loop because ``threading.Thread`` cannot
    await coroutines and we can't assume the request runs under an event
    loop the caller controls (FastAPI's threadpool / sync handlers).
    """
    if prewarm_disabled() or not user_id:
        return
    full = f"{owner}/{repo}"
    supplied_branch = (default_branch or "").strip() or None

    def _runner() -> None:
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
                _resolve_auth_token,
            )

            client = get_sandbox_client()
            token = _resolve_auth_token(user_id, full)
        except Exception:
            logger.exception(
                "pot_sandbox_provisioning: sandbox client unavailable for pot=%s repo=%s",
                pot_id,
                full,
            )
            return

        branch = supplied_branch or _resolve_default_branch(
            repo_url=repo_url, owner=owner, repo=repo, token=token
        )
        if not branch:
            # No reliable ref to prewarm against — the agent's lazy first
            # batch (with full per-pot auth resolution) will still clone on
            # demand. Better to skip than to guess `main` and crash on a
            # repo whose default is `master`/`develop`/etc.
            logger.info(
                "pot_sandbox_provisioning: default_branch unknown, skipping "
                "prewarm pot=%s repo=%s",
                pot_id,
                full,
            )
            return

        async def _prewarm() -> None:
            try:
                await client.ensure_repo_cache(
                    user_id=user_id,
                    repo=full,
                    base_ref=branch,
                    repo_url=repo_url,
                    auth_token=token,
                )
            except Exception:
                # Repo cache providers aren't wired on every backend
                # (Daytona today exposes only workspaces); fall through
                # to acquire_session which clones on first use.
                logger.info(
                    "pot_sandbox_provisioning: ensure_repo_cache skipped for "
                    "pot=%s repo=%s (likely no cache provider on this backend)",
                    pot_id,
                    full,
                )
            try:
                # ANALYSIS mode for the bootstrap path. The agent's
                # reconciliation batch picks the same workspace because
                # the facade keys on (user, pot_id, repo).
                from sandbox.domain.models import WorkspaceMode

                await client.acquire_session(
                    user_id=user_id,
                    project_id=pot_id,
                    repo=full,
                    branch=branch,
                    base_ref=branch,
                    auth_token=token,
                    repo_url=repo_url,
                    mode=WorkspaceMode.ANALYSIS,
                )
            except Exception as exc:
                # Best-effort: log known clone/auth failures at info (the
                # agent's lazy path still covers the request). Anything
                # else is unexpected and worth a real traceback.
                try:
                    from sandbox.domain.errors import (
                        RepoAuthFailed,
                        RepoCacheUnavailable,
                    )

                    expected = (RepoAuthFailed, RepoCacheUnavailable)
                except Exception:
                    expected = ()
                if expected and isinstance(exc, expected):
                    logger.info(
                        "pot_sandbox_provisioning: prewarm acquire skipped "
                        "pot=%s repo=%s ref=%s reason=%s",
                        pot_id,
                        full,
                        branch,
                        type(exc).__name__,
                    )
                else:
                    logger.exception(
                        "pot_sandbox_provisioning: acquire_session failed for "
                        "pot=%s repo=%s",
                        pot_id,
                        full,
                    )

        try:
            asyncio.run(_prewarm())
        except Exception:
            logger.exception(
                "pot_sandbox_provisioning: prewarm loop crashed pot=%s repo=%s",
                pot_id,
                full,
            )

    threading.Thread(
        target=_runner,
        name=f"sandbox-prewarm-{pot_id}-{owner}-{repo}",
        daemon=True,
    ).start()
