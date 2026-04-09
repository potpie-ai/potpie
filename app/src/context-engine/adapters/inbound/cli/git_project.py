"""Resolve pot_id from git ``origin`` + env maps + CLI ``pot use`` (no Potpie project API)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

from adapters.inbound.cli.credentials_store import (
    get_active_pot_id,
    resolve_cli_pot_ref,
)
from adapters.inbound.cli.env_bootstrap import load_cli_env
from bootstrap.http_projects import pot_map_from_env, repo_to_pot_map_from_env


def parse_owner_repo_from_remote(url: str) -> str | None:
    """Normalize ``git remote`` URL to ``owner/repo`` (GitHub/GitLab style path)."""
    u = url.strip()
    if not u:
        return None
    if u.startswith("git@"):
        if ":" not in u:
            return None
        _, path = u.split(":", 1)
        path = path.removesuffix(".git").strip()
        return path or None
    if "://" in u:
        p = urlparse(u)
        path = p.path.strip("/").removesuffix(".git")
        if not path:
            return None
        return path
    return None


def get_git_origin_remote_url(cwd: str | None = None) -> str | None:
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out or None


def _norm_repo(s: str) -> str:
    return s.strip().lower()


def resolve_pot_id_for_repo(owner_repo: str) -> str | None:
    """Match ``owner/repo`` against env maps (case-insensitive on repo key)."""
    want = _norm_repo(owner_repo)
    for k, v in repo_to_pot_map_from_env().items():
        if _norm_repo(k) == want:
            return str(v)
    active = get_active_pot_id()
    if active:
        for pid, repo in pot_map_from_env().items():
            if pid == active and _norm_repo(repo) == want:
                return str(pid)
    for pot_id, repo in pot_map_from_env().items():
        if _norm_repo(repo) == want:
            return str(pot_id)
    return None


@dataclass(slots=True)
class ParsedRemote:
    owner_repo: str
    provider: str
    provider_host: str


def parse_git_remote(url: str) -> ParsedRemote | None:
    """Best-effort provider/host + ``owner/repo`` from a ``git remote`` URL."""
    owner_repo = parse_owner_repo_from_remote(url)
    if not owner_repo:
        return None
    u = url.strip()
    host = ""
    if u.startswith("git@"):
        rest = u.split("@", 1)[1]
        host = rest.split(":", 1)[0] if ":" in rest else ""
    elif "://" in u:
        host = (urlparse(u).hostname or "").lower()
    provider = "git"
    hl = host.lower()
    if "github" in hl:
        provider = "github"
    elif "gitlab" in hl or "gitlab" in u.lower():
        provider = "gitlab"
    elif "bitbucket" in hl:
        provider = "bitbucket"
    return ParsedRemote(
        owner_repo=owner_repo,
        provider=provider,
        provider_host=host or "unknown",
    )


def resolve_pot_id_from_git_cwd(cwd: str | None = None) -> tuple[str | None, str]:
    """Return ``(pot_id, error_message)``. ``error_message`` is empty on success.

    Resolution order:

    1. ``context-engine pot use <uuid-or-name>`` (stored ``active_pot_id``; names from ``pot alias``) — global default for this machine.
    2. Else ``CONTEXT_ENGINE_REPO_TO_POT`` / ``CONTEXT_ENGINE_POTS`` for ``owner/repo`` from ``git`` ``origin``
       (under ``cwd``, default current directory).
    3. Else error (pass pot UUID explicitly, or set maps / ``pot use``).
    """
    load_cli_env()
    active = get_active_pot_id()
    if active:
        resolved, err = resolve_cli_pot_ref(active)
        if err:
            return None, err
        return resolved, ""

    url = get_git_origin_remote_url(cwd)
    if not url:
        return (
            None,
            "Could not read git remote `origin` (are you inside a git repository with origin set?). "
            "Run `context-engine pot use <pot-uuid>`, set CONTEXT_ENGINE_REPO_TO_POT / CONTEXT_ENGINE_POTS, "
            "or pass the pot UUID explicitly.",
        )
    owner_repo = parse_owner_repo_from_remote(url)
    if not owner_repo:
        return None, f"Could not parse owner/repo from remote URL: {url!r}"

    pid = resolve_pot_id_for_repo(owner_repo)
    if pid:
        return pid, ""

    return (
        None,
        f"No pot for repository {owner_repo!r}. "
        "Set CONTEXT_ENGINE_REPO_TO_POT or CONTEXT_ENGINE_POTS, "
        "run `context-engine pot use <pot-uuid>`, "
        "or pass the pot UUID as the first argument to `search` / `ingest`.",
    )
