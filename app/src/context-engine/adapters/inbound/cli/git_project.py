"""Resolve Potpie project_id from the current git repo + env maps."""

from __future__ import annotations

import os
import subprocess
from urllib.parse import urlparse

from adapters.inbound.cli.credentials_store import (
    get_stored_api_base_url,
    get_stored_api_key,
)
from adapters.inbound.cli.env_bootstrap import load_cli_env
from bootstrap.http_projects import project_map_from_env, repo_to_project_map_from_env


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


def get_git_current_branch(cwd: str | None = None) -> str | None:
    """Return the current branch name from ``git rev-parse --abbrev-ref HEAD``, or None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
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
    if not out or out == "HEAD":
        return None
    return out


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


def resolve_project_id_for_repo(owner_repo: str) -> str | None:
    """Match ``owner/repo`` against env maps (case-insensitive on repo key)."""
    want = _norm_repo(owner_repo)
    for k, v in repo_to_project_map_from_env().items():
        if _norm_repo(k) == want:
            return str(v)
    for proj_id, repo in project_map_from_env().items():
        if _norm_repo(repo) == want:
            return str(proj_id)
    return None


def _potpie_api_base_url_candidates() -> list[str]:
    """Order: env URL, then stored URL from ``login --url``, then port env, then localhost guesses."""
    explicit = (
        os.getenv("POTPIE_API_URL") or os.getenv("POTPIE_BASE_URL") or ""
    ).strip().rstrip("/")
    if explicit:
        return [explicit]
    stored = get_stored_api_base_url()
    if stored:
        return [stored]
    port = (os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT") or "").strip()
    if port:
        try:
            p = int(port)
            p = max(1, min(p, 65535))
        except ValueError:
            p = 8000
        return [f"http://127.0.0.1:{p}"]
    return ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]


def _potpie_error_detail(response: object, max_len: int = 500) -> str:
    """Best-effort extract FastAPI ``detail`` from a response body."""
    try:
        data = response.json()  # type: ignore[union-attr]
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        if isinstance(data, list):
            return str(data)[:max_len]
    except Exception:
        pass
    try:
        text = getattr(response, "text", None) or ""
        return text[:max_len]
    except Exception:
        return ""


def _potpie_request_headers() -> tuple[dict[str, str] | None, str]:
    """
    Build headers for ``GET /api/v2/projects/list``.

    API key: ``POTPIE_API_KEY`` env (highest precedence), else token from ``context-engine login``.
    """
    api_key = (os.getenv("POTPIE_API_KEY") or "").strip()
    if not api_key:
        api_key = get_stored_api_key()
    if not api_key:
        return (None, "")
    return ({"X-API-Key": api_key}, "")


def _try_potpie_project_list(owner_repo: str, cwd: str | None = None) -> tuple[str | None, str]:
    """
    Resolve project id via Potpie HTTP API (same data as the app UI).

    Uses ``GET .../api/v2/projects/list``. Loads the first ``.env`` walking up from ``cwd``
    before reading variables (so repo-local ``POTPIE_API_KEY`` works).

    Returns ``(project_id, error_message)``. Empty error means success.
    If auth is not configured, returns ``(None, "")`` so callers can fall through.
    """
    load_cli_env()
    headers, h_err = _potpie_request_headers()
    if h_err:
        return None, h_err
    if headers is None:
        return None, ""

    try:
        import httpx
    except ImportError:
        return (
            None,
            "httpx is required for Potpie API resolution (install package dependencies).",
        )

    candidates = _potpie_api_base_url_candidates()
    r = None
    last_url = ""
    last_conn: str | None = None
    for base in candidates:
        last_url = f"{base}/api/v2/projects/list"
        try:
            r = httpx.get(last_url, headers=headers, timeout=15.0)
            break
        except Exception as e:
            err_s = str(e).lower()
            if any(
                x in err_s
                for x in (
                    "connection",
                    "refused",
                    "timed out",
                    "unreachable",
                    "getaddrinfo",
                )
            ):
                last_conn = str(e)
                continue
            return None, f"Potpie API request failed ({last_url}): {e}"

    if r is None:
        return (
            None,
            f"Could not connect to Potpie at {candidates}. {last_conn or ''} "
            "Start the API or set POTPIE_API_URL / POTPIE_PORT.",
        )

    if r.status_code == 401:
        detail = _potpie_error_detail(r)
        hint = (
            "Set a valid Potpie API key: export POTPIE_API_KEY=... or run "
            "`context-engine login <token>` (create a key in the app)."
        )
        if detail:
            return (None, f"Potpie API 401: {detail}. {hint}")
        return (None, f"Potpie API HTTP 401 from {last_url}. {hint}")
    if r.status_code != 200:
        return (
            None,
            f"Potpie API HTTP {r.status_code} from {last_url}: {r.text[:400]}",
        )

    try:
        data = r.json()
    except ValueError as e:
        return None, f"Potpie API returned invalid JSON: {e}"

    if not isinstance(data, list):
        return None, "Potpie API /projects/list returned unexpected JSON (expected a list)."

    want = _norm_repo(owner_repo)
    candidates: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rid = item.get("id")
        rname = item.get("repo_name")
        if rname is not None and _norm_repo(str(rname)) == want and rid is not None:
            candidates.append(item)

    if len(candidates) == 1:
        return str(candidates[0]["id"]), ""

    if len(candidates) > 1:
        branch = get_git_current_branch(cwd)
        has_branch_in_api = any(
            str(c.get("branch_name") or "").strip() for c in candidates
        )
        if branch and has_branch_in_api:
            filtered = [
                c
                for c in candidates
                if str(c.get("branch_name") or "").strip() == branch
            ]
            if len(filtered) == 1:
                return str(filtered[0]["id"]), ""
            if len(filtered) == 0:
                return (
                    None,
                    f"No Potpie project matched repository {owner_repo!r} and "
                    f"current branch {branch!r}. Pass an explicit project UUID, "
                    "or register this branch in Potpie.",
                )
            return (
                None,
                f"Multiple Potpie projects matched repository {owner_repo!r} and branch "
                f"{branch!r}: {[str(c['id']) for c in filtered]!r}. "
                "Pass an explicit project UUID.",
            )
        return (
            None,
            f"Multiple Potpie projects matched repository {owner_repo!r}: "
            f"{[str(c['id']) for c in candidates]!r}. "
            "Pass an explicit project UUID, or match a single project by current git branch "
            "(upgrade Potpie so /api/v2/projects/list includes branch_name).",
        )

    return (
        None,
        f"No Potpie project with repo_name matching {owner_repo!r} for this user. "
        "Register the repo in Potpie, or set CONTEXT_ENGINE_REPO_TO_PROJECT.",
    )


def resolve_project_id_from_git_cwd(cwd: str | None = None) -> tuple[str | None, str]:
    """Return ``(project_id, error_message)``. ``error_message`` is empty on success."""
    load_cli_env()
    url = get_git_origin_remote_url(cwd)
    if not url:
        return (
            None,
            "Could not read git remote `origin` (are you inside a git repository with origin set?)",
        )
    owner_repo = parse_owner_repo_from_remote(url)
    if not owner_repo:
        return None, f"Could not parse owner/repo from remote URL: {url!r}"

    pid = resolve_project_id_for_repo(owner_repo)
    if pid:
        return pid, ""

    pid, err = _try_potpie_project_list(owner_repo, cwd)
    if pid:
        return pid, ""
    if err:
        return None, err

    return (
        None,
        f"No project mapping for repository {owner_repo!r}. "
        "Set CONTEXT_ENGINE_REPO_TO_PROJECT or CONTEXT_ENGINE_PROJECTS, "
        "or run `context-engine login <token>` / set POTPIE_API_KEY. "
        "See context-engine CLI README.",
    )
