"""One normalizer for repo identity, shared by the pot store and the setup seam.

The same working tree arrives spelled several ways — ``.``, an absolute path,
``git@github.com:Owner/Repo.git``, ``https://github.com/Owner/Repo`` — and every
place that had to recognise two of those spellings as one repo grew a private
copy of the rules. The copies then drifted: the setup orchestrator's had lost
the ``.lower()`` the others apply, so ``potpie setup`` persisted a source
location of ``github.com/Potpie-AI/Potpie`` where ``potpie source add repo .``
persisted ``github.com/potpie-ai/potpie`` for the identical repository. Nothing
caught it because every *lookup* re-normalized both sides — but a dedup that
compares stored strings would miss, and a dedup is exactly what setup needed to
stop appending a duplicate repo source on every re-run.

:func:`repo_identity_key` is canonical, not merely shared: it is the function
that produced every ``repo_defaults`` key currently on disk, so its output must
not move. Paths are resolved but deliberately keep their casing (filesystem
case semantics vary by platform); only remote refs are lower-cased.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def normalize_repo_ref(value: str) -> str | None:
    """Collapse a git remote to a comparable ``host/owner/repo``."""
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@") and ":" in raw:
        host, path = raw[4:].split(":", 1)
        return f"{host}/{path}".strip("/").lower()
    if "://" in raw:
        parsed = urlparse(raw)
        # Host and port, never `netloc`: netloc keeps the `user:password@`
        # prefix, so a remote cloned with an embedded token both keyed the repo
        # differently from every other spelling of it — defeating the dedup this
        # function exists for — and wrote that token into `pots.json` as a source
        # location. Credentials are not part of a repository's identity.
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            # A malformed port makes `parsed.port` raise rather than return
            # None; the host alone still identifies the repo.
            port = None
        if port:
            host = f"{host}:{port}"
        if host and parsed.path:
            return f"{host}/{parsed.path.strip('/')}".lower()
    return raw.strip("/").lower()


def repo_identity_key(value: str) -> str | None:
    """Stable local key for matching repo sources and repo→pot defaults.

    A path (relative, ``~``-rooted, or absolute) resolves to an absolute path;
    anything else is treated as a remote ref. Callers compare *keys*, never the
    stored spelling, because the stored spelling is whatever the user typed.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith((".", "~")) or Path(raw).is_absolute():
        return str(Path(raw).expanduser().resolve(strict=False))
    return normalize_repo_ref(raw)


__all__ = ["normalize_repo_ref", "repo_identity_key"]
