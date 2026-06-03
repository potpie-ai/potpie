"""Ensure the Potpie repo root is on ``sys.path`` so ``import app...`` works from the embedded CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_done = False


def ensure_potpie_repo_on_sys_path() -> None:
    """
    When ``context-engine`` runs from a Potpie checkout, the ``app`` package lives at
    ``<repo>/app/``. Console scripts do not add the repo root to ``sys.path`` by default, so
    ``import app.modules.context_graph.celery_job_queue`` fails unless Potpie was installed in a
    way that registers ``app`` (or we prepend the root here).

    Idempotent: walks up from this file's location and, if a Potpie demo host ``app/main.py``
    exists (``<root>/app`` or ``<root>/legacy/app``), prepends that directory to ``sys.path``.
    """
    global _done
    if _done:
        return
    here = Path(__file__).resolve()
    for i in range(min(16, len(here.parents))):
        root = here.parents[i]
        legacy_app = root / "legacy" / "app" / "main.py"
        if legacy_app.is_file():
            s = str(root / "legacy")
            if s not in sys.path:
                sys.path.insert(0, s)
            break
        if (root / "app" / "main.py").is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            break
    _done = True
