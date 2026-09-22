"""Compatibility import for the core repository identity normalizer.

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

The implementation lives in :mod:`potpie_context_core.repository_identity` so
core write lowering and engine/CLI reads use the same identity without reversing
the package dependency.  These re-exported function objects preserve existing
engine imports and the keys already stored in ``repo_defaults``.
"""

from __future__ import annotations

from potpie_context_core.repository_identity import (
    normalize_repo_ref,
    repo_identity_key,
)


__all__ = ["normalize_repo_ref", "repo_identity_key"]
