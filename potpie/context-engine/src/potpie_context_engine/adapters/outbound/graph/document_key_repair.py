"""Detect ``Document`` nodes still keyed by the legacy content hash.

``Document`` was a ``CONTENT_HASH`` entity before the resource store landed,
so its keys were minted as ``document:<hex>``. Promotion to ``SLUG_ALIAS``
kept the ``document`` key prefix, which is why those nodes remain valid and
correctly labelled — the validator only checks the prefix. What breaks is
*convergence*: re-ingesting the same source now mints ``document:<slug>``,
which will not merge with the old node, leaving a duplicate.

This is detection only. Rewriting an entity key is not safe here — no backend
can rewire claim edges onto a new key, so re-minting identity would orphan
every claim citing the old one. The repair target counts and samples; the
operator decides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from potpie_context_core.ports.graph.analytics import RepairFinding

DOCUMENT_KEY_TARGET = "document_keys"
DOCUMENT_KEY_TARGETS = frozenset(
    {DOCUMENT_KEY_TARGET, "document-keys", "document_key", "documents"}
)
DOCUMENT_KEY_PREFIX = "document:"
DOCUMENT_KEY_SAMPLE_SIZE = 5

# What a backend was able to enumerate, named in the finding so a zero count
# is never read as more than it is.
SCANNED_ENTITIES = "this pot's entities"
SCANNED_CLAIM_ENDPOINTS = "this pot's claim endpoints"
SCANNED_CLAIM_ENDPOINTS_TRUNCATED = "the first page of this pot's claim endpoints"

# The old ``CONTENT_HASH`` body: a hex digest truncated to 8-32 chars. This is
# a shape test, not a proof — a genuine slug can be all hex ("beadface",
# "deadbeef"), so a match is advisory and the next action says so.
_LEGACY_BODY_RE = re.compile(r"^[0-9a-f]{8,32}$")

_NEXT_ACTION = (
    "Re-state each document under a slug key (document:<slug>) and retire the "
    "legacy node once its claims point at the new one. Keys are never "
    "rewritten in place because no backend can rewire claim edges to a new "
    "key. Hex-shaped slugs are false positives — check the sample."
)


def wants_document_key_repair(targets: Sequence[str] = ()) -> bool:
    """Return true when a repair invocation should audit document keys."""
    if not targets:
        return True
    return any(t.strip().lower() in DOCUMENT_KEY_TARGETS for t in targets)


def is_legacy_document_key(entity_key: str) -> bool:
    """True when a ``document:`` key still carries a content-hash-shaped body.

    Single-segment keys only: ``document:a1b2c3d4`` matches,
    ``document:q3-review`` and ``document:a1b2c3d4:part`` do not.
    """
    key = (entity_key or "").strip()
    if not key.startswith(DOCUMENT_KEY_PREFIX):
        return False
    body = key[len(DOCUMENT_KEY_PREFIX) :]
    return _LEGACY_BODY_RE.fullmatch(body) is not None


def document_key_finding(
    entity_keys: Iterable[str],
    *,
    sample_size: int = DOCUMENT_KEY_SAMPLE_SIZE,
    scanned: str = SCANNED_ENTITIES,
) -> RepairFinding:
    """Classify a pot's entity keys into a detect-only repair finding.

    ``scanned`` names what the caller was able to enumerate. It is not
    decoration: a backend that can only reach claim endpoints cannot see a
    ``Document`` node whose claims were pruned, so a zero count from that
    backend means "none in what I looked at", not "none". Saying which is the
    difference between an audit and a clean bill of health that is not one.
    """
    legacy = sorted({key for key in entity_keys if is_legacy_document_key(key)})
    if not legacy:
        return RepairFinding(
            target=DOCUMENT_KEY_TARGET,
            count=0,
            detail=f"no legacy content-hash document keys among {scanned}",
        )
    return RepairFinding(
        target=DOCUMENT_KEY_TARGET,
        count=len(legacy),
        samples=tuple(legacy[: max(sample_size, 0)]),
        detail=(
            f"{len(legacy)} document key(s) among {scanned} still use the "
            f"legacy content-hash shape; a re-import mints document:<slug> and "
            f"will not converge with them"
        ),
        recommended_next_action=_NEXT_ACTION,
    )


__all__ = [
    "DOCUMENT_KEY_PREFIX",
    "DOCUMENT_KEY_SAMPLE_SIZE",
    "DOCUMENT_KEY_TARGET",
    "DOCUMENT_KEY_TARGETS",
    "SCANNED_CLAIM_ENDPOINTS",
    "SCANNED_CLAIM_ENDPOINTS_TRUNCATED",
    "SCANNED_ENTITIES",
    "document_key_finding",
    "is_legacy_document_key",
    "wants_document_key_repair",
]
