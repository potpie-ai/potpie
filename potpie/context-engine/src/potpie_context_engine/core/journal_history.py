"""Bounded commit history over a rebuildable header mirror.

Native receipts are authoritative. Recovery advances only a contiguous prefix;
neither newer rows nor timestamps can stand in for a missing sequence.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from potpie_context_engine.core.graph_journal import (
    CommitReceipt,
    JournalError,
    journal_hash,
    journal_json,
)


@dataclass(frozen=True, slots=True)
class CommitFilters:
    actor: str | None = None
    origin: str | None = None
    logical_key: str | None = None


_DEFAULT_FILTERS = CommitFilters()
_MAX_RESPONSE_BYTES = 1_000_000


def commit_header(receipt: CommitReceipt) -> dict[str, Any]:
    header = {
        "pot_id": receipt.pot_id,
        "commit_id": receipt.commit_id,
        "journal_generation": receipt.journal_generation,
        "sequence": receipt.sequence,
        "parent_commit_id": receipt.parent_commit_id,
        "actor": receipt.actor,
        "origin": receipt.origin,
        "message": receipt.message,
        "committed_at": receipt.committed_at.isoformat(),
        "required_access": receipt.required_access,
        "rollback_supported": receipt.rollback_supported,
        "unsupported_reason": receipt.unsupported_reason,
        "diff_complete": receipt.diff_complete,
        "affected_record_count": receipt.affected_record_count,
        "resource_operation_id": receipt.resource_operation_id,
        "touched_keys": sorted(
            {c.logical_key for c in (*receipt.semantic_changes, *receipt.audit_changes)}
        ),
        "receipt_hash": journal_hash(receipt),
    }
    header["header_hash"] = _header_hash(header)
    return header


def _header_hash(header: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in header.items() if key != "header_hash"}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_header(header: Mapping[str, Any]) -> None:
    if header.get("header_hash") != _header_hash(header):
        raise JournalError("mirror header hash mismatch")


def cursor_encode(
    *, pot_id: str, generation: str, filters: CommitFilters, before: int, ceiling: int
) -> str:
    raw = {
        "pot": pot_id,
        "generation": generation,
        "filters": _filter_hash(filters),
        "before": before,
        "ceiling": ceiling,
        "version": 1,
    }
    return (
        base64.urlsafe_b64encode(json.dumps(raw, sort_keys=True).encode())
        .decode()
        .rstrip("=")
    )


def _filter_hash(filters: CommitFilters) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "actor": filters.actor,
                "origin": filters.origin,
                "logical_key": filters.logical_key,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def cursor_decode(
    cursor: str, *, pot_id: str, generation: str, filters: CommitFilters
) -> tuple[int, int]:
    if len(cursor) > 2000:
        raise JournalError("invalid commit cursor")
    try:
        body = json.loads(
            base64.b64decode(
                cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True
            )
        )
        if (body["pot"], body["generation"], body["filters"], body["version"]) != (
            pot_id,
            generation,
            _filter_hash(filters),
            1,
        ):
            raise ValueError("cursor scope mismatch")
        before, ceiling = body["before"], body["ceiling"]
        if (
            type(before) is not int
            or type(ceiling) is not int
            or not 0 < before <= ceiling + 1
        ):
            raise ValueError("invalid cursor bounds")
        return before, ceiling
    except (ValueError, KeyError, TypeError) as exc:
        raise JournalError(
            "invalid commit cursor or filter/pot/generation mismatch"
        ) from exc


async def reconcile_history(
    journal, mirror, *, pot_id: str, max_batches: int = 10
) -> Mapping[str, Any]:
    """Bounded repair with durable progress committed by the mirror itself."""
    import asyncio

    state = await asyncio.to_thread(journal.journal_state, pot_id)
    if state is None:
        return {
            "head": None,
            "coverage_start": None,
            "indexed_through": 0,
            "indexing_lag": 0,
            "complete": False,
            "legacy_only": True,
        }
    progress = await mirror.progress_async(pot_id=pot_id, generation=state.generation)
    if progress > state.sequence:
        state = await asyncio.to_thread(journal.journal_state, pot_id)
        if state is None or progress > state.sequence:
            raise JournalError("mirror progress exceeds native journal HEAD")
    if progress:
        checkpoint = await mirror.header_at_async(
            pot_id=pot_id, generation=state.generation, sequence=progress
        )
        native_checkpoint = (
            (
                await asyncio.to_thread(
                    journal.get_receipt,
                    pot_id=pot_id,
                    commit_id=checkpoint["commit_id"],
                )
            )
            if checkpoint
            else None
        )
        if (
            checkpoint is None
            or native_checkpoint is None
            or commit_header(native_checkpoint) != dict(checkpoint)
        ):
            raise JournalError("mirror/native receipt hash mismatch at checkpoint")
    for _ in range(max_batches):
        if progress >= state.sequence:
            break
        receipts = await asyncio.to_thread(
            journal.read_receipts,
            pot_id=pot_id,
            generation=state.generation,
            after_sequence=progress,
            limit=100,
        )
        receipts = tuple(r for r in receipts if r.sequence <= state.sequence)
        if not receipts or [r.sequence for r in receipts] != list(
            range(progress + 1, progress + len(receipts) + 1)
        ):
            raise JournalError("native journal has a coverage gap")
        # Validate chain and pot/generation before accepting any batch.
        previous = None
        if progress:
            prior = await mirror.header_at_async(
                pot_id=pot_id, generation=state.generation, sequence=progress
            )
            if prior is None:
                raise JournalError("mirror checkpoint references a missing header")
            previous = prior["commit_id"]
            native_prior = await asyncio.to_thread(
                journal.get_receipt, pot_id=pot_id, commit_id=previous
            )
            if (
                native_prior is None
                or journal_hash(native_prior) != prior["receipt_hash"]
            ):
                raise JournalError("mirror/native receipt hash mismatch")
        for receipt in receipts:
            if (
                receipt.pot_id,
                receipt.journal_generation,
                receipt.parent_commit_id,
            ) != (pot_id, state.generation, previous):
                raise JournalError("native journal parent or scope mismatch")
            previous = receipt.commit_id
        await mirror.append_batch_async(
            headers=tuple(commit_header(r) for r in receipts),
            expected_progress=progress,
        )
        # Another replica may have won or progressed further. Re-read rather
        # than overwriting its checkpoint or skipping an unverified sequence.
        progress = await mirror.progress_async(
            pot_id=pot_id, generation=state.generation
        )
    latest = await asyncio.to_thread(journal.journal_state, pot_id)
    if latest is None or latest.generation != state.generation:
        raise JournalError("journal generation changed during recovery")
    state = latest
    return {
        "head": state.head,
        "journal_generation": state.generation,
        "native_sequence": state.sequence,
        "coverage_start": state.coverage_start,
        "indexed_through": progress,
        "indexing_lag": max(0, state.sequence - progress),
        "complete": progress >= state.sequence,
        "legacy_only": False,
        "resource_generation": state.resource_generation,
        "resource_operation_incomplete": state.resource_guard is not None,
        "max_page_size": min(200, state.limits.max_page_size),
    }


async def list_history(
    journal,
    mirror,
    *,
    pot_id: str,
    cursor: str | None = None,
    limit: int = 50,
    filters: CommitFilters = _DEFAULT_FILTERS,
):
    if not 1 <= limit <= 200:
        raise JournalError("commit page size must be between 1 and 200")
    coverage = await reconcile_history(journal, mirror, pot_id=pot_id)
    if coverage["legacy_only"]:
        return {"headers": (), "next_cursor": None, "coverage": coverage}
    if limit > coverage["max_page_size"]:
        raise JournalError("commit page exceeds configured page limit")
    generation = coverage["journal_generation"]
    ceiling = coverage["indexed_through"]
    before = ceiling + 1
    if cursor is not None:
        before, ceiling = cursor_decode(
            cursor, pot_id=pot_id, generation=generation, filters=filters
        )
        if ceiling > coverage["indexed_through"]:
            raise JournalError("cursor exceeds verified mirror coverage")
    page_progress, rows = await mirror.read_page_async(
        pot_id=pot_id,
        generation=generation,
        before_sequence=before,
        ceiling=ceiling,
        actor=filters.actor,
        origin=filters.origin,
        logical_key=filters.logical_key,
        limit=limit + 1,
    )
    if page_progress < coverage["indexed_through"]:
        raise JournalError(
            "History index is being rebuilt; retry the listing.",
            code="history_rebuilding",
        )
    for row in rows:
        verify_header(row)
    page, used_bytes = [], 0
    for row in rows[:limit]:
        size = len(journal_json(dict(row)).encode())
        if used_bytes + size > _MAX_RESPONSE_BYTES - 4000:
            if not page:
                raise JournalError("commit header exceeds listing response byte limit")
            break
        page.append(row)
        used_bytes += size
    more = len(rows) > len(page)
    next_cursor = (
        cursor_encode(
            pot_id=pot_id,
            generation=generation,
            filters=filters,
            before=page[-1]["sequence"],
            ceiling=ceiling,
        )
        if more
        else None
    )
    result = {"headers": page, "next_cursor": next_cursor, "coverage": coverage}
    if len(journal_json(result).encode()) > _MAX_RESPONSE_BYTES:
        raise JournalError("listing response exceeds byte limit")
    return result


def receipt_detail(
    journal, *, pot_id: str, commit_id: str, offset: int = 0, limit: int = 100
):
    """Native detail pagination is bounded independently of listing size."""
    if not 1 <= limit <= 200 or offset < 0:
        raise JournalError("invalid detail pagination")
    receipt = journal.get_receipt(pot_id=pot_id, commit_id=commit_id)
    if receipt is None:
        raise JournalError("commit is unavailable or outside coverage")
    state = journal.journal_state(pot_id)
    if state is None or limit > state.limits.max_detail_records:
        raise JournalError("invalid detail pagination: configured record limit")
    changes = (*receipt.semantic_changes, *receipt.audit_changes)
    page = changes[offset : offset + limit]
    ids = {c.record_id for c in page}
    for context in receipt.display_context:
        if context["record_id"] in ids:
            ids.update(context.get("endpoint_record_ids", ()))
    display = tuple(c for c in receipt.display_context if c["record_id"] in ids)
    if len(journal_json((page, display)).encode()) > min(
        1_000_000, state.limits.max_bytes
    ):
        raise JournalError(
            "detail page exceeds byte limit; request a smaller record page"
        )
    result = {
        "header": commit_header(receipt),
        "changes": changes[offset : offset + limit],
        "next_offset": offset + limit if offset + limit < len(changes) else None,
        "display_context": display,
        "historical_view": "partial changed fields with recorded context",
    }
    if len(journal_json(result).encode()) > min(
        _MAX_RESPONSE_BYTES, state.limits.max_bytes
    ):
        raise JournalError(
            "detail response exceeds byte limit; request a smaller record page"
        )
    return result


async def rebuild_history(journal, mirror, *, pot_id: str, max_batches: int = 10):
    """Explicit derived-index rebuild. Does not erase or alter native receipts."""
    import asyncio

    state = await asyncio.to_thread(journal.journal_state, pot_id)
    if state is None:
        raise JournalError("journal is not active")
    await mirror.reset_for_rebuild_async(pot_id=pot_id, generation=state.generation)
    return await reconcile_history(
        journal, mirror, pot_id=pot_id, max_batches=max_batches
    )
