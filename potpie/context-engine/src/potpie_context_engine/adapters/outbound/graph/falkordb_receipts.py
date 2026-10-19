"""Decode authoritative FalkorDB receipts, preserving unknown legacy metadata."""

from __future__ import annotations

import json

from potpie_context_engine.core.mutation_execution import MutationPersistenceError

from ._mutation_execution import (
    MutationExecutionReuseError,
    _mutation_result_from_dict,
)

RECEIPT_COLUMNS = (
    "r.fingerprint,r.payload,r.entities,r.edges,r.deletes,r.invalidations,"
    "r.affected_claim_record_ids"
)


def decode_receipt(row, *, pot_id, mutation_id, fingerprint):
    if len(row) != 7:
        raise MutationPersistenceError("incomplete native execution receipt")
    stored, payload, *native = row
    if stored != fingerprint:
        raise MutationExecutionReuseError(
            "mutation ID was already used with a different batch"
        )
    try:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise MutationPersistenceError("native execution receipt must be an object")
        # Historical journal receipts contained only their mutation summary.
        if set(raw) == {"mutation_summary"}:
            raw["ok"] = True
            raw["mutation_id"] = mutation_id
        summary = raw["mutation_summary"]
        for name, value in zip(
            (
                "entity_upserts_applied",
                "edge_upserts_applied",
                "edge_deletes_applied",
                "invalidations_applied",
            ),
            native[:4],
            strict=True,
        ):
            if type(value) is not int or value < 0:
                raise MutationPersistenceError("invalid native execution count")
            summary[name] = value
        if "execution" in raw:
            ids = native[4]
            if not isinstance(ids, list) or any(
                not isinstance(value, str) or not value for value in ids
            ):
                raise MutationPersistenceError(
                    "invalid native affected claim identities"
                )
            raw["execution"]["affected_claim_record_ids"] = sorted(set(ids))
        result = _mutation_result_from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise MutationPersistenceError("malformed native execution receipt") from exc
    if result is None or result.mutation_id != mutation_id:
        raise MutationPersistenceError(
            "native receipt identity differs from its payload"
        )
    if result.execution is not None and (
        result.execution.pot_id != pot_id
        or result.execution.batch_fingerprint != fingerprint
    ):
        raise MutationPersistenceError(
            "native receipt metadata differs from its identity"
        )
    return result
