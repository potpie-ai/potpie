"""Canonical snapshot import intent, identity and frozen execution context."""

from datetime import datetime, timezone
from typing import Any, Mapping
import uuid

from .graph_mutations import ProvenanceContext
from .graph_snapshot import normalize_snapshot_payload, validate_snapshot_identities
from .mutation_execution import PreparedMutation, batch_fingerprint, request_fingerprint
from .reconciliation import MutationBatch


def normalize_snapshot_import(
    payload: Mapping[str, Any], *, pot_id: str
) -> dict[str, Any]:
    incoming = normalize_snapshot_payload(payload, target_pot_id=pot_id)
    for entity in incoming["entities"]:
        entity["labels"] = sorted(set(entity["labels"]) | {"Entity"})
        props = entity["properties"]
        _validate_ids(props)
        props["record_id"] = (
            props.get("record_id")
            or props.get("uuid")
            or str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"potpie:import:entity:{pot_id}:{entity['key']}"
                )
            )
        )
    for claim in incoming["claims"]:
        props = claim["properties"]
        _validate_ids(props)
        explicit = claim.get("record_id")
        if explicit and props.get("record_id") and explicit != props["record_id"]:
            raise ValueError("snapshot claim has conflicting record_id values")
        claim["record_id"] = (
            explicit
            or props.get("record_id")
            or props.get("uuid")
            or str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "potpie:import:claim:" + request_fingerprint(claim),
                )
            )
        )
    validate_snapshot_identities(incoming)
    return incoming


def _validate_ids(props: Mapping[str, Any]) -> None:
    for key in ("record_id", "uuid"):
        value = props.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"snapshot {key} must be a nonempty string")


def prepare_snapshot_import(
    *,
    pot_id: str,
    payload: Mapping[str, Any],
    expected_version: int,
    mutation_id: str | None = None,
) -> tuple[MutationBatch, ProvenanceContext]:
    incoming = normalize_snapshot_import(payload, pot_id=pot_id)
    batch = MutationBatch(snapshot_import=incoming)
    fingerprint = batch_fingerprint(batch)
    identity = mutation_id or f"snapshot:{fingerprint}"
    prepared = PreparedMutation(
        pot_id=pot_id,
        mutation_id=identity,
        request_fingerprint=request_fingerprint({"snapshot_import": incoming}),
        batch_fingerprint=fingerprint,
        expected_version=expected_version,
        prepared_at=datetime.now(timezone.utc),
        target_claim_record_ids=tuple(row["record_id"] for row in incoming["claims"]),
    )
    return batch, ProvenanceContext(mutation_id=identity, prepared=prepared)
