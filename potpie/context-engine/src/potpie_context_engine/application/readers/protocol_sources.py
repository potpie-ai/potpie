"""Fetchable evidence checks for protocol layouts; graph refs alone prove nothing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from potpie_context_core.ports.resource_store import (
    ResourceStoreError,
    ResourceStorePort,
)


@dataclass
class ProtocolSources:
    store: ResourceStorePort | None
    pot_id: str
    digests: dict[str, str | None] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def prefetch(self, evidence: Iterable[dict[str, Any]]) -> None:
        refs = sorted(
            {
                item["source_ref"]
                for item in evidence
                if isinstance(item.get("source_ref"), str)
                and item["source_ref"].startswith("potpie://res/")
                and item["source_ref"] not in self.digests
            }
        )
        if self.store is None:
            return
        for start in range(0, len(refs), 256):
            batch = tuple(refs[start : start + 256])
            try:
                chunks = self.store.get_many(pot_id=self.pot_id, resource_ids=batch)
            except ResourceStoreError:
                # A failed multi-read cannot identify which individual ref failed.
                # Keep coverage conservative without a per-ref retry storm.
                self.failures.update(
                    {
                        ref: "missing" if len(batch) == 1 else "unavailable"
                        for ref in batch
                    }
                )
                chunks = ()
            self.digests.update({ref: None for ref in batch})
            self.digests.update(
                {
                    chunk.resource_id: hashlib.sha256(
                        chunk.text.encode("utf-8")
                    ).hexdigest()
                    for chunk in chunks
                }
            )

    def status(self, evidence: dict[str, Any]) -> str:
        ref = evidence.get("source_ref")
        if evidence.get("chunk_id", ref) != ref:
            return "source_mismatch"
        digest = evidence.get("chunk_digest") or evidence.get("digest")
        if (
            self.store is None
            or not isinstance(ref, str)
            or not ref.startswith("potpie://res/")
        ):
            return "unknown"
        if ref not in self.digests:
            self.prefetch([evidence])
        actual = self.digests[ref]
        if actual is None:
            return self.failures.get(ref, "missing")
        if not isinstance(digest, str) or not digest:
            return "unknown"
        if digest.removeprefix("sha256:") != actual:
            return "digest_mismatch"
        if not any(
            evidence.get(k) is not None
            for k in ("locator", "page", "section", "row", "chunk_id")
        ):
            return "unknown"
        return "verified"

    def evidence_status(self, evidence: list[dict[str, Any]]) -> str:
        states = [self.status(item) for item in evidence]
        if "verified" in states:
            return "verified"
        return next((state for state in states if state != "unknown"), "unknown")

    def entity_status(self, properties: dict, evidence: list[dict[str, Any]]) -> str:
        states = [self.evidence_status(evidence)]
        corrections = properties.get("correction_evidence", {})
        if isinstance(corrections, dict):
            states.extend(
                self.evidence_status(values)
                for values in corrections.values()
                if isinstance(values, list)
            )
        return next((state for state in states if state != "verified"), "verified")
