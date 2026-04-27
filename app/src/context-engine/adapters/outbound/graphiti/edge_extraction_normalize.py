"""Post-process Graphiti ``extract_edges`` output: reduce MODIFIED collapse, set lifecycle.

Monkey-patches ``graphiti_core.utils.maintenance.edge_operations.extract_edges`` once.
"""

from __future__ import annotations

import logging
from typing import Any

from domain.entity_schema import normalized_episodic_edge_allowlist
from domain.extraction_edges import (
    classify_episodic_edge,
    generic_modified_ratio_before_normalize,
)
from domain.reconciliation_flags import strict_extraction_enabled

logger = logging.getLogger(__name__)

_PATCH_INSTALLED = False

# Reject / log when too many edges are vague MODIFIED before normalization (regression guard).
_MAX_VAGUE_MODIFIED_RATIO = 0.25


def normalize_graphiti_extracted_edges(edges: list[Any], nodes: list[Any]) -> list[Any]:
    """Normalize relation names and stamp ``lifecycle_status`` on each edge."""
    if not edges:
        return edges

    uuid_to_labels: dict[str, tuple[str, ...]] = {}
    for n in nodes:
        uid = getattr(n, "uuid", None)
        if not uid:
            continue
        labels = tuple(getattr(n, "labels", None) or ())
        uuid_to_labels[str(uid)] = labels

    ratio = generic_modified_ratio_before_normalize(edges, uuid_to_labels)
    if ratio > _MAX_VAGUE_MODIFIED_RATIO:
        msg = (
            f"High vague MODIFIED rate ({ratio:.0%}) after extraction — "
            "check prompts or FACT_TYPES (edge-type collapse regression)."
        )
        if strict_extraction_enabled():
            raise RuntimeError(msg)
        logger.warning(msg)

    allowed = normalized_episodic_edge_allowlist()

    for edge in edges:
        fact = str(getattr(edge, "fact", "") or "")
        src = uuid_to_labels.get(str(getattr(edge, "source_node_uuid", "")), ())
        tgt = uuid_to_labels.get(str(getattr(edge, "target_node_uuid", "")), ())
        name = str(getattr(edge, "name", "") or "")

        attrs = getattr(edge, "attributes", None)
        base_attrs: dict[str, Any] = dict(attrs) if isinstance(attrs, dict) else {}
        existing_ls = (
            base_attrs.get("lifecycle_status")
            if isinstance(base_attrs.get("lifecycle_status"), str)
            else None
        )

        new_name, new_ls = classify_episodic_edge(
            name,
            fact,
            src,
            tgt,
            allowed_normalized_names=allowed,
            existing_lifecycle=existing_ls,
        )
        edge.name = new_name
        base_attrs["lifecycle_status"] = new_ls
        edge.attributes = base_attrs

    return edges


def install_extract_edges_normalize_patch() -> None:
    """Idempotent patch for Graphiti edge extraction."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    from graphiti_core.utils.maintenance import edge_operations as _eo

    _orig = _eo.extract_edges

    async def _wrapped(*args: Any, **kwargs: Any):
        edges = await _orig(*args, **kwargs)
        nodes = args[2] if len(args) > 2 else kwargs.get("nodes")
        if nodes is None:
            logger.warning("extract_edges patch: missing nodes; skipping normalize")
            return edges
        return normalize_graphiti_extracted_edges(list(edges), list(nodes))

    _eo.extract_edges = _wrapped  # type: ignore[method-assign]
    _PATCH_INSTALLED = True
