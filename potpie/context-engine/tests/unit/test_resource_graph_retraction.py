"""Unit tests for graph retraction mutations."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
    build_retract_mutations,
    element_entity_key,
    section_entity_key,
)

pytestmark = pytest.mark.unit


def test_build_retract_mutations_sections_and_elements() -> None:
    ops = build_retract_mutations(
        pot_id="pot-test",
        doc_slug="handbook",
        section_slugs=["page-1", "intro"],
        element_ids=["elem-00001"],
    )
    predicates = {(op["subject"]["key"], op["predicate"]) for op in ops}
    assert (section_entity_key("handbook", "page-1"), "SECTION_OF") in predicates
    assert (section_entity_key("handbook", "intro"), "RELATED_TO") in predicates
    assert (element_entity_key("handbook", "elem-00001"), "ELEMENT_OF") in predicates
