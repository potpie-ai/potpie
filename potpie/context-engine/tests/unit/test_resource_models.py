"""Unit tests for resource ingestion models."""

from __future__ import annotations

import pytest

from potpie_context_engine.domain.resource_models import (
    ResourceManifest,
    chunk_uri,
    parse_chunk_uri,
    validate_doc_slug,
)

pytestmark = pytest.mark.unit


def test_validate_doc_slug() -> None:
    assert validate_doc_slug("payments-oncall") == "payments-oncall"
    with pytest.raises(ValueError):
        validate_doc_slug("Bad_Slug")


def test_chunk_uri_roundtrip() -> None:
    uri = chunk_uri("payments-oncall", "rollback", 2)
    assert uri == "potpie://res/payments-oncall/rollback/0002"
    doc, section, seq = parse_chunk_uri(uri)
    assert doc == "payments-oncall"
    assert section == "rollback"
    assert seq == 2


def test_manifest_section_cap() -> None:
    chunks = [{"seq": i, "label": f"c{i}"} for i in range(6)]
    with pytest.raises(ValueError):
        ResourceManifest.model_validate(
            {
                "sections": [
                    {
                        "slug": "body",
                        "title": "Body",
                        "chunks": chunks,
                    }
                ]
            }
        )
