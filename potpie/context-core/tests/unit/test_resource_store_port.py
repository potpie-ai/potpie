"""Resource-store contract: id grammar, slug reuse, and RPC round-tripping."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from potpie_context_core.identity import is_valid_slug_body
from potpie_context_core.ports.resource_store import (
    DEFAULT_SECTION_SLUG,
    RESOURCE_CHUNK_MAX_CHARS,
    RESOURCE_CHUNK_TARGET_CHARS,
    RESOURCE_ID_INVALID,
    RESOURCE_SEQ_WIDTH,
    RESOURCE_SLUG_INVALID,
    RESOURCE_URI_PREFIX,
    Chunk,
    ChunkRef,
    DocumentManifest,
    ResourceId,
    ResourceStoreError,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
    require_resource_slug,
)

pytestmark = pytest.mark.unit


class TestFormatResourceId:
    """Ids are the canonical, zero-padded ``potpie://res/`` form."""

    def test_sequence_is_zero_padded(self) -> None:
        assert format_resource_id("q3-review", "capacity", 0) == (
            "potpie://res/q3-review/capacity/0000"
        )
        assert format_resource_id("q3-review", "capacity", 42) == (
            "potpie://res/q3-review/capacity/0042"
        )

    def test_sequences_wider_than_the_pad_still_render(self) -> None:
        assert format_resource_id("d", "s", 12345).endswith("/12345")

    def test_invalid_slugs_are_refused(self) -> None:
        for doc, section in (("Q3 Review", "body"), ("q3", "Body"), ("../up", "body")):
            with pytest.raises(ResourceStoreError) as exc:
                format_resource_id(doc, section, 0)
            assert exc.value.code == RESOURCE_SLUG_INVALID

    def test_negative_or_non_integer_sequence_is_refused(self) -> None:
        for seq in (-1, "0", 1.5, True):
            with pytest.raises(ResourceStoreError) as exc:
                format_resource_id("doc", "body", seq)  # type: ignore[arg-type]
            assert exc.value.code == RESOURCE_ID_INVALID


class TestParseResourceId:
    """Parsing is strict so parse and format are exact inverses."""

    def test_round_trips_with_format(self) -> None:
        for seq in (0, 7, 9999, 10000):
            resource_id = format_resource_id("q3-review", "capacity", seq)
            parsed = parse_resource_id(resource_id)
            assert parsed == ResourceId(doc="q3-review", section="capacity", seq=seq)
            assert format_resource_id(parsed.doc, parsed.section, parsed.seq) == (
                resource_id
            )

    def test_wrong_prefix_is_refused(self) -> None:
        for bad in ("q3-review/body/0000", "potpie://q3-review/body/0000", "", None):
            with pytest.raises(ResourceStoreError) as exc:
                parse_resource_id(bad)  # type: ignore[arg-type]
            assert exc.value.code == RESOURCE_ID_INVALID

    def test_wrong_segment_count_is_refused(self) -> None:
        for bad in ("doc/body", "doc/body/0000/extra", "doc"):
            with pytest.raises(ResourceStoreError) as exc:
                parse_resource_id(RESOURCE_URI_PREFIX + bad)
            assert exc.value.code == RESOURCE_ID_INVALID

    def test_invalid_slug_segments_are_refused(self) -> None:
        for bad in ("Doc/body/0000", "doc/Body/0000", "doc//0000", "/body/0000"):
            with pytest.raises(ResourceStoreError) as exc:
                parse_resource_id(RESOURCE_URI_PREFIX + bad)
            assert exc.value.code == RESOURCE_SLUG_INVALID

    def test_non_numeric_or_negative_sequence_is_refused(self) -> None:
        for bad in ("0abc", "abcd", "-001", "+001", "1_00", " 000", "٠٠٠٠"):
            with pytest.raises(ResourceStoreError) as exc:
                parse_resource_id(f"{RESOURCE_URI_PREFIX}doc/body/{bad}")
            assert exc.value.code == RESOURCE_ID_INVALID

    def test_unpadded_sequence_is_refused(self) -> None:
        # Only one spelling may address a chunk.
        for bad in ("0", "1", "00000", "00042"):
            with pytest.raises(ResourceStoreError) as exc:
                parse_resource_id(f"{RESOURCE_URI_PREFIX}doc/body/{bad}")
            assert exc.value.code == RESOURCE_ID_INVALID


class TestRequireResourceSlug:
    """The ``--doc`` / ``--section`` grammar is the graph's own slug grammar."""

    def test_accepts_what_the_identity_grammar_accepts(self) -> None:
        for slug in ("q3-review", "body", DEFAULT_SECTION_SLUG, "a", "s3-2026"):
            assert is_valid_slug_body(slug)
            assert require_resource_slug(slug) == slug

    def test_refuses_traversal_and_separators(self) -> None:
        for slug in ("..", "../up", "a/b", "a\\b", ".", "", " body", "BODY", None, 7):
            with pytest.raises(ResourceStoreError) as exc:
                require_resource_slug(slug)  # type: ignore[arg-type]
            assert exc.value.code == RESOURCE_SLUG_INVALID

    def test_error_names_the_segment_that_failed(self) -> None:
        with pytest.raises(ResourceStoreError) as exc:
            require_resource_slug("Bad Slug", kind="section")
        assert "section slug" in str(exc.value)
        assert exc.value.recommended_next_action


class TestChunkSizeContract:
    """The cap is a hard limit above the target, enforced at import."""

    def test_target_is_below_the_hard_cap(self) -> None:
        assert 0 < RESOURCE_CHUNK_TARGET_CHARS < RESOURCE_CHUNK_MAX_CHARS


class TestDtosSurviveRpc:
    """``daemon/rpc.py`` rebuilds DTOs with ``cls(**decoded_fields)``."""

    def test_every_field_is_accepted_by_the_constructor(self) -> None:
        chunk = Chunk(
            resource_id=format_resource_id("q3-review", "capacity", 3),
            doc="q3-review",
            section="capacity",
            seq=3,
            text="the text",
            chars=8,
            revision=2,
            source_ref="file:///q3.pdf",
            page=11,
            offset=204,
        )
        manifest = DocumentManifest(
            pot_id="pot_1",
            doc="q3-review",
            revision=2,
            source_ref="file:///q3.pdf",
            source_kind="pdf",
            sections=(
                SectionManifest(
                    slug="capacity",
                    title="Capacity",
                    summary="how much load the plan assumes",
                    ordinal=1,
                    content_hash="abc123",
                    chunks=(ChunkRef(seq=3, label="limits", page=11),),
                ),
            ),
            sections_added=("risks",),
            sections_kept=("capacity",),
            sections_removed=("appendix",),
            warnings=("section 'appendix' has no chunks",),
        )

        for value in (chunk, manifest, manifest.sections[0], ResourceId("d", "s", 0)):
            assert is_dataclass(value)
            rebuilt = type(value)(
                **{field.name: getattr(value, field.name) for field in fields(value)}
            )
            assert rebuilt == value

    def test_sequence_fields_are_tuples_not_sets(self) -> None:
        # rpc.encode() degrades set/frozenset to list, which would not decode
        # back to the declared type.
        for spec in (DocumentManifest, SectionManifest):
            for field in fields(spec):
                assert "set" not in str(field.type)


class TestResourceStoreError:
    """The error carries the four fields the CLI output contract needs."""

    def test_carries_code_message_detail_and_next_action(self) -> None:
        error = ResourceStoreError(
            RESOURCE_SLUG_INVALID,
            "document slug is not a valid slug",
            detail="Q3 Review",
            recommended_next_action="Use q3-review.",
        )
        assert error.code == RESOURCE_SLUG_INVALID
        assert str(error) == "document slug is not a valid slug"
        assert error.detail == "Q3 Review"
        assert error.recommended_next_action == "Use q3-review."

    def test_detail_and_next_action_are_optional(self) -> None:
        error = ResourceStoreError(RESOURCE_ID_INVALID, "bad id")
        assert (error.detail, error.recommended_next_action) == (None, None)


def test_seq_width_matches_the_documented_padding() -> None:
    assert RESOURCE_SEQ_WIDTH == 4
    assert format_resource_id("d", "s", 0).endswith("0000")
