"""Shared value contracts; each input has a stable, independently reported case."""

from dataclasses import fields, is_dataclass

import pytest

from potpie_context_engine.core.identity import (
    IdentityClass,
    IdentityError,
    IdentitySpec,
    get_identity,
    is_valid_slug_body,
    mint_entity_key,
    register_identity,
    validate_entity_key,
)
from potpie_context_engine.core.ontology_classifier import (
    build_signals,
    classify_entity,
)
from potpie_context_engine.core.ports.resource_store import (
    RESOURCE_CHUNK_MAX_CHARS,
    RESOURCE_CHUNK_TARGET_CHARS,
    RESOURCE_SEQ_WIDTH,
    RESOURCE_SLUG_INVALID,
    ResourceId,
    ResourceStoreError,
    format_resource_id,
    parse_resource_id,
    require_resource_slug,
)
from potpie_context_engine.core.source_references import (
    SourceReferenceRecord,
    assess_freshness,
    dedupe_source_references,
    normalize_resolve_mode,
    normalize_source_policy,
    source_policy_fallbacks,
    source_ref_key,
    source_reference_from_mapping,
    validate_source_reference_properties,
)
from tests.cases import values

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw, expected", values.RESOLVE_MODES)
def test_resolve_mode(raw: str | None, expected: str) -> None:
    assert normalize_resolve_mode(raw) == expected


@pytest.mark.parametrize("raw, expected", values.SOURCE_POLICIES)
def test_source_policy(raw: str | None, expected: str) -> None:
    assert normalize_source_policy(raw) == expected


@pytest.mark.parametrize("properties, invalid_fields", values.SOURCE_PROPERTIES)
def test_source_properties(properties: dict, invalid_fields: tuple[str, ...]) -> None:
    errors = validate_source_reference_properties(properties)
    assert len(errors) == len(invalid_fields)
    for field in invalid_fields:
        assert any(field in error for error in errors), (field, errors)


@pytest.mark.parametrize(
    "source_type, external_id, options, expected", values.SOURCE_KEYS
)
def test_source_key(
    source_type: str | None, external_id: str | None, options: dict, expected: str
) -> None:
    assert source_ref_key(source_type, external_id, **options) == expected


@pytest.mark.parametrize("mapping, expected, absent_hints", values.SOURCE_MAPPINGS)
def test_source_mapping(
    mapping: dict, expected: dict | None, absent_hints: tuple[str, ...]
) -> None:
    reference = source_reference_from_mapping(mapping)
    if expected is None:
        assert reference is None
        return
    assert reference is not None
    for field, value in expected.items():
        assert getattr(reference, field) == value, field
    assert not set(absent_hints).intersection(reference.resolver_hint)


@pytest.mark.parametrize("records, expected_indices", values.SOURCE_DEDUPLICATION)
def test_source_deduplication(
    records: tuple, expected_indices: tuple[int, ...]
) -> None:
    references = [
        SourceReferenceRecord(ref=ref, source_type="github", title=title)
        for ref, title in records
    ]
    assert dedupe_source_references(references) == [
        references[index] for index in expected_indices
    ]


@pytest.mark.parametrize("records, expected, latest_prefixes", values.SOURCE_FRESHNESS)
def test_source_freshness(
    records: tuple[dict, ...], expected: dict, latest_prefixes: tuple[str, ...]
) -> None:
    report = assess_freshness([SourceReferenceRecord(**record) for record in records])
    for field, value in expected.items():
        assert getattr(report, field) == value, field
    if latest_prefixes:
        assert report.last_graph_update is not None
        assert any(prefix in report.last_graph_update for prefix in latest_prefixes)


@pytest.mark.parametrize(
    "policy, records, required, forbidden, count", values.SOURCE_FALLBACKS
)
def test_source_fallbacks(
    policy: str,
    records: tuple[dict, ...],
    required: tuple,
    forbidden: tuple[str, ...],
    count: int | None,
) -> None:
    fallbacks = source_policy_fallbacks(
        source_policy=policy,
        refs=[SourceReferenceRecord(**record) for record in records],
    )
    if count is not None:
        assert len(fallbacks) == count
    for code, ref in required:
        assert any(
            item.code == code and (ref is None or item.ref == ref) for item in fallbacks
        )
    assert not set(forbidden).intersection(item.code for item in fallbacks)


@pytest.mark.parametrize(
    "labels, properties, outgoing, incoming, required, forbidden, empty",
    values.CLASSIFICATION,
)
def test_entity_classification(
    labels: tuple[str, ...],
    properties: dict,
    outgoing: tuple[str, ...],
    incoming: tuple[str, ...],
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
    empty: bool,
) -> None:
    result = classify_entity(
        build_signals(
            labels=labels,
            properties=properties,
            outgoing_edge_names=outgoing,
            incoming_edge_names=incoming,
        )
    )
    assert set(required).issubset(result)
    assert not set(forbidden).intersection(result)
    if empty:
        assert result == ()


@pytest.mark.parametrize("name, expected", values.SLUG_IDENTITIES)
def test_slug_identity(name: str, expected: str | None) -> None:
    spec = get_identity("Service")
    assert spec is not None
    if expected is None:
        with pytest.raises(IdentityError):
            mint_entity_key(spec, name=name)
    else:
        key = mint_entity_key(spec, name=name)
        assert key == expected
        assert is_valid_slug_body(key.split(":", 1)[1]) is True


@pytest.mark.parametrize("arguments, expected", values.EXTERNAL_IDENTITIES)
def test_external_identity(arguments: dict, expected: str | None) -> None:
    spec = get_identity("Activity")
    assert spec is not None
    if expected is None:
        with pytest.raises(IdentityError):
            mint_entity_key(spec, **arguments)
    else:
        assert mint_entity_key(spec, **arguments) == expected


@pytest.mark.parametrize("content, other, equal", values.CONTENT_IDENTITIES)
def test_content_identity(content: str, other: str | None, equal: bool | None) -> None:
    spec = get_identity("Decision")
    assert spec is not None
    if equal is None:
        with pytest.raises(IdentityError):
            mint_entity_key(spec, content=content)
    else:
        assert (
            mint_entity_key(spec, content=content)
            == mint_entity_key(spec, content=other)
        ) is equal


@pytest.mark.parametrize("label, arguments, key, expected", values.ENTITY_KEYS)
def test_entity_key(
    label: str, arguments: dict | None, key: str | None, expected: bool
) -> None:
    spec = get_identity(label)
    assert spec is not None
    if arguments is not None:
        key = mint_entity_key(spec, **arguments)
    assert validate_entity_key(spec, key) is expected


@pytest.mark.parametrize("body, expected", values.SLUG_BODIES)
def test_slug_body(body: object, expected: bool) -> None:
    assert is_valid_slug_body(body) is expected


@pytest.mark.parametrize("identity_class, conflict", values.IDENTITY_REGISTRATIONS)
def test_identity_registration(identity_class: IdentityClass, conflict: bool) -> None:
    spec = IdentitySpec(label="Service", klass=identity_class, key_prefix="service")
    if conflict:
        with pytest.raises(IdentityError):
            register_identity(spec)
    else:
        register_identity(spec)
        assert get_identity("Service") == spec


@pytest.mark.parametrize(
    "doc, section, sequence, expected, error", values.RESOURCE_FORMATS
)
def test_resource_id_format(
    doc: str, section: str, sequence: object, expected: str | None, error: str | None
) -> None:
    if error is not None:
        with pytest.raises(ResourceStoreError) as raised:
            format_resource_id(doc, section, sequence)
        assert raised.value.code == error
    else:
        assert RESOURCE_SEQ_WIDTH == 4
        assert format_resource_id(doc, section, sequence) == expected


@pytest.mark.parametrize("raw, expected, error", values.RESOURCE_PARSE)
def test_resource_id_parse(
    raw: str | None, expected: ResourceId | None, error: str | None
) -> None:
    if error is not None:
        with pytest.raises(ResourceStoreError) as raised:
            parse_resource_id(raw)
        assert raised.value.code == error
    else:
        parsed = parse_resource_id(raw)
        assert parsed == expected
        assert format_resource_id(parsed.doc, parsed.section, parsed.seq) == raw


@pytest.mark.parametrize("slug, kind, valid", values.RESOURCE_SLUGS)
def test_resource_slug(slug: object, kind: str | None, valid: bool) -> None:
    if valid:
        assert is_valid_slug_body(slug)
        assert require_resource_slug(slug) == slug
    else:
        options = {"kind": kind} if kind else {}
        with pytest.raises(ResourceStoreError) as raised:
            require_resource_slug(slug, **options)
        assert raised.value.code == RESOURCE_SLUG_INVALID
        if kind:
            assert f"{kind} slug" in str(raised.value)
            assert raised.value.recommended_next_action


def test_resource_chunk_bounds() -> None:
    assert 0 < RESOURCE_CHUNK_TARGET_CHARS < RESOURCE_CHUNK_MAX_CHARS


@pytest.mark.parametrize("value", values.RESOURCE_DTOS)
def test_resource_dto(value: object) -> None:
    assert is_dataclass(value)
    rebuilt = type(value)(
        **{field.name: getattr(value, field.name) for field in fields(value)}
    )
    assert rebuilt == value
    for field in fields(value):
        assert "set" not in str(field.type), field.name


@pytest.mark.parametrize("code, message, detail, next_action", values.RESOURCE_ERRORS)
def test_resource_error(
    code: str, message: str, detail: str | None, next_action: str | None
) -> None:
    error = ResourceStoreError(
        code, message, detail=detail, recommended_next_action=next_action
    )
    assert (error.code, str(error), error.detail, error.recommended_next_action) == (
        code,
        message,
        detail,
        next_action,
    )
