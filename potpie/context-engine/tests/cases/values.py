"""Value-contract cases; expectations are independent of production results."""

from pytest import param

from potpie_context_engine.core.identity import IdentityClass
from potpie_context_engine.core.ports.resource_store import (
    DEFAULT_SECTION_SLUG,
    RESOURCE_ID_INVALID,
    RESOURCE_SLUG_INVALID,
    Chunk,
    ChunkRef,
    DocumentManifest,
    ResourceId,
    SectionManifest,
)


RESOLVE_MODES = (
    *(
        param(mode, mode, id=f"known-{mode}")
        for mode in ("fast", "balanced", "deep", "verify")
    ),
    param(None, "fast", id="none"),
    param("", "fast", id="empty"),
    param("turbo", "fast", id="unknown"),
    param("  deep  ", "deep", id="whitespace"),
    param("FAST", "fast", id="uppercase"),
    param("Deep", "deep", id="mixed-case"),
)

SOURCE_POLICIES = (
    *(
        param(policy, policy, id=f"known-{policy}")
        for policy in (
            "references_only",
            "summary",
            "verify",
            "snippets",
            "full_if_needed",
        )
    ),
    param(None, "references_only", id="none"),
    param("aggressive", "references_only", id="unknown"),
    param("  SUMMARY  ", "summary", id="whitespace-and-case"),
)

# Invalid cases identify every offending field, rather than accepting any error.
SOURCE_PROPERTIES = (
    param({}, (), id="empty"),
    param(
        {
            "last_seen_at": "2024-01-01T00:00:00Z",
            "last_verified_at": "2024-06-01T12:00:00+00:00",
        },
        (),
        id="timestamps",
    ),
    param({"last_seen_at": "not-a-date"}, ("last_seen_at",), id="invalid-last-seen"),
    param(
        {"last_verified_at": "yesterday"},
        ("last_verified_at",),
        id="invalid-last-verified",
    ),
    *(
        param({"access": access}, (), id=f"access-{access}")
        for access in (
            "allowed",
            "unknown",
            "permission_denied",
            "source_unreachable",
            "missing",
        )
    ),
    param({"access": "denied_by_policy"}, ("access",), id="invalid-access"),
    param({"freshness": "moldy"}, ("freshness",), id="invalid-freshness"),
    param(
        {"verification_state": "maybe"},
        ("verification_state",),
        id="invalid-verification",
    ),
    param({"sync_status": "in_flight"}, ("sync_status",), id="invalid-sync-status"),
    param({"freshness_ttl_hours": 0}, ("freshness_ttl_hours",), id="zero-ttl"),
    param({"freshness_ttl_hours": -5}, ("freshness_ttl_hours",), id="negative-ttl"),
    param(
        {"freshness_ttl_hours": "three"}, ("freshness_ttl_hours",), id="non-integer-ttl"
    ),
    param({"freshness_ttl_hours": 24}, (), id="positive-ttl"),
    param(
        {"access": "bad", "freshness": "bad", "verification_state": "bad"},
        ("access", "freshness", "verification_state"),
        id="multiple-errors",
    ),
)

SOURCE_KEYS = (
    param(
        "github",
        "pr:42",
        {"source_system": "github.com"},
        "github.com:pr:42",
        id="explicit-system",
    ),
    param("jira", "JIRA-99", {}, "jira:JIRA-99", id="source-type"),
    param(
        "docs",
        None,
        {"uri": "https://docs.example.com/page"},
        "docs:https://docs.example.com/page",
        id="uri-fallback",
    ),
    param(None, None, {}, "unknown:unknown", id="missing-identifiers"),
    param("  github  ", "  pr:1  ", {}, "github:pr:1", id="whitespace"),
)

SOURCE_MAPPINGS = (
    param(
        {"source_type": "github", "external_id": "pr:1"},
        {
            "source_type": "github",
            "external_id": "pr:1",
            "ref": "github:pr:1",
            "fetchable": False,
        },
        (),
        id="minimal",
    ),
    param(
        {"ref": "explicit-ref-key", "source_type": "github", "external_id": "x"},
        {"ref": "explicit-ref-key"},
        (),
        id="explicit-ref",
    ),
    param(
        {"ref_type": "linear", "external_id": "L-1"},
        {"source_type": "linear"},
        (),
        id="ref-type",
    ),
    param({"kind": "doc", "external_id": "d-1"}, {"source_type": "doc"}, (), id="kind"),
    param({}, None, (), id="missing-identity"),
    param(
        {"source_type": "github", "external_id": "x", "access": "forbidden"},
        {"access": "unknown"},
        (),
        id="invalid-access",
    ),
    param(
        {"source_type": "github", "external_id": "x", "freshness": "rotten"},
        {"freshness": "unknown"},
        (),
        id="invalid-freshness",
    ),
    param(
        {"source_type": "github", "external_id": "x", "verification_state": "maybe"},
        {"verification_state": "unverified"},
        (),
        id="invalid-verification",
    ),
    param(
        {
            "source_type": "github",
            "external_id": "pr:1",
            "retrieval_uri": "https://github.com/org/repo/pull/1",
        },
        {"fetchable": True},
        (),
        id="fetchable-uri",
    ),
    param(
        {"source_type": "doc", "external_id": "d1", "name": "Architecture ADR"},
        {"title": "Architecture ADR"},
        (),
        id="name-title",
    ),
    param(
        {"source_type": "github", "external_id": "pr:1", "source_system": None},
        {},
        ("source_system",),
        id="resolver-omits-null",
    ),
    param(
        {"source_type": "linear", "external_id": "L-5", "provider": "linear.app"},
        {"source_system": "linear.app"},
        (),
        id="provider-system",
    ),
)

# Indices express both stable ordering and retention of the first full record.
SOURCE_DEDUPLICATION = (
    param((("a", None), ("b", None), ("a", None)), (0, 1), id="duplicates"),
    param((), (), id="empty"),
    param((("x", None), ("y", None), ("z", None)), (0, 1, 2), id="unique"),
    param((("same", "First"), ("same", "Second")), (0,), id="first-record-wins"),
)

SOURCE_FRESHNESS = (
    param((), {"status": "unknown", "last_graph_update": None}, (), id="empty"),
    param(
        (
            {
                "ref": "a",
                "source_type": "github",
                "freshness": "fresh",
                "verification_state": "verified",
                "sync_status": "synced",
                "last_verified_at": "2024-06-01T00:00:00Z",
            },
        ),
        {"status": "fresh", "stale_refs": []},
        (),
        id="fresh",
    ),
    param(
        ({"ref": "a", "source_type": "github", "freshness": "stale"},),
        {"status": "stale", "stale_refs": ["a"]},
        (),
        id="stale",
    ),
    param(
        (
            {"ref": "a", "source_type": "github", "freshness": "stale"},
            {"ref": "b", "source_type": "github", "freshness": "source_unreachable"},
        ),
        {"status": "source_unreachable"},
        (),
        id="unreachable-before-stale",
    ),
    param(
        ({"ref": "r", "source_type": "docs", "sync_status": "source_unreachable"},),
        {"status": "source_unreachable"},
        (),
        id="unreachable-sync",
    ),
    param(
        ({"ref": "r", "source_type": "docs", "access": "source_unreachable"},),
        {"status": "source_unreachable"},
        (),
        id="unreachable-access",
    ),
    param(
        ({"ref": "r", "source_type": "github", "verification_state": "unverified"},),
        {"status": "needs_verification", "needs_verification_refs": ["r"]},
        (),
        id="unverified",
    ),
    param(
        (
            {
                "ref": "a",
                "source_type": "github",
                "freshness": "fresh",
                "verification_state": "verified",
                "last_seen_at": "2024-01-01T00:00:00Z",
                "last_verified_at": "2024-06-01T00:00:00Z",
            },
            {
                "ref": "b",
                "source_type": "github",
                "freshness": "fresh",
                "verification_state": "verified",
                "last_seen_at": "2024-03-01T00:00:00Z",
                "last_verified_at": "2024-02-01T00:00:00Z",
            },
        ),
        {},
        ("2024-06", "2024-03"),
        id="latest-update",
    ),
    param(
        (
            {
                "ref": "a",
                "source_type": "github",
                "freshness": "fresh",
                "verification_state": "verified",
                "last_verified_at": "2024-06-01T00:00:00Z",
            },
            {
                "ref": "b",
                "source_type": "docs",
                "verification_state": "needs_verification",
            },
        ),
        {"status": "needs_verification"},
        (),
        id="verification-before-fresh",
    ),
)

SOURCE_FALLBACKS = (
    param(
        "references_only",
        ({"ref": "a", "source_type": "github"},),
        (),
        (),
        0,
        id="references-only",
    ),
    param("summary", (), (("no_source_references", None),), (), 1, id="no-references"),
    param(
        "summary",
        ({"ref": "a", "source_type": "github", "fetchable": False},),
        (("source_resolver_unavailable", None),),
        (),
        None,
        id="no-resolver",
    ),
    param(
        "verify",
        (
            {
                "ref": "a",
                "source_type": "github",
                "verification_state": "unverified",
                "fetchable": True,
            },
        ),
        (("source_unverified", "a"),),
        (),
        None,
        id="unverified",
    ),
    param(
        "verify",
        (
            {
                "ref": "a",
                "source_type": "github",
                "verification_state": "verified",
                "fetchable": True,
            },
        ),
        (),
        ("source_unverified",),
        None,
        id="verified",
    ),
    param(
        "nonexistent",
        ({"ref": "a", "source_type": "github"},),
        (),
        (),
        0,
        id="unknown-policy",
    ),
)

CLASSIFICATION = (
    param(
        ("Entity",),
        {},
        ("DEPLOYED_TO",),
        (),
        ("Service",),
        (),
        False,
        id="deployment-source",
    ),
    param(
        ("Entity",),
        {},
        (),
        ("DEPLOYED_TO",),
        ("Environment",),
        (),
        False,
        id="deployment-target",
    ),
    param(
        ("Entity",),
        {},
        ("DEFINED_IN",),
        (),
        ("Service",),
        (),
        False,
        id="definition-source",
    ),
    param(
        ("Entity",),
        {},
        (),
        ("DEFINED_IN",),
        ("Repository",),
        (),
        False,
        id="definition-target",
    ),
    param(("Entity",), {}, (), ("USES",), (), (), True, id="ambiguous-uses"),
    param(
        ("Entity",), {}, (), ("HOSTED_ON",), ("Cluster",), (), False, id="host-target"
    ),
    param(
        ("Entity",),
        {},
        (),
        ("MEMBER_OF",),
        ("Team",),
        (),
        False,
        id="membership-target",
    ),
    param(
        ("Entity",),
        {},
        (),
        ("deployed-to",),
        ("Environment",),
        (),
        False,
        id="normalized-edge",
    ),
    param(("Entity",), {}, (), ("OWNED_BY",), (), (), True, id="ambiguous-owner"),
    *(
        param(
            ("Entity",),
            {field: text},
            (),
            (),
            (label,),
            (),
            False,
            id=f"text-{label.lower()}",
        )
        for field, text, label in (
            ("name", "the auth service", "Service"),
            ("summary", "the platform monorepo", "Repository"),
            ("name", "production environment", "Environment"),
            ("summary", "the orders postgres database", "DataStore"),
            ("name", "the eks cluster", "Cluster"),
            ("name", "the identity team", "Team"),
            ("summary", "the code owner", "Person"),
        )
    ),
    param(
        ("Entity",),
        {"canonical_type": "Service"},
        (),
        (),
        ("Service",),
        (),
        False,
        id="explicit-type",
    ),
    param(
        ("Entity",),
        {"canonical_type": "NotAType"},
        (),
        (),
        (),
        (),
        True,
        id="unknown-type",
    ),
    param(
        ("Entity",),
        {"name": "auth service"},
        ("DEPLOYED_TO",),
        (),
        ("Service",),
        (),
        False,
        id="combined-signals",
    ),
    param(
        ("Entity", "Service"),
        {},
        ("DEPLOYED_TO",),
        (),
        (),
        ("Service",),
        False,
        id="existing-label",
    ),
)

SLUG_IDENTITIES = (
    *(
        param(name, "service:auth-svc", id=f"normalized-{index}")
        for index, name in enumerate(("auth-svc", "Auth-SVC", " Auth   SVC "))
    ),
    param("Auth Service", "service:auth-service", id="minted-slug-body"),
    *(
        param(name, None, id=f"empty-{index}")
        for index, name in enumerate(("", "   ", "!!!", "\t"))
    ),
)

EXTERNAL_IDENTITIES = (
    param(
        {"external_id": "1042", "extra_segments": ("github/pr",)},
        "activity:github/pr:1042",
        id="extra-segment",
    ),
    param(
        {"external_id": "abc123def456"},
        "activity:abc123def456",
        id="identifier-preserved",
    ),
    param({"name": "ignored", "external_id": None}, None, id="identifier-required"),
)

CONTENT_IDENTITIES = (
    param(
        "ADR-007: adopt Position B for canonical edges",
        "ADR-007: adopt Position B for canonical edges",
        True,
        id="deterministic",
    ),
    param("ADR-007", "ADR-008", False, id="different-content"),
    param("", None, None, id="content-required"),
)

ENTITY_KEYS = (
    param("Service", {"name": "checkout-v2"}, None, True, id="slug-round-trip"),
    param("Activity", {"external_id": "42"}, None, True, id="external-round-trip"),
    param(
        "Decision",
        {"content": "A note about a decision"},
        None,
        True,
        id="content-round-trip",
    ),
    param("Service", None, "feature:checkout-v2", False, id="wrong-prefix"),
    param("Service", None, "", False, id="empty-key"),
    param("Service", None, "service:", False, id="empty-body"),
)

SLUG_BODIES = (
    param("auth-service", True, id="minted-body"),
    *(
        param(body, False, id=f"invalid-{index}")
        for index, body in enumerate(
            ("service:auth", "Auth", "-auth", "auth-", "au th", "au_th", "", 7)
        )
    ),
)

IDENTITY_REGISTRATIONS = (
    param(IdentityClass.SLUG_ALIAS, False, id="identical-registration"),
    param(IdentityClass.EXTERNAL_ID, True, id="conflicting-registration"),
)

RESOURCE_FORMATS = (
    param(
        "q3-review",
        "capacity",
        0,
        "potpie://res/q3-review/capacity/0000",
        None,
        id="zero-padding",
    ),
    param(
        "q3-review",
        "capacity",
        42,
        "potpie://res/q3-review/capacity/0042",
        None,
        id="sequence-padding",
    ),
    param("d", "s", 12345, "potpie://res/d/s/12345", None, id="wide-sequence"),
    *(
        param(doc, section, 0, None, RESOURCE_SLUG_INVALID, id=f"invalid-slug-{index}")
        for index, (doc, section) in enumerate(
            (("Q3 Review", "body"), ("q3", "Body"), ("../up", "body"))
        )
    ),
    *(
        param(
            "doc",
            "body",
            seq,
            None,
            RESOURCE_ID_INVALID,
            id=f"invalid-sequence-{index}",
        )
        for index, seq in enumerate((-1, "0", 1.5, True))
    ),
)

RESOURCE_PARSE = (
    *(
        param(
            f"potpie://res/q3-review/capacity/{text}",
            ResourceId("q3-review", "capacity", seq),
            None,
            id=f"round-trip-{seq}",
        )
        for text, seq in (("0000", 0), ("0007", 7), ("9999", 9999), ("10000", 10000))
    ),
    *(
        param(value, None, RESOURCE_ID_INVALID, id=f"wrong-prefix-{index}")
        for index, value in enumerate(
            ("q3-review/body/0000", "potpie://q3-review/body/0000", "", None)
        )
    ),
    *(
        param(
            f"potpie://res/{body}",
            None,
            RESOURCE_ID_INVALID,
            id=f"segment-count-{index}",
        )
        for index, body in enumerate(("doc/body", "doc/body/0000/extra", "doc"))
    ),
    *(
        param(
            f"potpie://res/{body}",
            None,
            RESOURCE_SLUG_INVALID,
            id=f"invalid-slug-{index}",
        )
        for index, body in enumerate(
            ("Doc/body/0000", "doc/Body/0000", "doc//0000", "/body/0000")
        )
    ),
    *(
        param(
            f"potpie://res/doc/body/{seq}",
            None,
            RESOURCE_ID_INVALID,
            id=f"invalid-sequence-{index}",
        )
        for index, seq in enumerate(
            ("0abc", "abcd", "-001", "+001", "1_00", " 000", "٠٠٠٠")
        )
    ),
    *(
        param(
            f"potpie://res/doc/body/{seq}",
            None,
            RESOURCE_ID_INVALID,
            id=f"noncanonical-padding-{seq}",
        )
        for seq in ("0", "1", "00000", "00042")
    ),
)

RESOURCE_SLUGS = (
    *(
        param(slug, None, True, id=f"valid-{slug}")
        for slug in ("q3-review", "body", DEFAULT_SECTION_SLUG, "a", "s3-2026")
    ),
    *(
        param(slug, None, False, id=f"invalid-{index}")
        for index, slug in enumerate(
            ("..", "../up", "a/b", "a\\b", ".", "", " body", "BODY", None, 7)
        )
    ),
    param("Bad Slug", "section", False, id="named-segment-error"),
)

_SECTION = SectionManifest(
    slug="capacity",
    title="Capacity",
    summary="how much load the plan assumes",
    ordinal=1,
    content_hash="abc123",
    chunks=(ChunkRef(seq=3, label="limits", page=11),),
)
RESOURCE_DTOS = (
    param(
        Chunk(
            resource_id="potpie://res/q3-review/capacity/0003",
            doc="q3-review",
            section="capacity",
            seq=3,
            text="the text",
            chars=8,
            revision=2,
            source_ref="file:///q3.pdf",
            page=11,
            offset=204,
        ),
        id="chunk",
    ),
    param(
        DocumentManifest(
            pot_id="pot_1",
            doc="q3-review",
            revision=2,
            source_ref="file:///q3.pdf",
            source_kind="pdf",
            sections=(_SECTION,),
            sections_added=("risks",),
            sections_kept=("capacity",),
            sections_removed=("appendix",),
            warnings=("section 'appendix' has no chunks",),
        ),
        id="document",
    ),
    param(_SECTION, id="section"),
    param(ResourceId("d", "s", 0), id="resource-id"),
)

RESOURCE_ERRORS = (
    param(
        RESOURCE_SLUG_INVALID,
        "document slug is not a valid slug",
        "Q3 Review",
        "Use q3-review.",
        id="detailed",
    ),
    param(RESOURCE_ID_INVALID, "bad id", None, None, id="minimal"),
)
