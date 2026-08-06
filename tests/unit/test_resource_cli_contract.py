"""CLI contract coverage for ``potpie resource`` (P3).

The commands run against the real ``ResourceFacade`` over the in-memory store,
so what is asserted here is the contract an agent sees — payload fields, error
codes, exit codes, and the number of store round trips a read costs — not a
mock's call log.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, resource
from potpie.daemon.client import RemoteHostShell, RemoteSurface, _raise_remote_error
from potpie.daemon.main import _ALLOWED_RPC_SURFACES, _error_payload
from potpie.daemon.rpc import decode, encode
from potpie_context_core.ports.resource_store import (
    RESOURCE_CHUNK_MAX_CHARS,
    Chunk,
    ChunkRef,
    DocumentManifest,
    ResourceStoreError,
    ResourceStoreStatus,
    SectionManifest,
    format_resource_id,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.host.shell import ResourceFacade
from potpie_context_engine.testing import (
    InMemoryResourceStore,
    build_test_graph_runtime,
    write_import_directory,
)

pytestmark = pytest.mark.unit

DOC = "q3-review"


@pytest.fixture(autouse=True)
def _reset_json_mode():
    yield
    _common.set_json(False)


class _Pot:
    pot_id = "p"
    name = "default"
    active = True


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]

    def list_sources(self, *, pot_id):
        return []


class _CountingStore:
    """Wraps the in-memory store and counts calls, one per daemon round trip."""

    def __init__(self) -> None:
        self.inner = InMemoryResourceStore()
        self.calls: list[str] = []
        self.import_dirs: list[Path] = []

    def import_dir(self, **kwargs):
        self.calls.append("import_dir")
        self.import_dirs.append(kwargs["source_dir"])
        return self.inner.import_dir(**kwargs)

    def get_many(self, **kwargs):
        self.calls.append("get_many")
        return self.inner.get_many(**kwargs)

    def get(self, **kwargs):
        self.calls.append("get")
        return self.inner.get(**kwargs)

    def list(self, **kwargs):
        self.calls.append("list")
        return self.inner.list(**kwargs)

    def delete(self, **kwargs):
        self.calls.append("delete")
        return self.inner.delete(**kwargs)

    def purge_pot(self, pot_id):
        self.calls.append("purge_pot")
        return self.inner.purge_pot(pot_id)

    def status(self, **kwargs):
        self.calls.append("status")
        return self.inner.status(**kwargs)


class _Host:
    def __init__(self, store: _CountingStore, *, with_graph: bool = True) -> None:
        self.pots = _Pots()
        self.store = store
        # A real graph service, not a mock: P4's claim is that an import lands
        # through the ordinary write door, so the validator and lowerer have to
        # be in the loop for the assertion to mean anything.
        runtime = build_test_graph_runtime() if with_graph else None
        self.runtime = runtime
        self.graph = runtime.graph if runtime else None
        self.resources = ResourceFacade(
            store=store,
            graph=self.graph,
            claims=runtime.backend.claim_query if runtime else None,
        )


def _host(*, with_graph: bool = True) -> _Host:
    host = _Host(_CountingStore(), with_graph=with_graph)
    _common.set_host(host)
    return host


def _section(slug, *, chunks, summary="what this section covers", **overrides):
    section = {
        "slug": slug,
        "title": slug.replace("-", " ").title(),
        "summary": summary,
        "ordinal": 0,
        "content_hash": f"{slug}-v1",
        "chunks": chunks,
    }
    section.update(overrides)
    return section


def _import_dir(root: Path, sections=None, **kwargs) -> Path:
    return write_import_directory(
        root,
        sections
        or [
            _section(
                "body",
                chunks=[
                    {"label": "opening", "text": "alpha"},
                    {"label": "middle", "text": "beta"},
                    {"label": "closing", "text": "gamma"},
                ],
            )
        ],
        source_ref=kwargs.pop("source_ref", "file:///q3.pdf"),
        source_kind=kwargs.pop("source_kind", "pdf"),
    )


def _run(args, *, as_json=True):
    if as_json:
        _common.set_json(True)
    return CliRunner().invoke(resource.resource_app, args)


def _seed(tmp_path, sections=None) -> _Host:
    host = _host()
    directory = _import_dir(tmp_path / "in", sections)
    result = _run(["import", str(directory), "--doc", DOC])
    assert result.exit_code == 0, result.stdout
    host.store.calls.clear()
    return host


# --- import -----------------------------------------------------------------


def test_import_reports_the_document_and_its_sections(tmp_path):
    _host()
    directory = _import_dir(tmp_path / "in")

    result = _run(["import", str(directory), "--doc", DOC])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["doc"] == DOC
    assert payload["revision"] == 1
    assert payload["source_ref"] == "file:///q3.pdf"
    assert payload["source_kind"] == "pdf"
    assert payload["section_count"] == 1
    assert payload["chunk_count"] == 3
    assert payload["sections_added"] == ["body"]
    assert payload["sections_kept"] == []
    assert payload["sections_changed"] == []
    assert payload["sections_removed"] == []
    assert payload["summary_pending"] == []


def test_import_writes_the_document_structure_to_the_graph(tmp_path):
    _host()

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    payload = json.loads(result.stdout)["graph"]
    assert payload["written"] is True
    assert payload["status"] == "applied"
    assert payload["entity_key"] == f"document:{DOC}"
    # Document upsert + one SECTION_OF claim.
    assert payload["operations_applied"] == 2


def test_import_says_so_when_no_graph_is_wired(tmp_path):
    """Bytes without structure is a silent failure otherwise: ``get`` keeps
    working while search returns nothing, and only the warning says why."""
    _host(with_graph=False)

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    payload = json.loads(result.stdout)
    assert payload["graph"]["written"] is False
    assert payload["graph"]["status"] == "skipped"
    assert any("search cannot find it" in w for w in payload["warnings"])


def test_import_reports_written_false_when_claims_do_not_read_back(tmp_path):
    """``graph.written`` must come from a read, not from the writer's opinion.

    The writer counts submitted operations, so a mutation that applies onto an
    existing tombstone returns ``status: applied`` while the claim stays
    invisible to every read. That combination — bytes on disk, ``written:
    true``, nothing findable — is the exact failure the graph block exists to
    surface, so it is asserted against a claim query that reports the claim
    missing rather than against a hypothetical.
    """
    host = _host()
    real_find = host.runtime.backend.claim_query.find_claims

    def _find_claims(filter_):
        # Only the readback probe filters by claim_key; everything else is
        # left alone so the import itself behaves normally.
        if filter_.claim_key_in:
            return []
        return real_find(filter_)

    host.resources.claims = MagicMock()
    host.resources.claims.find_claims.side_effect = _find_claims

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["graph"]["status"] == "applied"
    assert payload["graph"]["written"] is False
    assert payload["graph"]["missing_claim_keys"]
    assert any("cannot be read back" in w for w in payload["warnings"])


def test_import_carries_chunk_ids_as_claim_evidence(tmp_path):
    """R13: a section's search hit already holds the ids ``get`` takes."""
    host = _host()

    _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    rows = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SECTION_OF",))
    )
    assert len(rows) == 1
    assert set(rows[0].source_refs) == {
        format_resource_id(DOC, "body", seq) for seq in (0, 1, 2)
    }


def test_import_never_puts_chunk_text_in_the_graph(tmp_path):
    """R1, asserted where it can actually be violated."""
    host = _host()
    directory = _import_dir(
        tmp_path / "in",
        [_section("body", chunks=[{"label": "opening", "text": "SECRET-PAYLOAD"}])],
    )

    _run(["import", str(directory), "--doc", DOC])

    rows = host.runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id="p"))
    assert rows
    assert "SECRET-PAYLOAD" not in json.dumps(
        [
            {
                "fact": row.fact,
                "description": row.description,
                "properties": dict(row.properties or {}),
            }
            for row in rows
        ]
    )


def test_reimport_retracts_a_section_that_disappeared(tmp_path):
    host = _host()
    first = _import_dir(
        tmp_path / "v1",
        [
            _section("body", chunks=[{"label": "opening", "text": "alpha"}]),
            _section("appendix", chunks=[{"label": "notes", "text": "beta"}]),
        ],
    )
    _run(["import", str(first), "--doc", DOC])
    second = _import_dir(
        tmp_path / "v2",
        [_section("body", chunks=[{"label": "opening", "text": "alpha"}])],
    )

    result = _run(["import", str(second), "--doc", DOC])

    payload = json.loads(result.stdout)
    assert payload["revision"] == 2
    assert payload["sections_removed"] == ["appendix"]
    assert payload["graph"]["written"] is True
    live = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SECTION_OF",))
    )
    assert [row.subject_key for row in live] == [f"docsection:{DOC}:body"]


def test_import_points_at_the_missing_scope_edge(tmp_path):
    """Quality factor 4: a document with no DOCUMENTS edge is findable by
    semantic luck alone, and import is the only step that knows it landed."""
    _host()

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    action = json.loads(result.stdout)["recommended_next_action"]
    assert "DOCUMENTS" in action and f"document:{DOC}" in action


def _link(host, subject_key: str, *, object_key: str = "service:payments-api") -> None:
    """Write the scope edge an agent would write, through the normal door."""
    workbench = host.runtime.workbench
    proposal = workbench.propose(
        {
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "knowledge",
                    "subject": {"key": subject_key},
                    "predicate": "DOCUMENTS",
                    "object": {"key": object_key, "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": format_resource_id(DOC, "body", 0)}],
                    "description": f"{subject_key} covers {object_key}",
                }
            ]
        },
        pot_id="p",
    )
    assert proposal.ok is True, proposal.detail
    assert workbench.commit(proposal.plan_id, pot_id="p", approved_by="test").ok


def test_import_stops_nudging_once_the_document_is_linked(tmp_path):
    """P1-4: the scope nudge has to be a check, not a constant.

    It fired on every import regardless of how many ``DOCUMENTS`` claims the
    document already had, which made it useless for the one thing
    ``resources.md`` asks it for — telling a linked document from an unlinked
    one.
    """
    host = _seed(tmp_path)
    _link(host, f"document:{DOC}")

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    action = json.loads(result.stdout)["recommended_next_action"]
    assert "DOCUMENTS" not in action
    assert "--include docs" in action


def test_import_counts_a_section_level_link_as_scope(tmp_path):
    """The skills teach linking one section rather than the whole document, so
    a section-only link is a linked document."""
    host = _seed(tmp_path)
    _link(host, f"docsection:{DOC}:body")

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    assert "DOCUMENTS" not in json.loads(result.stdout)["recommended_next_action"]


def test_import_nudges_again_when_the_last_scope_edge_is_retracted(tmp_path):
    """A re-import that drops the linked section takes the scope edge with it
    (P0-2), which puts the document back in the state the nudge is for."""
    host = _host()
    kept = _section("kept", chunks=[{"label": "a", "text": "alpha"}])
    dropped = _section("dropped", chunks=[{"label": "b", "text": "beta"}])
    assert (
        _run(
            [
                "import",
                str(_import_dir(tmp_path / "in1", [kept, dropped])),
                "--doc",
                DOC,
            ]
        ).exit_code
        == 0
    )
    _link(host, f"docsection:{DOC}:dropped")

    result = _run(["import", str(_import_dir(tmp_path / "in2", [kept])), "--doc", DOC])

    action = json.loads(result.stdout)["recommended_next_action"]
    assert "DOCUMENTS" in action and f"document:{DOC}" in action


def test_import_nudges_for_scope_when_no_claim_query_is_wired(tmp_path):
    """Unknown is not 'linked': with nothing to count, the nudge is the honest
    default — a document nothing can find is the costlier silence."""
    host = _host()
    host.resources.claims = None

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    assert "DOCUMENTS" in json.loads(result.stdout)["recommended_next_action"]


def test_import_scope_nudge_points_at_the_commands_the_catalog_lists(tmp_path):
    """It used to point at ``graph mutate``, which self-describes as a legacy
    transition wrapper and is not in the catalog's command list."""
    _host()

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", DOC])

    action = json.loads(result.stdout)["recommended_next_action"]
    assert "graph propose" in action and "graph commit" in action
    assert "graph mutate" not in action


def test_import_flags_sections_that_still_need_a_summary(tmp_path):
    _host()
    directory = _import_dir(
        tmp_path / "in",
        [_section("body", chunks=[{"label": "opening", "text": "alpha"}], summary="")],
    )

    result = _run(["import", str(directory), "--doc", DOC])

    payload = json.loads(result.stdout)
    assert payload["summary_pending"] == ["body"]
    assert "body" in payload["recommended_next_action"]


def test_import_resolves_a_relative_directory_before_the_daemon_hop(
    tmp_path, monkeypatch
):
    host = _host()
    _import_dir(tmp_path / "in")
    monkeypatch.chdir(tmp_path)

    result = _run(["import", "in", "--doc", DOC])

    assert result.exit_code == 0, result.stdout
    # The daemon has its own working directory; a relative path must not reach it.
    assert host.store.import_dirs[0].is_absolute()


def test_reimport_separates_kept_changed_added_and_removed(tmp_path):
    _host()
    _run(["import", str(_import_dir(tmp_path / "v1")), "--doc", DOC])
    second = _import_dir(
        tmp_path / "v2",
        [
            _section("body", chunks=[{"label": "opening", "text": "alpha"}]),
            _section(
                "risks",
                chunks=[{"label": "top risk", "text": "delta"}],
                ordinal=1,
            ),
        ],
    )
    # 'body' keeps its content_hash from v1 but 'risks' is new.
    result = _run(["import", str(second), "--doc", DOC])

    payload = json.loads(result.stdout)
    assert payload["revision"] == 2
    assert payload["sections_kept"] == ["body"]
    assert payload["sections_added"] == ["risks"]
    assert payload["sections_changed"] == []
    assert payload["sections_removed"] == []


def test_reimport_reports_a_changed_section_as_changed(tmp_path):
    _host()
    _run(["import", str(_import_dir(tmp_path / "v1")), "--doc", DOC])
    second = _import_dir(
        tmp_path / "v2",
        [
            _section(
                "body",
                chunks=[{"label": "opening", "text": "rewritten"}],
                content_hash="body-v2",
            )
        ],
    )

    payload = json.loads(_run(["import", str(second), "--doc", DOC]).stdout)

    assert payload["sections_changed"] == ["body"]
    assert payload["sections_kept"] == []


def test_oversized_chunk_is_refused_with_the_stores_own_code(tmp_path):
    _host()
    _run(["import", str(_import_dir(tmp_path / "v1")), "--doc", DOC])
    oversized = _import_dir(
        tmp_path / "big",
        [
            _section(
                "body",
                chunks=[
                    {"label": "huge", "text": "x" * (RESOURCE_CHUNK_MAX_CHARS + 1)}
                ],
            )
        ],
    )

    result = _run(["import", str(oversized), "--doc", DOC])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["code"] == "resource_chunk_too_large"
    assert payload["recommended_next_action"]
    # The prior revision is untouched.
    listed = json.loads(_run(["list", "--doc", DOC]).stdout)
    assert listed["chunk_count"] == 3


def test_missing_import_directory_is_a_manifest_error(tmp_path):
    _host()

    result = _run(["import", str(tmp_path / "nope"), "--doc", DOC])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "resource_manifest_invalid"


def test_invalid_document_slug_is_refused(tmp_path):
    _host()

    result = _run(["import", str(_import_dir(tmp_path / "in")), "--doc", "Q3 Review"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "resource_slug_invalid"


# --- get --------------------------------------------------------------------


def test_get_returns_the_documented_chunk_shape(tmp_path):
    _seed(tmp_path)

    result = _run(["get", format_resource_id(DOC, "body", 0)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    chunk = payload["chunks"][0]
    assert chunk == {
        "resource_id": format_resource_id(DOC, "body", 0),
        "doc": DOC,
        "section": "body",
        "seq": 0,
        "text": "alpha",
        "chars": 5,
        "revision": 1,
        "source_ref": "file:///q3.pdf",
        "page": None,
        "offset": None,
        "requested": True,
    }


def test_get_answers_several_ids_in_one_round_trip(tmp_path):
    host = _seed(tmp_path)
    ids = [format_resource_id(DOC, "body", seq) for seq in (2, 0)]

    payload = json.loads(_run(["get", *ids]).stdout)

    assert [row["text"] for row in payload["chunks"]] == ["gamma", "alpha"]
    assert host.store.calls == ["get_many"]


def test_get_with_neighbors_stays_one_round_trip(tmp_path):
    host = _seed(tmp_path)

    payload = json.loads(
        _run(["get", format_resource_id(DOC, "body", 1), "--with-neighbors"]).stdout
    )

    assert [row["seq"] for row in payload["chunks"]] == [0, 1, 2]
    assert [row["requested"] for row in payload["chunks"]] == [False, True, False]
    # One listing to learn the section's chunks, one batched read — never a
    # read per chunk.
    assert host.store.calls == ["list", "get_many"]


def test_get_with_neighbors_stops_at_the_section_boundary(tmp_path):
    _seed(
        tmp_path,
        [
            _section("body", chunks=[{"label": "only", "text": "alpha"}]),
            _section(
                "risks", chunks=[{"label": "top risk", "text": "delta"}], ordinal=1
            ),
        ],
    )

    payload = json.loads(
        _run(["get", format_resource_id(DOC, "body", 0), "--with-neighbors"]).stdout
    )

    assert [row["resource_id"] for row in payload["chunks"]] == [
        format_resource_id(DOC, "body", 0)
    ]


def test_get_deduplicates_overlapping_neighborhoods(tmp_path):
    _seed(tmp_path)
    ids = [format_resource_id(DOC, "body", seq) for seq in (0, 1)]

    payload = json.loads(_run(["get", *ids, "--with-neighbors"]).stdout)

    assert [row["seq"] for row in payload["chunks"]] == [0, 1, 2]


def test_get_of_an_unknown_chunk_reports_not_found(tmp_path):
    _seed(tmp_path)

    result = _run(["get", format_resource_id(DOC, "body", 9)])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "resource_not_found"


def test_get_with_neighbors_of_an_unknown_chunk_still_reports_not_found(tmp_path):
    _seed(tmp_path)

    result = _run(["get", format_resource_id(DOC, "body", 9), "--with-neighbors"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["code"] == "resource_not_found"
    # The miss is the store's, not a neighborhood invented from a listing.
    assert format_resource_id(DOC, "body", 9) in payload["message"]


def test_get_of_a_malformed_id_reports_the_id_error(tmp_path):
    _seed(tmp_path)

    result = _run(["get", "res/q3-review/body/0"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "resource_id_invalid"


def test_human_get_prints_chunk_text_verbatim(tmp_path):
    _seed(
        tmp_path,
        [
            _section(
                "body",
                chunks=[{"label": "opening", "text": "first para\n\nsecond para"}],
            )
        ],
    )
    _common.set_json(False)

    result = _run(["get", format_resource_id(DOC, "body", 0)], as_json=False)

    assert result.exit_code == 0, result.stdout
    # Blank lines and all: this command returns evidence, it does not format it.
    assert "first para\n\nsecond para" in result.stdout


# --- list -------------------------------------------------------------------


def test_list_returns_chunk_ids_and_labels(tmp_path):
    _seed(tmp_path)

    payload = json.loads(_run(["list", "--doc", DOC]).stdout)

    assert payload["section_count"] == 1
    assert payload["chunk_count"] == 3
    section = payload["sections"][0]
    assert section["slug"] == "body"
    assert [chunk["label"] for chunk in section["chunks"]] == [
        "opening",
        "middle",
        "closing",
    ]
    assert section["chunks"][0]["resource_id"] == format_resource_id(DOC, "body", 0)


def test_list_can_narrow_to_one_section(tmp_path):
    _seed(
        tmp_path,
        [
            _section("body", chunks=[{"label": "opening", "text": "alpha"}]),
            _section(
                "risks", chunks=[{"label": "top risk", "text": "delta"}], ordinal=1
            ),
        ],
    )

    payload = json.loads(_run(["list", "--doc", DOC, "--section", "risks"]).stdout)

    assert [row["slug"] for row in payload["sections"]] == ["risks"]


def test_list_of_an_unknown_document_reports_not_found(tmp_path):
    _seed(tmp_path)

    result = _run(["list", "--doc", "absent"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "resource_not_found"


# --- rm ---------------------------------------------------------------------


def test_rm_without_confirm_removes_nothing(tmp_path):
    host = _seed(tmp_path)

    result = _run(["rm", DOC])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["code"] == "confirmation_required"
    assert "delete" not in host.store.calls


def test_rm_with_confirm_removes_the_document(tmp_path):
    _seed(tmp_path)

    result = _run(["rm", DOC, "--confirm"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["removed"] is True
    assert payload["graph_retracted"] is True
    assert json.loads(_run(["list", "--doc", DOC]).stdout)["code"] == (
        "resource_not_found"
    )


def test_rm_retracts_section_claims_in_the_graph(tmp_path):
    """P5: ``rm`` must not leave SECTION_OF claims pointing at deleted chunks."""
    host = _seed(tmp_path)
    assert host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SECTION_OF",))
    )

    result = _run(["rm", DOC, "--confirm"])

    assert result.exit_code == 0, result.stdout
    live = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("SECTION_OF",))
    )
    assert live == []


def test_rm_of_an_absent_document_is_a_no_op(tmp_path):
    _seed(tmp_path)

    payload = json.loads(_run(["rm", "absent", "--confirm"]).stdout)

    assert payload["removed"] is False
    assert payload["graph_retracted"] is False


# --- dependent claims (P0-2) ------------------------------------------------


def test_rm_retracts_agent_written_claims_about_the_sections(tmp_path):
    """A deleted document must not keep answering through somebody else's claim.

    ``rm`` retracts ``SECTION_OF``, which is the structure this store wrote. But
    the skills tell agents to link individual sections to what they cover, and
    those ``DOCUMENTS`` claims used to survive the delete — still ranking in
    ``search --include docs``, still handing out chunk ids that now return
    ``resource_not_found``.
    """
    host = _seed(tmp_path)
    workbench = host.runtime.workbench
    proposal = workbench.propose(
        {
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "knowledge",
                    "subject": {"key": f"docsection:{DOC}:body"},
                    "predicate": "DOCUMENTS",
                    "object": {"key": "service:payments-api", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": format_resource_id(DOC, "body", 0)}],
                    "description": "the body section covers the payments api",
                }
            ]
        },
        pot_id="p",
    )
    assert proposal.ok is True, proposal.detail
    assert workbench.commit(proposal.plan_id, pot_id="p", approved_by="test").ok

    def live_documents():
        return host.runtime.backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id="p", predicate_in=("DOCUMENTS",))
        )

    assert len(live_documents()) == 1

    assert _run(["rm", DOC, "--confirm"]).exit_code == 0

    assert live_documents() == [], "a DOCUMENTS claim outlived the chunks it cites"


def test_reimport_retracts_claims_about_a_section_it_drops(tmp_path):
    """Same defect on the normal path: a re-import that drops one section.

    This is how it is actually reached — nobody has to delete a document for a
    section to disappear; re-importing a changed source is enough.
    """
    host = _host()
    kept = _section("kept", chunks=[{"label": "a", "text": "alpha"}])
    dropped = _section("dropped", chunks=[{"label": "b", "text": "beta"}])
    first = _import_dir(tmp_path / "in1", [kept, dropped])
    assert _run(["import", str(first), "--doc", DOC]).exit_code == 0

    workbench = host.runtime.workbench
    proposal = workbench.propose(
        {
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "knowledge",
                    "subject": {"key": f"docsection:{DOC}:dropped"},
                    "predicate": "DOCUMENTS",
                    "object": {"key": "service:payments-api", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": format_resource_id(DOC, "dropped", 0)}],
                    "description": "the dropped section covers the payments api",
                }
            ]
        },
        pot_id="p",
    )
    assert proposal.ok is True, proposal.detail
    assert workbench.commit(proposal.plan_id, pot_id="p", approved_by="test").ok

    second = _import_dir(tmp_path / "in2", [kept])
    assert _run(["import", str(second), "--doc", DOC]).exit_code == 0

    live = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("DOCUMENTS",))
    )
    assert live == [], "a claim about a dropped section stayed live"


def test_claims_about_sections_that_stay_are_left_alone(tmp_path):
    """The cleanup must be surgical: only the departing section's claims go."""
    host = _host()
    kept = _section("kept", chunks=[{"label": "a", "text": "alpha"}])
    dropped = _section("dropped", chunks=[{"label": "b", "text": "beta"}])
    first = _import_dir(tmp_path / "in1", [kept, dropped])
    assert _run(["import", str(first), "--doc", DOC]).exit_code == 0

    workbench = host.runtime.workbench
    for slug in ("kept", "dropped"):
        proposal = workbench.propose(
            {
                "operations": [
                    {
                        "op": "assert_claim",
                        "subgraph": "knowledge",
                        "subject": {"key": f"docsection:{DOC}:{slug}"},
                        "predicate": "DOCUMENTS",
                        "object": {"key": "service:payments-api", "type": "Service"},
                        "truth": "source_observation",
                        "evidence": [{"source_ref": format_resource_id(DOC, slug, 0)}],
                        "description": f"the {slug} section covers the payments api",
                    }
                ]
            },
            pot_id="p",
        )
        assert workbench.commit(proposal.plan_id, pot_id="p", approved_by="test").ok

    second = _import_dir(tmp_path / "in2", [kept])
    assert _run(["import", str(second), "--doc", DOC]).exit_code == 0

    live = host.runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("DOCUMENTS",))
    )
    assert [row.subject_key for row in live] == [f"docsection:{DOC}:kept"]


# --- transport --------------------------------------------------------------


def test_resources_is_an_allowed_rpc_surface():
    assert "resources" in _ALLOWED_RPC_SURFACES
    assert isinstance(RemoteHostShell(rpc=MagicMock()).resources, RemoteSurface)


@pytest.mark.parametrize(
    "value",
    [
        DocumentManifest(
            pot_id="p",
            doc=DOC,
            revision=2,
            source_ref="file:///q3.pdf",
            source_kind="pdf",
            sections=(
                SectionManifest(
                    slug="body",
                    title="Body",
                    summary="what it covers",
                    ordinal=0,
                    content_hash="body-v1",
                    chunks=(ChunkRef(seq=0, label="opening", page=1),),
                ),
            ),
            sections_added=("body",),
            warnings=("something advisory",),
        ),
        Chunk(
            resource_id=format_resource_id(DOC, "body", 0),
            doc=DOC,
            section="body",
            seq=0,
            text="alpha",
            chars=5,
            revision=2,
            source_ref="file:///q3.pdf",
            page=1,
        ),
        ResourceStoreStatus(kind="local", ready=True, location="/x", documents=3),
    ],
    ids=["manifest", "chunk", "status"],
)
def test_resource_dtos_survive_the_daemon_encoding(value):
    # The daemon rebuilds DTOs by module path with cls(**fields); a field these
    # cannot round-trip breaks every resource command in daemon mode only.
    restored = decode(encode(value))

    assert restored == value
    assert type(restored) is type(value)


def test_a_store_error_keeps_its_code_across_the_daemon_hop():
    payload = _error_payload(
        ResourceStoreError(
            "resource_chunk_too_large",
            "chunk 0 is 9000 chars",
            detail="/tmp/out/body/0000.txt",
            recommended_next_action="Split the chunk on a paragraph boundary.",
        )
    )

    with pytest.raises(ValueError) as raised:
        _raise_remote_error(payload)

    assert getattr(raised.value, "code") == "resource_chunk_too_large"
    assert getattr(raised.value, "detail") == "/tmp/out/body/0000.txt"
    assert "paragraph boundary" in getattr(raised.value, "recommended_next_action")


def test_a_plain_validation_error_still_arrives_without_a_code():
    payload = _error_payload(ValueError("nope"))

    with pytest.raises(ValueError) as raised:
        _raise_remote_error(payload)

    assert getattr(raised.value, "code", None) is None
