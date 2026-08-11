"""ResourceIndexPort capability conformance.

Every index profile satisfies the same contract, and the suite gates each
assertion on the capability the profile *declares* — so ``none`` passes by
honestly answering nothing and ``sqlite_hybrid`` passes by actually retrieving.
New profiles drop into ``RUNNABLE_PROFILES``.

The suite deliberately runs ``sqlite_hybrid`` twice: once as configured, and
once with the embedder switched off. The second run is the labeled-degradation
case, which is the one that regresses silently — a hybrid profile that quietly
becomes lexical looks identical from the outside until recall is measured.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.resources.index import (
    KNOWN_PROFILES,
    build_resource_index,
    default_resource_index_profile,
)
from potpie_context_core.ports.resource_index import (
    MATCH_MODE_DISABLED,
    ResourceIndexError,
)
from potpie_context_engine.testing.conformance import (
    build_conformance_document,
    run_resource_index_conformance,
)

RUNNABLE_PROFILES = ["sqlite_hybrid", "sqlite_fts", "none"]


@pytest.mark.parametrize("profile", RUNNABLE_PROFILES)
def test_resource_index_conformance(profile, tmp_path):
    run_resource_index_conformance(
        lambda: build_resource_index(profile, home=tmp_path / profile)
    )


def test_hybrid_without_an_embedder_degrades_in_a_labeled_way(tmp_path):
    """No embedder must mean "lexical, and here is why" — never silent hybrid."""
    # Constructed directly: ``build_resource_index`` fills in the bundled
    # embedder when none is passed, which is the right default and the wrong
    # thing for this test.
    from potpie_context_engine.adapters.outbound.resources.index.sqlite_hybrid import (
        SqliteHybridResourceIndex,
    )

    index = SqliteHybridResourceIndex(home=tmp_path, embedder=None)
    caps = index.capabilities()
    assert caps.lexical is True
    assert caps.semantic is False and caps.hybrid is False
    assert caps.match_mode() == "lexical"

    manifest, chunks = build_conformance_document()
    report = index.index_document(pot_id="p", manifest=manifest, chunks=chunks)
    # Nothing is pending, because nothing will ever embed it. Reporting a
    # backlog here would send a caller to wait on a drain that cannot run.
    assert report.pending_embeddings == 0

    found = index.search(pot_id="p", query="ERR_QUOTA_EXCEEDED", limit=5)
    assert found.hits and found.match_mode == "lexical"
    assert all(hit.similarity is None for hit in found.hits)

    drained = index.drain()
    assert drained.embedded == 0 and drained.detail

    status = index.status(pot_id="p")
    assert status.ready is True
    assert "embedder" in (status.detail or "")


def test_none_profile_never_raises(tmp_path):
    """The disabled profile answers every method rather than failing a read."""
    index = build_resource_index("none", home=tmp_path)
    manifest, chunks = build_conformance_document()

    assert (
        index.index_document(pot_id="p", manifest=manifest, chunks=chunks).chunks == 0
    )
    result = index.search(pot_id="p", query="anything", limit=5)
    assert result.hits == () and result.match_mode == MATCH_MODE_DISABLED
    assert index.drop_document(pot_id="p", slug="q3-review") is False
    assert index.purge_pot("p") is False
    assert index.drain().embedded == 0
    assert index.status().ready is False


def test_unknown_profile_is_refused_not_defaulted():
    """A typo must fail loudly; silently working search would hide the setting."""
    with pytest.raises(ResourceIndexError) as excinfo:
        build_resource_index("sqlite_fts5")
    assert excinfo.value.code == "resource_index_profile_unknown"
    assert "sqlite_fts" in (excinfo.value.detail or "")


def test_profile_selection_prefers_env_then_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXT_ENGINE_RESOURCE_INDEX", "sqlite_fts")
    assert default_resource_index_profile() == "sqlite_fts"
    # A blank value falls through rather than selecting an empty profile —
    # the same rule ``default_backend_profile`` applies.
    monkeypatch.setenv("CONTEXT_ENGINE_RESOURCE_INDEX", "   ")
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    assert default_resource_index_profile() in KNOWN_PROFILES
