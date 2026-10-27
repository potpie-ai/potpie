"""Process-kill gates. Server tests require an explicitly named private container."""

import multiprocessing
import os
import signal
import subprocess
import time
import uuid

import pytest
from potpie_context_engine.core.graph_journal import JournalError
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationBatch


def _backend(profile, location):
    if profile == "embedded":
        from pathlib import Path

        from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
            EmbeddedGraphBackend,
        )

        return EmbeddedGraphBackend(home=Path(location))
    from falkordb import FalkorDB

    from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
        FalkorDBGraphBackend,
    )

    class Settings:
        def is_enabled(self):
            return True

    port, name = location
    graph = FalkorDB(host="127.0.0.1", port=port).select_graph(name)
    return FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)


def _write(backend, mutation_id, summary):
    return backend.mutation.compare_and_apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:a", ("Entity", "Service"), {"summary": summary})
            ]
        ),
        expected_pot_id="p",
        expected_version=backend.mutation.current_version("p"),
        provenance_context=ProvenanceContext(mutation_id=mutation_id),
    )


def _kill_writer(profile, location, phase):
    backend = _backend(profile, location)
    if phase == "resource":
        backend.journal.begin_resource(
            pot_id="p", operation_id="crashed-resource", owner="old-worker"
        )
        os.kill(os.getpid(), signal.SIGKILL)
    if profile == "embedded":
        from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
            EmbeddedGraphBackend,
        )

        original = EmbeddedGraphBackend._write_state

        def kill(self, *args, **kwargs):
            if phase == "after":
                original(self, *args, **kwargs)
            os.kill(os.getpid(), signal.SIGKILL)

        EmbeddedGraphBackend._write_state = kill
    else:
        from potpie_context_engine.adapters.outbound.graph.falkordb_journal import (
            FalkorJournal,
        )

        original = FalkorJournal._execute

        def kill(self, *args, **kwargs):
            if phase == "after":
                original(self, *args, **kwargs)
            os.kill(os.getpid(), signal.SIGKILL)

        FalkorJournal._execute = kill
    _write(backend, "crash", "changed")


@pytest.mark.parametrize("profile", ("embedded", "falkordb"))
@pytest.mark.parametrize("phase", ("before", "after", "resource"))
def test_killed_writer_and_native_restart_preserve_atomic_receipt(
    profile, phase, tmp_path
):
    container = None
    if profile == "embedded":
        location = str(tmp_path)
    else:
        container = os.environ.get("POTPIE_JOURNAL_CRASH_CONTAINER")
        port = os.environ.get("POTPIE_JOURNAL_CRASH_PORT")
        if not container or not port:
            pytest.skip("provide a disposable AOF-enabled FalkorDB container and port")
        if not container.startswith("pie-journal-crash-"):
            pytest.fail("crash tests require an explicitly disposable container name")
        location = (int(port), uuid.uuid4().hex)
    backend = _backend(profile, location)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    _write(backend, "base", "before")
    child = multiprocessing.get_context("spawn").Process(
        target=_kill_writer, args=(profile, location, phase)
    )
    child.start()
    child.join(timeout=20)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("writer did not reach publication boundary")
    assert child.exitcode == -signal.SIGKILL
    if container:
        subprocess.run(
            ["docker", "kill", "--signal=KILL", container],
            check=True,
            capture_output=True,
        )
        subprocess.run(["docker", "start", container], check=True, capture_output=True)
        import redis

        deadline = time.monotonic() + 20
        client = redis.Redis(host="127.0.0.1", port=location[0])
        while True:
            try:
                client.ping()
                break
            except redis.exceptions.ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        assert client.config_get("appendonly")["appendonly"] == "yes"
        assert client.config_get("appendfsync")["appendfsync"] == "always"
    restarted = _backend(profile, location)
    expected = "changed" if phase == "after" else "before"
    assert (
        restarted.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == expected
    )
    receipt = restarted.journal.get_receipt(pot_id="p", commit_id="crash")
    assert (receipt is not None) == (phase == "after")
    assert restarted.journal.journal_state("p").sequence == (
        2 if phase == "after" else 1
    )
    if phase == "resource":
        assert (
            restarted.journal.journal_state("p").resource_guard.operation_id
            == "crashed-resource"
        )
        with pytest.raises(JournalError, match="guard"):
            _write(restarted, "unsafe", "changed")
        restarted.journal.recover_resource_guard(
            pot_id="p",
            operation_id="crashed-resource",
            expected_owner="old-worker",
            new_owner="recovery-worker",
            verify_worker_stopped=lambda _: child.exitcode == -signal.SIGKILL,
        )
        receipt = restarted.journal.complete_resource(
            pot_id="p", operation_id="crashed-resource", owner="recovery-worker"
        )
        assert not receipt.rollback_supported
        assert restarted.journal.journal_state("p").resource_guard is None
    else:
        assert _write(restarted, "crash", "changed").ok
    assert restarted.journal.journal_state("p").sequence == 2
    if profile == "falkordb":
        restarted.journal.graph.delete()
