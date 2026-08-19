"""The base ``potpie`` distribution is a remote-only client.

Everything that runs a graph-native store on this machine — the FalkorDB
driver, the daemon's HTTP server, the local embedder, the ingestion clients —
lives behind an extra. These cover what the CLI does when the extras were not
taken, which is the state no other test file is ever in: the dev venv and every
CI job install ``--all-extras``, so a regression here is invisible everywhere
else until it reaches a user's first run.

The absences are simulated by hiding the module from ``importlib.util.find_spec``
rather than by uninstalling anything, because that is precisely the question the
production code asks.
"""

from __future__ import annotations

import importlib.util

import pytest

from potpie.cli import hosts
from potpie.daemon.process import launcher
from potpie_context_engine.bootstrap import host_wiring


@pytest.fixture()
def without_modules(monkeypatch):
    """Make named modules invisible to spec probes, leaving imports alone."""

    def _hide(*names: str) -> None:
        hidden = set(names)
        real = importlib.util.find_spec

        def _find_spec(name: str, package: str | None = None):
            if name in hidden:
                return None
            return real(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", _find_spec)

    return _hide


# --- the local backend degrades instead of crashing --------------------------


def test_default_backend_is_graph_native_when_the_driver_is_there(
    without_modules, monkeypatch
) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)

    assert host_wiring.default_backend_profile() == "falkordb_lite"


def test_default_backend_falls_back_to_embedded_without_the_driver(
    without_modules, monkeypatch
) -> None:
    """A profile that is certain to fail on first use is not a default.

    ``embedded`` is a real, JSON-persisted backend with no third-party
    dependencies, so a base install keeps a working local graph rather than a
    crash two layers down in a driver import.
    """
    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)
    without_modules("redislite")

    assert host_wiring.default_backend_profile() == "embedded"


def test_an_explicitly_named_backend_is_never_second_guessed(
    without_modules, monkeypatch
) -> None:
    """If an operator named a profile, a loud failure about it is the answer."""
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    without_modules("redislite")

    assert host_wiring.default_backend_profile() == "falkordb_lite"


# --- the local host runs in process when there is no daemon to run -----------


def test_local_host_goes_in_process_without_the_daemon_extra(
    without_modules, monkeypatch
) -> None:
    """Otherwise every local command tried to start a daemon that cannot import.

    A bare ``potpie use mine`` died with "Cannot reach the local host" plus
    advice to run ``potpie setup`` — a repair that cannot work, because the
    missing piece is a package.
    """
    monkeypatch.delenv("CONTEXT_ENGINE_HOST_MODE", raising=False)
    without_modules("uvicorn")
    built: list[str] = []
    monkeypatch.setattr(
        host_wiring, "build_host_shell", lambda *a, **k: built.append("in_process")
    )

    hosts._build_local_host()

    assert built == ["in_process"]


def test_local_host_uses_the_daemon_when_the_extra_is_installed(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_HOST_MODE", raising=False)

    host = hosts._build_local_host()

    assert type(host).__name__ == "RemoteHostShell"


def test_daemon_server_installed_needs_every_module() -> None:
    """Half the server is not a server; one missing module is enough."""
    assert hosts.daemon_server_installed() is True


def test_daemon_server_installed_is_false_when_one_is_missing(
    without_modules,
) -> None:
    without_modules("fastapi")

    assert hosts.daemon_server_installed() is False


# --- failures name a package -------------------------------------------------


def test_daemon_crash_message_names_the_extra() -> None:
    hint = launcher._missing_extra_hint(
        "ModuleNotFoundError: No module named 'uvicorn'"
    )

    assert "potpie[daemon]" in hint


def test_daemon_crash_message_stays_quiet_about_unrelated_imports() -> None:
    """A blanket handler would report a real bug as packaging advice."""
    hint = launcher._missing_extra_hint(
        "ModuleNotFoundError: No module named 'potpie.daemon.oops'"
    )

    assert hint == ""


def test_daemon_crash_message_stays_quiet_about_other_faults() -> None:
    assert (
        launcher._missing_extra_hint("OSError: [Errno 48] Address already in use") == ""
    )


def test_missing_local_driver_names_the_extra_and_the_managed_way_out() -> None:
    from potpie_context_core.errors import GraphSubstrateUnavailable
    from potpie_context_engine.adapters.outbound.graph import falkordb_writer

    with pytest.raises(GraphSubstrateUnavailable) as exc:
        with falkordb_writer._naming_the_missing_extra("redislite"):
            raise ModuleNotFoundError("No module named 'redislite'", name="redislite")

    assert "potpie[local]" in exc.value.recommended_next_action
    assert "managed" in exc.value.recommended_next_action


def test_an_import_error_from_inside_the_driver_is_not_repackaged() -> None:
    """Reporting a circular import as "install potpie[local]" hides a real bug."""
    from potpie_context_engine.adapters.outbound.graph import falkordb_writer

    with pytest.raises(ModuleNotFoundError) as exc:
        with falkordb_writer._naming_the_missing_extra("redislite"):
            raise ModuleNotFoundError(
                "No module named 'something_else'", name="something_else"
            )

    assert exc.value.name == "something_else"


# --- the diagnostic agrees with the read path --------------------------------


def test_backend_readiness_reports_the_missing_driver(without_modules) -> None:
    """Readiness used to be answered from what was *wired*.

    ``writer.enabled`` is set during construction and stays true whether or not
    a handle can ever be opened, so ``backend doctor`` said ``ready: true`` with
    every capability true and the very next read crashed.
    """
    from potpie_context_engine.adapters.outbound.graph.backends import (
        falkordb_backend,
    )

    without_modules("redislite")

    class _Settings:
        def falkordb_mode(self) -> str:
            return "lite"

    class _Writer:
        enabled = True

    mutation = falkordb_backend._FalkorDBMutation(
        settings=_Settings(), writer=_Writer()
    )
    readiness = mutation.readiness("pot")

    assert readiness.ready is False
    assert not any(readiness.capability_ready.values())
    assert "potpie[local]" in readiness.detail


def test_backend_readiness_is_ready_when_the_driver_is_there() -> None:
    from potpie_context_engine.adapters.outbound.graph.backends import (
        falkordb_backend,
    )

    class _Settings:
        def falkordb_mode(self) -> str:
            return "lite"

    class _Writer:
        enabled = True

    mutation = falkordb_backend._FalkorDBMutation(
        settings=_Settings(), writer=_Writer()
    )
    readiness = mutation.readiness("pot")

    assert readiness.ready is True
    assert readiness.capability_ready["mutation"] is True


# --- the distribution's own metadata -----------------------------------------


def test_base_dependencies_carry_no_local_backend_or_server() -> None:
    """The one assertion that fails the moment someone re-adds `[all]`."""
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(root.read_text())["project"]
    base = " ".join(project["dependencies"])

    assert "[all]" not in base
    for absent in ("fastapi", "uvicorn", "falkordb", "sentence-transformers", "pillow"):
        assert absent not in base, f"{absent} is a base dependency again"
    extras = project["optional-dependencies"]
    assert "potpie[daemon]" in extras["local"], "local must imply daemon"
