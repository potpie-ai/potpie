from host.daemon_runtime.health import HealthRegistrar
from domain.ports.daemon.shell import HealthStatus


def test_register_and_get():
    h = HealthRegistrar()
    h.set("transport:http", HealthStatus.STARTING)
    h.set("transport:http", HealthStatus.READY)
    assert h.get("transport:http") is HealthStatus.READY


def test_aggregate_all_ready():
    h = HealthRegistrar()
    h.set("a", HealthStatus.READY)
    h.set("b", HealthStatus.READY)
    assert h.aggregate() is HealthStatus.READY


def test_aggregate_degraded_dominates():
    h = HealthRegistrar()
    h.set("a", HealthStatus.READY)
    h.set("b", HealthStatus.DEGRADED)
    assert h.aggregate() is HealthStatus.DEGRADED


def test_aggregate_starting_when_anything_starting():
    h = HealthRegistrar()
    h.set("a", HealthStatus.READY)
    h.set("b", HealthStatus.STARTING)
    assert h.aggregate() is HealthStatus.STARTING


def test_snapshot_returns_copy():
    h = HealthRegistrar()
    h.set("a", HealthStatus.READY)
    snap = h.snapshot()
    snap["a"] = HealthStatus.DEGRADED
    assert h.get("a") is HealthStatus.READY


def test_aggregate_empty_is_starting():
    h = HealthRegistrar()
    assert h.aggregate() is HealthStatus.STARTING


def test_aggregate_stopped_when_only_stopped():
    h = HealthRegistrar()
    h.set("a", HealthStatus.STOPPED)
    assert h.aggregate() is HealthStatus.STOPPED
