from dataclasses import is_dataclass
from domain.ports.daemon.shell import (
    Transport, Component, ServiceBackend,
    ServiceSpec, ReadyProbe, RestartPolicy, HealthStatus,
)


def test_protocols_are_runtime_checkable():
    class ConcreteTransport:
        def bind(self, ctx): ...
        async def serve(self, ops): ...
        async def stop(self): ...
        def health(self): ...

    class ConcreteComponent:
        name = "x"
        async def on_start(self, ctx): ...
        async def on_stop(self): ...
        def health(self): ...
        def operations(self): return []

    class ConcreteBackend:
        async def start(self, spec, ctx): ...
        async def stop(self, spec): ...
        async def probe(self, spec): ...

    assert isinstance(ConcreteTransport(), Transport)
    assert isinstance(ConcreteComponent(), Component)
    assert isinstance(ConcreteBackend(), ServiceBackend)

    class NotATransport:
        pass
    assert not isinstance(NotATransport(), Transport)


def test_service_spec_is_dataclass():
    spec = ServiceSpec(
        name="g", backend="subprocess",
        config={"command": ["echo", "hi"]},
        ready=ReadyProbe(kind="tcp", target="127.0.0.1:1"),
        endpoint="tcp://127.0.0.1:1",
    )
    assert is_dataclass(ServiceSpec)
    assert spec.restart is RestartPolicy.ON_FAILURE
    assert spec.depends_on == []


def test_health_status_values():
    assert HealthStatus.STARTING.value == "starting"
    assert HealthStatus.READY.value == "ready"
    assert HealthStatus.DEGRADED.value == "degraded"
    assert HealthStatus.STOPPED.value == "stopped"
