import logging
import pathlib
import pytest
from host.daemon_runtime.context import ShellContext, ServiceEndpoints


def test_service_endpoints_register_and_resolve():
    se = ServiceEndpoints()
    se.set("graph-db", "bolt://127.0.0.1:7687")
    assert se.get("graph-db") == "bolt://127.0.0.1:7687"
    assert se.resolve("service:graph-db") == "bolt://127.0.0.1:7687"
    # passthrough for non-service URIs
    assert se.resolve("sqlite:///x.db") == "sqlite:///x.db"


def test_service_endpoints_unknown_raises():
    se = ServiceEndpoints()
    with pytest.raises(KeyError) as ei:
        se.resolve("service:missing")
    assert "missing" in str(ei.value)


def test_service_endpoints_remove():
    se = ServiceEndpoints()
    se.set("a", "tcp://x")
    se.remove("a")
    with pytest.raises(KeyError):
        se.get("a")
    se.remove("a")  # idempotent — no error


def test_shell_context_construct(tmp_path: pathlib.Path):
    ctx = ShellContext(
        config={"k": 1},
        data_dir=tmp_path,
        logger=logging.getLogger("t"),
        endpoints=ServiceEndpoints(),
    )
    assert ctx.config == {"k": 1}
    assert ctx.data_dir == tmp_path
    assert ctx.shutdown.is_set() is False
