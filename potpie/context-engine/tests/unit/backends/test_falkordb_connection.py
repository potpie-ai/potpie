from types import SimpleNamespace

from potpie_context_engine.adapters.outbound.graph.falkordb_connection import (
    transaction_client,
)


def test_server_driver_uses_its_redis_connection():
    redis = object()
    assert (
        transaction_client(SimpleNamespace(client=SimpleNamespace(connection=redis)))
        is redis
    )


def test_lite_redis_client_accepts_a_null_connection_attribute():
    redis = SimpleNamespace(connection=None)
    assert transaction_client(SimpleNamespace(client=redis)) is redis


def test_direct_redis_client_has_no_connection_attribute():
    redis = object()
    assert transaction_client(SimpleNamespace(client=redis)) is redis
