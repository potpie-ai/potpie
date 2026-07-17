"""``potpie_context_engine.adapters.inbound.daemon_http`` — the daemon's inbound HTTP transport, serving the
``OperationRegistry`` over a Unix domain socket (or TCP) with generic ``/op/{name}``
dispatch. Knows nothing about specific operations.
"""
