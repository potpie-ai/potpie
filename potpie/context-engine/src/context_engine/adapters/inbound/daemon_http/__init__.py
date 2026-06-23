"""Daemon inbound HTTP transport.

Serves the ``OperationRegistry`` over a Unix domain socket (or TCP) with generic
``/op/{name}`` dispatch. Knows nothing about specific operations.
"""
