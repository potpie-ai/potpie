"""Daemon shell ports: transport-agnostic contracts for the local daemon host.

These are the hexagonal ports the in-process daemon runtime
(``potpie.context_engine.host.daemon_runtime``) and its adapters bind to: the
operation contract (``operations``), the three runtime ports plus health
(``shell``), and the managed-service value objects (``service``).
"""
