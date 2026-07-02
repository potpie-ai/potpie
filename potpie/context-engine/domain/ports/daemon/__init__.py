"""Daemon shell ports: transport-agnostic contracts for the local daemon host.

These are the hexagonal ports the product daemon runtime and its adapters bind
to: the operation contract (``operations``), the three runtime ports plus health
(``shell``), and the managed-service value objects (``service``).
"""
