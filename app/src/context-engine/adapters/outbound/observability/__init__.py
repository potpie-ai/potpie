"""Observability adapters (concrete :class:`ObservabilityPort` impls).

The port + NoOp default live in ``domain.ports.observability``. These
adapters are imported lazily from ``bootstrap.container`` so the package
stays installable without the ``observability`` extra.
"""
