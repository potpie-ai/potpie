"""Observability adapters (concrete :class:`ObservabilityPort` impls).

The port + NoOp default live in ``potpie_context_engine.domain.ports.observability``. These
adapters are imported lazily from ``potpie_context_engine.bootstrap.ingestion_server`` so the package
stays installable without the ``observability`` extra.
"""
