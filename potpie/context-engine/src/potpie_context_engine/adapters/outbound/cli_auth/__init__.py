"""Outbound adapters for the CLI auth subsystem.

Persistence (file credential store), HTTP transport, and the
provider auth/flow clients (GitHub, Firebase, Potpie, Linear, Atlassian) that
talk to external systems. The inbound CLI command surfaces under
``adapters/inbound/cli`` drive these via their ports
(:class:`~potpie_context_engine.domain.ports.cli_auth.credentials.CredentialStore`,
:class:`~potpie_context_engine.adapters.outbound.cli_auth.http.HttpClient`). Nothing here imports from
``potpie_context_engine.adapters.inbound``.
"""
