"""Outbound adapters for the CLI auth subsystem.

Persistence (file credential store), HTTP transport, and the
provider auth/flow clients (GitHub, Firebase, Potpie, Linear, Atlassian) that
talk to external systems. Product CLI command surfaces drive these via their ports
(:class:`~domain.ports.cli_auth.credentials.CredentialStore`,
:class:`~adapters.outbound.cli_auth.http.HttpClient`). Nothing here imports from
inbound CLI code.
"""
