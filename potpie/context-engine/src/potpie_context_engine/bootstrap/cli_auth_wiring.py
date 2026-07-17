"""Composition root for the CLI auth credential seam.

The auth/credential surface deliberately does not route through ``HostShell`` (it
is interactive, inbound-adapter credential acquisition), but the *persistence*
contract is a core-owned port (:class:`~potpie_context_engine.domain.ports.cli_auth.credentials.CredentialStore`).
This module is the one place that picks the concrete implementation, so inbound
command code depends on the port — never on an adapter.

The default is the file-backed store; tests inject an in-memory fake via
``commands/_common.set_store`` instead.
"""

from __future__ import annotations

from potpie_context_engine.domain.ports.cli_auth.credentials import CredentialStore


def build_credential_store() -> CredentialStore:
    """Construct the production file-backed ``CredentialStore``."""
    from potpie_context_engine.adapters.outbound.cli_auth.credentials import FileCredentialStore

    return FileCredentialStore()


__all__ = ["build_credential_store"]
