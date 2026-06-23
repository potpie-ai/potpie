"""Composition root for the CLI auth credential seam.

The auth/credential surface deliberately does not route through ``HostShell`` (it
is interactive, inbound-adapter credential acquisition), but the *persistence*
contract is a core-owned port (:class:`~domain.ports.cli_auth.credentials.CredentialStore`).
This module is the one place that picks the concrete implementation, so inbound
command code depends on the port — never on an adapter.

The default is the keychain-backed store; tests inject an in-memory fake via
``commands/_common.set_store`` instead.
"""

from __future__ import annotations

from potpie.context_engine.domain.ports.cli_auth.credentials import CredentialStore


def build_credential_store() -> CredentialStore:
    """Construct the production ``CredentialStore`` (keychain + config file)."""
    from potpie.context_engine.adapters.outbound.cli_auth.credentials import KeyringCredentialStore

    return KeyringCredentialStore()


__all__ = ["build_credential_store"]
