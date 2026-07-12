"""``AuthService`` — the local auth/identity lifecycle seam.

Local OSS works without auth; this seam exists so the auth owner can add local
token/socket credentials (and the managed profile can bind ``potpie cloud
login``) behind a stable interface. In setup it is a **soft** step: an unbuilt
``init_local`` is reported ``not_implemented`` and never blocks first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from potpie_context_engine.domain.lifecycle import StepResult


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    """Who the host thinks the caller is."""

    subject: str
    mode: str  # local | managed | none
    detail: str | None = None


class AuthService(Protocol):
    """Local identity/credential provisioning."""

    def init_local(self) -> StepResult:
        """Provision local credentials (token/socket). Soft setup step."""
        ...

    def whoami(self) -> AuthIdentity: ...

    def logout(self) -> None: ...


__all__ = ["AuthIdentity", "AuthService"]
