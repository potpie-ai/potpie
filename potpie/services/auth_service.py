"""``LocalAuthService`` — local no-auth identity.

Local OSS works without auth, so ``whoami`` reports a ``none`` identity and
``init_local`` is a fail-closed stub: the auth owner fills it with real local
token/socket credential provisioning. Because the setup orchestrator runs auth
as a **soft** step, the unbuilt ``init_local`` is reported ``not_implemented``
and never blocks first run.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_core.domain.errors import CapabilityNotImplemented
from potpie_context_core.domain.lifecycle import StepResult
from potpie_context_engine.domain.ports.services.auth import AuthIdentity


@dataclass(slots=True)
class LocalAuthService:
    """Local identity stub; managed auth binds ``potpie cloud login`` later."""

    profile: str = "local"

    def init_local(self) -> StepResult:
        raise CapabilityNotImplemented(
            "host.auth.init_local",
            detail="local auth/identity provisioning not implemented",
            recommended_next_action="local OSS works without auth; managed profile uses 'potpie cloud login'",
        )

    def whoami(self) -> AuthIdentity:
        return AuthIdentity(
            subject="local", mode="none", detail="local OSS; no auth required"
        )

    def logout(self) -> None:
        return None


__all__ = ["LocalAuthService"]
