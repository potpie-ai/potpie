"""Canonical domain-error encoding for the protocol-v1 daemon boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from potpie_context_engine.contracts import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)


@dataclass(frozen=True, slots=True)
class EncodedDomainError:
    code: str
    message: str
    details: dict[str, Any]
    retryable: bool = False


def encode_domain_error(exc: Exception) -> EncodedDomainError | None:
    if isinstance(exc, CapabilityNotImplemented):
        return EncodedDomainError(
            code="ENGINE_CAPABILITY_NOT_IMPLEMENTED",
            message=str(exc),
            details={
                "capability": exc.capability,
                "detail": exc.detail,
                "recommended_next_action": exc.recommended_next_action,
            },
        )
    if isinstance(exc, PotNotFound):
        return EncodedDomainError(
            code="ENGINE_POT_NOT_FOUND",
            message=str(exc),
            details={},
        )
    if isinstance(exc, ContextEngineDisabled):
        return EncodedDomainError(
            code="ENGINE_DISABLED",
            message=str(exc),
            details={},
        )
    if isinstance(exc, ValueError):
        return EncodedDomainError(
            code="ENGINE_VALIDATION_ERROR",
            message=str(exc),
            details={
                "detail": getattr(exc, "detail", None),
                "recommended_next_action": getattr(
                    exc, "recommended_next_action", None
                ),
            },
        )
    return None


def decode_domain_error(
    *, code: str, message: str, details: Mapping[str, Any]
) -> Exception | None:
    if code == "ENGINE_CAPABILITY_NOT_IMPLEMENTED":
        capability = str(details.get("capability") or "unknown")
        detail = details.get("detail")
        action = details.get("recommended_next_action")
        return CapabilityNotImplemented(
            capability,
            detail=str(detail) if detail is not None else None,
            recommended_next_action=str(action) if action is not None else None,
        )
    if code == "ENGINE_POT_NOT_FOUND":
        return PotNotFound(message)
    if code == "ENGINE_DISABLED":
        return ContextEngineDisabled(message)
    if code == "ENGINE_VALIDATION_ERROR":
        exc = ValueError(message)
        detail = details.get("detail")
        action = details.get("recommended_next_action")
        if detail is not None:
            setattr(exc, "detail", detail)
        if action is not None:
            setattr(exc, "recommended_next_action", action)
        return exc
    return None


__all__ = ["EncodedDomainError", "decode_domain_error", "encode_domain_error"]
