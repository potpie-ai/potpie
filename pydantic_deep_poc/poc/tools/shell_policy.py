"""Shell policy definitions to avoid circular imports."""

from __future__ import annotations

from enum import Enum


class ShellPolicy(Enum):
    """Shell command policy levels for role-based access control."""

    UNRESTRICTED = "unrestricted"
    READ_ONLY = "read_only"
    VALIDATE_ONLY = "validate_only"
    FORBIDDEN = "forbidden"
