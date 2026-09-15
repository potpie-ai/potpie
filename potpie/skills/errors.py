"""Domain errors raised by the skills installation flow."""

from __future__ import annotations


class UnknownAgentTargetError(ValueError):
    """Raised when no global skills-install target exists for an agent."""


class InvalidSkillsInstallPathError(ValueError):
    """Raised when a skills install path points to a file instead of a directory."""
