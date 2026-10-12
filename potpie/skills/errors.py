"""Domain errors raised by the skills installation flow."""

from __future__ import annotations


class UnknownAgentTargetError(ValueError):
    """Raised when no global skills-install target exists for an agent."""


class InvalidSkillsInstallPathError(ValueError):
    """Raised when a skills install path points to a file instead of a directory."""


class UnwritableSkillsTargetError(ValueError):
    """Raised when a harness directory cannot be written or cleaned up.

    Carries the directory to fix as ``recommended_next_action`` so the CLI can
    print the repair instead of an unclassified internal error.
    """

    def __init__(self, message: str, *, recommended_next_action: str) -> None:
        super().__init__(message)
        self.recommended_next_action = recommended_next_action
