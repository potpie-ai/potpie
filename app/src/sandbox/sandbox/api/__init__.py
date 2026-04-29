"""Public client surface for `app/src/sandbox`.

Callers in the rest of potpie should import from here, not from
`sandbox.application` or `sandbox.domain`. Everything else is an implementation
detail of the providers and the service.
"""

from sandbox.api.client import SandboxClient
from sandbox.api.types import (
    FileEntry,
    GitStatus,
    Hit,
    WorkspaceHandle,
)

__all__ = [
    "FileEntry",
    "GitStatus",
    "Hit",
    "SandboxClient",
    "WorkspaceHandle",
]
