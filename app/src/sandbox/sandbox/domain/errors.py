"""Domain-level sandbox errors; translate these at API or CLI boundaries."""


class SandboxCoreError(Exception):
    """Base for all sandbox core errors."""


class WorkspaceError(SandboxCoreError):
    """Base for workspace/repository persistence failures."""


class WorkspaceNotFound(WorkspaceError):
    """Workspace id is unknown to the configured store/provider."""


class WorkspaceLocked(WorkspaceError):
    """Workspace is already being mutated by another operation."""


class WorkspaceDirty(WorkspaceError):
    """Requested operation cannot proceed on a dirty workspace."""


class RepoCacheUnavailable(WorkspaceError):
    """Repository cache could not be created or fetched."""


class RepoAuthFailed(WorkspaceError):
    """Repository clone/fetch failed due to authentication or authorization."""


class InvalidWorkspacePath(WorkspaceError):
    """A requested path escapes or does not belong to the workspace root."""


class RuntimeErrorBase(SandboxCoreError):
    """Base for runtime provider failures."""


class RuntimeNotFound(RuntimeErrorBase):
    """Runtime id is unknown to the configured store/provider."""


class RuntimeUnavailable(RuntimeErrorBase):
    """Runtime backend is unavailable or cannot satisfy the request."""


class RuntimeUnauthorized(RuntimeErrorBase):
    """Runtime backend rejected credentials or access."""


class RuntimeTimeout(RuntimeErrorBase):
    """Runtime command or lifecycle operation timed out."""


class RuntimeConflict(RuntimeErrorBase):
    """Runtime lifecycle conflict, such as creating an already existing runtime."""


class RuntimeResourceLimit(RuntimeErrorBase):
    """Runtime backend rejected resource hints or quota was exceeded."""


class RuntimeCommandRejected(RuntimeErrorBase):
    """Command was rejected before execution by policy or adapter limitations."""


class GitPlatformError(SandboxCoreError):
    """Base for git-platform (PR/review/comment) failures."""


class PullRequestFailed(GitPlatformError):
    """The git platform refused or could not create the requested PR."""


class GitPlatformNotConfigured(GitPlatformError):
    """No `GitPlatformProvider` is wired into the service."""

