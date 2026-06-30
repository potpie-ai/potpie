"""Tool registry exceptions for programmatic error handling."""


class RegistryError(Exception):
    """Raised when a registry operation fails (e.g. unknown allow-list, invalid category)."""

    pass
