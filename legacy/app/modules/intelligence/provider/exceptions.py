class UnsupportedProviderError(RuntimeError):
    """Raised when the selected provider cannot satisfy the requested capability."""

    def __init__(self, message: str):
        super().__init__(message)
