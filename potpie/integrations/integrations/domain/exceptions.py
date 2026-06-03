"""Integration domain exceptions (expected flows, not all are errors)."""


class LinearOrganizationAlreadyIntegratedError(Exception):
    """Raised when OAuth completes for a Linear org that already has an integration row."""

    def __init__(self, integration_id: str) -> None:
        self.integration_id = integration_id
        msg = (
            "Linear organization is already integrated. "
            f"Existing integration ID: {integration_id}. "
            "Delete the existing integration first if you want to reconnect."
        )
        super().__init__(msg)
