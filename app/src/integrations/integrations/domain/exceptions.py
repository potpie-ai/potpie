"""Integration domain exceptions.

Layering rule: anything that crosses the HTTP boundary or the Celery
worker boundary is mapped to one of these. The HTTP adapter is the
*only* layer that knows how to translate them into status codes;
inner layers raise them, never catch-and-rewrap.
"""

from __future__ import annotations


class IntegrationError(Exception):
    """Base for all integration-domain errors."""


# -- OAuth / install ---------------------------------------------------


class LinearOrganizationAlreadyIntegratedError(IntegrationError):
    """The (user, organization) pair already has a Linear integration row.

    Raised by ``save_linear_integration`` on retry of an OAuth callback
    for an already-installed workspace. The HTTP adapter maps this to a
    redirect with ``status=already_integrated``; the frontend surfaces
    a non-error UI ("you already connected this workspace").
    """

    def __init__(self, integration_id: str) -> None:
        self.integration_id = integration_id
        super().__init__(
            "Linear organization is already integrated. "
            f"Existing integration ID: {integration_id}."
        )


# -- Webhook ingestion -------------------------------------------------


class WebhookError(IntegrationError):
    """Base for webhook-handling errors."""


class WebhookSignatureError(WebhookError):
    """Webhook signature missing or did not match the configured secret.

    HTTP layer maps to 401. We never include the expected signature in
    the message — only that verification failed.
    """


class WebhookPayloadError(WebhookError):
    """Webhook body could not be parsed or was missing required fields.

    HTTP layer maps to 400. Used for malformed JSON, missing
    ``organizationId``, unknown envelope shape.
    """


class WebhookProcessingError(WebhookError):
    """An unexpected failure happened while routing a webhook to pots.

    HTTP layer maps to 500. Use only when no more specific error
    applies; the underlying exception should be the ``__cause__``.
    """
