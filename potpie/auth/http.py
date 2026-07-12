"""Injectable HTTP transport for the CLI auth subsystem.

One place that constructs httpx clients (a single timeout, optional base URL) and
translates *transport* failures (connect/timeout/etc.) into :class:`AuthHttpError`.
Flows depend on the :class:`HttpClient` Protocol, so tests inject a fake instead of
monkeypatching ``httpx``.

HTTP *status codes* are returned untouched — each provider keeps its own
status/body message extraction (GitHub reads ``error_description``, Firebase reads
the body repr, Atlassian classifies gateway responses), which is genuinely
provider-specific and does not belong in the shared transport.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import httpx

from potpie.auth.errors import CliAuthError

_DEFAULT_TIMEOUT = 30.0


class AuthHttpError(CliAuthError):
    """A transport failure, or a non-JSON body where JSON was required."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@runtime_checkable
class HttpClient(Protocol):
    """The HTTP surface the auth flows need (a subset of ``httpx.Client``)."""

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...

    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def delete(self, url: str, **kwargs: Any) -> httpx.Response: ...


def read_json(response: httpx.Response) -> Any:
    """Parse a response body as JSON, or raise :class:`AuthHttpError` (non-JSON)."""
    try:
        return response.json()
    except Exception as exc:  # httpx/json raise a few distinct types
        raise AuthHttpError(
            "Response body was not valid JSON.",
            status_code=response.status_code,
            body=response.text,
        ) from exc


class AuthHttpClient:
    """Reusable httpx-backed :class:`HttpClient` with one timeout + error translation.

    Holds a lazily-created ``httpx.Client`` reused across a flow (connection
    pooling for e.g. the GitHub device-code poll loop). Transport errors become
    :class:`AuthHttpError`; HTTP status codes pass through. Usable as a context
    manager; tests may inject a pre-built ``httpx.Client`` (e.g. with a mock
    transport) via ``client=``.
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _ensure(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _full_url(self, url: str) -> str:
        if not self._base or url.startswith(("http://", "https://")):
            return url
        return f"{self._base}{url if url.startswith('/') else '/' + url}"

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        client = self._ensure()
        try:
            return client.request(method, self._full_url(url), **kwargs)
        except httpx.RequestError as exc:
            raise AuthHttpError(f"HTTP request failed: {exc}") from exc

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> AuthHttpClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["AuthHttpClient", "AuthHttpError", "HttpClient", "read_json"]
