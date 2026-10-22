"""Small REST transport for the resource index's Qdrant projection.

Using HTTP keeps the transport independent of the native grpcio/NumPy wheels
pulled in by qdrant-client. The index owns retries and durable publication;
this layer reports success only after Qdrant confirms an applied write.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Sequence
from urllib.parse import quote

import httpx


class QdrantError(RuntimeError):
    """A safe diagnostic: never includes credentials, request or response data."""


class QdrantHttp:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        timeout: float = 5,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        try:
            parsed = httpx.URL(url)
        except (httpx.InvalidURL, ValueError, TypeError):
            raise QdrantError("Qdrant URL is invalid") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise QdrantError(
                "Qdrant URL must use HTTP(S) without credentials, query or fragment"
            )
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise QdrantError("Qdrant timeout must be a finite positive number")
        try:
            seconds = float(timeout)
        except OverflowError:
            raise QdrantError(
                "Qdrant timeout must be a finite positive number"
            ) from None
        if not math.isfinite(seconds) or seconds <= 0:
            raise QdrantError("Qdrant timeout must be a finite positive number")
        self._url = str(parsed).rstrip("/")
        self._timeout = seconds
        self._headers = {"api-key": api_key} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._timeout, follow_redirects=False, trust_env=False
        )

    @staticmethod
    def _collection_path(name: str) -> str:
        if not isinstance(name, str) or not name or name in {".", ".."}:
            raise QdrantError("Qdrant collection name is invalid")
        return f"/collections/{quote(name, safe='')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_status: tuple[int, ...] = (),
    ) -> Any:
        if self._client.is_closed:
            raise QdrantError("Qdrant HTTP client is closed")
        try:
            response = self._client.request(
                method,
                self._url + path,
                headers=self._headers,
                json=body,
                params=params,
                timeout=self._timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise QdrantError("Qdrant request timed out") from None
        except httpx.HTTPError:
            raise QdrantError("Qdrant service could not be reached") from None
        except (ValueError, TypeError):
            raise QdrantError("Qdrant request could not be encoded") from None
        if response.status_code in allow_status:
            return None
        if not 200 <= response.status_code < 300:
            raise QdrantError(f"Qdrant request failed (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError:
            raise QdrantError("Qdrant returned invalid JSON") from None
        if (
            not isinstance(data, dict)
            or data.get("status") != "ok"
            or data.get("result") is None
        ):
            raise QdrantError("Qdrant returned an invalid response envelope")
        return data["result"]

    def ensure_collection(
        self,
        name: str,
        dimensions: int,
        *,
        on_missing: Callable[[], None] | None = None,
    ) -> bool:
        """Validate the collection and return whether it was initially missing.

        An initially missing collection requires this index to republish its
        points even if another caller wins the creation race. ``on_missing``
        must durably mark that backlog before remote creation, so a lost
        creation response or process exit cannot conceal an empty collection.
        """
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions < 1
        ):
            raise QdrantError("Qdrant vector dimensions must be a positive integer")
        path = self._collection_path(name)
        result = self._request("GET", path, allow_status=(404,))
        was_missing = result is None
        if result is None:
            if on_missing is not None:
                on_missing()
            # A concurrent process may create this collection between GET and
            # PUT. Only an explicit conflict is tolerated, then validated.
            created = self._request(
                "PUT",
                path,
                body={"vectors": {"size": dimensions, "distance": "Cosine"}},
                params={"timeout": math.ceil(self._timeout)},
                allow_status=(409,),
            )
            if created is not None and created is not True:
                raise QdrantError("Qdrant did not confirm collection creation")
            result = self._request("GET", path)
        try:
            vectors = result["config"]["params"]["vectors"]
        except (TypeError, KeyError):
            raise QdrantError(
                "Qdrant returned an invalid collection configuration"
            ) from None
        if (
            not isinstance(vectors, dict)
            or type(vectors.get("size")) is not int
            or vectors["size"] != dimensions
            or vectors.get("distance") != "Cosine"
        ):
            raise QdrantError(
                "Qdrant collection vector size or distance is incompatible"
            )
        return was_missing

    def _write(
        self,
        method: str,
        path: str,
        body: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        result = self._request(
            method,
            path,
            body=body,
            params={"wait": "true", "timeout": math.ceil(self._timeout)},
            allow_status=(404,) if missing_ok else (),
        )
        if result is None and missing_ok:
            return
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise QdrantError("Qdrant did not confirm completion of the write")

    def upsert(self, collection: str, points: Sequence[dict[str, Any]]) -> None:
        if points:
            self._write(
                "PUT",
                self._collection_path(collection) + "/points",
                {"points": list(points)},
            )

    def delete(self, collection: str, point_ids: Sequence[str | int]) -> None:
        if point_ids:
            self._write(
                "POST",
                self._collection_path(collection) + "/points/delete",
                {"points": list(point_ids)},
                missing_ok=True,
            )

    def query(
        self,
        collection: str,
        vector: Sequence[float],
        filter: dict[str, Any],
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0:
            raise QdrantError("Qdrant query limit or offset is invalid")
        result = self._request(
            "POST",
            self._collection_path(collection) + "/points/query",
            body={
                "query": list(vector),
                "filter": filter,
                "limit": limit,
                "offset": offset,
                "with_payload": True,
                "with_vector": False,
                "params": {"exact": True},
            },
            params={"timeout": math.ceil(self._timeout)},
        )
        if not isinstance(result, dict) or not isinstance(result.get("points"), list):
            raise QdrantError("Qdrant returned an invalid query result")
        hits = []
        for point in result["points"]:
            if (
                not isinstance(point, dict)
                or type(point.get("id")) not in {int, str}
                or (isinstance(point["id"], str) and not point["id"])
                or (isinstance(point["id"], int) and not 0 <= point["id"] < 2**64)
                or type(point.get("score")) not in {int, float}
                or "payload" not in point
                or (
                    point["payload"] is not None
                    and not isinstance(point["payload"], dict)
                )
            ):
                raise QdrantError("Qdrant returned an invalid scored point")
            try:
                score = float(point["score"])
            except OverflowError:
                raise QdrantError("Qdrant returned an invalid scored point") from None
            if not math.isfinite(score):
                raise QdrantError("Qdrant returned an invalid scored point")
            hits.append(
                {
                    "id": point["id"],
                    "score": score,
                    "payload": point["payload"] or {},
                }
            )
        return hits

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
