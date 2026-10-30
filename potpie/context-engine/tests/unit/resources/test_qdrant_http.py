"""REST contracts and failure boundaries for the resource vector projection."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from potpie_context_engine.adapters.outbound.resources.index.qdrant_http import (
    QdrantError,
    QdrantHttp,
)


def ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "result": result})


def collection(size: int = 3, distance: str = "Cosine") -> httpx.Response:
    return ok({"config": {"params": {"vectors": {"size": size, "distance": distance}}}})


@pytest.fixture
def make_transport():
    clients: list[httpx.Client] = []

    def make(handler: Callable[[httpx.Request], httpx.Response], **kwargs):
        client = httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=True
        )
        clients.append(client)
        return QdrantHttp("https://qdrant.example/prefix", client=client, **kwargs)

    yield make
    for client in clients:
        client.close()


def test_existing_collection_validates_without_mutation(make_transport):
    requests: list[httpx.Request] = []

    def handle(request):
        requests.append(request)
        return collection()

    transport = make_transport(handle, api_key="private-api-key", timeout=1.5)
    assert transport.ensure_collection("collection/name", 3) is False
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert request.url.raw_path == b"/prefix/collections/collection%2Fname"
    assert request.headers["api-key"] == "private-api-key"
    assert set(request.extensions["timeout"].values()) == {1.5}


@pytest.mark.parametrize("create_status", [200, 409])
def test_missing_collection_creates_and_verifies_even_after_race(
    make_transport, create_status
):
    requests: list[httpx.Request] = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(404)
        if len(requests) == 2:
            return ok(True) if create_status == 200 else httpx.Response(409)
        return collection()

    transport = make_transport(handle, timeout=1.2)
    assert transport.ensure_collection("resources", 3) is True
    assert [request.method for request in requests] == ["GET", "PUT", "GET"]
    assert json.loads(requests[1].content) == {
        "vectors": {"size": 3, "distance": "Cosine"}
    }
    assert requests[1].url.params["timeout"] == "2"


@pytest.mark.parametrize(
    "response",
    [
        collection(4),
        collection(distance="Dot"),
        ok({}),
        ok([]),
        ok(
            {
                "config": {
                    "params": {"vectors": {"named": {"size": 3, "distance": "Cosine"}}}
                }
            }
        ),
    ],
)
def test_incompatible_collection_is_rejected(make_transport, response):
    transport = make_transport(lambda request: response)
    with pytest.raises(QdrantError, match="collection"):
        transport.ensure_collection("resources", 3)


def test_failed_create_never_silently_uses_other_collection(make_transport):
    statuses = iter([httpx.Response(404), httpx.Response(500, text="sensitive body")])
    transport = make_transport(lambda request: next(statuses))
    with pytest.raises(QdrantError, match=r"^Qdrant request failed \(HTTP 500\)$"):
        transport.ensure_collection("resources", 3)


def test_false_creation_result_is_not_success(make_transport):
    responses = iter([httpx.Response(404), ok(False)])
    transport = make_transport(lambda request: next(responses))
    with pytest.raises(QdrantError, match="confirm collection creation"):
        transport.ensure_collection("resources", 3)


def test_missing_callback_precedes_creation_even_when_create_response_is_lost(
    make_transport,
):
    events = []

    def handle(request):
        events.append(request.method)
        if request.method == "GET":
            return httpx.Response(404)
        raise httpx.ReadTimeout("response lost", request=request)

    transport = make_transport(handle)
    with pytest.raises(QdrantError, match="timed out"):
        transport.ensure_collection(
            "resources", 3, on_missing=lambda: events.append("reset")
        )
    assert events == ["GET", "reset", "PUT"]


def test_failed_missing_callback_prevents_remote_creation(make_transport):
    events = []

    def handle(request):
        events.append(request.method)
        return httpx.Response(404)

    def reset():
        raise RuntimeError("local transaction failed")

    with pytest.raises(RuntimeError, match="local transaction failed"):
        make_transport(handle).ensure_collection("resources", 3, on_missing=reset)
    assert events == ["GET"]


def test_existing_collection_never_calls_missing_callback(make_transport):
    def unexpected():
        pytest.fail("existing collection must not reset publication state")

    transport = make_transport(lambda request: collection())
    assert transport.ensure_collection("resources", 3, on_missing=unexpected) is False


@pytest.mark.parametrize("dimensions", [True, 0, -1, 1.2])
def test_invalid_dimensions_do_not_send_request(make_transport, dimensions):
    def unexpected(request):
        pytest.fail("unexpected HTTP request")

    with pytest.raises(QdrantError, match="dimensions"):
        make_transport(unexpected).ensure_collection("resources", dimensions)


def test_writes_wait_for_completion_and_preserve_points(make_transport):
    requests: list[httpx.Request] = []

    def handle(request):
        requests.append(request)
        return ok({"operation_id": 1, "status": "completed"})

    transport = make_transport(handle)
    points = [
        {
            "id": "a703461d-4c32-4c47-8387-8dbd2ccfb4cf",
            "vector": [0.1, 0.2, 0.3],
            "payload": {"index_id": "local", "revision": "revision-1"},
        }
    ]
    transport.upsert("resources", points)
    transport.delete("resources", [points[0]["id"]])
    assert [request.method for request in requests] == ["PUT", "POST"]
    assert requests[0].url.path.endswith("/points")
    assert requests[1].url.path.endswith("/points/delete")
    assert json.loads(requests[0].content) == {"points": points}
    assert json.loads(requests[1].content) == {"points": [points[0]["id"]]}
    for request in requests:
        assert dict(request.url.params) == {"wait": "true", "timeout": "5"}


@pytest.mark.parametrize("result", [{"status": "acknowledged"}, {}, True])
def test_write_requires_completed_acknowledgement(make_transport, result):
    transport = make_transport(lambda request: ok(result))
    with pytest.raises(QdrantError, match="confirm completion"):
        transport.upsert("resources", [{"id": 1, "vector": [1.0]}])


def test_delete_missing_collection_is_idempotent_but_upsert_is_not(make_transport):
    transport = make_transport(lambda request: httpx.Response(404))
    transport.delete("resources", [1])
    with pytest.raises(QdrantError, match="HTTP 404"):
        transport.upsert("resources", [{"id": 1, "vector": [1.0]}])


def test_empty_writes_do_not_send_requests(make_transport):
    def unexpected(request):
        pytest.fail("unexpected HTTP request")

    transport = make_transport(unexpected)
    transport.upsert("resources", [])
    transport.delete("resources", [])


def test_query_uses_exact_search_and_preserves_scope_pagination_payload(make_transport):
    requests: list[httpx.Request] = []
    payload = {"index_id": "local", "resource_id": "resource", "revision": "revision-1"}

    def handle(request):
        requests.append(request)
        return ok(
            {
                "points": [
                    {"id": 123, "score": 1, "payload": payload},
                    {
                        "id": "a703461d-4c32-4c47-8387-8dbd2ccfb4cf",
                        "score": 0.4,
                        "payload": None,
                    },
                ]
            }
        )

    scope = {"must": [{"key": "index_id", "match": {"value": "local"}}]}
    transport = make_transport(handle)
    hits = transport.query("resources", [1.0, 0.0, 0.0], scope, 2, offset=4)
    assert hits == [
        {"id": 123, "score": 1.0, "payload": payload},
        {"id": "a703461d-4c32-4c47-8387-8dbd2ccfb4cf", "score": 0.4, "payload": {}},
    ]
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/points/query")
    assert json.loads(request.content) == {
        "query": [1.0, 0.0, 0.0],
        "filter": scope,
        "limit": 2,
        "offset": 4,
        "with_payload": True,
        "with_vector": False,
        "params": {"exact": True},
    }


@pytest.mark.parametrize("result", [[], {}, {"points": None}, {"points": {}}])
def test_invalid_query_result_is_rejected(make_transport, result):
    transport = make_transport(lambda request: ok(result))
    with pytest.raises(QdrantError, match="query result"):
        transport.query("resources", [1.0], {}, 1)


@pytest.mark.parametrize(
    "point",
    [
        {},
        None,
        {"id": 1, "score": 1},
        {"id": True, "score": 1, "payload": {}},
        {"id": "", "score": 1, "payload": {}},
        {"id": -1, "score": 1, "payload": {}},
        {"id": 2**64, "score": 1, "payload": {}},
        {"id": 1, "score": True, "payload": {}},
        {"id": 1, "score": "1", "payload": {}},
        {"id": 1, "score": 1, "payload": []},
    ],
)
def test_invalid_scored_points_are_rejected(make_transport, point):
    transport = make_transport(lambda request: ok({"points": [point]}))
    with pytest.raises(QdrantError, match="scored point"):
        transport.query("resources", [1.0], {}, 1)


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_query_scores_are_rejected(make_transport, score):
    body = (
        '{"status":"ok","result":{"points":[{"id":1,"score":'
        + score
        + ',"payload":{}}]}}'
    )
    transport = make_transport(lambda request: httpx.Response(200, text=body))
    with pytest.raises(QdrantError, match="scored point"):
        transport.query("resources", [1.0], {}, 1)


def test_unrepresentable_query_score_is_safe_error(make_transport):
    transport = make_transport(
        lambda request: ok({"points": [{"id": 1, "score": 10**400, "payload": {}}]})
    )
    with pytest.raises(QdrantError, match="scored point"):
        transport.query("resources", [1.0], {}, 1)


@pytest.mark.parametrize(
    "limit,offset", [(0, 0), (True, 0), (1.5, 0), (1, -1), (1, True)]
)
def test_invalid_pagination_is_rejected_before_network(make_transport, limit, offset):
    def unexpected(request):
        pytest.fail("unexpected HTTP request")

    with pytest.raises(QdrantError, match="limit or offset"):
        make_transport(unexpected).query("resources", [1.0], {}, limit, offset)


@pytest.mark.parametrize(
    "response,message",
    [
        (
            httpx.Response(
                503, text="private-api-key https://secret-host/ secret-payload"
            ),
            "HTTP 503",
        ),
        (httpx.Response(200, text="secret-payload"), "invalid JSON"),
        (
            httpx.Response(200, json={"status": {"error": "secret-payload"}}),
            "response envelope",
        ),
        (ok(None), "response envelope"),
        (httpx.Response(307, headers={"Location": "https://secret-host/"}), "HTTP 307"),
    ],
)
def test_errors_never_include_response_or_redirect_secrets(
    make_transport, response, message
):
    requests = []

    def handle(request):
        requests.append(request)
        return response

    transport = make_transport(handle, api_key="private-api-key")
    with pytest.raises(QdrantError, match=message) as error:
        transport.ensure_collection("secret-payload", 3)
    assert len(requests) == 1
    assert "secret" not in str(error.value)
    assert "private-api-key" not in str(error.value)


@pytest.mark.parametrize(
    "error_type,message",
    [
        (httpx.ReadTimeout, "timed out"),
        (httpx.ConnectError, "could not be reached"),
    ],
)
def test_transport_errors_hide_original_exception(make_transport, error_type, message):
    def handle(request):
        raise error_type("private-api-key https://secret-host/", request=request)

    with pytest.raises(QdrantError, match=message) as error:
        make_transport(handle).ensure_collection("resources", 3)
    assert error.value.__suppress_context__
    assert "secret" not in str(error.value)


def test_nonfinite_request_is_safe_encoding_error(make_transport):
    def unexpected(request):
        pytest.fail("unexpected HTTP request")

    with pytest.raises(QdrantError, match="could not be encoded"):
        make_transport(unexpected).query("resources", [float("nan")], {}, 1)


@pytest.mark.parametrize(
    "url",
    [
        "file:///secret-path",
        "https://user:secret@qdrant.example",
        "https://qdrant.example?secret=value",
        "https://qdrant.example#secret",
        "not a URL",
    ],
)
def test_unsafe_base_urls_are_rejected_without_echoing_input(url):
    with pytest.raises(QdrantError) as error:
        QdrantHttp(url)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, 10**400])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(QdrantError, match="timeout"):
        QdrantHttp("http://localhost:6333", timeout=timeout)


def test_close_only_closes_owned_client(make_transport):
    transport = make_transport(lambda request: collection())
    transport.close()
    assert transport.ensure_collection("resources", 3) is False

    owned = QdrantHttp("http://localhost:6333")
    owned.close()
    owned.close()
    with pytest.raises(QdrantError, match="closed"):
        owned.ensure_collection("resources", 3)
