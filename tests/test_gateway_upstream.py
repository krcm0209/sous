"""The forwarder in isolation: a Starlette app whose only handler calls
Upstream.forward, driven through httpx's in-process transport, with
FakeUpstream on the far side. Header and byte fidelity live here; what only a
real socket shows (incremental relay, hang-ups) is in test_gateway_http.py."""

import asyncio
import gzip
import json
import sys

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from sous.gateway.upstream import TIMEOUT, VIA, Upstream, request_headers, response_headers
from tests.fake_upstream import METHODS, FakeUpstream


def _proxy_app(upstream: Upstream, *, buffered: bool) -> Starlette:
    """The two ways routes.py calls forward(): with the body it already read
    (the Messages routes) or streaming it through (the catch-all)."""

    async def handler(request: Request) -> Response:
        body = await request.body() if buffered else None
        return await upstream.forward(request, body)

    return Starlette(routes=[Route("/{path:path}", handler, methods=METHODS)])


def _send(app, method: str, path: str, headers=None, content=None) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8383"
        ) as client:
            return await client.request(method, path, headers=headers, content=content)

    return asyncio.run(go())


def _hand_built_request(headers: list[tuple[bytes, bytes]], **scope) -> Request:
    """A Request no HTTP client could produce for us: an exact header list with
    nothing an outer client would add of its own, or a request target uvicorn's
    h11 would 400 before it ever became an ASGI scope."""

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/hello",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8383),
            **scope,
        },
        receive,
    )


def _failing_upstream(exc: Exception) -> Upstream:
    def raise_it(request: httpx.Request) -> httpx.Response:
        raise exc

    return Upstream("https://upstream.test", transport=httpx.MockTransport(raise_it))


# --- pure header filtering -----------------------------------------------------------


def test_request_headers_drop_host_hop_by_hop_and_optionally_content_length():
    raw = [
        (b"Host", b"127.0.0.1:8383"),
        (b"Connection", b"close, X-Hop"),
        (b"X-Hop", b"1"),
        (b"Keep-Alive", b"timeout=5"),
        (b"Transfer-Encoding", b"chunked"),
        (b"TE", b"trailers"),
        (b"Upgrade", b"h2c"),
        (b"Proxy-Connection", b"keep-alive"),
        (b"Content-Length", b"99"),
        (b"Authorization", b"Bearer x"),
        (b"Anthropic-Beta", b"a"),
        (b"Anthropic-Beta", b"b"),
    ]
    assert request_headers(raw, drop_content_length=True) == [
        (b"Authorization", b"Bearer x"),
        (b"Anthropic-Beta", b"a"),
        (b"Anthropic-Beta", b"b"),
    ]
    assert request_headers(raw, drop_content_length=False) == [
        (b"Content-Length", b"99"),
        (b"Authorization", b"Bearer x"),
        (b"Anthropic-Beta", b"a"),
        (b"Anthropic-Beta", b"b"),
    ]


def test_response_headers_drop_hop_by_hop_and_server_owned_and_add_via():
    raw = [
        (b"date", b"Mon, 01 Jan 2029 00:00:00 GMT"),
        (b"server", b"cloudflare"),
        (b"transfer-encoding", b"chunked"),
        (b"connection", b"keep-alive"),
        (b"content-type", b"text/event-stream"),
        (b"request-id", b"req_1"),
        (b"anthropic-ratelimit-requests-remaining", b"9"),
        (b"set-cookie", b"a=1"),
        (b"set-cookie", b"b=2"),
    ]
    assert response_headers(raw) == [
        (b"content-type", b"text/event-stream"),
        (b"request-id", b"req_1"),
        (b"anthropic-ratelimit-requests-remaining", b"9"),
        (b"set-cookie", b"a=1"),
        (b"set-cookie", b"b=2"),
        (b"via", b"1.1 sous"),
    ]


# --- request fidelity -------------------------------------------------------------


def test_forwards_method_path_query_and_body_bytes_verbatim():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream(), buffered=True)
    # Deliberately not JSON the gateway would accept: the forwarder must never parse.
    body = b'{"model": "claude-opus-5", "n": 1e999, "x": NaN,   "spaced" : true}'
    r = _send(app, "POST", "/v1/messages?beta=true", content=body)
    assert r.status_code == 200
    assert r.json() == {"upstream": True}
    (seen,) = fake.requests
    assert (seen["method"], seen["path"], seen["query"]) == ("POST", "/v1/messages", "beta=true")
    assert seen["body"] == body


def test_end_to_end_headers_travel_unmodified_in_order_with_duplicates():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream(), buffered=True)
    headers = [
        ("authorization", "Bearer sk-ant-oat01-canary"),
        ("anthropic-beta", "oauth-2025-04-20"),
        ("anthropic-beta", "claude-code-20250219"),
        ("anthropic-version", "2023-06-01"),
        ("x-unknown-future-header", "kept"),
        ("user-agent", "claude-cli/2.1.247"),
        ("content-type", "application/json"),
    ]
    _send(app, "POST", "/v1/messages", headers=headers, content=b"{}")
    (seen,) = fake.requests
    ours = {name for name, _ in headers}
    assert [(n, v) for n, v in seen["headers"] if n in ours] == headers


def test_host_is_the_upstreams_and_hop_by_hop_headers_do_not_cross():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream("https://upstream.test:8443"), buffered=True)
    _send(
        app,
        "POST",
        "/v1/messages",
        headers=[
            ("connection", "keep-alive, x-per-hop"),
            ("keep-alive", "timeout=5"),
            ("x-per-hop", "gone"),
            ("te", "trailers"),
            ("proxy-connection", "keep-alive"),
            ("upgrade", "h2c"),
            ("x-kept", "yes"),
        ],
        content=b"{}",
    )
    (seen,) = fake.requests
    names = [n for n, _ in seen["headers"]]
    for gone in ("keep-alive", "x-per-hop", "te", "proxy-connection", "upgrade"):
        assert gone not in names, gone
    assert ("x-kept", "yes") in seen["headers"]
    assert dict(seen["headers"])["host"] == "upstream.test:8443"


def test_a_buffered_body_is_sent_with_a_recomputed_content_length():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream(), buffered=True)
    _send(app, "POST", "/v1/messages", content=b'{"a":1}')
    (seen,) = fake.requests
    headers = dict(seen["headers"])
    assert headers["content-length"] == "7"
    assert "transfer-encoding" not in headers


def test_a_streamed_body_keeps_the_clients_content_length():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream(), buffered=False)
    r = _send(app, "PUT", "/api/frame/upload", content=b"x" * 5000)
    assert r.status_code == 200
    (seen,) = fake.requests
    assert seen["body"] == b"x" * 5000
    assert dict(seen["headers"])["content-length"] == "5000"


def test_bodiless_requests_send_no_body_framing():
    fake = FakeUpstream()
    app = _proxy_app(fake.upstream(), buffered=False)
    assert _send(app, "GET", "/api/oauth/usage").status_code == 200
    assert _send(app, "HEAD", "/api/hello").status_code == 200
    for seen in fake.requests:
        names = [n for n, _ in seen["headers"]]
        assert "transfer-encoding" not in names and "content-length" not in names, seen


def test_upstream_cookies_are_never_stored_or_replayed():
    """httpx's client keeps a cookie jar: left alone, it eats every upstream
    Set-Cookie (Cloudflare's __cf_bm on any Anthropic response) and adds a
    Cookie header to the NEXT forwarded request — one the client never sent,
    and shared across every local client the daemon serves."""
    fake = FakeUpstream()
    fake.reply = Response(
        b'{"upstream": true}',
        media_type="application/json",
        headers={"set-cookie": "__cf_bm=abc; Path=/"},
    )
    upstream = fake.upstream()
    app = _proxy_app(upstream, buffered=True)
    first = _send(app, "POST", "/v1/messages", content=b"{}")
    second = _send(app, "POST", "/v1/messages", content=b"{}")
    names = [n for n, _ in fake.requests[1]["headers"]]
    assert "cookie" not in names, fake.requests[1]["headers"]
    # The relay itself is untouched: the client still sees what the upstream set.
    assert first.headers["set-cookie"] == "__cf_bm=abc; Path=/"
    assert second.headers["set-cookie"] == "__cf_bm=abc; Path=/"
    asyncio.run(upstream.aclose())


def test_a_clients_own_cookie_header_passes_through():
    """Refusing to store cookies must not stop forwarding one: a Cookie the
    client sent is an ordinary end-to-end header."""
    fake = FakeUpstream()
    upstream = fake.upstream()
    app = _proxy_app(upstream, buffered=True)
    _send(app, "POST", "/v1/messages", headers=[("cookie", "mine=1")], content=b"{}")
    (seen,) = fake.requests
    assert ("cookie", "mine=1") in seen["headers"]
    asyncio.run(upstream.aclose())


# --- response fidelity --------------------------------------------------------------


def test_status_headers_and_raw_body_are_relayed_with_via():
    fake = FakeUpstream()
    payload = gzip.compress(
        b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'
    )
    fake.reply = Response(
        payload,
        status_code=429,
        headers={
            "content-type": "application/json",
            "content-encoding": "gzip",
            "retry-after": "7",
            "request-id": "req_abc",
            "server": "fake-origin",
            "date": "Mon, 01 Jan 2029 00:00:00 GMT",
        },
    )
    app = _proxy_app(fake.upstream(), buffered=True)
    r = _send(app, "POST", "/v1/messages", headers={"accept-encoding": "gzip"}, content=b"{}")
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.headers["request-id"] == "req_abc"
    assert r.headers.get_list("via") == [VIA]
    # The body crossed as the gzip bytes the upstream produced, described by
    # the headers it produced — the client (httpx here) is what decodes it.
    assert r.headers["content-encoding"] == "gzip"
    assert r.headers["content-length"] == str(len(payload))
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert "server" not in r.headers
    assert "date" not in r.headers


def test_repeated_response_headers_survive():
    fake = FakeUpstream()
    reply = Response(b"{}", media_type="application/json")
    reply.raw_headers.extend([(b"set-cookie", b"a=1"), (b"set-cookie", b"b=2")])
    fake.reply = reply
    app = _proxy_app(fake.upstream(), buffered=True)
    r = _send(app, "POST", "/v1/messages", content=b"{}")
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"]


# --- failures of the forwarding itself -------------------------------------------------


def test_an_unreachable_upstream_is_a_502_that_quotes_nothing_from_the_wire():
    app = _proxy_app(_failing_upstream(httpx.ConnectError("secret-detail")), buffered=True)
    r = _send(app, "POST", "/v1/messages", content=b"{}")
    assert r.status_code == 502
    assert r.json() == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "could not reach upstream upstream.test: ConnectError",
        },
    }
    assert r.headers["via"] == VIA
    assert "secret-detail" not in r.text


def test_an_upstream_timeout_is_a_504():
    app = _proxy_app(_failing_upstream(httpx.ReadTimeout("t")), buffered=True)
    r = _send(app, "POST", "/v1/messages", content=b"{}")
    assert r.status_code == 504
    assert r.json()["error"] == {"type": "api_error", "message": "upstream upstream.test timed out"}
    assert r.headers["via"] == VIA


def test_client_never_adds_a_credential_or_follows_a_redirect():
    """These are the module's actual security boundary, not just the docstring's
    claim of one, and none of the other tests in this file would fail if any
    were flipped: trust_env=False keeps a stray ~/.netrc or HTTPS_PROXY from
    ever seeing the OAuth bearer token this process forwards; follow_redirects
    =False stops httpx from replaying Authorization to whatever a 3xx names,
    which only the real client should ever decide to do; the empty default
    header set keeps httpx from asking for gzip on the client's behalf (so
    _relay never hands back compressed bytes to a client that never said it
    could decompress them) or identifying the request as anything but what the
    client sent; TIMEOUT and the connection limits are exactly the values the
    module's own comments justify."""
    upstream = FakeUpstream().upstream()
    client = upstream._client
    assert client.trust_env is False
    assert client.follow_redirects is False
    assert list(client.headers.items()) == []
    assert client.timeout == TIMEOUT
    assert httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=None) == TIMEOUT
    # FakeUpstream forces httpx.ASGITransport (no socket, no connection pool),
    # so the pool's own limits are checked against a real-transport client
    # instead. httpx.Limits(max_connections=None, ...) reaches httpcore as
    # sys.maxsize, not None -- httpcore itself makes that substitution -- so
    # that is what an unbounded pool asserts here.
    real = Upstream("https://upstream.test")
    pool = real._client._transport._pool  # ty: ignore[unresolved-attribute]
    assert pool._max_connections == sys.maxsize
    assert pool._max_keepalive_connections == 20
    asyncio.run(real.aclose())
    asyncio.run(upstream.aclose())


def test_nothing_the_client_did_not_send_is_added():
    """The upstream must see the client's headers and nothing else: httpx's
    own defaults (accept, user-agent, accept-encoding, connection) would
    describe the request as httpx's rather than as the client made it. Hand
    built rather than sent through a client, because any real client adds its
    own defaults and the point here is what SOUS adds."""
    fake = FakeUpstream()
    upstream = fake.upstream()
    request = _hand_built_request([(b"host", b"127.0.0.1:8383"), (b"x-only", b"1")])
    response = asyncio.run(upstream.forward(request, None))
    assert response.status_code == 200
    (seen,) = fake.requests
    assert dict(seen["headers"]) == {"host": "upstream.test", "x-only": "1"}
    asyncio.run(upstream.aclose())


def test_a_non_ascii_request_target_is_a_400_not_a_raise():
    """forward()'s contract is that it never raises. Building the upstream URL
    is part of that: a non-ASCII raw_path makes httpx.URL raise
    UnicodeDecodeError, which is not an httpx.HTTPError and would escape.
    Driven through a hand-built scope because no HTTP client will send this —
    uvicorn's h11 400s a non-ASCII target before it ever becomes an ASGI scope.
    """
    fake = FakeUpstream()
    upstream = fake.upstream()
    request = _hand_built_request([(b"host", b"127.0.0.1:8383")], path="/\udcff", raw_path=b"/\xff")
    response = asyncio.run(upstream.forward(request, None))
    assert response.status_code == 400
    assert response.headers["via"] == VIA
    assert json.loads(bytes(response.body))["error"]["message"] == "malformed request target"
    assert fake.requests == []
    asyncio.run(upstream.aclose())


def test_aclose_closes_the_client():
    upstream = FakeUpstream().upstream()
    asyncio.run(upstream.aclose())
    assert upstream._client.is_closed
