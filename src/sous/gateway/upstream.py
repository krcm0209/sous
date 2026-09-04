"""Forwarding of everything the gateway does not serve itself.

A transparent HTTP/1.1 proxy to one fixed origin: the bytes the client sent
are the bytes the upstream gets, end-to-end headers travel unmodified (in
their original order, duplicates included), and the upstream's status,
headers and body come back untouched. Only what a proxy must change changes:
`Host`, the hop-by-hop headers of RFC 9110 §7.6.1, and a `Via` on the way
back. Never logs anything; never adds a credential (no netrc, no proxy
environment — trust_env is off).
"""

from __future__ import annotations

import http.cookiejar
from collections.abc import AsyncIterator
from urllib.parse import quote

import anyio
import httpx
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

# RFC 9110 §7.6.1: hop-by-hop headers describe one connection, not the
# message, and must not be forwarded — plus whatever `Connection` itself
# names. proxy-connection is the pre-standard spelling some clients still send.
_HOP_BY_HOP = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"proxy-connection",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)
# uvicorn prepends its own Date and Server to every response; forwarding the
# upstream's copies would send both.
_SERVER_OWNED = frozenset({b"date", b"server"})
# RFC 9110 §7.6.3. On every response this module produces — relayed or its
# own error — so a client (the `sous claude` launcher) can tell "the gateway
# forwarded this" from "the daemon has no such route". Never added to a
# request: nothing that could make the upstream treat a proxied request
# differently goes up.
VIA = "1.1 sous"
# connect: a dead network fails fast. read: the gap between bytes, not the
# whole response — Anthropic streams pings while it thinks, and its own SDK
# defaults to 600s. write: per chunk of a body up to 32 MiB. pool: connections
# are unlimited below, so nothing ever waits for one.
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=None)


def _connection_named(raw: list[tuple[bytes, bytes]]) -> frozenset[bytes]:
    names: set[bytes] = set()
    for name, value in raw:
        if name.lower() == b"connection":
            names.update(t.strip().lower() for t in value.split(b",") if t.strip())
    return frozenset(names)


def request_headers(
    raw: list[tuple[bytes, bytes]], *, drop_content_length: bool
) -> list[tuple[bytes, bytes]]:
    """The client's headers as the upstream should see them: same order, same
    duplicates, minus Host (httpx derives it from the upstream URL) and the
    hop-by-hop set. Content-Length goes too when the body was buffered — httpx
    recomputes it from the exact bytes, so a client that lied cannot desync
    the upstream connection."""
    drop = _HOP_BY_HOP | _connection_named(raw) | {b"host"}
    if drop_content_length:
        drop = drop | {b"content-length"}
    return [(name, value) for name, value in raw if name.lower() not in drop]


def response_headers(raw: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """The upstream's headers as the client should see them, plus Via.
    Content-Length and Content-Encoding stay: the body is relayed as the raw
    bytes they describe."""
    drop = _HOP_BY_HOP | _connection_named(raw) | _SERVER_OWNED
    kept = [(name, value) for name, value in raw if name.lower() not in drop]
    kept.append((b"via", VIA.encode("ascii")))
    return kept


def _has_body(request: Request) -> bool:
    # httpx frames any iterable content as chunked when no length is declared;
    # a GET or HEAD must go out with no body at all, not an empty chunked one.
    length = request.headers.get("content-length")
    if length is not None:
        return length != "0"
    return "transfer-encoding" in request.headers


def _target_path(request: Request) -> bytes:
    # raw_path is the percent-encoded path exactly as received (ASGI leaves
    # it optional, so fall back to re-encoding the decoded one); the query
    # string is always raw bytes.
    path = request.scope.get("raw_path") or quote(request.url.path).encode("ascii")
    query = request.scope.get("query_string", b"")
    return path + b"?" + query if query else path


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": "api_error", "message": message}},
        status_code=status,
        headers={"via": VIA},
    )


async def _relay(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
        # An upstream failure mid-body is deliberately NOT caught: the headers
        # are already on the wire, and returning here would let Starlette end
        # the body normally — on a chunked response (any SSE stream) that is a
        # complete, valid, silently truncated answer, and an SSE stream missing
        # message_stop looks exactly like a finished one. Propagating instead
        # makes uvicorn abort the connection without the terminating chunk, so
        # the client sees a broken response and retries. uvicorn logs the
        # exception; the traceback carries no header or body value.
    finally:
        # Also reached when the client hangs up: Starlette cancels the
        # streaming task, and inside a cancelled scope an await is skipped
        # unless shielded — the upstream connection would otherwise stay open
        # (and the upstream generating) until its read timeout.
        with anyio.CancelScope(shield=True):
            await response.aclose()


class Upstream:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._base = httpx.URL(base_url)
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=TIMEOUT,
            # One Claude Code session opens a handful of connections; a cap
            # would only ever turn a burst into an untimed pool wait.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
            # A 3xx is the upstream's answer; the client follows it, not sous.
            follow_redirects=False,
            # No ~/.netrc, no HTTPS_PROXY: sous must never add a credential
            # the client did not send, and must never route the ones it did
            # send anywhere but the configured origin.
            trust_env=False,
        )
        # httpx's default headers (accept, accept-encoding, connection,
        # user-agent) are added to every request that did not carry them, so
        # the upstream would see a request described as httpx's rather than as
        # the client made it — including a gzip it never asked for, whose raw
        # relayed bytes it could not decompress. Nothing is lost by not asking:
        # a server does not compress for a client that sent no Accept-Encoding.
        self._client.headers.clear()
        # httpx has no "no cookies" switch: its jar stores every upstream
        # Set-Cookie (Cloudflare sets __cf_bm on Anthropic responses) and adds
        # a Cookie header to later requests — one the client never sent, and
        # shared by every local client this one daemon serves. An empty
        # allowed_domains policy is the documented way to make a jar refuse
        # everything; a Cookie the client did send still crosses as the
        # ordinary end-to-end header it is.
        self._client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))

    @property
    def host(self) -> str:
        return self._base.host

    async def forward(self, request: Request, body: bytes | None) -> Response:
        """Relay `request` to the upstream. `body` is the already-read body
        (the routes that had to look inside it) or None to stream the client's
        body through untouched. Never raises: every failure of the forwarding
        itself is an Anthropic-shaped error carrying Via."""
        content: bytes | AsyncIterator[bytes] | None
        if body is not None:
            content = body
        elif _has_body(request):
            content = request.stream()
        else:
            content = None
        try:
            upstream_request = self._client.build_request(
                request.method,
                self._base.copy_with(raw_path=_target_path(request)),
                headers=request_headers(request.headers.raw, drop_content_length=body is not None),
                content=content,
            )
            response = await self._client.send(upstream_request, stream=True)
        except httpx.InvalidURL, UnicodeDecodeError:
            # Building the URL is inside the try because neither of these is an
            # httpx.HTTPError, so both would escape the "never raises" contract.
            # Unreachable through uvicorn: h11 rejects a non-ASCII request
            # target with its own 400 before it ever becomes an ASGI scope.
            return _error(400, "malformed request target")
        except httpx.TimeoutException:
            return _error(504, f"upstream {self.host} timed out")
        except httpx.HTTPError as e:
            # ConnectError, RemoteProtocolError, ...: the class name says
            # which without quoting anything from the wire.
            return _error(502, f"could not reach upstream {self.host}: {type(e).__name__}")
        except ClientDisconnect:
            # The client left while its body was still streaming up. Nobody
            # reads this; it keeps the disconnect from being a traceback.
            return _error(400, "client disconnected mid-body")
        relayed = StreamingResponse(_relay(response), status_code=response.status_code)
        # Raw pairs, not a mapping: a Mapping would collapse repeated headers.
        relayed.raw_headers = response_headers(response.headers.raw)
        return relayed

    async def aclose(self) -> None:
        await self._client.aclose()
