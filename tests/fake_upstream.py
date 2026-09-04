"""A recording stand-in for api.anthropic.com behind httpx's in-process ASGI
transport. Shared by the forwarder's own tests and the routing tests; the
real-socket tests in test_gateway_http.py serve a Starlette app of their own
under uvicorn instead, because this transport buffers whole responses."""

from __future__ import annotations

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from sous.gateway.upstream import Upstream

METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


class FakeUpstream:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply: Response = Response(b'{"upstream": true}', media_type="application/json")
        self.app = Starlette(routes=[Route("/{path:path}", self.handle, methods=METHODS)])

    async def handle(self, request: Request) -> Response:
        self.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                # Raw pairs: order and repeats are part of what is under test.
                "headers": [(n.decode(), v.decode()) for n, v in request.headers.raw],
                "body": await request.body(),
            }
        )
        return self.reply

    def upstream(self, base_url: str = "https://upstream.test") -> Upstream:
        return Upstream(base_url, transport=httpx.ASGITransport(app=self.app))
