"""The gateway's HTTP surface: Anthropic-shaped routes on the daemon's app.

Never logs a request body or a header value. Never executes a tool: tool_use
blocks go back to Claude Code, whose permission system runs them (toolexec.py
is not in this path).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import sys
import threading
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from mcp.server import MCPServer
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from sous.config import SousConfig
from sous.engine.base import Delta, EngineManager, GenerationStalled
from sous.gateway.convert import (
    ChatRequest,
    RequestError,
    parse_count_tokens_request,
    parse_messages_request,
)
from sous.gateway.response import TurnAssembler, new_message_id
from sous.gateway.turn import (
    GatewayBusy,
    PromptTooLong,
    TurnAbandoned,
    TurnResult,
    TurnRunner,
)
from sous.protocol import ToolSet

# Anthropic's own request cap. The MCP transport's 4 MiB limit wraps only the
# /mcp handler; custom routes get nothing unless they enforce it themselves.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
# Claude Code disconnects when nothing arrives for a while (oMLX saw it on
# 90k-token prefills). Both official SDKs drop `ping` events in the SSE
# iterator before their accumulator sees anything, so pings are safe anywhere
# in the stream — including before message_start, which is where the lock
# wait, the model load and the prefill all happen.
PING_INTERVAL_SECONDS = 10

# A ServerSentEvent carries its own separator (the response-level `sep` only
# applies to dicts and strings), and its default is "\r\n"; every frame here
# says "\n" so the whole stream is one canonical shape.
_SEP = "\n"
_PING = ServerSentEvent(event="ping", data='{"type": "ping"}', sep=_SEP)

# Custom routes get none of the Host validation the /mcp transport applies to
# loopback binds; without it a web page whose hostname re-resolves to
# 127.0.0.1 could drive the local model. Same allow-list as the SDK's.
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
# The SDK checks Origin as well, and Host alone does not cover what it covers:
# a cross-origin fetch with Content-Type: text/plain is a CORS simple request,
# so it skips the preflight and arrives with a perfectly legitimate loopback
# Host. The page cannot read the reply, but the turn it starts holds the
# gateway lock, the engine lock and a prompt-cache slot for a whole generation
# timeout — a drive-by DoS of the daemon. urlsplit unwraps the IPv6 brackets a
# netloc carries, so these are bare addresses.
_ALLOWED_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")


def _log(message: str) -> None:
    print(f"sous gateway: {message}", file=sys.stderr, flush=True)


def _error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}}, status_code=status
    )


def _classify(exc: Exception) -> tuple[int, str, str]:
    """(status, error type, message) for a failure while turning."""
    if isinstance(exc, PromptTooLong):
        return 400, "invalid_request_error", str(exc)
    if isinstance(exc, GatewayBusy):
        return 529, "overloaded_error", str(exc)
    if isinstance(exc, GenerationStalled):
        return 500, "api_error", str(exc)
    return 500, "api_error", f"generation failed: {exc}"


async def _read_json(request: Request) -> object:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise RequestError(
            413, "request_too_large", f"request body exceeds {MAX_REQUEST_BYTES} bytes"
        )
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise RequestError(
                    413, "request_too_large", f"request body exceeds {MAX_REQUEST_BYTES} bytes"
                )
            chunks.append(chunk)
    except ClientDisconnect:
        # Nobody will read this response; it exists so the disconnect is not
        # a 500 with a traceback in the daemon log.
        raise RequestError(400, "invalid_request_error", "client disconnected mid-body") from None
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        raise RequestError(400, "invalid_request_error", "request body is not valid JSON") from None


def _check_loopback(request: Request) -> None:
    """Host and Origin, the pair the /mcp transport checks on a loopback bind."""
    host = request.headers.get("host", "").lower()
    # Strip the port; an IPv6 literal keeps its brackets.
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    else:
        host = host.split(":", 1)[0]
    if host not in _ALLOWED_HOSTS:
        raise RequestError(403, "permission_error", "the gateway serves loopback hosts only")
    # Only a browser sends Origin, and only a browser can be someone else's
    # page: absent (Claude Code, httpx, curl) passes untouched. `null` — a
    # sandboxed frame or a file:// page — has no hostname and is refused.
    origin = request.headers.get("origin")
    if origin is None:
        return
    try:
        # An unbalanced IPv6 bracket makes urlsplit raise instead of
        # returning; a malformed Origin is refused like any foreign one.
        hostname = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        hostname = ""
    if hostname not in _ALLOWED_ORIGIN_HOSTS:
        raise RequestError(403, "permission_error", "the gateway serves loopback origins only")


def _frame(event: dict) -> ServerSentEvent:
    return ServerSentEvent(
        event=event["type"], data=json.dumps(event, separators=(",", ":")), sep=_SEP
    )


class _QueueSink:
    """Relays a turn's progress from its threads onto the event loop's queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self._loop = loop
        self._queue = queue

    def put(self, item: tuple) -> None:
        # A closed loop means the daemon is shutting down: nobody is listening.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def started(self, input_tokens: int) -> None:
        self.put(("started", input_tokens))

    def delta(self, delta: Delta) -> None:
        self.put(("delta", delta))


class _NullSink:
    def started(self, input_tokens: int) -> None: ...

    def delta(self, delta: Delta) -> None: ...


class Gateway:
    def __init__(self, engines: EngineManager, config: SousConfig):
        self._config = config
        self._runner = TurnRunner(engines, config)
        # Turns get their own pool, never asyncio's default executor. A turn
        # drains to completion after its client is gone, so a shared pool would
        # mean one generation-long thread starving every asyncio.to_thread user
        # in the daemon; here the drain outlives the event loop harmlessly,
        # because asyncio.run's teardown joins only the DEFAULT executor (3.14
        # waits THREAD_JOIN_TIMEOUT = 300s there). In production that join is
        # not what bounds shutdown — uvicorn re-raises the captured SIGTERM
        # inside serve(), long before teardown (see GRACEFUL_SHUTDOWN_SECONDS
        # in server.py) — but off the main thread, which is how the real-server
        # test drives it, it is: the pool is what lets that test observe the
        # graceful bound instead of the drain.
        self._turns = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="sous-gateway-turn")

    async def hello(self, request: Request) -> Response:
        # Claude Code probes HEAD /api/hello at startup (gate 1, O3).
        try:
            _check_loopback(request)
        except RequestError as e:
            return JSONResponse(e.body(), status_code=e.status)
        return Response(status_code=200)

    async def count_tokens(self, request: Request) -> Response:
        try:
            _check_loopback(request)
            chat = parse_count_tokens_request(await _read_json(request))
            self._check_model(chat.model)
        except RequestError as e:
            return JSONResponse(e.body(), status_code=e.status)
        try:
            count = await asyncio.get_running_loop().run_in_executor(
                self._turns, self._runner.count_tokens, chat.messages, chat.tools
            )
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            return _error_response(*_classify(e))
        return JSONResponse({"input_tokens": count})

    async def messages(self, request: Request) -> Response:
        try:
            _check_loopback(request)
            chat = parse_messages_request(await _read_json(request))
            self._check_model(chat.model)
        except RequestError as e:
            _log(f"POST /v1/messages status={e.status} error={e.error_type}")
            return JSONResponse(e.body(), status_code=e.status)
        if chat.dropped_server_tools:
            _log(
                f"dropped {len(chat.dropped_server_tools)} server-side tool(s) not executable "
                f"locally: {', '.join(repr(t) for t in chat.dropped_server_tools)}"
            )
        assembler = TurnAssembler(
            new_message_id(), chat.model, ToolSet.from_tools(chat.tools, strict=False)
        )
        if chat.stream:
            return EventSourceResponse(
                self._stream(chat, assembler),
                ping=PING_INTERVAL_SECONDS,
                ping_message_factory=lambda: _PING,
                sep=_SEP,
            )
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._turns,
                self._runner.run,
                chat.messages,
                chat.tools,
                chat.max_tokens,
                _NullSink(),
            )
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            status, error_type, message = _classify(e)
            _log(
                f"POST /v1/messages model={chat.model} stream=0 status={status} error={error_type}"
            )
            return _error_response(status, error_type, message)
        assembler.start(result.input_tokens)
        assembler.finish(result.text, result.output_tokens, result.finish_reason)
        self._log_turn(chat, result, assembler, stream=False)
        return JSONResponse(assembler.message())

    def _check_model(self, model: str) -> None:
        # Phase 2 forwards other models upstream; until then they are simply
        # not served here, in the vocabulary the client already understands.
        if model not in self._config.gateway_local_models:
            raise RequestError(404, "not_found_error", f"model: {model}")

    async def _stream(
        self, chat: ChatRequest, assembler: TurnAssembler
    ) -> AsyncIterator[ServerSentEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sink = _QueueSink(loop, queue)
        abandoned = threading.Event()

        def turn() -> None:
            # Completion and failure both travel through the queue, so this
            # generator has exactly one thing to wait on — and a client that
            # disconnects cancels only the waiting, never a started turn: the
            # thread drains the generation to completion regardless.
            try:
                sink.put(
                    (
                        "done",
                        self._runner.run(
                            chat.messages, chat.tools, chat.max_tokens, sink, abandoned
                        ),
                    )
                )
            except TurnAbandoned:
                _log(f"POST /v1/messages model={chat.model} stream=1 abandoned while queued")
            except Exception as e:  # noqa: BLE001 — relayed as an in-band error event
                sink.put(("error", e))

        loop.run_in_executor(self._turns, turn)
        try:
            yield _PING
            while True:
                kind, value = await queue.get()
                if kind == "started":
                    for event in assembler.start(value):
                        yield _frame(event)
                elif kind == "delta":
                    for event in assembler.feed(value):
                        yield _frame(event)
                elif kind == "done":
                    for event in assembler.finish(
                        value.text, value.output_tokens, value.finish_reason
                    ):
                        yield _frame(event)
                    self._log_turn(chat, value, assembler, stream=True)
                    return
                else:
                    status, error_type, message = _classify(value)
                    _log(
                        f"POST /v1/messages model={chat.model} stream=1 status=200 "
                        f"error={error_type}"
                    )
                    yield _frame(
                        {"type": "error", "error": {"type": error_type, "message": message}}
                    )
                    return
        finally:
            # Reached on a normal finish, on the client disconnecting (the
            # CancelledError lands at queue.get) and on generator close: a turn
            # still waiting for the lock sees this and never starts.
            abandoned.set()

    def _log_turn(
        self, chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
    ) -> None:
        _log(
            f"POST /v1/messages model={chat.model} stream={int(stream)} status=200 "
            f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
            f"stop={assembler.stop_reason} cache={'hit' if result.cache_hit else 'miss'} "
            f"reused_tokens={result.reused_tokens} seconds={result.seconds:.1f}"
        )


def mount_gateway(mcp: MCPServer, engines: EngineManager, config: SousConfig) -> Gateway:
    """Register the Anthropic-compatible routes on the daemon's Starlette app.
    custom_route adds bare routes: no auth (loopback only, like /mcp), no
    body limit (enforced above), no DNS-rebinding check (the /mcp transport's
    settings do not reach here)."""
    # sse-starlette logs every frame it sends at DEBUG — the model's reply,
    # verbatim. The daemon runs at INFO, but the no-bodies-in-logs rule must not
    # depend on that: pin the library's logger above DEBUG where the frames are
    # made. Here rather than at import, because server.py imports this module
    # unconditionally and a disabled gateway must not reconfigure a logger.
    logging.getLogger("sse_starlette").setLevel(logging.INFO)
    gateway = Gateway(engines, config)
    mcp.custom_route("/v1/messages", methods=["POST"])(gateway.messages)
    mcp.custom_route("/v1/messages/count_tokens", methods=["POST"])(gateway.count_tokens)
    mcp.custom_route("/api/hello", methods=["GET", "HEAD"])(gateway.hello)
    return gateway
