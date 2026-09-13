"""The gateway's HTTP surface: Anthropic-shaped routes on the daemon's app.

Never logs a request body or a header value. Never executes a tool: tool_use
blocks go back to Claude Code, whose permission system runs them (toolexec.py
is not in this path). Requests for any other model — and every path it has no
route for — are forwarded to [gateway].upstream_url by gateway/upstream.py,
byte for byte.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from mcp.server import MCPServer
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, RedirectResponse, Response

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
from sous.gateway.upstream import SynthesizedError, Upstream
from sous.protocol import ToolSet

# Anthropic's own request cap. The MCP transport's 4 MiB limit wraps only the
# /mcp handler; custom routes get nothing unless they enforce it themselves.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
# A schema a real client sends nests a handful of levels. Beyond this the
# body is hostile or broken either way, and — same reasoning as protocol.py's
# _MAX_ARGUMENT_DEPTH — a body that decodes here on this machine's stack
# would still blow up the chat template's tojson encoder re-serializing it on
# a smaller one. RecursionError is not a reliable "too deep" signal because
# the C recursion guard is stack-size dependent, not payload-size dependent.
MAX_BODY_DEPTH = 128
# Claude Code disconnects when nothing arrives for a while (oMLX saw it on
# 90k-token prefills). Both official SDKs drop `ping` events in the SSE
# iterator before their accumulator sees anything, so pings are safe anywhere
# in the stream — including before message_start, which is where the lock
# wait, the model load and the prefill all happen.
PING_INTERVAL_SECONDS = 10
# The turn pool's executor queue is unbounded: without this, a burst beyond
# _turns' worker count sits in that queue holding its parsed request while
# GatewayBusy's timeout has not even started — an untimed wait instead of a
# real 529. Bounds memory held by queued/draining requests and gives Claude
# Code's hybrid mode (a handful of subagents at once) a 529 both official
# SDKs already retry with backoff, instead of hanging until the client gives
# up. Counts turns running, queued on TurnRunner._lock, or draining after a
# disconnect (see Gateway._stream) — not just requests in the executor queue.
MAX_PENDING_TURNS = 8
# Same failure mode as MAX_PENDING_TURNS, one pool over: count_tokens' executor
# queue is unbounded too, and each queued count holds its parsed body (up to
# 32 MiB) while the 2 _counts workers serialize on the tokenizer. Without this,
# a burst of counts queues unboundedly instead of getting a timely 529.
MAX_PENDING_COUNTS = 8

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

# How much of the dropped-tool set the log names. Anthropic's identifiers are
# short and few, so these bounds only ever bite on a client sending junk.
_LOG_TYPES = 8
_LOG_TYPE_CHARS = 64

# Claude Code appends a bracketed beta suffix to a model id on some requests
# (`sous-local[1m]` for the 1M-context beta, seen live in the Phase 1 exit);
# the id inside is what routing matches. One suffix, at the very end, no
# nesting — anything else is a different id.
_MODEL_SUFFIX_RE = re.compile(r"\[[^\[\]]*\]$")
# Bounds for the two identifiers a forwarded request contributes to its log
# line. Both are client-controlled strings, so anything that is not a short,
# printable, space-free token is logged as "-".
_LOG_ID_CHARS = 64
_LOG_PATH_CHARS = 80

# Where the SDK mounts the MCP transport: `streamable_http_path`, whose default
# is "/mcp" in mcp/server/mcpserver/server.py's streamable_http_app (server.py
# calls it without overriding). It is an exact Route, so a trailing slash never
# matched it — Starlette's redirect_slashes used to answer /mcp/ with a 307,
# and the catch-all below now matches it instead. See Gateway.passthrough.
_MCP_PATH = "/mcp"


_logger = logging.getLogger("sous.gateway")


def _log(message: str, level: int = logging.INFO) -> None:
    """One metadata line per event, at a level that means something: INFO for
    a request served or forwarded, WARNING for one sous refused (a 4xx, a
    529, a client gone while queued, tools it had to drop), ERROR for a
    failure sous produced (a 5xx from an exception, an unreachable upstream).
    The root handler (sous.logs) adds the timestamp and the name."""
    _logger.log(level, message)


def _status_level(status: int) -> int:
    # 529 is back-pressure both official SDKs retry with backoff, not a fault.
    return logging.ERROR if status >= 500 and status != 529 else logging.WARNING


def _lcp_region(lcp: int, lo: int, hi: int) -> str:
    """Which part of the render a miss diverged in, given the two fork
    boundaries the probe verified: below the tools boundary the tool array
    changed, below the header the system text did, else the conversation.
    With one boundary known (a tool block under the fork floor, or a template
    that renders the tools inside the system block) nothing below it tells the
    two apart. `-` when the probe never ran (no fork budget) or there was no
    slot to compare with (an lcp of 0)."""
    if not hi or not lcp:
        return "-"
    if lo and lcp < lo:
        return "tools"
    if lcp < hi:
        return "system" if lo else "tools-or-system"
    return "conversation"


def _rate(count: int, seconds: float) -> str:
    return "-" if seconds <= 0 else f"{count / seconds:.1f}"


def _opt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _elapsed(since: float) -> str:
    return f" seconds={time.monotonic() - since:.1f}"


def _error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}}, status_code=status
    )


def _log_refused(path: str, e: RequestError, received: float) -> JSONResponse:
    """The shape shared by every RequestError caught before a chat request is
    parsed (no `model=` yet): one metadata line, one Anthropic-shaped body."""
    _log(
        f"POST {path} status={e.status} error={e.error_type}{_elapsed(received)}",
        level=_status_level(e.status),
    )
    return JSONResponse(e.body(), status_code=e.status)


def _log_token(value: object, limit: int) -> str:
    if (
        isinstance(value, str)
        and 0 < len(value) <= limit
        and value.isprintable()
        and not any(c.isspace() for c in value)
    ):
        return value
    return "-"


def _model_label(chat: ChatRequest) -> str:
    # Routing matches the suffix-stripped id against config, but chat.model
    # keeps the id exactly as the client sent it — client text, not config
    # text, since `_MODEL_SUFFIX_RE` accepts anything between the brackets.
    # Bound it like any other client-controlled string before it reaches a
    # log line; the response echo (TurnAssembler) keeps the raw id.
    return _log_token(chat.model, _LOG_ID_CHARS)


def _classify(exc: Exception) -> tuple[int, str, str]:
    """(status, error type, message) for a failure while turning."""
    if isinstance(exc, PromptTooLong):
        return 400, "invalid_request_error", str(exc)
    if isinstance(exc, GatewayBusy):
        return 529, "overloaded_error", str(exc)
    if isinstance(exc, GenerationStalled):
        return 500, "api_error", str(exc)
    return 500, "api_error", f"generation failed: {exc}"


def _reject_constant(name: str) -> None:
    # json.loads accepts NaN/Infinity/-Infinity by default; the real API does
    # not, and a body that is not JSON must take the 400 below, not a turn.
    raise ValueError(f"non-JSON constant {name}")


def _finite_float(text: str) -> float:
    # An overflowing literal (1e999) is an ordinary number to the scanner and
    # becomes inf — the same non-JSON value by another spelling.
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {text}")
    return value


def _depth_exceeds(value: object, limit: int) -> bool:
    """Iterative depth walk over nested dicts/lists — never recursive, or
    this would reintroduce the exact stack-size dependence the cap exists to
    remove. Kept local to routes.py rather than shared with protocol.py's
    _check_depth: two tiny iterative loops are cheaper than a cross-module
    dependency between the gateway and the protocol parser. `value` itself is
    depth 1 when it is a container; a scalar never adds depth."""
    worklist: list[tuple[object, int]] = [(value, 1)]
    while worklist:
        obj, depth = worklist.pop()
        if isinstance(obj, dict):
            if depth > limit:
                return True
            worklist.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if depth > limit:
                return True
            worklist.extend((v, depth + 1) for v in obj)
    return False


async def _read_body(request: Request) -> bytes:
    declared = request.headers.get("content-length", "")
    # isdecimal, not isdigit: "²" is a digit to str but not to int(), and a
    # 5000-digit value trips int()'s digit limit — either would be a bare
    # ValueError here, outside the shaped-error handling. Anything longer
    # than the cap's own digit count is oversized without converting it.
    if declared.isdecimal() and (
        len(declared) > len(str(MAX_REQUEST_BYTES)) or int(declared) > MAX_REQUEST_BYTES
    ):
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
    return b"".join(chunks)


def _decode_json(raw: bytes) -> object:
    try:
        return json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
    except ValueError, RecursionError:
        # RecursionError is the backstop for a body deeper than the
        # interpreter's stack allows to even finish decoding — _check_depth
        # is the real contract for "too deep".
        raise RequestError(400, "invalid_request_error", "request body is not valid JSON") from None


def _check_depth(body: object) -> None:
    if _depth_exceeds(body, MAX_BODY_DEPTH):
        raise RequestError(
            400, "invalid_request_error", f"request body nests deeper than {MAX_BODY_DEPTH} levels"
        )


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

    # The queue is drained into SSE events the client is already reading —
    # a warm-cache failure after any delta cannot be replayed cold.
    replay_safe = False

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
    # Non-streaming: no byte of a delta ever reaches the client, so a
    # warm-cache failure may still be retried cold.
    replay_safe = True

    def started(self, input_tokens: int) -> None: ...

    def delta(self, delta: Delta) -> None: ...


class Gateway:
    def __init__(
        self, engines: EngineManager, config: SousConfig, upstream: Upstream | None = None
    ):
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
        # Tied to MAX_PENDING_TURNS rather than left at the default worker
        # count: the turn line's queue_s is only a complete accounting of the
        # wait when every admitted turn can start on its own worker at once —
        # otherwise a burst would queue inside the pool too, off the clock,
        # on whatever core count the host happens to have. max() guards a
        # test driving admission to zero (BoundedSemaphore(0) refuses every
        # turn outright): the pool itself still needs at least one thread.
        self._turns = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, MAX_PENDING_TURNS), thread_name_prefix="sous-gateway-turn"
        )
        # count_tokens never takes TurnRunner._lock, so it must never queue
        # behind a turn parked waiting on that lock inside a saturated _turns
        # pool. A dedicated pool also keeps it off asyncio's default executor
        # — a count can load the model, seconds of work on a large prompt.
        self._counts = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sous-gateway-count"
        )
        # Read here, not at class-definition time, so tests can monkeypatch
        # the module constant before building the app.
        self._pending = threading.BoundedSemaphore(MAX_PENDING_TURNS)
        self._pending_counts = threading.BoundedSemaphore(MAX_PENDING_COUNTS)
        self._upstream = upstream or Upstream(config.gateway_upstream_url)
        # The event loop holds its tasks weakly; these log a turn whose client
        # left mid-stream (_stream), and must outlive the request that made them.
        self._drains: set[asyncio.Task] = set()

    def close(self, timeout: float = 2.0) -> None:
        """Best-effort shutdown, called from the app's lifespan hook. Never
        waits on a draining generation: a turn in flight keeps the runner's
        lock, so its session and thread are left alone here (cancel_futures
        drops only work that never started; the running turn's own future
        finishes on its own, off a daemon thread, after the process is asked
        to exit) — the running turn drops the session itself when it
        finishes, once TurnRunner.close has marked the runner as closing.
        """
        closed = self._runner.close(timeout)
        self._turns.shutdown(wait=False, cancel_futures=True)
        self._counts.shutdown(wait=False, cancel_futures=True)
        _log("closed" if closed else "close deferred: turn in progress")

    async def aclose(self) -> None:
        """close() plus the upstream client; what the app's lifespan awaits."""
        self.close()
        await self._upstream.aclose()

    async def passthrough(self, request: Request) -> Response:
        """Everything the gateway has no route of its own for — /api/hello,
        /api/oauth/usage, event logging, whatever Claude Code adds next —
        streams through to the upstream untouched."""
        try:
            _check_loopback(request)
        except RequestError as e:
            return JSONResponse(e.body(), status_code=e.status)
        # /mcp/ is the MCP transport's path, not the upstream's: forwarding it
        # would put an MCP client's JSON-RPC body on the wire to Anthropic.
        # Give back exactly what Starlette's redirect_slashes gave before this
        # catch-all shadowed it — a 307, which preserves the method and body.
        path = request.url.path
        if path != _MCP_PATH and path.rstrip("/") == _MCP_PATH:
            query = request.url.query
            return RedirectResponse(f"{_MCP_PATH}?{query}" if query else _MCP_PATH, status_code=307)
        return await self._forward(request, None, "-")

    def _route(self, raw: bytes) -> tuple[object | None, str]:
        """(decoded body, model) when the body names a model served here;
        (None, model-for-the-log) when it is the upstream's — including a body
        that does not decode or names no model: those are the upstream's to
        judge, in its own words. Only a claimed body is ever re-serialized, so
        only a claimed body gets the depth check (in the caller)."""
        try:
            body = _decode_json(raw)
        except RequestError:
            return None, "-"
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str):
            return None, "-"
        # Exact first, then suffix-stripped: a configured id may itself end in
        # brackets, and stripping alone would make it permanently unroutable.
        local = self._config.gateway_local_models
        if model in local or _MODEL_SUFFIX_RE.sub("", model, count=1) in local:
            return body, model
        return None, _log_token(model, _LOG_ID_CHARS)

    async def _forward(self, request: Request, body: bytes | None, model: str) -> Response:
        started = time.monotonic()
        response = await self._upstream.forward(request, body)
        # `seconds` is time to the upstream's headers — for a stream, the body
        # is still flowing when this line is written.
        _log(
            f"upstream {request.method} {_log_token(request.url.path, _LOG_PATH_CHARS)} "
            f"model={model} status={response.status_code} "
            f"seconds={time.monotonic() - started:.1f}",
            level=_status_level(response.status_code)
            if isinstance(response, SynthesizedError)
            else logging.INFO,
        )
        return response

    async def count_tokens(self, request: Request) -> Response:
        received = time.monotonic()
        try:
            _check_loopback(request)
            raw = await _read_body(request)
        except RequestError as e:
            return _log_refused("/v1/messages/count_tokens", e, received)
        body, model = self._route(raw)
        if body is None:
            return await self._forward(request, raw, model)
        try:
            _check_depth(body)
            chat = parse_count_tokens_request(body)
        except RequestError as e:
            return _log_refused("/v1/messages/count_tokens", e, received)
        # Same bounded-admission reasoning as messages()'s self._pending: a
        # queued count still holds its parsed body while the 2 _counts
        # workers serialize on the tokenizer, so bound it before any bytes
        # are touched rather than let the executor queue grow unboundedly.
        if not self._pending_counts.acquire(blocking=False):
            _log(
                f"POST /v1/messages/count_tokens model={_model_label(chat)} "
                f"status=529 error=overloaded_error{_elapsed(received)}",
                level=logging.WARNING,
            )
            return _error_response(529, "overloaded_error", "too many token counts queued")
        try:
            result = await self._submit(
                self._counts,
                self._pending_counts,
                self._runner.count_tokens,
                chat.messages,
                chat.tools,
            )
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            status, error_type, message = _classify(e)
            _log(
                f"POST /v1/messages/count_tokens model={_model_label(chat)} status={status} "
                f"error={error_type}{_elapsed(received)}",
                level=_status_level(status),
            )
            return _error_response(status, error_type, message)
        _log(
            f"POST /v1/messages/count_tokens model={_model_label(chat)} "
            f"input_tokens={result.count} load_s={result.load_seconds:.1f} "
            f"count_s={result.seconds:.1f}{_elapsed(received)}"
        )
        return JSONResponse({"input_tokens": result.count})

    async def messages(self, request: Request) -> Response:
        received = time.monotonic()
        try:
            _check_loopback(request)
            raw = await _read_body(request)
        except RequestError as e:
            return _log_refused("/v1/messages", e, received)
        body, model = self._route(raw)
        if body is None:
            return await self._forward(request, raw, model)
        try:
            _check_depth(body)
            chat = parse_messages_request(body)
        except RequestError as e:
            return _log_refused("/v1/messages", e, received)
        if chat.dropped_tool_types:
            # Anthropic's type identifiers are a bounded, useful set, but a
            # client can put any string there, so cap the line: repr escapes a
            # forged newline, the slice bounds a padded one.
            shown = [repr(t[:_LOG_TYPE_CHARS]) for t in chat.dropped_tool_types[:_LOG_TYPES]]
            hidden = len(chat.dropped_tool_types) - len(shown)
            if hidden:
                shown.append(f"… (+{hidden} more)")
            _log(
                f"dropped {len(chat.dropped_tool_types)} tool(s) with no client-supplied "
                f"schema (Anthropic server-side or built-in): {', '.join(shown)}",
                level=logging.WARNING,
            )
        assembler = TurnAssembler(
            new_message_id(), chat.model, ToolSet.from_tools(chat.tools, strict=False)
        )
        # Acquired as late as possible, right before either branch submits
        # work, so almost nothing sits between acquire and dispatch — and
        # what does is guarded below, so a failure there still releases.
        if not self._pending.acquire(blocking=False):
            _log(
                f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} "
                f"stream={int(chat.stream)} status=529 error=overloaded_error{_elapsed(received)}",
                level=logging.WARNING,
            )
            return _error_response(529, "overloaded_error", "too many turns queued")
        if chat.stream:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sink = _QueueSink(loop, queue)
            abandoned = threading.Event()

            def turn() -> None:
                # Completion and failure both travel through the queue, so
                # the consumer below has exactly one thing to wait on — and a
                # client that disconnects cancels only that wait, never a
                # started turn: the thread drains the generation to
                # completion regardless. Submitted here, before the response
                # is even constructed, so the slot's lifetime is tied to this
                # turn running to completion — never to whether sse-starlette
                # ever iterates the generator's body. A client gone before
                # the first iteration still gets its turn run to completion
                # (nothing sets `abandoned` for it, so this is the same
                # drain-to-completion rule as a disconnect after streaming
                # starts, now bounded by MAX_PENDING_TURNS either way) and
                # the slot is released by the done-callback regardless.
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
                    _log(
                        f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} "
                        f"stream=1 status=499 error=abandoned{_elapsed(received)}",
                        level=logging.WARNING,
                    )
                    # Ends the drain _stream left waiting for this turn's outcome.
                    sink.put(("abandoned", None))
                except Exception as e:  # noqa: BLE001 — relayed as an in-band error event
                    sink.put(("error", e))

            self._submit(self._turns, self._pending, turn)
            return EventSourceResponse(
                self._stream(chat, assembler, queue, abandoned, received),
                ping=PING_INTERVAL_SECONDS,
                ping_message_factory=lambda: _PING,
                sep=_SEP,
            )
        future = self._submit(
            self._turns,
            self._pending,
            self._runner.run,
            chat.messages,
            chat.tools,
            chat.max_tokens,
            _NullSink(),
        )
        try:
            result = await future
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            status, error_type, message = _classify(e)
            _log(
                f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} "
                f"stream=0 status={status} error={error_type}{_elapsed(received)}",
                level=_status_level(status),
            )
            return _error_response(status, error_type, message)
        assembler.start(result.input_tokens)
        assembler.finish(
            result.text, result.output_tokens, result.finish_reason, result.reused_tokens
        )
        self._log_turn(chat, result, assembler, stream=False)
        return JSONResponse(assembler.message())

    def _submit(self, executor, slots, fn, *args) -> asyncio.Future:
        """Submit a job to `executor` and tie `slots`' release to the
        concurrent.futures.Future's completion — the job's REAL completion —
        not to the asyncio wrapper run_in_executor hands back, and not to
        anything further downstream (an awaited result, an SSE body
        sse-starlette may never iterate). Assumes the caller already holds
        the slot being released — acquire happens once before this is
        called, in messages() for a turn and in count_tokens() for a count.

        The asyncio wrapper is wrong to hang the callback off: asyncio.run_in_
        executor's future is `cf`-shaped only via asyncio.wrap_future, and if
        the caller instead cancels the *request task* awaiting that future,
        the wrapper future is marked done (cancelled) immediately — while the
        submitted job keeps running on its executor thread, because cancelling
        a wrapper never stops a running executor job. The done-callback would
        then fire, and the slot would be released while the turn is still
        draining, bypassing MAX_PENDING_TURNS for as long as cancelled
        requests keep arriving.

        Submitting directly to the concurrent.futures.Future and wrapping it
        ourselves fixes that: `cf` only completes when the executor job itself
        finishes, or when `cf.cancel()` succeeds — which is possible only
        while the job has not started (still queued in the pool). asyncio.
        wrap_future chains cancellation downward (cancelling the wrapper calls
        `cf.cancel()`), so a non-streaming request cancelled while its turn is
        still queued cancels the job outright — it never runs, its
        done-callback fires on that cancellation, and the slot is released
        correctly. A request cancelled while its turn is already running
        leaves the job draining, and the slot is held until the drain ends.
        Both are the intended semantics.

        `cf.add_done_callback` runs the callback on the completing worker
        thread (or the cancelling thread, for a queued-and-cancelled job) —
        threading.BoundedSemaphore.release is safe from either.
        """
        try:
            cf = executor.submit(fn, *args)
        except Exception:
            # Submission itself failed (e.g. the pool refused new work): no
            # future exists to release the slot on completion, so release it
            # here instead of leaking it.
            slots.release()
            raise
        cf.add_done_callback(lambda _f: slots.release())
        return asyncio.wrap_future(cf)

    async def _stream(
        self,
        chat: ChatRequest,
        assembler: TurnAssembler,
        queue: asyncio.Queue,
        abandoned: threading.Event,
        received: float,
    ) -> AsyncIterator[ServerSentEvent]:
        # Pure consumer: the turn was already submitted (and the pending slot
        # already tied to its future) by messages() before this generator was
        # even constructed — see the comment at that submission site.
        outcome_seen = False
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
                    outcome_seen = True
                    events = assembler.finish(
                        value.text, value.output_tokens, value.finish_reason, value.reused_tokens
                    )
                    # Before the last frames, so a client that hangs up on
                    # them still leaves its turn on record.
                    self._log_turn(chat, value, assembler, stream=True)
                    for event in events:
                        yield _frame(event)
                    return
                else:
                    outcome_seen = True
                    status, error_type, message = _classify(value)
                    self._log_stream_failure(chat, assembler, value, received)
                    yield _frame(
                        {"type": "error", "error": {"type": error_type, "message": message}}
                    )
                    return
        finally:
            # Reached on a normal finish, on the client disconnecting (the
            # CancelledError lands at queue.get) and on generator close: a turn
            # still waiting for the lock sees this and never starts.
            abandoned.set()
            # A turn that already started drains to completion with nobody
            # reading its queue; without a reader its outcome never reaches
            # the log, and the next turn's queue_s waits on something unnamed.
            # (No running loop means the daemon is going down: nobody to log.)
            if not outcome_seen:
                with contextlib.suppress(RuntimeError):
                    drain = asyncio.get_running_loop().create_task(
                        self._log_when_drained(chat, assembler, queue, received)
                    )
                    self._drains.add(drain)
                    drain.add_done_callback(self._drains.discard)

    async def _log_when_drained(
        self, chat: ChatRequest, assembler: TurnAssembler, queue: asyncio.Queue, received: float
    ) -> None:
        """The line `_stream` would have written, for a client that left while
        its turn ran: the rest of the queue, consumed without a reader."""
        while True:
            kind, value = await queue.get()
            if kind == "done":
                assembler.finish(
                    value.text, value.output_tokens, value.finish_reason, value.reused_tokens
                )
                self._log_turn(chat, value, assembler, stream=True)
                return
            if kind == "error":
                self._log_stream_failure(chat, assembler, value, received)
                return
            if kind == "abandoned":
                return  # still queued when the client left: turn() logged its 499

    def _log_stream_failure(
        self, chat: ChatRequest, assembler: TurnAssembler, exc: Exception, received: float
    ) -> None:
        status, error_type, _ = _classify(exc)
        # status=200 because the SSE headers already went out; the level
        # follows the failure itself.
        _log(
            f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} stream=1 "
            f"status=200 error={error_type}{_elapsed(received)}",
            level=_status_level(status),
        )

    def _log_turn(
        self, chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
    ) -> None:
        cache = "fork" if result.forked else "hit" if result.cache_hit else "miss"
        # A hit retried cold took no slot in the end: its took_len is 0.
        took = (
            "none"
            if cache == "miss" or not result.took_len
            else f"{'fork' if cache == 'fork' else 'turn'}@{result.took_len}"
        )
        lo, hi = result.bounds
        # A hit already says how much was reused; a miss says how far the
        # render agreed with the closest slot and in which region — the
        # number that tells a changed tool array from a changed system text.
        diag = ""
        if cache == "miss":
            bounds = ",".join(str(b) for b in (lo, hi) if b)
            diag = (
                f" lcp={result.lcp} lcp_region={_lcp_region(result.lcp, lo, hi)} bounds=[{bounds}]"
            )
        _log(
            f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} "
            f"stream={int(stream)} status=200 "
            f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
            f"stop={assembler.stop_reason} cache={cache} took={took} "
            f"reused_tokens={result.reused_tokens} prefilled_tokens={result.prefilled_tokens} "
            f"forks={result.forks} evicted={result.evictions} pressure={result.pressure_evictions} "
            f"load_s={result.load_seconds:.1f} queue_s={result.queue_seconds:.1f} "
            f"engine_wait_s={result.engine_wait_seconds:.1f} "
            f"tokenize_s={result.tokenize_seconds:.1f} ttft_s={_opt(result.ttft_seconds)} "
            f"prefill_s={result.prefill_seconds:.1f} decode_s={result.decode_seconds:.1f} "
            f"prefill_tps={_rate(result.prefilled_tokens, result.prefill_seconds)} "
            f"decode_tps={_rate(result.output_tokens, result.decode_seconds)} "
            f"seconds={result.seconds:.1f}{diag} "
            f"tools={chat.tools_hash or '-'} system={chat.system_hash or '-'}"
        )


def mount_gateway(
    mcp: MCPServer, engines: EngineManager, config: SousConfig, *, upstream: Upstream | None = None
) -> Gateway:
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
    # Same rule, one layer down: httpx logs "HTTP Request: <method> <full URL>"
    # at INFO — the upstream URL including its query string — and httpcore
    # logs response header values verbatim at DEBUG. Neither is hypothetical
    # at INFO: MCPServer.__init__ calls the SDK's configure_logging("INFO"),
    # which basicConfig's a stderr handler onto the root logger, so both reach
    # the daemon log unless pinned above where they say those things.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    gateway = Gateway(engines, config, upstream)
    mcp.custom_route("/v1/messages", methods=["POST"])(gateway.messages)
    mcp.custom_route("/v1/messages/count_tokens", methods=["POST"])(gateway.count_tokens)
    # Registered last and matched last: the SDK appends custom routes after
    # its /mcp mount, so this can never shadow the MCP transport — and a
    # method the two routes above do not take (GET /v1/messages) falls
    # through to here and gets the upstream's own answer for it.
    mcp.custom_route(
        "/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )(gateway.passthrough)
    return gateway
