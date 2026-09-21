"""The frame conventions the daemon's two event streams share — the
endpoint's `POST /v1/messages` and the monitor's `GET /sous/events` — so the
two are one shape on the wire: one separator, one keepalive interval, one
ping frame."""

from __future__ import annotations

from sse_starlette import ServerSentEvent

# Claude Code disconnects when nothing arrives for a while (oMLX saw it on
# 90k-token prefills). Both official SDKs drop `ping` events in the SSE
# iterator before their accumulator sees anything, so pings are safe anywhere
# in the stream — including before message_start, which is where the lock
# wait, the model load and the prefill all happen. The status stream keeps
# the same interval, for a client behind something that closes a silent
# connection.
PING_INTERVAL_SECONDS = 10

# A ServerSentEvent carries its own separator (the response-level `sep` only
# applies to dicts and strings), and its default is "\r\n"; every frame says
# "\n" so each stream is one canonical shape.
SEP = "\n"
PING = ServerSentEvent(event="ping", data='{"type": "ping"}', sep=SEP)
