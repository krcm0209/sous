"""The Host/Origin check every custom route on the daemon's loopback bind
applies — the pair the MCP transport checks on /mcp, which custom routes get
none of. Shared by the gateway's routes and the daemon's own /sous/ routes."""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.requests import Request

from sous.gateway.convert import RequestError

# Without the Host check a web page whose hostname re-resolves to 127.0.0.1
# could drive the local model. Same allow-list as the SDK's.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
# The SDK checks Origin as well, and Host alone does not cover what it covers:
# a cross-origin fetch with Content-Type: text/plain is a CORS simple request,
# so it skips the preflight and arrives with a perfectly legitimate loopback
# Host. The page cannot read the reply, but the turn it starts holds the
# gateway lock, the engine lock and a prompt-cache slot for a whole generation
# timeout — a drive-by DoS of the daemon. urlsplit unwraps the IPv6 brackets a
# netloc carries, so these are bare addresses.
ALLOWED_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")


def check_loopback(request: Request) -> None:
    """Refuse (403, permission_error) a Host that is not loopback or an
    Origin that is not — a browser's, never Claude Code's or curl's."""
    host = request.headers.get("host", "").lower()
    # Strip the port; an IPv6 literal keeps its brackets.
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    else:
        host = host.split(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        raise RequestError(403, "permission_error", "loopback hosts only")
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
    if hostname not in ALLOWED_ORIGIN_HOSTS:
        raise RequestError(403, "permission_error", "loopback origins only")
