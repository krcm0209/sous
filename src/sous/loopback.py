"""The Host/Origin/fetch-metadata check every custom route on the daemon's
loopback bind applies — the Host/Origin pair is what the MCP transport checks
on /mcp, which custom routes get none of; the fetch-metadata rule goes one
step further. Shared by the gateway's routes and the daemon's own /sous/
routes."""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.requests import Request

from sous.api.convert import RequestError

# Without the Host check a web page whose hostname re-resolves to 127.0.0.1
# could drive the local model. Same allow-list as the SDK's.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
# The SDK checks Origin as well, and Host alone does not cover what it covers:
# a cross-origin fetch with Content-Type: text/plain is a CORS simple request,
# so it skips the preflight and arrives with a perfectly legitimate loopback
# Host. The page cannot read the reply, but the turn it starts holds the
# gateway lock, the engine lock and a prompt-cache slot for a whole generation
# timeout — a drive-by DoS of the daemon. urlsplit unwraps the IPv6 brackets a
# netloc carries, so these are bare addresses. Any port passes: this rule is
# the fallback for a browser without fetch metadata, and the fetch-metadata
# rule below is the stricter of the two.
ALLOWED_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")
# A browser omits Origin on a cross-origin no-cors GET — an <iframe>, <script>
# or <img> src, a no-cors fetch — so such a request from a visited page
# carries a legitimate loopback Host and nothing the Origin rule can see; on
# GET /sous/events that one request buys a perpetual status stream. A browser
# with fetch metadata (every current one: Chrome 76, Firefox 90, Safari 16.4,
# Baseline since 2023; an older one sends none, and its no-cors GET still
# passes) sends Sec-Fetch-Site on every request, navigations and
# sub-resources alike, and a page's script cannot set it (a forbidden request
# header). Absent is every non-browser client — Claude Code's Bun binary,
# httpx and curl send no fetch metadata at all. `none` is a navigation the
# user started in the browser itself (the address bar, a bookmark, a link
# opened from another app) and is kept through redirects, so a link that
# redirects here opens one visible tab on the daemon — the cost of the user
# typing the URL. `same-origin` is a page the daemon itself served, and it
# serves none. `same-site` — a page on another loopback port — is refused
# too: the daemon has no browser client, and a local dev server runs a third
# party's script as readily as a remote page does. The value is a lowercase
# token by definition and is matched exactly.
ALLOWED_FETCH_SITES = ("none", "same-origin")


def check_loopback(request: Request) -> None:
    """Refuse (403, permission_error) a Host that is not loopback, an Origin
    that is not, or fetch metadata from a page that is not the daemon's own
    — a browser's, never Claude Code's or curl's."""
    host = request.headers.get("host", "").lower()
    # Strip the port; an IPv6 literal keeps its brackets.
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    else:
        host = host.split(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        raise RequestError(403, "permission_error", "loopback hosts only")
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ALLOWED_FETCH_SITES:
        raise RequestError(403, "permission_error", "same-origin pages only")
    # Absent (Claude Code, httpx, curl) passes untouched. `null` — a
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


# Every verb the gateway's catch-all forwards and the daemon's /sous/ 404
# answers: the two lists must stay one, or a verb would fall through one
# and reach the other.
ALL_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
