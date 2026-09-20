"""The Host/Origin/fetch-metadata guard shared by the gateway's routes and
the daemon's /sous/ routes: what it lets through and what it refuses."""

import pytest
from starlette.requests import Request

from sous.api.convert import RequestError
from sous.loopback import check_loopback


def _request(host: str | None, origin: str | None = None, site: str | None = None) -> Request:
    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    if site is not None:
        headers.append((b"sec-fetch-site", site.encode()))
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": headers, "query_string": b""}
    )


@pytest.mark.parametrize("host", ["127.0.0.1:8383", "localhost", "[::1]:8383", "LOCALHOST:1"])
def test_loopback_hosts_pass(host):
    check_loopback(_request(host))


@pytest.mark.parametrize("host", ["sous.example:8383", "10.0.0.5", "", None])
def test_other_hosts_are_refused(host):
    with pytest.raises(RequestError) as exc:
        check_loopback(_request(host))
    assert exc.value.status == 403 and exc.value.error_type == "permission_error"


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:5173", "http://localhost", "http://[::1]:9", None]
)
def test_loopback_and_absent_origins_pass(origin):
    check_loopback(_request("127.0.0.1", origin))


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://[::1", ""])
def test_foreign_and_malformed_origins_are_refused(origin):
    with pytest.raises(RequestError) as exc:
        check_loopback(_request("127.0.0.1", origin))
    assert exc.value.status == 403


@pytest.mark.parametrize("site", [None, "none", "same-origin"])
def test_absent_typed_and_same_origin_fetch_metadata_pass(site):
    """No fetch metadata is every non-browser client; `none` is a navigation
    the user started in the browser itself; `same-origin` is the daemon's own
    origin, the only page a browser may fetch from."""
    check_loopback(_request("127.0.0.1", site=site))


@pytest.mark.parametrize("site", ["cross-site", "same-site", "", "elsewhere", "Same-Origin"])
def test_other_fetch_metadata_is_refused(site):
    """A browser omits Origin on a no-cors GET — an iframe, script or img
    src, a no-cors fetch — but a current browser sends Sec-Fetch-Site on
    every request, and a page's script cannot set it. Another loopback port
    is another site's page: the daemon serves none of its own. The value is
    a lowercase token by definition: a cased one is refused, not folded."""
    with pytest.raises(RequestError) as exc:
        check_loopback(_request("127.0.0.1", site=site))
    assert exc.value.status == 403 and exc.value.error_type == "permission_error"
    assert "same-origin" in exc.value.message


def test_fetch_metadata_is_checked_whether_or_not_origin_is_sent():
    """A CORS request carries both headers; the Origin rule passing must not
    let a cross-site request with a loopback Origin (another local port's
    page) through, and a refused Origin stays refused with `none`."""
    with pytest.raises(RequestError):
        check_loopback(_request("127.0.0.1", "http://127.0.0.1:5173", "same-site"))
    with pytest.raises(RequestError):
        check_loopback(_request("127.0.0.1", "https://evil.example", "none"))


def test_the_allow_lists_are_exactly_the_loopback_names():
    from sous.loopback import ALLOWED_FETCH_SITES, ALLOWED_HOSTS, ALLOWED_ORIGIN_HOSTS

    assert set(ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}
    assert set(ALLOWED_ORIGIN_HOSTS) == {"127.0.0.1", "localhost", "::1"}
    assert set(ALLOWED_FETCH_SITES) == {"none", "same-origin"}
