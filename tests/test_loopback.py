"""The Host/Origin guard shared by the gateway's routes and the daemon's
/sous/ routes: what it lets through and what it refuses."""

import pytest
from starlette.requests import Request

from sous.gateway.convert import RequestError
from sous.loopback import check_loopback


def _request(host: str | None, origin: str | None = None) -> Request:
    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
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


def test_the_allow_lists_are_exactly_the_loopback_names():
    from sous.loopback import ALLOWED_HOSTS, ALLOWED_ORIGIN_HOSTS

    assert set(ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}
    assert set(ALLOWED_ORIGIN_HOSTS) == {"127.0.0.1", "localhost", "::1"}
