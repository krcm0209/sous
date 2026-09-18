import json

from sous.tune.daemon import ready_for_tune


def _status(**engine):
    doc = {
        "engine": {"loaded": False, "loading": False, "holders": 0, **engine},
        "inflight": [],
        "queue": {"queued": 0, "running": 0},
    }
    return doc


def _request(status_doc, unload=(200, {"unloaded": True, "reason": None})):
    calls = []

    def request(port, method, path, json=None):
        calls.append((method, path))
        if path == "/sous/status":
            return 200, json_bytes(status_doc)
        if path == "/sous/unload":
            code, body = unload
            return code, json_bytes(body)
        raise AssertionError(path)

    return request, calls


def json_bytes(obj):
    return json.dumps(obj).encode()


def test_no_daemon_is_ready():
    r = ready_for_tune(8383, request=lambda *a, **k: (0, b""), port_open=lambda p: False)
    assert r.ready and "no daemon" in r.reason


def test_a_loaded_idle_daemon_is_asked_to_unload():
    request, calls = _request(_status(loaded=True))
    r = ready_for_tune(8383, request=request, port_open=lambda p: True)
    assert r.ready and "released" in r.reason
    assert calls == [("GET", "/sous/status"), ("POST", "/sous/unload")]


def test_a_held_model_a_running_task_and_a_turn_each_refuse_without_unloading():
    for doc, word in (
        (_status(loaded=True, holders=1), "sous claude"),
        ({**_status(loaded=True), "queue": {"queued": 0, "running": 1}}, "busy"),
        ({**_status(loaded=True), "inflight": [{"id": "msg_1"}]}, "busy"),
    ):
        request, calls = _request(doc)
        r = ready_for_tune(8383, request=request, port_open=lambda p: True)
        assert r.ready is False and word in r.reason
        assert calls == [("GET", "/sous/status")]


def test_a_refused_unload_reports_the_daemons_reason():
    request, _ = _request(
        _status(loading=True),
        unload=(
            409,
            {
                "type": "error",
                "error": {"type": "conflict_error", "message": "a generation is in flight"},
            },
        ),
    )
    r = ready_for_tune(8383, request=request, port_open=lambda p: True)
    assert r.ready is False and r.reason == "a generation is in flight"


def test_something_else_on_the_port_is_not_ready():
    r = ready_for_tune(8383, request=lambda *a, **k: (404, b"nope"), port_open=lambda p: True)
    assert r.ready is False and "404" in r.reason
