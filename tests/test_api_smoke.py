import importlib.util
from pathlib import Path

from sous.api.convert import chat_tools


def _smoke():
    path = Path(__file__).resolve().parents[1] / "scripts" / "api_smoke.py"
    spec = importlib.util.spec_from_file_location("api_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_smoke_request_is_one_the_endpoint_serves_locally():
    body = _smoke().build_request()
    assert body["model"] == "sous-local"
    assert body["stream"] is True
    tools, dropped = chat_tools(body["tools"])
    assert len(tools) == 1
    assert not dropped
    assert body["messages"][0]["role"] == "user"
